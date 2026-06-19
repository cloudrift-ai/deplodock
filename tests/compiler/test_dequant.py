"""W4A16 (AWQ-INT4) weight-only quantization — numerics, codegen, and e2e.

Default suite is network- / GPU-free: the numerics oracle is checked against a
tiny synthetic packed fixture built in-test, and the integer/cast codegen is
asserted by rendering the leaf stmts directly. The ``compressed_tensors`` parity
assertion is an optional, package-gated test. The end-to-end run (toy
``DequantLinear`` compiled + executed vs eager) is CUDA-gated.

See ``plans/w4a16-quantization-support.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.ir.frontend.ir import (
    DequantLinearOp,
    QuantScheme,
    dequant_weight_numpy,
)

from .conftest import requires_cuda

# ---------------------------------------------------------------------------
# Synthetic packed fixture — the numerics oracle (no network, no GPU).
# ---------------------------------------------------------------------------


def _pack_along(values: np.ndarray, axis: int, per: int, num_bits: int) -> np.ndarray:
    """Pack ``per`` unsigned ``num_bits`` nibbles per int32 along ``axis``
    (the interleaved ``i::per`` layout): out[..., j, ...] holds element
    ``j``'s nibble at bit ``num_bits*(j % per)`` of lane ``j // per``."""
    n = values.shape[axis]
    packed_shape = list(values.shape)
    packed_shape[axis] = n // per
    out = np.zeros(packed_shape, dtype=np.int64)
    for j in range(n):
        src = np.take(values, j, axis=axis)
        dst_idx = [slice(None)] * values.ndim
        dst_idx[axis] = j // per
        out[tuple(dst_idx)] |= src.astype(np.int64) << (num_bits * (j % per))
    return out.astype(np.int32)


def make_fixture(out_f=32, in_f=128, group=32, num_bits=4, symmetric=False, seed=0):
    """Build a synthetic packed W4A16 linear + its expected fp16 weight."""
    rng = np.random.RandomState(seed)
    per = 32 // num_bits
    groups = in_f // group
    lo, hi = 0, 1 << num_bits

    nib = rng.randint(lo, hi, size=(out_f, in_f)).astype(np.int64)
    scale = (rng.rand(out_f, groups).astype(np.float32) * 0.05 + 0.01).astype(np.float16)
    weight_packed = _pack_along(nib, axis=1, per=per, num_bits=num_bits)

    if symmetric:
        weight_zero_point = None
        offset = 1 << (num_bits - 1)
        deq_int = nib - offset
    else:
        zp = rng.randint(lo, hi, size=(out_f, groups)).astype(np.int64)
        weight_zero_point = _pack_along(zp, axis=0, per=per, num_bits=num_bits)
        deq_int = nib - np.repeat(zp, group, axis=1)

    expected = (deq_int.astype(np.float32) * np.repeat(scale.astype(np.float32), group, axis=1)).astype(np.float16)
    scheme = QuantScheme(num_bits=num_bits, group_size=group, packed_dim=1, symmetric=symmetric)
    return dict(weight_packed=weight_packed, weight_scale=scale, weight_zero_point=weight_zero_point, scheme=scheme, expected=expected)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("group", [32, 128])
def test_dequant_weight_numpy_oracle(symmetric, group):
    fx = make_fixture(out_f=32, in_f=256, group=group, symmetric=symmetric, seed=1)
    got = dequant_weight_numpy(fx["weight_packed"], fx["weight_scale"], fx["weight_zero_point"], fx["scheme"])
    assert got.dtype == np.float16
    assert got.shape == fx["expected"].shape
    # Integer-exact unpack; the only error is fp16 scale rounding (already in
    # ``expected``), so this must match bit-for-bit.
    np.testing.assert_array_equal(got, fx["expected"])


def test_dequant_weight_per_tensor_group():
    """Per-tensor scale (group == in_features → one group)."""
    fx = make_fixture(out_f=16, in_f=64, group=64, symmetric=False, seed=2)
    got = dequant_weight_numpy(fx["weight_packed"], fx["weight_scale"], fx["weight_zero_point"], fx["scheme"])
    np.testing.assert_array_equal(got, fx["expected"])


def test_dequant_linear_op_forward_matches_dense_matmul():
    fx = make_fixture(out_f=24, in_f=96, group=32, symmetric=False, seed=3)
    x = (np.random.RandomState(4).randn(1, 5, 96) * 0.1).astype(np.float16)
    op = DequantLinearOp(has_bias=False, num_bits=4, group_size=32, packed_dim=1, symmetric=False)
    got = op.forward(x, fx["weight_packed"], fx["weight_scale"], fx["weight_zero_point"])
    ref = x.astype(np.float32) @ fx["expected"].astype(np.float32).T
    np.testing.assert_allclose(got.astype(np.float32), ref, rtol=2e-3, atol=2e-3)
    assert op.infer_output_shape([x.shape, fx["weight_packed"].shape]) == (1, 5, 24)


def test_torch_twin_matches_numpy_oracle():
    torch = pytest.importorskip("torch")
    from deplodock.compiler.trace.quantized import _dequant_weight_torch

    fx = make_fixture(out_f=32, in_f=128, group=32, symmetric=False, seed=5)
    got = dequant_weight_numpy(fx["weight_packed"], fx["weight_scale"], fx["weight_zero_point"], fx["scheme"])
    twin = _dequant_weight_torch(
        torch.from_numpy(fx["weight_packed"]),
        torch.from_numpy(fx["weight_scale"]),
        torch.from_numpy(fx["weight_zero_point"]),
        num_bits=4,
        group_size=32,
        symmetric=False,
    )
    np.testing.assert_array_equal(twin.numpy(), got)


def test_compressed_tensors_parity():
    """Optional, package-gated cross-check against the upstream dequant."""
    ct = pytest.importorskip("compressed_tensors")
    torch = pytest.importorskip("torch")
    try:
        from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
        from compressed_tensors.quantization.lifecycle.forward import dequantize
    except Exception:  # noqa: BLE001 — package layout varies across versions
        pytest.skip(f"compressed_tensors {ct.__version__} dequantize API not found")

    fx = make_fixture(out_f=32, in_f=128, group=32, symmetric=False, seed=6)
    # Reconstruct the unpacked (signed) weight + zp the way compressed_tensors
    # expects: it dequantizes from already-unpacked int tensors.
    per = 8
    nib = np.empty((32, 128), dtype=np.int64)
    wp = fx["weight_packed"].astype(np.int64)
    for i in range(per):
        nib[:, i::per] = (wp >> (4 * i)) & 0xF
    zp_unpacked = np.empty((32, 4), dtype=np.int64)
    zpp = fx["weight_zero_point"].astype(np.int64)
    for i in range(per):
        zp_unpacked[i::per, :] = (zpp >> (4 * i)) & 0xF

    args = QuantizationArgs(num_bits=4, group_size=32, symmetric=False, strategy=QuantizationStrategy.GROUP)
    ref = dequantize(
        x_q=torch.from_numpy(nib).to(torch.int8),
        scale=torch.from_numpy(fx["weight_scale"]),
        zero_point=torch.from_numpy(zp_unpacked).to(torch.int8),
        args=args,
    )
    got = dequant_weight_numpy(fx["weight_packed"], fx["weight_scale"], fx["weight_zero_point"], fx["scheme"])
    np.testing.assert_allclose(got.astype(np.float32), ref.numpy().astype(np.float32), rtol=2e-3, atol=2e-3)


# ---------------------------------------------------------------------------
# Codegen — integer unpack + int→fp16 cast render (no GPU).
# ---------------------------------------------------------------------------


def _render(stmt, ssa_dtypes=None, literal_ssa=None):
    from deplodock.compiler.ir.stmt.base import RenderCtx

    ctx = RenderCtx()
    ctx.ssa_dtypes.update(ssa_dtypes or {})
    # Mirror ``render_body``'s pre-pass: an inlined int constant is stamped
    # i32 at its use site (so a downstream shift takes the native-int path).
    for n, v in (literal_ssa or {}).items():
        ctx.literal_ssa[n] = v
        ctx.ssa_dtypes[n] = "i32" if isinstance(v, int) and not isinstance(v, bool) else "f32"
    return "\n".join(stmt.render(ctx))


def test_int_shift_and_mask_render_as_int():
    from deplodock.compiler.dtype import I32
    from deplodock.compiler.ir.stmt.leaves import Assign

    shift = _render(
        Assign(name="lane", op="right_shift", args=("packed", "c"), dtype=I32), ssa_dtypes={"packed": "i32"}, literal_ssa={"c": 4}
    )
    assert "int lane = packed >> 4;" in shift
    assert ".0f" not in shift  # never a float shift amount

    mask = _render(Assign(name="nib", op="bitwise_and", args=("lane", "m"), dtype=I32), ssa_dtypes={"lane": "i32"}, literal_ssa={"m": 15})
    assert "int nib = lane & 15;" in mask


def test_cast_renders_int2half_directly():
    from deplodock.compiler.dtype import F16
    from deplodock.compiler.ir.stmt.leaves import Assign

    out = _render(Assign(name="v", op="copy", args=("in0",), dtype=F16), ssa_dtypes={"in0": "i32"})
    assert "__half v = __int2half_rn(in0);" in out
    # Direct conversion — NOT the f32-promote detour.
    assert "__half2float" not in out
    assert "__float2half(__int2float_rn" not in out


def test_i32_select_renders_int():
    from deplodock.compiler.dtype import I32
    from deplodock.compiler.ir.expr import Literal, Var
    from deplodock.compiler.ir.stmt.leaves import Select, SelectBranch

    sel = Select(
        name="nib",
        branches=(SelectBranch("l0", Var("a0").lt(Literal(1, "int"))), SelectBranch("l1", Literal(1, "int"))),
        dtype=I32,
    )
    out = _render(sel, ssa_dtypes={"l0": "i32", "l1": "i32"})
    assert out.strip().startswith("int nib =")
    assert "(int)" in out and "float" not in out


def test_render_target_int_conversions():
    from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget

    t = CudaRenderTarget()
    assert t.type_name("i32") == "int"
    assert t.convert("x", "i32", "f16") == "__int2half_rn(x)"
    assert t.convert("x", "i32", "f32") == "__int2float_rn(x)"
    assert t.has_native_op("right_shift", "i32")
    assert t.has_native_op("subtract", "i32")


# ---------------------------------------------------------------------------
# Substitution walker (no GPU).
# ---------------------------------------------------------------------------


def test_substitution_replaces_targets_and_skips_ignore():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from deplodock.compiler.trace.quantized import apply_quant_substitution

    fx = make_fixture(out_f=16, in_f=64, group=32, symmetric=False, seed=7)

    def _quant_linear():
        m = nn.Module()
        m.register_buffer("weight_packed", torch.from_numpy(fx["weight_packed"]))
        m.register_buffer("weight_scale", torch.from_numpy(fx["weight_scale"]))
        m.register_buffer("weight_zero_point", torch.from_numpy(fx["weight_zero_point"]))
        m.bias = None
        return m

    model = nn.Module()
    model.config = type("Cfg", (), {})()
    model.config.quantization_config = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": {"num_bits": 4, "group_size": 32, "symmetric": False}}},
        "ignore": ["lm_head"],
    }
    model.proj = _quant_linear()
    model.lm_head = _quant_linear()

    n = apply_quant_substitution(model)
    assert n == 1  # proj substituted, lm_head ignored
    assert type(model.proj).__name__ == "DequantLinear"
    assert type(model.lm_head).__name__ != "DequantLinear"


def test_substitution_noop_for_unquantized_model():
    pytest.importorskip("torch")
    import torch.nn as nn

    from deplodock.compiler.trace.quantized import apply_quant_substitution

    model = nn.Module()
    model.config = type("Cfg", (), {})()
    model.lin = nn.Linear(8, 8)
    assert apply_quant_substitution(model) == 0


# ---------------------------------------------------------------------------
# End-to-end (CUDA-gated): trace → compile → run vs eager.
# ---------------------------------------------------------------------------


def _build_quant_module(fx, out_f):
    import torch
    import torch.nn as nn

    from deplodock.compiler.trace.quantized import _build_dequant_linear

    holder = nn.Module()
    holder.register_buffer("weight_packed", torch.from_numpy(fx["weight_packed"]))
    holder.register_buffer("weight_scale", torch.from_numpy(fx["weight_scale"]))
    zp = fx["weight_zero_point"]
    holder.register_buffer("weight_zero_point", torch.from_numpy(zp if zp is not None else fx["weight_packed"][: max(out_f // 8, 1)]))
    holder.bias = None
    return _build_dequant_linear(holder, fx["scheme"])


def _compile_and_run(mod, x):
    """Trace → compile → bind → run a DequantLinear; return (compiled, out, ref)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.loader.binder import bind_constants, declared_const_dtypes, source_array_at_dtype
    from deplodock.compiler.trace.torch import trace_module_with_constants

    ref = mod(x).detach().numpy().astype(np.float32)
    graph, _ = trace_module_with_constants(mod, (x,))
    compiled = CudaBackend().compile(graph)

    in_id = list(compiled.inputs)[0]
    input_data = {in_id: x.detach().cpu().numpy().astype(compiled.nodes[in_id].output.dtype.np, copy=False)}
    declared = declared_const_dtypes(compiled)
    src = {p: source_array_at_dtype(t, declared.get(p)) for p, t in mod.named_buffers(remove_duplicate=False) if t is not None}
    input_data.update(bind_constants(compiled, src))

    out = list(CudaBackend().run(compiled, input_data=input_data)[0].outputs.values())[-1]
    return compiled, np.asarray(out).reshape(ref.shape).astype(np.float32), ref


@requires_cuda
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("group", [32, 128])  # per-group — the granularities real AWQ models use
def test_dequant_linear_e2e_cuda(group, symmetric):
    import torch

    fx = make_fixture(out_f=64, in_f=256, group=group, symmetric=symmetric, seed=8)
    mod = _build_quant_module(fx, out_f=64)
    x = torch.randn(1, 8, 256, dtype=torch.float16)
    compiled, out, ref = _compile_and_run(mod, x)

    # The compiled kernel dequantizes in integer math — no float-shift bug.
    import re

    sources_cu = "\n".join(getattr(n.op, "kernel_source", "") or "" for n in compiled.nodes.values())
    assert ">> " in sources_cu and "& 15" in sources_cu
    assert not re.search(r">> [0-9]+\.[0-9]+f", sources_cu)

    # fp16-eps tolerance — the int round-trip is exact, error is only fp16
    # accumulation (the same bound a plain fp16 linear gets).
    np.testing.assert_allclose(out, ref, rtol=6e-3, atol=6e-3)


def _gmem_buffers(compiled):
    """``{node_id: (bytes, shape, dtype_name)}`` over a compiled graph's buffers."""
    out = {}
    for nid, node in compiled.nodes.items():
        t = node.output
        n = 1
        for d in t.shape:
            n *= d.as_static() if hasattr(d, "as_static") else int(d)
        shape = tuple(d.as_static() if hasattr(d, "as_static") else int(d) for d in t.shape)
        out[nid] = (n * t.dtype.nbytes, shape, t.dtype.name)
    return out


@requires_cuda
def test_dequant_fuses_no_materialized_weight_vram():
    """Phase 3: the dequant cone fuses into the matmul — gmem holds only packed
    int32 + scale + zp, never a materialized [out, in] fp16 weight slab."""
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module_with_constants

    out_f, in_f = 512, 512
    fx = make_fixture(out_f=out_f, in_f=in_f, group=32, symmetric=False, seed=9)
    qmod = _build_quant_module(fx, out_f=out_f)
    x = torch.randn(1, 16, in_f, dtype=torch.float16)
    qcompiled, qout, qref = _compile_and_run(qmod, x)
    np.testing.assert_allclose(qout, qref, rtol=6e-3, atol=6e-3)

    bufs = _gmem_buffers(qcompiled)
    # No materialized dense fp16 weight (the unpack intermediates never hit gmem).
    dense_shape = tuple(sorted((out_f, in_f)))
    materialized = [(nid, shape) for nid, (_, shape, dt) in bufs.items() if dt == "f16" and tuple(sorted(shape)) == dense_shape]
    assert materialized == [], f"unexpected materialized fp16 weight buffer(s): {materialized}"
    # The largest buffer is the packed int32 weight, 4x smaller than dense fp16.
    biggest = max(bufs.values())
    assert biggest[2] == "i32" and biggest[1] == (out_f, in_f // 8)

    # Total deplodock gmem footprint is far below a same-shape dense fp16 linear.
    import torch.nn as nn

    dcompiled = CudaBackend().compile(trace_module_with_constants(nn.Linear(in_f, out_f, bias=False).half(), (x,))[0])
    q_bytes = sum(b for b, _, _ in bufs.values())
    d_bytes = sum(b for b, _, _ in _gmem_buffers(dcompiled).values())
    assert q_bytes < d_bytes / 2.5, f"expected >2.5x gmem reduction, got {d_bytes / q_bytes:.2f}x ({q_bytes} vs {d_bytes})"


def test_per_channel_single_group_raises_clear_error():
    """Single-group quant (per-channel / per-tensor-along-K) is a documented
    Phase-2 gap; it must fail with a clear message, not a cryptic IR error."""
    pytest.importorskip("torch")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module_with_constants

    fx = make_fixture(out_f=16, in_f=32, group=32, symmetric=False, seed=3)  # group == in_features → 1 group
    mod = _build_quant_module(fx, out_f=16)
    graph, _ = trace_module_with_constants(mod, (torch.randn(1, 4, 32, dtype=torch.float16),))
    with pytest.raises(NotImplementedError, match="single-group"):
        CudaBackend().compile(graph)
