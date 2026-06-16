"""End-to-end MMA matmul correctness — M8 of
``plans/mma-fragment-factorization.md``.

Verifies the s16816 ``mma.sync`` F16 path produces correct output across
realistic shapes + both output dtypes (F32 accumulator-direct-store, F16
acc with __half2 store). Pins the warp-tier ``mma_m16n8k16_f16`` atom +
a multi-warp 128×128 tile (single-warp mma.sync is pruned — ldmatrix is
smem→register only) and compares against the f32 matmul reference within
fp16 tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline
from deplodock.compiler.pipeline.knob import is_warp

from .conftest import dyn_M, requires_cuda

# Route every test in this module to the single ``cuda`` xdist_group
# (``tests/conftest.py::_is_cuda_item`` detects the ``"CUDA not available"``
# skipif reason) so they run sequentially on one worker — the host has one
# GPU and scattering CUDA tests across workers exhausts the device context.
# The per-test ``_supports_mma_sync`` skipif still gates the sm_80+ arch
# requirement on top of this.
pytestmark = [requires_cuda]


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    """The s16816 ``mma.sync.aligned.m16n8k16`` op + ``ldmatrix`` need
    sm_80+ (Ampere and later)."""
    if not _has_cuda():
        return False
    import cupy as cp

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 80


def _matmul_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    i = Axis("i", M)
    j = Axis("j", N)
    k = Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _matmul_graph(*, M: int, N: int, K: int, out_dtype: DataType) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(
        op=_matmul_loop_op(M=M, N=N, K=K),
        inputs=["a", "b"],
        output=Tensor("c", (M, N), dtype=out_dtype),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


def _np_dtype(dt: DataType):
    return {F16: np.float16, F32: np.float32}[dt]


# Default-on MMA matches the `MMA` knob's unset default (`mma_mode()`; per
# `plans/mma-fragment-factorization.md` post-M5). The pin is still set
# in tests so the fixture is robust against env-var clobbering during
# parallel pytest runs (xdist).
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize(
    ("M", "N", "K", "out_dtype"),
    [
        (128, 128, 128, F32),
        (256, 256, 128, F32),
        # F16-output cases — the per-lane __half2 epilogue downconverts the
        # fp32 accumulator into the narrower output buffer.
        (128, 128, 128, F16),
        (256, 256, 128, F16),
        # Skewed shape catches asymmetric N_b × WN × FN arithmetic — both dtypes.
        (128, 256, 128, F32),
        (128, 256, 128, F16),
    ],
)
def test_mma_matmul_matches_f32_reference(M: int, N: int, K: int, out_dtype: DataType, shape_mode, monkeypatch):
    """An F16×F16 matmul compiled via the mma.sync path agrees with the f32
    reference within fp16 tolerance — for both F16 and F32 output dtypes, and
    for both a STATIC M and a DYNAMIC (symbolic ``seq_len``) M run at the same
    runtime M (the masked-tile path; ``M ∈ {128, 256}`` are tile divisors).

    Pins the warp-tier ``mma_m16n8k16_f16`` atom + a multi-warp 128×128
    tile (``WM=2 WN=2 FM=4 FN=8 BK=2`` — N-tile = WN·FN·atom_n = 2·8·8 = 128,
    M-tile = WM·FM·atom_m = 2·4·16 = 128). Single-warp mma.sync is pruned
    (ldmatrix is smem→register only), so the pin forces the staged path."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")

    Mg = dyn_M(shape_mode, M)
    g = _matmul_graph(M=Mg, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "expected warp-tier mma.sync variant"

    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "ldmatrix.sync.aligned" in src, "mma.sync operands must load via ldmatrix"
    assert "wmma::" not in src, "the mma.sync path must not mix in legacy wmma intrinsics"
    # Accumulation is ALWAYS fp32 (matches cuBLAS / PyTorch fp16-GEMM
    # precision). For an F16 output buffer the per-lane epilogue downconverts
    # the fp32 accumulator via the vectorized ``__floats2half2_rn`` store.
    if out_dtype == F16:
        assert "__floats2half2_rn" in src, "f16 output needs the fp32→fp16 __half2 downconvert epilogue"
    else:
        assert "__floats2half2_rn" not in src

    # Launch through the real backend (``CudaBackend.run`` → ``run_program``)
    # rather than a hand-rolled ``RawModule`` call. The warp-tier picker may
    # land on a TMA-staged variant (signature ``…, const CUtensorMap* a_desc,
    # b_desc``); a 3-arg cubin launch then passes garbage descriptor pointers
    # and segfaults. The backend's ``_prebuild_descriptors`` binds every
    # ``CUtensorMap`` and opts into dynamic smem, so any geometry — TMA or
    # gmem-direct — launches correctly. The render+source asserts above still
    # cover the KERNEL-stage IR; this only fixes the execution path. The dynamic
    # graph resolves ``seq_len`` from the (M, K) input array shape.
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_matmul_graph(M=Mg, N=N, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32)
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    # F32 acc: tight tolerance (≤1e-2). F16 acc: looser (~ K · max-prod-mag · f16-eps);
    # for our 0.1-scale inputs and K ≤ 256, ~5e-2 covers the worst-case drift.
    tol = 5e-2 if out_dtype == F16 else 1e-2
    assert diff < tol, f"M={M} N={N} K={K} out={out_dtype.name} max-abs-err {diff}"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_mma_default_on_picks_warp_variant(monkeypatch):
    """With DEPLODOCK_MMA defaulted ON (the MMA knob reads default True),
    a no-env run of an MMA-eligible multi-warp F16 matmul still emits the
    warp variant. Guards against accidental default flips and confirms
    priority ordering (warp variants outrank scalar in the fork tree).
    mma.sync auto-enumerates on sm_90+; the 128² shape is multi-warp so it
    survives the single-warp prune."""
    monkeypatch.delenv("DEPLODOCK_MMA", raising=False)

    g = _matmul_graph(M=128, N=128, K=128, out_dtype=F16)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "MMA should be on by default"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_mma_disabled_falls_back_to_scalar(monkeypatch):
    """When DEPLODOCK_MMA=0 is set explicitly, the planner drops the
    warp-tier branch and emits only scalar register-tile variants."""
    monkeypatch.setenv("DEPLODOCK_MMA", "0")

    g = _matmul_graph(M=128, N=128, K=128, out_dtype=F16)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    # Scalar fallback: the leaf carries the MMA OFF sentinel ("0"), not a real
    # atom kind — ``is_warp`` is False (no warp-tier variant emitted).
    assert not is_warp(kop.knobs) and kop.knobs.get("MMA") == "0", "warp variant should not be emitted when DEPLODOCK_MMA=0"
    # Scalar path should stamp BN/BM.
    assert "BN" in kop.knobs and "BM" in kop.knobs


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_atom_cell_carries_through_staging(monkeypatch):
    """``tile/011_lower_atom_cell`` fuses the matmul compute into an ``Mma``
    right after ``partition_loops``; the ``Mma`` (carrying the atom kind +
    naming its A/B operand Loads by SSA value) rides the staging passes, and
    ``kernel/005_lower_atom_tile`` recovers each operand's role from the ``Mma``
    to emit the ``ldmatrix`` + ``mma.sync`` chain. Run the tile chain, confirm
    the ``Mma`` survives staging and still names its two operand Loads, then the
    full chain confirms the kernel still emits the s16816 instruction."""
    from deplodock.compiler.ir.kernel.render import render_kernelop
    from deplodock.compiler.ir.stmt import Load, Mma
    from deplodock.compiler.pipeline import TILE_PASSES

    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")

    # After the full tile chain (partition + lower-atom-cell + staging) the cell
    # is an Mma whose A/B names resolve to two (plain) Loads of the staged smem.
    g_tile = _matmul_graph(M=128, N=128, K=128, out_dtype=F32)
    g_tile = Pipeline.build(TILE_PASSES).run(g_tile)
    top = g_tile.nodes["c"].op
    mmas = [s for s in top.body.iter() if isinstance(s, Mma)]
    assert mmas, "the matmul compute should be an Mma carried through staging"
    assert all(m.atom.name == "mma_m16n8k16_f16" for m in mmas)
    # The Mma names its operands; each names a real Load (operand Loads are plain).
    load_names = {ld.names[0] for ld in top.body.iter() if isinstance(ld, Load)}
    for m in mmas:
        assert m.a in load_names and m.b in load_names and m.a != m.b

    # Full chain still produces the s16816 kernel.
    g = _matmul_graph(M=128, N=128, K=128, out_dtype=F32)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert not any(isinstance(s, Mma) for s in kop.body.iter()), "every Mma must be lowered before render"
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "mma.sync.aligned.m16n8k16" in src
