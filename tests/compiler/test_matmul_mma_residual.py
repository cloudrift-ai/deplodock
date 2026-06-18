"""MMA fragment-store pointwise epilogue — the general fold, gated in the negative.

A matmul fused with a pointwise epilogue (``out = f(a @ b, ...)`` — residual adds, bias / scale broadcasts,
activations) used to be locked out of the tensor-core tier: the mma path stores the accumulator *fragment*
(``RegStore``), so scalar epilogue Assigns have no accumulator SSA name to read (0 of 74 tuned Qwen3 down_proj
variants on tensor cores; 29 us vs 8 us cuBLAS — ``plans/qwen3-embedding-layer0-tune-findings.md`` finding 3).

The fold (CUTLASS epilogue-visitor pattern): each lane knows the (row, col) of its 4 C-fragment elements, so
``RegStore`` evaluates the whole chain per element in f32 — leaf operands load at the element's own coordinates
(per-dim ``m``/``n``/``fixed`` roles at each buffer's own stride, so transposed / broadcast operands read
correctly) and the ops render via the scalar ``op_to_expr`` translation.

Eligibility is the NEGATIVE rule (``tile/_atom.classify_fragment_epilogue``): the backward slice from the Write
to the accumulator is foldable unless it contains an ineligible operation / dependency — accumulator consumed
inside a reduce loop, multiple accumulators, multiple / vector Writes, escaping slice values, non-Load leaves,
in-kernel-produced leaf buffers, unconvertible dtypes, ops without a rendering, or leaf index dims the lane
arithmetic can't reproduce (reduce axes, mixed cell axes). ``kernel/005_lower_atom_tile`` re-runs the same
classifier on the tile body, strips the scalar stmts, and rides the chain on the ``RegStore``. The fold relies on
the warp tier's pre-existing v1 SPLITK = 1 invariant (the ``Cond(K_s == 0)`` residual gate is scalar-tier-only —
locked in by a test below).

Harness mirrors ``test_matmul_mma.py`` (hand-built LoopOp with canonical ``a[i,k] / b[k,j]`` operand orientation);
the gating tests pin the sm_120 target so they run GPU-less like ``passes/test_use_tma_gates.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler import target as target_mod
from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign, Stmt
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

from .conftest import requires_cuda, requires_sm90


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 80


def _epilogue_graph(
    *, M: int, N: int, K: int, out_dtype: DataType = F16, epilogue: tuple[Stmt, ...], extra_inputs: dict | None = None
) -> Graph:
    """Matmul ``acc = sum_k a[i,k] * b[k,j]`` followed by the given post-reduce
    ``epilogue`` stmts (Loads / Assigns / the Write). ``extra_inputs`` adds
    operand tensors beyond the standard ``r`` (M, N) residual."""
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    lop = LoopOp(
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
                            *epilogue,
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("r", (M, N), dtype=F16), node_id="r")
    inputs = ["a", "b", "r"]
    for name, shape in (extra_inputs or {}).items():
        g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape, dtype=F16), node_id=name)
        inputs.append(name)
    g.add_node(op=lop, inputs=inputs, output=Tensor("c", (M, N), dtype=out_dtype), node_id="c")
    g.inputs = inputs
    g.outputs = ["c"]
    return g


def _residual_add(res_index=None) -> tuple[Stmt, ...]:
    """The matmul_add residual: ``c[i,j] = acc + r[<res_index>]``."""
    idx = tuple(res_index) if res_index is not None else (Var("i"), Var("j"))
    return (
        Load(name="r_v", input="r", index=idx),
        Assign(name="v", op=ElementwiseImpl("add"), args=("acc", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
    )


@pytest.fixture
def _sm120_target():
    """Pin the sm_120 codegen path so the warp-tier passes fire without a live device."""
    target_mod.set_target((12, 0))
    try:
        yield
    finally:
        target_mod.set_target(None)


def _pin_warp(monkeypatch, *, FM: int = 1) -> None:
    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", str(FM))
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")


def _compile(g: Graph):
    return Pipeline.build(KERNEL_PASSES).run(g).nodes["c"].op


def _render(g2_op, g: Graph | None = None) -> str:
    from deplodock.compiler.ir.kernel.render import render_kernelop

    # Tensors resolve from the kernel op's own populated I/O.
    tensors = {**g2_op.inputs, **g2_op.outputs}
    return render_kernelop(g2_op, tensors=tensors)


# --- admitted shapes (compile-only, GPU-less) --------------------------------------


def test_residual_epilogue_admits_warp_tier(monkeypatch, _sm120_target):
    """The matmul_add residual lowers on the mma.sync path with the chain folded
    into the fragment store and the scalar epilogue stripped."""
    _pin_warp(monkeypatch)
    kop = _compile(_epilogue_graph(M=32, N=1024, K=3072, epilogue=_residual_add()))
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    src = _render(kop)
    assert "mma.sync.aligned.m16n8k16" in src
    assert "__half2float(r[" in src, "residual leaf must be loaded in the fragment store"
    assert "float v = " not in src, "the scalar epilogue Assign must be stripped"


def test_pointwise_chain_with_broadcast_admits_warp_tier(monkeypatch, _sm120_target):
    """A multi-op chain with a column-broadcast leaf — ``c = relu(acc) * s[j] + r``
    — is pointwise per element, so it folds (relu → multiply → add, the scale
    loaded at the element's own column with no row motion)."""
    epilogue = (
        Load(name="s_v", input="s", index=(Var("j"),)),
        Load(name="r_v", input="r", index=(Var("i"), Var("j"))),
        Assign(name="t0", op=ElementwiseImpl("relu"), args=("acc",)),
        Assign(name="t1", op=ElementwiseImpl("multiply"), args=("t0", "s_v")),
        Assign(name="v", op=ElementwiseImpl("add"), args=("t1", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
    )
    _pin_warp(monkeypatch)
    kop = _compile(_epilogue_graph(M=32, N=1024, K=3072, epilogue=epilogue, extra_inputs={"s": (1024,)}))
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    src = _render(kop)
    assert "__half2float(s[" in src, "broadcast leaf must load in the fragment store"
    assert "fmaxf(0.0f" in src, "relu must render through op_to_expr"


def test_transposed_residual_admits_warp_tier(monkeypatch, _sm120_target):
    """A residual loaded transposed (``r[j, i]``) is still per-element
    addressable — the dim roles swap (n at dim 0, m at dim 1) and the offsets
    apply at the residual's own strides."""
    _pin_warp(monkeypatch)
    kop = _compile(_epilogue_graph(M=128, N=128, K=128, epilogue=_residual_add(res_index=(Var("j"), Var("i")))))
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"


def test_multiply_epilogue_admits_warp_tier(monkeypatch, _sm120_target):
    """``v = multiply(acc, r)`` is pointwise — admitted under the general rule
    (the old add-only whitelist rejected it). Nonlinearity only matters for
    SPLITK > 1, which the warp tier pins to 1."""
    epilogue = (
        Load(name="r_v", input="r", index=(Var("i"), Var("j"))),
        Assign(name="v", op=ElementwiseImpl("multiply"), args=("acc", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
    )
    _pin_warp(monkeypatch)
    kop = _compile(_epilogue_graph(M=32, N=1024, K=3072, epilogue=epilogue))
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"


def test_epilogue_warp_rows_stay_splitk_one(monkeypatch, _sm120_target):
    """The fragment fold relies on the warp tier's v1 SPLITK = 1 invariant
    (``_enumerate_warp_matmul_impl`` hard-codes it — the ``Cond(K_s == 0)``
    residual gate of ``015_gate_splitk_residual`` is a scalar-tier-only shape):
    even a pinned SPLITK=2 keeps the warp row at SPLITK = 1."""
    _pin_warp(monkeypatch)
    monkeypatch.setenv("DEPLODOCK_SPLITK", "2")
    kop = _compile(_epilogue_graph(M=32, N=1024, K=3072, epilogue=_residual_add()))
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    assert kop.knobs.get("SPLITK") == 1


# --- blocked dependencies (compile-only, GPU-less) ----------------------------------


def test_mixed_cell_axis_index_blocks_fold(monkeypatch, _sm120_target):
    """A leaf indexed by an expression mixing the output cell axes (``r2[i+j]``)
    has no per-element lane reproduction — stays scalar."""
    epilogue = (
        Load(name="r_v", input="r2", index=(BinaryExpr("+", Var("i"), Var("j")),)),
        Assign(name="v", op=ElementwiseImpl("add"), args=("acc", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
    )
    _pin_warp(monkeypatch)
    kop = _compile(_epilogue_graph(M=128, N=128, K=128, epilogue=epilogue, extra_inputs={"r2": (256,)}))
    assert kop.knobs.get("MMA") != "mma_m16n8k16_f16"


def test_multi_accumulator_epilogue_blocks_fold(monkeypatch, _sm120_target):
    """An epilogue combining two accumulators (two K loops → ``add(acc, acc2)``,
    the gated-MLP shape) needs a multi-fragment fold — not wired; stays scalar."""
    i, j = Axis("i", 128), Axis("j", 128)
    k1, k2 = Axis("k1", 128), Axis("k2", 128)
    lop = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k1,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k1"))),
                                    Load(name="b_v", input="b", index=(Var("k1"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Loop(
                                axis=k2,
                                body=(
                                    Load(name="a2_v", input="a", index=(Var("i"), Var("k2"))),
                                    Load(name="b2_v", input="b", index=(Var("k2"), Var("j"))),
                                    Assign(name="p2", op=ElementwiseImpl("multiply"), args=("a2_v", "b2_v")),
                                    Accum(name="acc2", value="p2"),
                                ),
                            ),
                            Assign(name="v", op=ElementwiseImpl("add"), args=("acc", "acc2")),
                            Write(output="c", index=(Var("i"), Var("j")), value="v"),
                        ),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (128, 128), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (128, 128), dtype=F16), node_id="b")
    g.add_node(op=lop, inputs=["a", "b"], output=Tensor("c", (128, 128), dtype=F16), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    _pin_warp(monkeypatch)
    kop = _compile(g)
    assert kop.knobs.get("MMA") != "mma_m16n8k16_f16"


def test_escaping_epilogue_value_blocks_fold(monkeypatch, _sm120_target):
    """A slice value consumed outside the epilogue (a second Write of the chain
    value) can't be stripped — stays scalar."""
    epilogue = (
        Load(name="r_v", input="r", index=(Var("i"), Var("j"))),
        Assign(name="v", op=ElementwiseImpl("add"), args=("acc", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
        Write(output="c2", index=(Var("i"), Var("j")), value="v"),
    )
    g = _epilogue_graph(M=128, N=128, K=128, epilogue=epilogue)
    g.add_node(op=InputOp(), inputs=[], output=Tensor("c2", (128, 128), dtype=F16), node_id="c2")
    _pin_warp(monkeypatch)
    kop = _compile(g)
    assert kop.knobs.get("MMA") != "mma_m16n8k16_f16"


# --- device ------------------------------------------------------------------------


# ``requires_cuda`` routes the device tests onto the single ``cuda`` xdist group
# (one GPU — see ``test_matmul_mma.py``); the skipif adds the sm_80+ arch gate.
@requires_cuda
@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize(
    ("M", "N", "K", "FM", "out_dtype"),
    [
        (32, 1024, 3072, 1, F16),  # the Qwen3 down_proj + residual shape
        (128, 256, 128, 4, F16),  # FM>1: per-cell replication rewrites the leaf indices per fragment cell
        (128, 256, 128, 4, F32),  # f32 output: chain still computes in f32, no downconvert
    ],
)
def test_residual_mma_matches_reference(M: int, N: int, K: int, FM: int, out_dtype: DataType, monkeypatch):
    """The fused matmul+residual on the mma.sync path matches the f32 reference."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    _pin_warp(monkeypatch, FM=FM)
    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    r = (np.random.randn(M, N) * 0.5).astype(np.float16)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(M=M, N=N, K=K, out_dtype=out_dtype, epilogue=_residual_add()))
    kop = compiled.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "expected the warp-tier variant"
    out, _ = be.run(compiled, input_data={"a": a, "b": b, "r": r})

    expected = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32)
    result = out.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"M={M} N={N} K={K} FM={FM} out={out_dtype.name} max-abs-err {diff}"


@requires_cuda
@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("FM", [1, 4])
def test_chain_epilogue_mma_matches_reference(FM: int, monkeypatch):
    """``c = relu(acc) * s[j] + r[i,j]`` — multi-op chain with a broadcast leaf
    and a residual, on the mma.sync path, matches the f32 reference (also at
    FM>1, where the per-cell replication offsets every leaf index)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    M, N, K = 128, 256, 128
    epilogue = (
        Load(name="s_v", input="s", index=(Var("j"),)),
        Load(name="r_v", input="r", index=(Var("i"), Var("j"))),
        Assign(name="t0", op=ElementwiseImpl("relu"), args=("acc",)),
        Assign(name="t1", op=ElementwiseImpl("multiply"), args=("t0", "s_v")),
        Assign(name="v", op=ElementwiseImpl("add"), args=("t1", "r_v")),
        Write(output="c", index=(Var("i"), Var("j")), value="v"),
    )
    _pin_warp(monkeypatch, FM=FM)
    np.random.seed(7)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    r = (np.random.randn(M, N) * 0.5).astype(np.float16)
    s = (np.random.randn(N) * 0.5).astype(np.float16)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(M=M, N=N, K=K, epilogue=epilogue, extra_inputs={"s": (N,)}))
    kop = compiled.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "expected the warp-tier variant"
    out, _ = be.run(compiled, input_data={"a": a, "b": b, "r": r, "s": s})

    acc = a.astype(np.float32) @ b.astype(np.float32)
    expected = np.maximum(acc, 0.0) * s.astype(np.float32)[None, :] + r.astype(np.float32)
    result = out.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"FM={FM} max-abs-err {diff}"


@requires_cuda
@requires_sm90
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
def test_transposed_residual_mma_matches_reference(monkeypatch):
    """``c = acc + r[j, i]`` — the swapped dim roles apply the row / col motion
    at the transposed operand's own strides."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    M = N = K = 128
    _pin_warp(monkeypatch, FM=4)
    np.random.seed(3)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    r = (np.random.randn(N, M) * 0.5).astype(np.float16)

    be = CudaBackend()
    compiled = be.compile(_epilogue_graph(M=M, N=N, K=K, epilogue=_residual_add(res_index=(Var("j"), Var("i")))))
    kop = compiled.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "expected the warp-tier variant"
    out, _ = be.run(compiled, input_data={"a": a, "b": b, "r": r})

    expected = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32).T
    result = out.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"max-abs-err {diff}"
