"""MMA fragment-store residual epilogue — the ``matmul_add`` fold.

A matmul fused with a residual add (``out = a @ b + r`` — e.g. the Qwen3 down_proj + residual-add fusion) used to be
locked out of the tensor-core tier entirely: the mma path stores the accumulator *fragment* (``RegStore``), so the
scalar epilogue ``v = add(acc, r)`` had no accumulator SSA name to read, and the eligibility gate in
``tile/_atom.py`` rejected any accumulator-consuming epilogue (0 of 74 tuned variants on tensor cores; 29 us vs
8 us cuBLAS — ``plans/qwen3-embedding-layer0-tune-findings.md`` finding 3).

The fold (CUTLASS epilogue-functor pattern): each lane already knows the (row, col) of its 4 C-fragment elements,
so ``RegStore`` loads the residual at those same coordinates, adds in f32, and downconverts with the store.
``tile/_atom._is_foldable_residual_epilogue`` admits exactly this shape to the warp tier (single ``add``, residual
Load indexed identically to the Write); ``kernel/005_lower_atom_tile._scan_epilogue`` strips the scalar Load /
Assign and rides the residual on the ``RegStore``. The fold relies on the warp tier's pre-existing v1 SPLITK = 1
invariant (the ``Cond(K_s == 0)`` residual gate is scalar-tier-only — locked in by a test below).

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
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

from .conftest import requires_cuda


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


def _residual_matmul_graph(*, M: int, N: int, K: int, out_dtype: DataType = F16, epilogue_op: str = "add", res_index=None) -> Graph:
    """``c[i,j] = epilogue_op(sum_k a[i,k]*b[k,j], r[<res_index>])`` — the fused
    matmul + pointwise-epilogue LoopOp. Default ``res_index`` is the Write's
    own ``(i, j)`` (the foldable matmul_add); tests pass a transposed index to
    exercise the index-mismatch gate."""
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    r_idx = res_index if res_index is not None else (Var("i"), Var("j"))
    lop = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="r_v", input="r", index=tuple(r_idx)),
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Assign(name="v", op=ElementwiseImpl(epilogue_op), args=("acc", "r_v")),
                            Write(output="c", index=(Var("i"), Var("j")), value="v"),
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
    g.add_node(op=lop, inputs=["a", "b", "r"], output=Tensor("c", (M, N), dtype=out_dtype), node_id="c")
    g.inputs = ["a", "b", "r"]
    g.outputs = ["c"]
    return g


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


# --- gating (compile-only, GPU-less) ----------------------------------------------


def test_residual_epilogue_admits_warp_tier(monkeypatch, _sm120_target):
    """The matmul_add shape lowers on the mma.sync path with the residual folded
    into the fragment store and the scalar epilogue stripped (no reference to
    the undefined accumulator SSA name survives)."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    _pin_warp(monkeypatch)
    g = Pipeline.build(KERNEL_PASSES).run(_residual_matmul_graph(M=32, N=1024, K=3072))
    kop = g.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "mma.sync.aligned.m16n8k16" in src
    assert "+ __half2float(r[" in src, "residual must be folded into the fragment store"
    assert "float v = " not in src, "the scalar epilogue Assign must be stripped"


def test_epilogue_warp_rows_stay_splitk_one(monkeypatch, _sm120_target):
    """The fragment fold relies on the warp tier's v1 SPLITK = 1 invariant
    (``_enumerate_warp_matmul_impl`` hard-codes it — the ``Cond(K_s == 0)``
    residual gate of ``015_gate_splitk_residual`` is a scalar-tier-only shape):
    even a pinned SPLITK=2 keeps the warp row at SPLITK = 1."""
    _pin_warp(monkeypatch)
    monkeypatch.setenv("DEPLODOCK_SPLITK", "2")
    g = Pipeline.build(KERNEL_PASSES).run(_residual_matmul_graph(M=32, N=1024, K=3072))
    kop = g.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16"
    assert kop.knobs.get("SPLITK") == 1


def test_nonlinear_epilogue_still_gated(monkeypatch, _sm120_target):
    """``v = multiply(acc, r)`` is not the foldable shape — the kernel stays on
    the scalar register-tile path even with the atom pinned."""
    _pin_warp(monkeypatch)
    g = Pipeline.build(KERNEL_PASSES).run(_residual_matmul_graph(M=32, N=1024, K=3072, epilogue_op="multiply"))
    assert g.nodes["c"].op.knobs.get("MMA") != "mma_m16n8k16_f16"


def test_differently_indexed_residual_still_gated(monkeypatch, _sm120_target):
    """A residual loaded at a different index than the Write (transposed) can't
    reuse the fragment elements' (row, col) coordinates — stays scalar."""
    _pin_warp(monkeypatch)
    g = Pipeline.build(KERNEL_PASSES).run(_residual_matmul_graph(M=128, N=128, K=128, res_index=(Var("j"), Var("i"))))
    assert g.nodes["c"].op.knobs.get("MMA") != "mma_m16n8k16_f16"


# --- device ------------------------------------------------------------------------


# ``requires_cuda`` routes the device test onto the single ``cuda`` xdist group
# (one GPU — see ``test_matmul_mma.py``); the skipif adds the sm_80+ arch gate.
@requires_cuda
@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize(
    ("M", "N", "K", "FM", "out_dtype"),
    [
        (32, 1024, 3072, 1, F16),  # the Qwen3 down_proj + residual shape
        (128, 256, 128, 4, F16),  # FM>1: per-cell replication rewrites res_index per fragment cell
        (128, 256, 128, 4, F32),  # f32 output: residual still added in f32, no downconvert
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
    compiled = be.compile(_residual_matmul_graph(M=M, N=N, K=K, out_dtype=out_dtype))
    kop = compiled.nodes["c"].op
    assert kop.knobs.get("MMA") == "mma_m16n8k16_f16", "expected the warp-tier variant"
    out, _ = be.run(compiled, input_data={"a": a, "b": b, "r": r})

    expected = a.astype(np.float32) @ b.astype(np.float32) + r.astype(np.float32)
    result = out.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    assert diff < 1e-2, f"M={M} N={N} K={K} FM={FM} out={out_dtype.name} max-abs-err {diff}"
