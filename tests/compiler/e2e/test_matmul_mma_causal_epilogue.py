"""Causal-mask (coord-predicated ``Select``) folded into the mma.sync fragment store.

The attention scores kernel applies a causal mask after the matmul:
``out[m,n] = (n <= m) ? acc*scale : mask_fill``. The mask is a ``Select`` whose
predicate compares the output coordinates — foldable into the fragment store
because each lane's 4 elements own known (row, col) of the C tile. Without the
fold the consumer drops to the scalar tier (the QK^T's binding blocker, see
``plans/qwen3-embedding-0.6b-layer0-low-performer-analysis.md`` Finding 1).

Uses a CANONICAL matmul (``b[k,n]``) to isolate the epilogue fold from the
transposed-B path; the real QK^T composes both (transposed-B split consumer +
this causal-mask fold).
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dtype import F16, F32, DataType
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign, Select, SelectBranch
from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline
from deplodock.compiler.pipeline.knob import mma_atom

from ..conftest import dyn_M, requires_cuda, requires_sm90

# ``requires_sm90`` skips below sm_90: this suite forces the mma.sync warp tier
# (ldmatrix), which is non-functional on sm_80-89 (host ldmatrix fault). It
# deploys / is validated on sm_90+.
pytestmark = [requires_cuda, requires_sm90]


def _supports_mma_sync() -> bool:
    try:
        import cupy as cp
    except Exception:  # noqa: BLE001
        return False
    if not cp.cuda.is_available():
        return False
    return int(cp.cuda.Device().compute_capability) >= 80


def _causal_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    """``out[i,j] = (j <= i) ? acc*scale : fill`` with ``acc = Σ_k a[i,k] b[k,j]``."""
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="sc", input="scale", index=(Literal(0, "int"),)),
                            Load(name="mz", input="mask_zero", index=(Literal(0, "int"),)),
                            Load(name="mf", input="mask_fill", index=(Literal(0, "int"),)),
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            # causal mask: keep (n<=m) → mask_zero(0); else mask_fill.
                            Select(
                                name="msk",
                                branches=(
                                    SelectBranch(value="mz", select=BinaryExpr("<=", Var("j"), Var("i"))),
                                    SelectBranch(value="mf", select=BinaryExpr(">", Var("j"), Var("i"))),
                                ),
                            ),
                            Assign(name="scaled", op=ElementwiseImpl("multiply"), args=("acc", "sc")),
                            Assign(name="out", op=ElementwiseImpl("add"), args=("msk", "scaled")),
                            Write(output="c", index=(Var("i"), Var("j")), value="out"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _causal_graph(*, M: int, N: int, K: int, out_dtype: DataType) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N), dtype=F16), node_id="b")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("scale", (1,), dtype=F32), node_id="scale")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("mask_zero", (1,), dtype=F32), node_id="mask_zero")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("mask_fill", (1,), dtype=F32), node_id="mask_fill")
    g.add_node(
        op=_causal_loop_op(M=M, N=N, K=K),
        inputs=["a", "b", "scale", "mask_zero", "mask_fill"],
        output=Tensor("c", (M, N), dtype=out_dtype),
        node_id="c",
    )
    g.inputs = ["a", "b", "scale", "mask_zero", "mask_fill"]
    g.outputs = ["c"]
    return g


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("out_dtype", [F32, F16])
@pytest.mark.parametrize("M", [128, 130])
def test_causal_mask_epilogue_mma(M: int, out_dtype: DataType, shape_mode, monkeypatch):
    """A matmul + causal-mask ``Select`` epilogue compiles to mma.sync and the
    folded per-element ternary matches the masked f32 reference — static M,
    dynamic (symbolic) M, and a straddling M=130 (the per-element guard + the
    per-element causal coords both active)."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")

    if shape_mode == "static" and M % 16 != 0:
        pytest.skip("static non-divisible M has no fixed MMA tile (masking is symbolic-axis only)")
    N, K = 128, 128
    Mg = dyn_M(shape_mode, M)
    g = _causal_graph(M=Mg, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16", "causal-mask matmul must reach the warp tier"
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "mma.sync.aligned.m16n8k16" in src
    assert "?" in src, "the causal mask must render as a per-element ternary"

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(3)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(K, N) * 0.1).astype(np.float16)
    scale = np.array([0.5], dtype=np.float32)
    mask_zero = np.array([0.0], dtype=np.float32)
    mask_fill = np.array([-100.0], dtype=np.float32)
    be = CudaBackend()
    g_run = be.compile(_causal_graph(M=Mg, N=N, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b, "scale": scale, "mask_zero": mask_zero, "mask_fill": mask_fill})

    scaled = (a.astype(np.float32) @ b.astype(np.float32)) * 0.5
    rows = np.arange(M)[:, None]
    cols = np.arange(N)[None, :]
    expected = np.where(cols <= rows, 0.0, -100.0) + scaled
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    tol = 5e-2 if out_dtype == F16 else 1e-2
    assert diff < tol, f"M={M} out={out_dtype.name} max-abs-err {diff}"
