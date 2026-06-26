"""Transposed-B (Q@K^T) MMA matmul correctness.

The attention scores matmul ``C[m,n] = Σ_k A[m,k] · B[n,k]`` has BOTH operands
carrying the reduce axis K in their LAST index dim (``A[m,k]`` and ``B[n,k]`` —
B is N×K, the transpose of the canonical ``B[k,n]``). This is the native
``mma.row.col`` layout (B col-major K×N == physically ``[n,k]``), but the
operand classifier used to reject it (``b_load is None`` → scalar tier), so the
dominant attention kernel never reached the tensor cores. See
``plans/qwen3-embedding-0.6b-layer0-low-performer-analysis.md`` Finding 1.

Mirrors ``test_matmul_mma.py`` but with the transposed-B operand layout and an
``a @ b.T`` reference. Pins the warp-tier atom + a multi-warp tile.
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
from deplodock.compiler.pipeline.knob import mma_atom

from .conftest import dyn_M, requires_cuda, requires_sm90

# ``requires_sm90`` skips below sm_90: this suite forces the mma.sync warp tier
# (ldmatrix), which is non-functional on sm_80-89 (host ldmatrix fault). It
# deploys / is validated on sm_90+.
pytestmark = [requires_cuda, requires_sm90]


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

    cap = cp.cuda.Device().compute_capability
    return int(cap) >= 80


def _qkt_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    """``C[i,j] = Σ_k A[i,k] · B[j,k]`` — B indexed (j, k): K in the LAST dim."""
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
                                    Load(name="b_v", input="b", index=(Var("j"), Var("k"))),
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


def _qkt_graph(*, M: int, N: int, K: int, out_dtype: DataType) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (N, K), dtype=F16), node_id="b")
    g.add_node(
        op=_qkt_loop_op(M=M, N=N, K=K),
        inputs=["a", "b"],
        output=Tensor("c", (M, N), dtype=out_dtype),
        node_id="c",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["c"]
    return g


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize(
    ("M", "N", "K", "out_dtype"),
    [
        (128, 128, 128, F32),
        (256, 256, 128, F32),
        (128, 128, 128, F16),
        (128, 256, 128, F32),
        (128, 256, 128, F16),
    ],
)
def test_transposed_b_mma_matches_reference(M: int, N: int, K: int, out_dtype: DataType, shape_mode, monkeypatch):
    """A transposed-B (Q@K^T) F16×F16 matmul compiled via mma.sync agrees with
    the ``a @ b.T`` f32 reference, for STATIC and DYNAMIC (masked) M."""
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")

    Mg = dyn_M(shape_mode, M)
    g = _qkt_graph(M=Mg, N=N, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16", "transposed-B Q@K^T must reach the warp-tier mma.sync variant"

    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "mma.sync.aligned.m16n8k16" in src, "the s16816 mma.sync instruction must be emitted"
    assert "ldmatrix.sync.aligned" in src, "mma.sync operands must load via ldmatrix"

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(42)
    a = (np.random.randn(M, K) * 0.1).astype(np.float16)
    b = (np.random.randn(N, K) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_qkt_graph(M=Mg, N=N, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32).T
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    tol = 5e-2 if out_dtype == F16 else 1e-2
    assert diff < tol, f"M={M} N={N} K={K} out={out_dtype.name} max-abs-err {diff}"


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync.m16n8k16 needs CUDA + sm_80+")
@pytest.mark.parametrize("seq", [128, 130, 200])
@pytest.mark.parametrize("out_dtype", [F32, F16])
def test_transposed_b_mma_symbolic_mn(seq: int, out_dtype: DataType, monkeypatch):
    """The real Q@K^T case: BOTH M and N are the SAME symbolic ``seq_len`` (the
    scores matrix is [seq, seq]), with the reduce K static (head_dim). A
    straddling runtime ``seq`` (130, 200 — not a tile multiple) exercises the
    masked-tile clamps the divisor sizes above never trigger: A's staged-slab M
    clamp AND the transposed-B operand's gmem-direct ``dpl_mma_load_b_gmem_trans_nclamp``
    on the symbolic N. Compiled once (symbolic), run at each runtime seq."""
    from deplodock.compiler.dim import Dim
    from deplodock.compiler.ir.kernel.render import render_kernelop

    monkeypatch.setenv("DEPLODOCK_MMA", "mma_m16n8k16_f16")
    monkeypatch.setenv("DEPLODOCK_WM", "2")
    monkeypatch.setenv("DEPLODOCK_WN", "2")
    monkeypatch.setenv("DEPLODOCK_FM", "4")
    monkeypatch.setenv("DEPLODOCK_FN", "8")
    monkeypatch.setenv("DEPLODOCK_BK", "2")

    K = 128
    s = Dim("seq_len")  # one Dim instance → M and N are the same symbol
    g = _qkt_graph(M=s, N=s, K=K, out_dtype=out_dtype)
    g = Pipeline.build(KERNEL_PASSES).run(g)
    kop = g.nodes["c"].op
    assert mma_atom(kop.knobs) == "mma_m16n8k16_f16", "symbolic-MN Q@K^T must reach the warp-tier mma.sync variant"
    tensors = {nid: n.output for nid, n in g.nodes.items() if hasattr(n.output, "shape")}
    src = render_kernelop(kop, tensors=tensors)
    assert "dpl_mma_load_b_gmem_trans" in src, "transposed-B must use the gmem-direct trans helper"

    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415

    np.random.seed(7)
    a = (np.random.randn(seq, K) * 0.1).astype(np.float16)
    b = (np.random.randn(seq, K) * 0.1).astype(np.float16)
    be = CudaBackend()
    g_run = be.compile(_qkt_graph(M=s, N=s, K=K, out_dtype=out_dtype))
    run_result, _ = be.run(g_run, input_data={"a": a, "b": b})

    expected = a.astype(np.float32) @ b.astype(np.float32).T
    result = run_result.outputs["c"].astype(np.float32)
    diff = np.abs(result - expected).max()
    tol = 5e-2 if out_dtype == F16 else 1e-2
    assert diff < tol, f"seq={seq} K={K} out={out_dtype.name} max-abs-err {diff}"
