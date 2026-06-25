"""Intra-CTA cooperative combine over a general monoid (``Monoid``) carrier.

Step 2 of ``plans/atomic-free-monoid-combine.md``: a cooperative-K reduce whose
carrier is a tuple-valued ``Monoid`` (online-softmax ``(m, l)``) — not an
``Accum`` — splits across the CTA's threads and merges the per-thread partial
states via the carrier's ``combine_states`` (the materializer's ``emit_combine`` →
the tuple-aware ``WarpShuffle`` / ``TreeHalve``), instead of
running one thread serially. Verifies accuracy on GPU against numpy softmax for
both the warp-shuffle path (BR ≤ warp) and the smem-tree path (BR > warp), and
that the emitted kernel actually carries the cross-thread monoid combine.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.stmt import Assign, Init, Load, Loop, Monoid, Write


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def _ml_combine(m: str, ll: str, s: str) -> Monoid:
    """The online-softmax ``(m, l)`` monoid: state max + denominator, folding one
    score ``s``. Authors both ``merge`` (fold a partial) and ``combine_states``
    (merge two partition states) — the asymmetric monoid can't auto-derive."""
    merge = (
        Assign(f"{m}_mx", "maximum", (m, s)),
        Assign(f"{m}_dm", "subtract", (m, f"{m}_mx")),
        Assign(f"{m}_al", "exp", (f"{m}_dm",)),
        Assign(f"{m}_ds", "subtract", (s, f"{m}_mx")),
        Assign(f"{m}_p", "exp", (f"{m}_ds",)),
        Assign(f"{m}_lm", "multiply", (ll, f"{m}_al")),
        Assign(ll, "add", (f"{m}_lm", f"{m}_p")),
        Assign(m, "copy", (f"{m}_mx",)),
    )
    mb, lb = f"{m}__o", f"{ll}__o"
    combine_states = (
        Assign(f"{m}_cmx", "maximum", (m, mb)),
        Assign(f"{m}_cda", "subtract", (m, f"{m}_cmx")),
        Assign(f"{m}_ca", "exp", (f"{m}_cda",)),
        Assign(f"{m}_cdb", "subtract", (mb, f"{m}_cmx")),
        Assign(f"{m}_cb", "exp", (f"{m}_cdb",)),
        Assign(f"{m}_cla", "multiply", (ll, f"{m}_ca")),
        Assign(f"{m}_clb", "multiply", (lb, f"{m}_cb")),
        Assign(ll, "add", (f"{m}_cla", f"{m}_clb")),
        Assign(m, "copy", (f"{m}_cmx",)),
    )
    return Monoid(
        state=(m, ll),
        partial=(s,),
        merge=merge,
        identity=(Literal(-1e30), Literal(0.0)),
        commutative=True,
        axes=("k",),
        state_b=(mb, lb),
        combine_states=combine_states,
    )


def _softmax_combine_graph(rows: int, k: int) -> Graph:
    """Row-wise softmax ``out[r, k] = exp(x[r,k] - m_r) / l_r`` where ``(m_r, l_r)``
    is produced by a streaming ``(m, l)`` ``Monoid`` reduce over the K axis — the
    reduce a cooperative split parallelizes across the CTA's threads."""
    body = (
        Loop(
            axis=Axis("row", Dim(rows)),
            body=(
                Init(name="m", op=ElementwiseImpl("maximum"), dtype="f32"),
                Init(name="l", op=ElementwiseImpl("add"), dtype="f32"),
                Loop(
                    axis=Axis("k", Dim(k)),
                    body=(
                        Load(name="s", input="x", index=(Var("row"), Var("k"))),
                        _ml_combine("m", "l", "s"),
                    ),
                ),
                Loop(
                    axis=Axis("k2", Dim(k)),
                    body=(
                        Load(name="s2", input="x", index=(Var("row"), Var("k2"))),
                        Assign(name="d", op="subtract", args=("s2", "m")),
                        Assign(name="e", op="exp", args=("d",)),
                        Assign(name="o", op="divide", args=("e", "l")),
                        Write(output="out", index=(Var("row"), Var("k2")), value="o"),
                    ),
                ),
            ),
        ),
    )
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (Dim(rows), Dim(k))), node_id="x")
    g.add_node(op=LoopOp(body=body), inputs=["x"], output=Tensor("out", (Dim(rows), Dim(k))), node_id="out")
    g.inputs = ["x"]
    g.outputs = ["out"]
    return g


def _ref_softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.mark.skipif(not _has_cuda(), reason="cooperative monoid combine runs on CUDA")
@pytest.mark.parametrize("br", [32, 128])
def test_cooperative_combine_softmax_matches_numpy(monkeypatch, br: int) -> None:
    """A cooperative-K ``Monoid`` reduce (BR>1) matches numpy softmax — warp-
    shuffle path at BR=32, smem-tree path at BR=128."""
    monkeypatch.setenv("DEPLODOCK_BR", str(br))
    monkeypatch.setenv("DEPLODOCK_BN", "1")
    monkeypatch.setenv("DEPLODOCK_BM", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rows, k = 8, 256
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((rows, k)) * 3.0).astype(np.float32)
    be = CudaBackend()
    compiled = be.compile(_softmax_combine_graph(rows, k))
    out = be.run(compiled, input_data={"x": x})[0].outputs["out"]
    np.testing.assert_allclose(np.asarray(out).reshape(rows, k), _ref_softmax(x), atol=1e-5)


@pytest.mark.skipif(not _has_cuda(), reason="cooperative monoid combine runs on CUDA")
def test_cooperative_combine_emits_monoid_combine(monkeypatch) -> None:
    """With BR pinned, the kernel carries the cross-thread monoid combine
    (``__shfl_xor_sync`` butterfly over the full state) — not a serial reduce."""
    monkeypatch.setenv("DEPLODOCK_BR", "32")
    monkeypatch.setenv("DEPLODOCK_BN", "1")
    monkeypatch.setenv("DEPLODOCK_BM", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    compiled = CudaBackend().compile(_softmax_combine_graph(8, 256))
    src = "\n".join(n.op.kernel_source for n in compiled.nodes.values() if getattr(n.op, "kernel_source", None))
    assert "__shfl_xor_sync" in src
