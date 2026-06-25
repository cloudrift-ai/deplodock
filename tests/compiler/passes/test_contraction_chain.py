"""The carried contraction chain — Phase 1a of the tensor-core streaming flash
(``plans/tensor-core-streaming-flash-mma.md``).

A streaming-flash nest is a ``Monoid`` carrier streaming over a *nested* QK^T
contraction. Unification 3 reads it as a **chain on a shared axis**: ``kv`` is the
dual-role hinge — free-output of the inner QK^T contraction, reduce of the outer P@V
contraction (embedded in the carrier's ``O = O·α + p·v``) and of the carrier. These
tests pin the **derived view** ``IterDag.chain`` exposing that structure; it is a
projection of the body (computed on demand), so a non-streaming nest yields ``None``
and a flash nest yields the hinge + the inner SEMIRING contraction + the carrier.

All CPU — no CUDA, no lowering. The streaming ``LoopOp`` is built directly by the
recognizer's ``build_flash_frag`` (the same body the GPU flash tests compile)."""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Load, Monoid
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.loop.recognize._flash import build_flash_frag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag


def _flash_loop(*, causal: bool = False, group: int = 1) -> LoopOp:
    """The fused streaming-flash ``LoopOp`` for a small ``(B,H,S,D)`` SDPA — the
    body ``iter_dag`` reads. Built straight from the recognizer so the test needs
    no torch trace."""
    B, H, S, D = 1, 2, 16, 8
    shp = tuple(Dim(d) for d in (B, H, S, D))
    out = Tensor("o", (B, H, S, D), F32)
    frag = build_flash_frag("q", "k", "v", shp, shp, shp, out, causal=causal, group=group)
    return frag.nodes["o"].op


# ``LoopOp`` canonicalizes axis / SSA names on construction (``kv`` -> ``a3``, ``s`` ->
# ``v1``), so the tests key on STRUCTURE — extent, algebra, def-use — never literal names.
_S, _D = 16, 8  # the streaming KV extent (hinge) and the head-dim (inner QK^T reduce)


def test_streaming_flash_exposes_the_chain():
    dag = iter_dag(_flash_loop())
    assert dag.streaming, "a flash nest must be streaming"
    chain = dag.chain
    assert chain is not None, "a streaming flash nest must expose the carried contraction chain"

    # The hinge carries the online-softmax Monoid (the streaming reduce + the outer P@V
    # contraction live here); the inner QK^T is a nested SEMIRING reduce over the head-dim.
    assert isinstance(chain.carrier, Monoid)
    assert chain.hinge.algebra is AlgebraKind.MONOID
    assert chain.hinge.extent == _S
    assert chain.inner.algebra is AlgebraKind.SEMIRING
    assert chain.inner.extent == _D
    # The inner contraction is nested directly inside the hinge (the carried chain).
    assert chain.inner.parent is not None and chain.inner.parent.axis.name == chain.hinge_name


def test_chain_hinge_is_dual_role():
    """The hinge is dual-role: the reduce of the carrier (and P@V) AND a *free output*
    of the inner QK^T contraction (it indexes K inside the QK^T body but is not the QK^T
    reduce axis)."""
    dag = iter_dag(_flash_loop())
    chain = dag.chain
    hinge, inner = chain.hinge_name, chain.inner.loop

    # The inner contraction reduces the head-dim, not the hinge.
    assert inner.axis.name != hinge
    # ...yet the hinge indexes a Load in the inner body (K[kv, dd]) — its free output.
    inner_load_vars = {v for ld in inner.body.iter_of_type(Load) for e in ld.index for v in e.free_vars()}
    assert hinge in inner_load_vars, "the hinge must be a free output (index var) of the inner QK^T contraction"


def test_chain_score_is_the_carrier_partial():
    """The score edge is the inner contraction's result the carrier folds — the
    carrier's first partial (the INLINE edge value 1c materializes)."""
    dag = iter_dag(_flash_loop())
    chain = dag.chain
    assert chain.score == chain.carrier.partial[0]


def test_causal_chain_still_exposes_the_chain():
    """A causal mask folds a masked score; the chain still reports the carrier's
    first partial (whatever the carrier folds, not a hard-coded name)."""
    dag = iter_dag(_flash_loop(causal=True))
    chain = dag.chain
    assert chain is not None
    assert chain.score == chain.carrier.partial[0]
    assert chain.hinge.algebra is AlgebraKind.MONOID
    assert chain.inner.algebra is AlgebraKind.SEMIRING


def test_plain_reduce_has_no_chain():
    """A flat (non-streaming) reduce is not a chain — ``chain`` is ``None`` there,
    so the view stays specific to the carried-contraction shape."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 64)), node_id="x")
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", (4, 1)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    out = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target((12, 0)))
    lo = next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    dag = iter_dag(lo)
    assert not dag.streaming
    assert dag.chain is None
