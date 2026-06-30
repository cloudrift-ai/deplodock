"""Free-axis thread-tile offer invariants — ``_moves.thread_offers``.

The bare ``≈_THREAD_TARGET`` sort ties a degenerate ``(BN=1, BM=256)`` tile with a
balanced ``(16, 16)`` (both 256 threads) and, on choice order, emits the degenerate
one first. Since emission order *is* the cold greedy pick (and the search's first
trajectory), a SEMIRING matmul would sample / deploy a ``BN=1`` (no-coalescing) or
``BM=1`` (no M-reuse) tile 2–3× off the golden band. The ``balanced`` flag
(``090_thread_tile`` sets it for SEMIRING) drops the degenerate-aspect tiles and
leads with a square-ish, coalesced (``BN >= BM``) tile; MAP (pointwise) keeps the
bandwidth-bound wide-N order. All CPU — no CUDA, no lowering past the loop dialect.
"""

from __future__ import annotations

from emmy.compiler import dtype as _dt
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import LOOP_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, iter_dag
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, thread_offers

_CC = (12, 0)


def _dag(g: Graph) -> IterDag:
    out = Pipeline.build(LOOP_PASSES).run(g, ctx=Context.from_target(_CC))
    lo = next(n.op for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    return iter_dag(lo)


def _matmul_dag(m: int, n: int, k: int) -> IterDag:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (m, k), _dt.get("f32")), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (k, n), _dt.get("f32")), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (m, n), _dt.get("f32")), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return _dag(g)


def _pointwise_dag(m: int, n: int) -> IterDag:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (m, n), _dt.get("f32")), node_id="x")
    g.add_node(ElementwiseOp(op="relu"), ["x"], Tensor("o", (m, n), _dt.get("f32")), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    return _dag(g)


def test_balanced_matmul_drops_degenerate_and_leads_balanced():
    """A SEMIRING 2-D matmul tile: no axis collapsed to 1, and the option-0 (cold
    greedy) tile is balanced with ``BN >= BM``."""
    offers = thread_offers(_matmul_dag(512, 512, 512), Budget(), balanced=True)
    assert offers, "expected legal thread tiles"
    assert all(t_n > 1 and t_m > 1 for t_n, t_m in offers), f"degenerate tile in {offers[:6]}"
    t_n, t_m = offers[0]
    assert t_n >= t_m, f"leading tile not BN>=BM: {offers[0]}"
    assert max(t_n, t_m) // min(t_n, t_m) <= 2, f"leading tile not balanced: {offers[0]}"


def test_bare_matmul_emits_degenerate_first():
    """Regression guard: without ``balanced`` the bare sort still emits a degenerate
    ``BN=1`` / ``BM=1`` tile tied-first — the bug ``balanced`` fixes."""
    bare = thread_offers(_matmul_dag(512, 512, 512), Budget())
    assert any(t_n == 1 or t_m == 1 for t_n, t_m in bare[:1]), f"expected degenerate-first in bare order: {bare[:3]}"


def test_pointwise_unchanged_by_default():
    """MAP (pointwise) is bandwidth-bound: the default (``balanced=False``) order is
    untouched, so a wide-N tile is still reachable / not reordered away."""
    dag = _pointwise_dag(2048, 2048)
    assert thread_offers(dag, Budget()) == thread_offers(dag, Budget(), balanced=False)
