"""The block-DAG seed of the demoted-matmul cut (``seed_demoted``, R7 step 2.5).

``plans/dag-edge-placement-split-as-enumeration.md``: a demoted matmul is not one
"unbuildable" block — it is two clean blocks fusion glued together (a MONOID/MAP
producer ``--xn-->`` a SEMIRING matmul consumer), and "fused vs split" is the
**placement** of that ``xn`` edge. ``seed_demoted`` builds exactly that block-DAG seed.
These tests pin the representation: the right blocks, the derived edge, and the derived
placement.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import Placement, Schedule
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract import seed_demoted
from tests.compiler.passes.test_cut_offers import _f32_matmul_graph, _norm_linear_graph, _pointwise_graph

_CC = (12, 0)


def _demoted_loop_node(graph):
    """The fused demoted ``LoopOp`` node from the LOOP passes — the seed input."""
    out = Pipeline.build(LOOP_PASSES).run(graph, ctx=Context.from_target(_CC))
    node = next(n for n in out.nodes.values() if type(n.op).__name__ == "LoopOp")
    return out, node


def _seed(graph):
    out, node = _demoted_loop_node(graph)
    return seed_demoted(node.op, graph=out, node_id=node.id, out_tensor=node.output)


def test_demoted_matmul_seeds_as_monoid_to_semiring_dag():
    """``norm→linear`` seeds as two blocks: a MONOID rmsnorm producer feeding a SEMIRING
    matmul consumer through a derived ``xn`` edge."""
    tg = _seed(_norm_linear_graph())
    assert tg is not None
    assert len(tg.blocks) == 2
    prod, cons = tg.blocks
    assert prod.carrier.kind is AlgebraKind.MONOID  # the rmsnorm cooperative reduce
    assert cons.carrier.kind is AlgebraKind.SEMIRING  # the clean matmul

    # the producer→consumer intermediate edge is derived (not stored)
    xn_edges = [e for e in tg.edges if e.src == prod.name and e.dst == cons.name]
    assert len(xn_edges) == 1
    xn = xn_edges[0]
    # the consumer reads the materialized xn as a plain buffer (so it is a clean gemm
    # that can reach the warp tier — the whole point of the cut)
    assert xn.buffer in {p.buffer for p in cons.reads}
    assert xn.buffer not in {p.buffer for p in cons.writes}


def test_demoted_seed_edge_placement_is_derived():
    """The ``xn`` edge's placement is the derived ``Schedule`` view: GMEM by default
    (each block its own launch group — the two-launch cut), and GMEM under an explicit
    cross-group launch. SMEM/INLINE (the fused edge) would put them in one group."""
    tg = _seed(_norm_linear_graph())
    prod, cons = tg.blocks
    xn = next(e for e in tg.edges if e.src == prod.name and e.dst == cons.name)
    assert tg.placement(xn) is Placement.GMEM
    gmem = replace(tg, schedule=Schedule(launch={prod.name: 0, cons.name: 1}))
    assert gmem.placement(xn) is Placement.GMEM
    # a hypothetical fused (same-group) schedule reads as non-GMEM — the edge would
    # live in registers/smem inside one kernel (the future SMEM/INLINE placement)
    fused = replace(tg, schedule=Schedule(launch={prod.name: 0, cons.name: 0}))
    assert fused.placement(xn) is not Placement.GMEM


def test_clean_matmul_does_not_seed_demoted():
    """A clean matmul / pointwise body is not a demotion — ``seed_demoted`` declines
    (the fission has no cone to extract), so these flow through the normal single-block
    seed."""
    assert _seed(_f32_matmul_graph()) is None
    assert _seed(_pointwise_graph()) is None
