"""``010_recognize`` stamps the recognized :class:`Skeleton` onto the lifted ``TileOp``.

Drives the loop passes to the fused ``LoopOp``, then applies the recognize rewrite directly (the
``lowering/tile`` pass bundles recognition + scheduling, and scheduling rebuilds the ``TileOp``,
so we assert at the recognition boundary where the skeleton is produced and later consumed).
"""

from __future__ import annotations

import importlib

from deplodock.compiler import dtype as _dt
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import LOOP_PASSES, Match, Pipeline

_recognize = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_recognize")
F16 = _dt.get("f16")


def _recognized(graph: Graph) -> TileOp:
    fused = Pipeline.build(LOOP_PASSES).run(graph)
    node = next(n for n in fused.nodes.values() if type(n.op).__name__ == "LoopOp")
    out = _recognize.rewrite(Match(graph=fused, root_node_id=node.id, rule=None), node)
    assert isinstance(out, TileOp)
    return out


def test_recognize_stamps_contraction_skeleton():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("xn", (64, 64), F16), node_id="xn")
    g.add_node(InputOp(), [], Tensor("w", (64, 64), F16), node_id="w")
    g.add_node(MatmulOp(), ["xn", "w"], Tensor("o", (64, 64), F16), node_id="o")
    g.inputs, g.outputs = ["xn", "w"], ["o"]
    tile = _recognized(g)
    assert tile.skeleton is not None
    red = tile.skeleton.root.reduce
    assert red is not None and red.contraction is True
    assert red.binding is not None  # the matmul operands bind to mma A/B roles
    assert red.carrier.twist.family == "id"  # Semiring.as_monoid() — the K-as-reduce normalization
