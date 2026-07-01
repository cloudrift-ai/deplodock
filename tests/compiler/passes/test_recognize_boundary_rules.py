"""The Loop-IR → Tile-IR boundary fires for every kernel kind.

``lowering/tile/010_recognize`` is the sole recognizer that lifts a
``LoopOp`` into the tile IR (a ``Map`` / ``Reduction`` / ``Contraction``
node). These assert it fires on the two simplest kinds — pointwise and
reduce — transitively proving the axes got lifted and the kernel entered
the tile dialect (no planner / launch-geometry fallback needed).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def test_recognize_fires_on_pointwise(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert "recognize" in recording_dump.fired_rules("lowering/tile")


def test_recognize_fires_on_reduction(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert "recognize" in recording_dump.fired_rules("lowering/tile")
