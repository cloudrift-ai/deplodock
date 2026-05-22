"""Tests for the ``launch_geometry`` rule (``001_launch_geometry``).

Two halves:

1. **Firing tests** — like the matmul / reduction rule-firing tests:
   build a frontend graph, run the full ``TILE_PASSES``, assert
   ``launch_geometry`` shows up in ``recording_dump.fired_rules``.
2. **Behavior tests** — build a ``LoopOp`` graph that exercises a
   specific launch_geometry branch (outer-chain stripping for reduction
   / pointwise, sibling output-Loop lifting), then run only the
   ``launch_geometry`` rule via ``Pipeline.build(select={"launch_geometry"}).run(...)``
   so the resulting ``Tile.axes`` reflects launch_geometry alone — no
   subsequent planner partition or downstream rewrite to obscure the
   assertion.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.ir.tile.ir import BIND_THREAD, SerialTile, ThreadTile, Tile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- firing on frontend graphs ---------------------------------------


def test_launch_geometry_fires_on_pointwise(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "launch_geometry" in recording_dump.fired_rules("lowering/tile")


def test_launch_geometry_fires_on_reduction(recording_dump):
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "launch_geometry" in recording_dump.fired_rules("lowering/tile")


# --- behavior tests: build LoopOp graphs, run launch_geometry in isolation ---
#
# We construct the input as a ``LoopOp`` directly (skipping the
# frontend / lifting / fusion stages) and run the full pipeline with
# ``select={"launch_geometry"}`` so only the launch_geometry rule fires. The resulting
# ``Tile.axes`` is exactly what launch_geometry produced — no THREAD/BLOCK
# partition, no StridedLoop conversion.
#
# Axis names get rewritten to ``a0/a1/...`` by ``LoopOp`` normalization,
# so assertions key on extents rather than original names.


def _wrap_loopop(loop_op: LoopOp, *, output_id: str = "o", output_shape: tuple = (1,)) -> Graph:
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=loop_op, inputs=["x"], output=Tensor(output_id, output_shape), node_id=output_id)
    g.inputs = ["x"]
    g.outputs = [output_id]
    return g


def _run_only_launch_geometry(g: Graph) -> TileOp:
    """Run the full pipeline with only the ``launch_geometry`` rule enabled and
    return the (single) resulting ``TileOp``."""
    out = Pipeline.build(TILE_PASSES, select={"launch_geometry"}).run(g)
    tile_ops = [n.op for n in out.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tile_ops) == 1
    return tile_ops[0]


def _reduction_loopop() -> LoopOp:
    """``y[i] = sum_k x[i, k]`` for shape (4, 8)."""
    i = Axis("i", 4)
    k = Axis("k", 8)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=k,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="acc"),
                ),
            ),
        ),
    )


def test_launch_geometry_strips_outer_chain_for_reduction():
    """Outer free-Loop chain (just ``i`` here) becomes ``ThreadTile.axes``;
    the inner reduce ``k`` Loop survives in the body as a ``SerialTile``."""
    tile_op = _run_only_launch_geometry(_wrap_loopop(_reduction_loopop()))
    tile = next(s for s in tile_op.body if isinstance(s, ThreadTile))
    assert sorted(int(ax.extent) for ax in tile.axes) == [4]
    body_serial = [s for s in tile.body if isinstance(s, SerialTile)]
    assert len(body_serial) == 1 and body_serial[0].is_reduce


def test_launch_geometry_handles_pointwise():
    """No reduce — outer chain strips into ``ThreadTile.axes``; body is the
    leaf Load/Write pair."""
    i = Axis("i", 4)
    pointwise = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Load(name="x_v", input="x", index=(Var("i"),)),
                    Write(output="o", index=(Var("i"),), value="x_v"),
                ),
            ),
        ),
    )
    tile_op = _run_only_launch_geometry(_wrap_loopop(pointwise))
    tile = next(s for s in tile_op.body if isinstance(s, ThreadTile))
    assert sorted(int(ax.extent) for ax in tile.axes) == [4]
    assert not any(isinstance(s, (SerialTile, Loop)) for s in tile.body)


def test_launch_geometry_preserves_kernel_name():
    """The ``rewrite`` entry point assigns a kernel name based on the
    LoopOp shape and node id."""
    tile_op = _run_only_launch_geometry(_wrap_loopop(_reduction_loopop(), output_id="reduce"))
    assert tile_op.name.startswith("k_reduce_")


def test_launch_geometry_lifts_sibling_output_loop():
    """SDPA-shaped tail: ``d`` Loop sits as a top-level sibling to the
    reduce; its body Writes ``out[i, d]``. Both ``i=4`` (outer chain)
    and ``d=16`` (lifted) end up in ``Tile.axes`` as THREAD; the
    reduce ``k=8`` Loop survives in the body."""
    i = Axis("i", 4)
    k = Axis("k", 8)
    d = Axis("d", 16)
    loop_op = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=k,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Loop(
                        axis=d,
                        body=(
                            Load(name="v_v", input="x", index=(Var("i"), Var("d"))),
                            Write(output="o", index=(Var("i"), Var("d")), value="v_v"),
                        ),
                    ),
                ),
            ),
        ),
    )
    tile_op = _run_only_launch_geometry(_wrap_loopop(loop_op, output_shape=(4, 16)))
    tile = next(s for s in tile_op.body if isinstance(s, ThreadTile))
    assert sorted(int(ax.extent) for ax in tile.axes) == [4, 16]
    body_serial = [s for s in tile.body if isinstance(s, SerialTile)]
    assert len(body_serial) == 1 and body_serial[0].is_reduce
    assert int(body_serial[0].axis.extent) == 8
