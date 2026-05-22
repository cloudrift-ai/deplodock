"""Tests for the partition planner pass.

After the planner-emits-tiles refactor the planner constructs typed tile
flavors directly; the ``Role`` enum and ``Loop.role`` / ``StridedLoop.role``
fields it used to communicate decisions to downstream passes were
deleted. These tests cover the surviving surface: planner fires on
pointwise, and ``006a_register_tile_planned`` replicates a synthetic
``RegisterTile`` body. Earlier tests that built ``Loop(role=…)`` graphs
were deleted in the same refactor.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- Planner stub ----------------------------------------------------


def test_planner_fires_on_pointwise(recording_dump, monkeypatch):
    """M16: the planner partitions pointwise kernels too (BLOCK/THREAD
    on output axes). The old ``DEPLODOCK_PLANNER`` env gate is gone."""
    monkeypatch.delenv("DEPLODOCK_PLANNER", raising=False)
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=loop_op_pointwise(), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "partition_planner" in recording_dump.fired_rules("lowering/tile")


def loop_op_pointwise() -> LoopOp:
    i = Axis("i", 4)
    j = Axis("j", 8)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
                        ),
                    ),
                ),
            ),
        ),
    )


# --- 006a register_tile_planned (M2 plumbing → M4 active) -----------


def test_006a_replicates_register_tagged_body_loop():
    """When a TileOp body contains a ``RegisterTile``,
    ``006a_register_tile_planned`` replicates the body by the tile's
    axis extent (σ: axis → literal(i)), unwraps the wrapper, and stamps
    ``FM`` / ``FN`` so the legacy ``008_register_tile`` falls through
    via its ``FN-in-knobs`` idempotence check.

    Tests the simplest synthetic case — no Stages, just one
    ``RegisterTile`` wrapping a Load+Write body."""
    a_outer = Axis("a_outer", 8)
    a_reg = Axis("a_reg", 4)
    tile = ThreadTile(
        axes=(a_outer,),
        body=(
            RegisterTile(
                axes=(a_reg,),
                body=(
                    Load(name="x_v", input="x", index=(Var("a_outer"), Var("a_reg"))),
                    Write(output="o", index=(Var("a_outer"), Var("a_reg")), value="x_v"),
                ),
            ),
        ),
    )
    tile_op = TileOp(body=(tile,), name="t")

    g = Graph()
    _input(g, "x", (8, 4))
    g.add_node(op=tile_op, inputs=["x"], output=Tensor("o", (8, 4)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES, select={"register_tile_planned"}).run(g)
    new_tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    # The RegisterTile wrapper is gone; its body got replicated 4 times.
    new_tile = next(s for s in new_tile_op.body if isinstance(s, ThreadTile))
    register_tiles = [s for s in new_tile.body if isinstance(s, RegisterTile)]
    assert register_tiles == [], "RegisterTile wrapper should have been unwrapped"
    # 4 Loads + 4 Writes after replication.
    body_loads = list(new_tile.body.iter_of_type(Load))
    body_writes = [s for s in new_tile.body if isinstance(s, Write)]
    assert len(body_loads) == 4
    assert len(body_writes) == 4
    # The σ should have substituted ``a_reg`` Var with the literal index
    # in each replica — no surviving ``a_reg`` references.
    for w in body_writes:
        vars_in_index = {v for e in w.index for v in e.free_vars()}
        assert "a_reg" not in vars_in_index, f"unreplicated a_reg in Write: {w.index}"
    # FM gets stamped from the outermost (and only) RegisterTile axis extent.
    assert new_tile_op.knobs.get("FM") == 4
