"""Tests for the partition planner mechanism (M1 scope).

M1 establishes the role-tag infrastructure:

- ``Role`` enum + ``role`` field on ``Loop`` / ``StridedLoop``, excluded
  from ``Body.structural_key`` so adding tags doesn't invalidate cached
  autotune entries.
- ``000_partition_planner`` exists as a stub (always skips); subsequent
  milestones populate it.
- ``001_tileify`` honors ``Role.REGISTER`` by stopping the outer free-Loop
  chain lift and skipping sibling-output-loop lifting for tagged loops.

These tests construct synthetic IR and assert the new code paths fire
correctly. The planner itself is a no-op in M1, so it's not exercised
end-to-end here — that comes in M2 / M3 when each pass migrates.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Role
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import BIND_THREAD, Tile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- Role field round-trip -------------------------------------------


def test_role_field_default_is_none():
    """A freshly-constructed Loop has ``role=None`` so existing call
    sites that omit the field keep working unchanged."""
    loop = Loop(axis=Axis("i", 4), body=())
    assert loop.role is None


def test_with_bodies_preserves_role():
    """``Stmt.with_bodies`` is the canonical body-rewrite primitive (used
    by every pass that descends into nested bodies). It must propagate
    role tags or the planner's annotations would silently disappear on
    the first downstream rewrite."""
    original = Loop(axis=Axis("i", 4), body=(), role=Role.REGISTER)
    rewritten = original.with_bodies((Body(()),))
    assert isinstance(rewritten, Loop)
    assert rewritten.role is Role.REGISTER


def test_role_excluded_from_structural_key():
    """Adding a role tag must not change ``Body.structural_key`` —
    otherwise the planner cold-starts the autotune cache on every
    kernel before producing any actual decisions."""
    untagged = Body((Loop(axis=Axis("i", 4), body=()),))
    tagged = Body((Loop(axis=Axis("i", 4), body=(), role=Role.REGISTER),))
    assert untagged.structural_key() == tagged.structural_key()


# --- Tileify honors REGISTER tag -------------------------------------


def _wrap_loopop(loop_op: LoopOp) -> Graph:
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=loop_op, inputs=["x"], output=Tensor("o", (4,)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    return g


def _run_only_tileify(g: Graph) -> TileOp:
    out = Pipeline.build(TILE_PASSES, select={"tileify"}).run(g)
    tile_ops = [n.op for n in out.nodes.values() if isinstance(n.op, TileOp)]
    assert len(tile_ops) == 1
    return tile_ops[0]


def test_tileify_stops_at_register_tagged_loop():
    """Outer free-Loop chain: when an inner Loop carries ``Role.REGISTER``,
    tileify must stop lifting at the previous level. The REGISTER Loop
    stays in the body for ``008_register_tile`` to replicate."""
    outer = Axis("i", 4)
    inner = Axis("i_i", 2)
    loop_op = LoopOp(
        body=(
            Loop(
                axis=outer,
                body=(
                    Loop(
                        axis=inner,
                        role=Role.REGISTER,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("i_i"))),
                            Write(output="o", index=(Var("i"), Var("i_i")), value="x_v"),
                        ),
                    ),
                ),
            ),
        ),
    )
    tile_op = _run_only_tileify(_wrap_loopop(loop_op))
    tile = next(s for s in tile_op.body if isinstance(s, Tile))
    # Outer ``i`` lifted to Tile.axes; inner REGISTER ``i_i`` stays in body.
    assert sorted(int(ba.axis.extent) for ba in tile.axes) == [4]
    assert all(ba.bind == BIND_THREAD for ba in tile.axes)
    body_loops = [s for s in tile.body if isinstance(s, Loop)]
    assert len(body_loops) == 1
    assert body_loops[0].role is Role.REGISTER
    assert int(body_loops[0].axis.extent) == 2


def test_tileify_skips_register_sibling_output_loop():
    """A top-level body free Loop tagged REGISTER must NOT be lifted by
    ``_lift_output_loops`` even if its subtree writes the loop axis —
    the tag means "this is register-tile inner, not a launch dim."
    """
    # Outer reduce so tileify's chain-strip stops before the sibling level.
    i = Axis("i", 4)
    k = Axis("k", 8)
    reg = Axis("r", 2)
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
                    # Sibling free Loop tagged REGISTER. Writes index ``r`` —
                    # without the tag, ``_lift_output_loops`` would pull it
                    # into Tile.axes.
                    Loop(
                        axis=reg,
                        role=Role.REGISTER,
                        body=(Write(output="o", index=(Var("i"), Var("r")), value="acc"),),
                    ),
                ),
            ),
        ),
    )
    tile_op = _run_only_tileify(_wrap_loopop(loop_op))
    tile = next(s for s in tile_op.body if isinstance(s, Tile))
    # Only ``i`` (outer chain) ends up in Tile.axes — ``r`` stays in body.
    assert sorted(int(ba.axis.extent) for ba in tile.axes) == [4]
    body_loops = [s for s in tile.body if isinstance(s, Loop)]
    # Body has the reduce Loop and the REGISTER Loop.
    register_loops = [loop for loop in body_loops if loop.role is Role.REGISTER]
    assert len(register_loops) == 1
    assert int(register_loops[0].axis.extent) == 2


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
    """When a TileOp body contains a ``Role.REGISTER`` Loop,
    ``006a_register_tile_planned`` replicates the body by the loop's
    extent (σ: axis → literal(i)), unwraps the wrapper, and stamps
    ``FM`` / ``FN`` so the legacy ``008_register_tile`` falls through
    via its ``FN-in-knobs`` idempotence check.

    Tests the simplest synthetic case — no Stages, just one REGISTER
    Loop wrapping a Load+Write body."""
    from deplodock.compiler.ir.axis import BIND_THREAD, BoundAxis
    from deplodock.compiler.ir.stmt import Tile
    from deplodock.compiler.ir.tile.ir import TileOp

    a_outer = Axis("a_outer", 8)
    a_reg = Axis("a_reg", 4)
    tile = Tile(
        axes=(BoundAxis(axis=a_outer, bind=BIND_THREAD),),
        body=(
            Loop(
                axis=a_reg,
                role=Role.REGISTER,
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

    # The REGISTER Loop wrapper is gone; its body got replicated 4 times.
    new_tile = next(s for s in new_tile_op.body if isinstance(s, Tile))
    body_loops = [s for s in new_tile.body if isinstance(s, Loop)]
    assert body_loops == [], "REGISTER Loop wrapper should have been unwrapped"
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
    # FM gets stamped from the outermost (and only) REGISTER axis extent.
    assert new_tile_op.knobs.get("FM") == 4


# --- (006 chunk_reduce STAGE_INNER guard test deleted alongside the
#       legacy pass — planner owns chunk-reduce now.)
