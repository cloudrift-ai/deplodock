"""Tests for masked tiles — output-axis extents with no power-of-2 divisor.

When the partition planner faces an output axis like ``vocab=151669`` whose
``_TUNE_AXIS_CHOICES`` divisor set is ``{1}``, it still picks a normal
``BN`` and emits ``Axis.real_extent`` on the block axis plus a ``Cond``
predicate wrapping the σ-rewritten body. The Cond predicate references the
register-tile axis, so ``010_split_register_axes`` must replicate the Cond
itself (not just descend into its body) to avoid leaving dangling Var refs.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Cond, Load, Write
from deplodock.compiler.ir.tile.ir import RegisterTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def test_planner_admits_non_divisor_n_with_real_extent(recording_dump):
    """N=47 has no divisor in ``_TUNE_AXIS_CHOICES`` other than 1. The planner
    should still emit a TileOp whose N block axis carries ``real_extent=47``
    and whose body contains a ``Cond`` masking OOB lanes."""
    g = Graph()
    _input(g, "a", (256, 64))
    _input(g, "b", (64, 47))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (256, 47)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    # Find any axis with real_extent set — should be the N block axis at 47.
    real_extents = []
    for stmt in tile_op.body.iter():
        for ax in getattr(stmt, "axes", ()):
            if isinstance(ax, Axis) and ax.real_extent is not None:
                real_extents.append((ax.name, ax.real_extent))
    assert 47 in [e for _, e in real_extents], f"expected real_extent=47 on a block axis, got {real_extents}"

    # The body should contain at least one Cond (the mask) referencing 47.
    conds = list(tile_op.body.iter_of_type(Cond))
    assert conds, "expected at least one mask Cond wrapping the σ-rewritten body"
    pred_text = conds[0].cond.pretty()
    assert "47" in pred_text, f"mask predicate should reference real extent 47, got {pred_text!r}"


def test_split_register_axes_replicates_cond_with_axis_dep_predicate():
    """When a Cond's predicate references the register-tile axis being
    replicated, the pass must replicate the entire Cond (not just descend
    into its body). Each replica's predicate gets the σ-substituted literal
    so NVRTC sees fully-resolved conditions — and there is no dangling
    reference to a no-longer-defined register-axis Var."""
    a_thread = Axis("a_thread", 4)
    a_reg = Axis("a_reg", 3)
    inner_body = (
        Cond(
            cond=BinaryExpr("<", BinaryExpr("+", BinaryExpr("*", Var("a_thread"), Literal(3, "int")), Var("a_reg")), Literal(10, "int")),
            body=(
                Load(name="x_v", input="x", index=(Var("a_thread"), Var("a_reg"))),
                Write(output="o", index=(Var("a_thread"), Var("a_reg")), value="x_v"),
            ),
        ),
    )
    tile = ThreadTile(
        axes=(a_thread,),
        body=(RegisterTile(axes=(a_reg,), body=inner_body),),
    )
    tile_op = TileOp(body=(tile,), name="t")

    g = Graph()
    _input(g, "x", (4, 3))
    g.add_node(op=tile_op, inputs=["x"], output=Tensor("o", (4, 3)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, select={"split_register_axes"}).run(g)
    new_tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    new_tile = next(s for s in new_tile_op.body if isinstance(s, ThreadTile))

    # RegisterTile is gone, body is replicated 3 times — so 3 Conds (not 1).
    assert not any(isinstance(s, RegisterTile) for s in new_tile.body)
    conds = [s for s in new_tile.body if isinstance(s, Cond)]
    assert len(conds) == 3, f"expected 3 replicated Conds (one per a_reg literal), got {len(conds)}"

    # Each replica's predicate should have a_reg substituted to its literal —
    # no surviving ``a_reg`` Var references in any predicate.
    for c in conds:
        free = set(c.cond.free_vars())
        assert "a_reg" not in free, f"a_reg should be σ-substituted, got predicate {c.cond.pretty()!r}"
