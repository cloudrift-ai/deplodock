"""Tests for Tile-IR node helpers (``iter_body`` etc.).

Tests for the ``partition_loops`` rule itself (Loop-IR ``LoopOp`` →
Tile-IR ``TileOp``) live in ``tests/compiler/passes/test_partition_planner_rules.py``.
"""

from __future__ import annotations

from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.loop import Accum, Axis, Load, Loop
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile.ir import ThreadTile


def test_iter_body_walks_into_tile():
    i = Axis("i", 4)
    k = Axis("k", 8)
    inner = Loop(
        axis=k,
        body=(
            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
        ),
    )
    blk = ThreadTile(axes=(i,), body=(inner,))
    seen = list(Body((blk,)).iter())
    assert any(isinstance(s, Accum) for s in seen)
    assert seen[0] is blk
