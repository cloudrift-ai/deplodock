"""Tests for Tile-IR node helpers (``iter_body`` etc.).

Tests for the ``tileify`` rule itself (Loop-IR ``LoopOp`` → Tile-IR
``TileOp``) live in ``tests/compiler/passes/test_tileify_rules.py``.
"""

from __future__ import annotations

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop
from deplodock.compiler.ir.tile.ir import BIND_THREAD, BoundAxis, Tile, iter_body


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
    blk = Tile(axes=(BoundAxis(axis=i, bind=BIND_THREAD),), body=(inner,))
    seen = list(iter_body((blk,)))
    assert any(isinstance(s, Accum) for s in seen)
    assert seen[0] is blk
