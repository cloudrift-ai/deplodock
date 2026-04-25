"""Tests for the ``Tile`` structural node and the ``Block`` → ``Tile``
materialization.

``Block`` is the high-level pre-materialization form produced by
``lower_naive``; ``Tile`` is the low-level post-materialization form
that render consumes. These tests exercise the node directly and the
transition between them.
"""

from __future__ import annotations

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import BIND_THREAD, Block, BoundLoop, Enclosure, Tile, TileOp, iter_body
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.ir.tile.render import render_tileop


def _reduction_loopop() -> LoopOp:
    """LoopOp computing ``y[i] = sum_k x[i, k]`` for shape (4, 8)."""
    i = Axis("i", 4)
    k = Axis("k", 8)
    body = (
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
                Write(output="y", index=(Var("i"),), value="acc"),
            ),
        ),
    )
    return LoopOp(body=body)


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
    tile = Tile(live_axes=(i,), extents=(4,), body=(inner,))
    seen = list(iter_body((tile,)))
    assert any(isinstance(s, Accum) for s in seen)
    assert any(isinstance(s, Load) for s in seen)
    assert seen[0] is tile


def test_iter_body_walks_into_block():
    i = Axis("i", 4)
    k = Axis("k", 8)
    inner = Loop(
        axis=k,
        body=(
            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
        ),
    )
    blk = Block(output_axes=(i,), output_bind=BIND_THREAD, body=(inner,))
    seen = list(iter_body((blk,)))
    assert any(isinstance(s, Accum) for s in seen)
    assert seen[0] is blk


def test_lower_naive_produces_block_for_reduction():
    """``lower_naive`` builds a logical ``Block`` with BoundLoops; Enclosure/Tile land later."""
    tile_op = lower_naive(_reduction_loopop(), kernel_name="reduce")
    blocks = [s for s in tile_op.body if isinstance(s, Block)]
    assert len(blocks) == 1
    blk = blocks[0]
    assert len(blk.output_axes) == 1
    assert blk.output_bind == BIND_THREAD
    # Block body holds the logical compute — BoundLoop + Write — no lowered machinery.
    assert any(isinstance(s, BoundLoop) for s in blk.body)
    assert any(isinstance(s, Write) for s in blk.body)
    assert not any(isinstance(s, (Enclosure, Tile)) for s in blk.body)


def test_lower_naive_produces_block_for_pointwise():
    """Pointwise kernel also produces a ``Block`` (no Accum → no Tile
    after materialization)."""
    i = Axis("i", 4)
    pointwise = LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Load(name="x_v", input="x", index=(Var("i"),)),
                    Write(output="y", index=(Var("i"),), value="x_v"),
                ),
            ),
        ),
    )
    tile_op = lower_naive(pointwise, kernel_name="pw")
    blocks = [s for s in tile_op.body if isinstance(s, Block)]
    assert len(blocks) == 1


def test_tile_renders_as_passthrough():
    """Render output is identical with-Tile vs without-Tile (one-thread-per-slot)."""
    i = Axis("i", 4)
    k = Axis("k", 8)
    inner_stmts = (
        Loop(
            axis=k,
            body=(
                Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
            ),
        ),
        Write(output="y", index=(Var("i"),), value="acc"),
    )
    encl_no_tile = Enclosure(thread_axes=(i,), block_axes=(), body=inner_stmts)
    encl_with_tile = Enclosure(
        thread_axes=(i,),
        block_axes=(),
        body=(Tile(live_axes=(i,), extents=(4,), body=inner_stmts),),
    )
    shapes = {"x": (4, 8), "y": (4,)}
    src_a = render_tileop(TileOp(body=(encl_no_tile,), name="r"), shapes=shapes)
    src_b = render_tileop(TileOp(body=(encl_with_tile,), name="r"), shapes=shapes)
    assert src_a == src_b
