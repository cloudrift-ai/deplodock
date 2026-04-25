"""Tests for the ``Tile`` cooperative-block Stmt node.

``Tile`` is the structural marker for a cooperative block: its body
shares a per-block scratch indexed by ``live_axes``. Currently every
block runs at one-thread-per-slot, so render is a transparent
concatenation over the body and behaves identically to placing the
body's statements directly under the enclosing ``Enclosure``.

These tests exercise the node directly:
- ``iter_body`` walks into ``Tile.body``.
- ``lower_naive`` wraps the inner body of any reduction-bearing kernel
  in a ``Tile`` whose ``live_axes`` are the surviving thread axes.
- The renderer produces the same CUDA whether or not the inner body is
  wrapped in a ``Tile`` (pass-through invariant for the current
  one-thread-per-slot configuration).
"""

from __future__ import annotations

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import Enclosure, Tile, TileOp, iter_body
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


def test_lower_naive_wraps_reduction_in_tile():
    tile_op = lower_naive(_reduction_loopop(), kernel_name="reduce")
    enclosures = [s for s in tile_op.body if isinstance(s, Enclosure)]
    assert len(enclosures) == 1
    encl = enclosures[0]
    assert len(encl.body) == 1
    assert isinstance(encl.body[0], Tile)
    tile = encl.body[0]
    assert tile.live_axes == encl.thread_axes
    assert tile.extents == tuple(int(ax.extent) for ax in encl.thread_axes)
    assert tile.extents == (4,)


def test_lower_naive_no_tile_for_pointwise():
    """Pointwise body has no Accum, so no Tile is created."""
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
    enclosures = [s for s in tile_op.body if isinstance(s, Enclosure)]
    assert len(enclosures) == 1
    assert not any(isinstance(s, Tile) for s in enclosures[0].body)


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
