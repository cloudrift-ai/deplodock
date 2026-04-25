"""Tests for Tile IR ``Block`` / ``BoundLoop`` and the Kernel-IR lowering.

Tile IR is the schedule-decision layer produced by ``lower_naive``. The
materialization pass converts it to Kernel IR (``KernelOp`` with
``Enclosure`` / ``Smem`` / ``StridedLoop`` / ``TreeHalve``). These
tests exercise the nodes directly and the pipeline boundary.
"""

from __future__ import annotations

from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tile.ir import BIND_SERIAL, BIND_THREAD, Block, BoundAxis, BoundLoop, TileOp, iter_body
from deplodock.compiler.ir.tile.lower import lower_naive


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


def test_iter_body_walks_into_block():
    i = Axis("i", 4)
    k = Axis("k", 8)
    inner = BoundLoop(
        axis=BoundAxis(axis=k, bind=BIND_SERIAL),
        body=(
            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
        ),
    )
    blk = Block(axes=(BoundAxis(axis=i, bind=BIND_THREAD),), body=(inner,))
    seen = list(iter_body((blk,)))
    assert any(isinstance(s, Accum) for s in seen)
    assert seen[0] is blk


def test_lower_naive_produces_block_for_reduction():
    """``lower_naive`` builds a logical ``Block`` with BoundLoops — no Kernel-IR yet."""
    tile_op = lower_naive(_reduction_loopop(), kernel_name="reduce")
    blocks = [s for s in tile_op.body if isinstance(s, Block)]
    assert len(blocks) == 1
    blk = blocks[0]
    assert len(blk.axes) == 1
    assert blk.axes[0].bind == BIND_THREAD
    assert any(isinstance(s, BoundLoop) for s in blk.body)
    assert any(isinstance(s, Write) for s in blk.body)


def test_lower_naive_produces_block_for_pointwise():
    """Pointwise kernel also produces a ``Block``."""
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


def test_tileop_container_preserves_name():
    """``TileOp`` is a graph-node container; name and body pass through."""
    tile_op = lower_naive(_reduction_loopop(), kernel_name="reduce")
    assert isinstance(tile_op, TileOp)
    assert tile_op.name == "reduce"
