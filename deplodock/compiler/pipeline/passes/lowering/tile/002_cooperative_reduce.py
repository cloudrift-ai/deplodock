"""Cooperative-reduce strategy — moves output axes from ``thread_axes``
to ``block_axes`` on a ``Block`` and flips inner ``BoundLoop`` bindings
so threads cooperate.

Reads the logical ``Block`` produced by ``lower_naive`` (default:
``thread_axes=output_axes`` / ``block_axes=()``, inner ``BoundLoop``s
all ``BIND_SERIAL``) and rewrites it in place:

- Move axes from ``Block.thread_axes`` to ``Block.block_axes``: each
  CUDA block now owns one output slot.
- Every inner ``BoundLoop`` (reduction or free output) →
  ``BIND_BLOCK_STRIDED`` so threads cooperate on the axis.
- After each reduce ``BoundLoop`` (one whose immediate body contains an
  ``Accum``), a ``Combine(name, op, via=BLOCK_REDUCE)`` sibling is
  inserted so materialization emits the cross-thread combine.

Post-rewrite example (softmax)::

    Block(thread_axes=(), block_axes=(i,), body=(
      BoundLoop(k1, bind=BIND_BLOCK_STRIDED, body=(Load, Accum("acc_max", max))),
      Combine("acc_max", max, BLOCK_REDUCE),
      BoundLoop(k2, bind=BIND_BLOCK_STRIDED, body=(Load, Assign, Assign, Accum("acc_sum", add))),
      Combine("acc_sum", add, BLOCK_REDUCE),
      BoundLoop(k3, bind=BIND_BLOCK_STRIDED, body=(Load, Assign, Assign, Assign, Write)),
    ))

Trigger conditions:

- TileOp body has exactly one ``Block``.
- ``Block.thread_axes`` is 1D and ``block_axes`` is empty (idempotence).
- ``Block.body`` contains at least one reduce ``BoundLoop`` whose
  immediate body has exactly one ``Accum``.
- The first reduce BoundLoop's axis extent ≥ ``COOP_THRESHOLD``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, BoundAxis
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.ir.tile.ir import (
    COMBINE_BLOCK_REDUCE,
    Block,
    BoundLoop,
    Combine,
    Stmt,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

COOP_THRESHOLD = 128


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body = _maybe_rewrite(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    blocks = [(i, s) for i, s in enumerate(body) if isinstance(s, Block)]
    if len(blocks) != 1:
        return None
    idx, blk = blocks[0]
    if blk.block_axes:
        return None  # idempotence — already cooperative

    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Block) -> Block | None:
    if len(blk.thread_axes) != 1:
        return None

    reduce_loops = [s for s in blk.body if isinstance(s, BoundLoop) and _is_reduce(s)]
    if not reduce_loops:
        return None
    if int(reduce_loops[0].axis.extent) < COOP_THRESHOLD:
        return None
    for rl in reduce_loops:
        if sum(1 for s in rl.body if isinstance(s, Accum)) != 1:
            return None  # multi-Accum per Loop — online algorithms, punted

    # --- Flip bindings on every inner BoundLoop, insert Combine after reductions ---
    # Free output BoundLoops (no Accum in body) iterate output axes that are
    # cooperatively walked across the block's threads — lift those axes into
    # ``Block.axes`` with ``BIND_BLOCK_STRIDED`` so Block.axes documents the
    # full output shape. The BoundLoop stays in the body to drive iteration.
    new_body: list[Stmt] = []
    strided_output_axes: list[BoundAxis] = []
    for s in blk.body:
        if isinstance(s, BoundLoop):
            cooperative_axis = BoundAxis(axis=s.axis, bind=BIND_BLOCK_STRIDED)
            new_body.append(replace(s, axis=cooperative_axis))
            if _is_reduce(s):
                accum = next(a for a in s.body if isinstance(a, Accum))
                new_body.append(Combine(name=accum.name, op=accum.op, via=COMBINE_BLOCK_REDUCE))
            else:
                strided_output_axes.append(cooperative_axis)
        else:
            new_body.append(s)

    # Original output axes flip to BIND_BLOCK; cooperatively-walked axes
    # appended as BIND_BLOCK_STRIDED.
    new_axes = (
        *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in blk.axes),
        *strided_output_axes,
    )
    return Block(axes=new_axes, body=tuple(new_body))


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)
