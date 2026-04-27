"""Cooperative-reduce — convert each serial reduce ``Loop`` to a
``StridedLoop`` driven by a cooperative thread axis ``t``.

Fires only on Tiles that ``002_blockify_thread_axes`` left untouched
(i.e., the Tile still has all-thread axes and no block axes — typical
for kernels whose output is too small to blockify, e.g. RMSNorm with
1D output below ``BLOCK_TG``).

Post-rewrite shape (single output axis ``i``)::

    Tile(axes=(t=THREAD, i=BLOCK), body=(
      StridedLoop(k1, start=t, step=BLOCK_SIZE, body=(... Accum)),
      Combine(acc, op=...),
      ...
    ))
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Loop, StridedLoop
from deplodock.compiler.ir.tile.ir import BLOCK_SIZE, Combine, Stmt, Tile, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    blocks = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(blocks) != 1:
        return None
    idx, blk = blocks[0]
    if blk.block_axes:
        return None  # blockify already claimed the launch geometry
    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Tile) -> Tile | None:
    if len(blk.thread_axes) != 1:
        return None
    reduce_loops = [s for s in blk.body if isinstance(s, Loop) and s.is_reduce]
    if not reduce_loops:
        return None
    if int(reduce_loops[0].axis.extent) < BLOCK_SIZE:
        return None
    for rl in reduce_loops:
        if sum(1 for s in rl.body if isinstance(s, Accum)) != 1:
            return None  # multi-Accum per Loop punted

    t_axis = Axis("t", BLOCK_SIZE)
    t_start = Var(t_axis.name)
    step = Literal(BLOCK_SIZE, "int")

    new_body: list[Stmt] = []
    for s in blk.body:
        if isinstance(s, Loop):
            new_body.append(StridedLoop(axis=s.axis, start=t_start, step=step, body=s.body))
            if s.is_reduce:
                accum = next(a for a in s.body if isinstance(a, Accum))
                new_body.append(Combine(name=accum.name, op=accum.op))
        else:
            new_body.append(s)

    new_axes = (
        BoundAxis(axis=t_axis, bind=BIND_THREAD),
        *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in blk.axes),
    )
    return Tile(axes=new_axes, body=tuple(new_body))
