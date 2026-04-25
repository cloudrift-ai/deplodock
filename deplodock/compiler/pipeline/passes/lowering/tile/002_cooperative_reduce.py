"""Cooperative-reduce strategy — flips bindings on a ``Block`` to turn
serial per-thread reductions into multi-phase cooperative ones.

Reads the logical ``Block`` produced by ``lower_naive`` (default
``output_bind=BIND_THREAD``, inner ``BoundLoop``s all ``BIND_SERIAL``)
and rewrites it in place:

- ``Block.output_bind`` → ``BIND_BLOCK`` (one CUDA block per output slot).
- Every inner ``BoundLoop`` (reduction or free output) → ``BIND_STRIDED``
  so threads cooperate on the axis.
- After each reduce ``BoundLoop`` (one whose immediate body contains an
  ``Accum``), a ``Combine(name, op, via=SMEM_TREE_HALVE)`` sibling is
  inserted so materialization emits the cross-thread combine.

Post-rewrite example (softmax)::

    Block(output_axes=(i,), output_bind=BIND_BLOCK, body=(
      BoundLoop(k1, bind=BIND_STRIDED, body=(Load, Accum("acc_max", max))),
      Combine("acc_max", max, SMEM_TREE_HALVE),
      BoundLoop(k2, bind=BIND_STRIDED, body=(Load, Assign, Assign, Accum("acc_sum", add))),
      Combine("acc_sum", add, SMEM_TREE_HALVE),
      BoundLoop(k3, bind=BIND_STRIDED, body=(Load, Assign, Assign, Assign, Write)),
    ))

Trigger conditions:

- TileOp body has exactly one ``Block``.
- ``Block.output_axes`` is 1D and ``output_bind`` is still the default
  ``BIND_THREAD`` (idempotence check).
- ``Block.body`` contains at least one reduce ``BoundLoop`` whose
  immediate body has exactly one ``Accum``.
- The first reduce BoundLoop's axis extent ≥ ``COOP_THRESHOLD``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import Accum
from deplodock.compiler.ir.tile.ir import (
    BIND_BLOCK,
    BIND_STRIDED,
    BIND_THREAD,
    COMBINE_SMEM_TREE_HALVE,
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
    if blk.output_bind != BIND_THREAD:
        return None  # idempotence / already-rewritten

    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Block) -> Block | None:
    if len(blk.output_axes) != 1:
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
    new_body: list[Stmt] = []
    for s in blk.body:
        if isinstance(s, BoundLoop):
            new_body.append(replace(s, bind=BIND_STRIDED))
            if _is_reduce(s):
                accum = next(a for a in s.body if isinstance(a, Accum))
                new_body.append(Combine(name=accum.name, op=accum.op, via=COMBINE_SMEM_TREE_HALVE))
        else:
            new_body.append(s)

    return Block(output_axes=blk.output_axes, output_bind=BIND_BLOCK, body=tuple(new_body))


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)
