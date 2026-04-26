"""Cooperative-reduce strategy — splits each cooperative axis into
``(chunk_outer, t)`` and rewrites the body's references to the original
axis as ``chunk * BLOCK_SIZE + t``.

Reads the logical ``Tile`` produced by ``lower_naive`` (default:
``thread_axes=output_axes`` / ``block_axes=()``, inner ``BoundLoop``s
all ``BIND_SERIAL``) and rewrites:

- The Tile's output axis (``thread_axes``) flips to ``BIND_BLOCK`` —
  one CUDA block per output slot.
- A synthetic cooperative thread axis ``t`` (``BLOCK_SIZE`` threads)
  is added to ``Tile.axes`` as ``BIND_THREAD``.
- Each inner ``BoundLoop`` (extent must be a multiple of BLOCK_SIZE)
  becomes a serial loop over ``chunk = old_axis.extent / BLOCK_SIZE``
  iterations. The body's references to the original axis are
  σ-substituted to ``chunk * BLOCK_SIZE + t``.
- After each reduce ``BoundLoop``, a ``Combine(name, op)`` sibling is
  inserted; materialization emits the cross-thread tree-halve over
  smem indexed by ``t``.

Post-rewrite example (softmax)::

    Tile(axes=(t=THREAD, i=BLOCK), body=(
      BoundLoop(k1_chunk=SERIAL, body=(
        Load input[i, k1_chunk*256 + t],
        Accum("acc_max", max),
      )),
      Combine("acc_max", max),
      BoundLoop(k2_chunk=SERIAL, body=(
        Load input[i, k2_chunk*256 + t],
        Assign, Assign,
        Accum("acc_sum", add),
      )),
      Combine("acc_sum", add),
      BoundLoop(k3_chunk=SERIAL, body=(
        Load input[i, k3_chunk*256 + t],
        Assign, Assign, Assign,
        Write merged[i, k3_chunk*256 + t],
      )),
    ))

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is 1D and ``block_axes`` is empty (idempotence).
- ``Tile.body`` contains at least one reduce ``BoundLoop`` whose
  immediate body has exactly one ``Accum``.
- Every body BoundLoop's axis extent is divisible by ``BLOCK_SIZE``
  (the explicit-split form requires it; non-divisible extents would
  need a residue-tail story that no current strategy wants).
- The first reduce BoundLoop's axis extent ≥ ``BLOCK_SIZE``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.ir.tile.ir import (
    BLOCK_SIZE,
    BoundLoop,
    Combine,
    Stmt,
    Tile,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]


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
    blocks = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(blocks) != 1:
        return None
    idx, blk = blocks[0]
    if blk.block_axes:
        return None  # idempotence — already cooperative

    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Tile) -> Tile | None:
    if len(blk.thread_axes) != 1:
        return None

    reduce_loops = [s for s in blk.body if isinstance(s, BoundLoop) and _is_reduce(s)]
    if not reduce_loops:
        return None
    if int(reduce_loops[0].axis.extent) < BLOCK_SIZE:
        return None
    for rl in reduce_loops:
        if sum(1 for s in rl.body if isinstance(s, Accum)) != 1:
            return None  # multi-Accum per Loop — online algorithms, punted

    # Every body BoundLoop must split cleanly by BLOCK_SIZE for the
    # explicit-split form to apply.
    for s in blk.body:
        if isinstance(s, BoundLoop) and int(s.axis.extent) % BLOCK_SIZE != 0:
            return None

    t_axis = Axis("t", BLOCK_SIZE)

    # For each inner BoundLoop, split axis → (chunk, t). Body references
    # to the original axis become ``chunk * BLOCK_SIZE + t``. Reduce
    # loops get a Combine sibling for the cross-thread tree-halve.
    new_body: list[Stmt] = []
    for s in blk.body:
        if isinstance(s, BoundLoop):
            old_axis = s.axis.axis
            chunk_axis = Axis(f"{old_axis.name}_chunk", int(old_axis.extent) // BLOCK_SIZE)
            sigma = Sigma({old_axis.name: Var(chunk_axis.name) * Literal(BLOCK_SIZE, "int") + Var(t_axis.name)})
            new_inner = tuple(stmt.rewrite(_id, sigma) for stmt in s.body)
            new_body.append(BoundLoop(axis=BoundAxis(axis=chunk_axis, bind=BIND_SERIAL), body=new_inner))
            if _is_reduce(s):
                accum = next(a for a in s.body if isinstance(a, Accum))
                new_body.append(Combine(name=accum.name, op=accum.op))
        else:
            new_body.append(s)

    # ``Tile.axes`` carries strictly launch geometry: the synthesized
    # ``t`` THREAD axis plus the original output axes as BLOCK.
    new_axes = (
        BoundAxis(axis=t_axis, bind=BIND_THREAD),
        *(BoundAxis(axis=ba.axis, bind=BIND_BLOCK) for ba in blk.axes),
    )
    return Tile(axes=new_axes, body=tuple(new_body))


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


def _id(name: str) -> str:
    return name
