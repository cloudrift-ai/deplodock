"""Cooperative-reduce strategy — converts each cooperative axis from a
serial ``Loop`` to a ``StridedLoop`` driven by the cooperative thread
axis ``t``.

Reads the logical ``Tile`` produced by ``lower_naive`` (default:
``thread_axes=output_axes`` / ``block_axes=()``, inner Loops all
serial) and rewrites:

- The Tile's output axis (``thread_axes``) flips to ``BIND_BLOCK`` —
  one CUDA block per output slot.
- A synthetic cooperative thread axis ``t`` (``BLOCK_SIZE`` threads)
  is added to ``Tile.axes`` as ``BIND_THREAD``.
- Each inner ``Loop`` becomes a ``StridedLoop(axis, start=Var("t"),
  step=BLOCK_SIZE)`` — threads of the block stride through the axis.
  Body indices stay as ``Var(axis.name)`` (no rewriting); the strided
  iteration is encoded by the loop construct itself.
- After each reduce loop, a ``Combine(name, op)`` sibling is inserted;
  materialization emits the cross-thread tree-halve over smem indexed
  by ``t``.

Post-rewrite example (softmax)::

    Tile(axes=(t=THREAD, i=BLOCK), body=(
      StridedLoop(k1, start=t, step=256, body=(
        Load input[i, k1],
        Accum("acc_max", max),
      )),
      Combine("acc_max", max),
      StridedLoop(k2, start=t, step=256, body=(
        Load input[i, k2],
        Assign, Assign,
        Accum("acc_sum", add),
      )),
      Combine("acc_sum", add),
      StridedLoop(k3, start=t, step=256, body=(
        Load input[i, k3],
        Assign, Assign, Assign,
        Write merged[i, k3],
      )),
    ))

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is non-empty and ``block_axes`` is empty
  (idempotence). Multi-axis output tiles (e.g. softmax over the last
  dim of a 4D tensor) are handled by promoting all existing thread
  axes to ``BIND_BLOCK`` alongside the synthetic cooperative ``t``.
- ``Tile.body`` contains at least one reduce ``Loop`` whose immediate
  body has exactly one ``Accum``.
- The first reduce Loop's axis extent ≥ ``BLOCK_SIZE``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Loop, StridedLoop
from deplodock.compiler.ir.tile.ir import (
    BLOCK_SIZE,
    Combine,
    Stmt,
    Tile,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    blocks = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(blocks) != 1:
        raise RuleSkipped(f"need exactly one Tile in TileOp.body, found {len(blocks)}")
    idx, blk = blocks[0]
    if blk.block_axes:
        raise RuleSkipped("Tile already cooperative (block_axes non-empty)")

    new_blk = _rewrite_block(blk)
    if new_blk is None:
        return None
    return body[:idx] + (new_blk,) + body[idx + 1 :]


def _rewrite_block(blk: Tile) -> Tile | None:
    if not blk.thread_axes:
        raise RuleSkipped("Tile has no thread_axes to convert to BLOCK")

    reduce_loops = [loop for loop in blk.loops if loop.is_reduce]
    if not reduce_loops:
        raise RuleSkipped("Tile body has no reduce Loop")
    if int(reduce_loops[0].axis.extent) < BLOCK_SIZE:
        raise RuleSkipped(f"first reduce-axis extent {int(reduce_loops[0].axis.extent)} < BLOCK_SIZE={BLOCK_SIZE}")
    for rl in reduce_loops:
        if sum(1 for s in rl.body if isinstance(s, Accum)) != 1:
            raise RuleSkipped(f"reduce Loop {rl.axis.name!r} has multiple Accums (online algorithm — punted)")

    t_axis = Axis("t", BLOCK_SIZE)
    t_start = Var(t_axis.name)
    step = Literal(BLOCK_SIZE, "int")

    # Each body Loop becomes a StridedLoop driven by t. Body untouched.
    # Reduce loops get a Combine sibling for the cross-thread tree-halve.
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
