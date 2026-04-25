"""Cooperative-block reduction strategy.

Rewrites a 1D-reduction ``TileOp`` so each CUDA block owns one output
slot and threads in the block cooperate on the reduction axis via a
``__shared__`` tree-halve.

Pre-rewrite shape (produced by ``001_lower_loopop`` + ``lower_naive``)::

    Enclosure(thread_axes=(i,), block_axes=(),
      Tile(live_axes=(i,), extents=(M,),
        Loop(axis=k, body=(Load, Accum)),
        Write(out, index=(i,), value=acc),
      ))

Post-rewrite shape::

    Enclosure(thread_axes=(t,), block_axes=(i,),
      Tile(live_axes=(i,), extents=(M,),
        Smem(name=<acc>_smem, extents=(BLOCK,)),
        StridedLoop(axis=k, start=Var(t), step=BLOCK, body=(Load, Accum)),
        Write(<acc>_smem, index=(t,), value=acc),     # store partial
        Sync,
        TreeHalve(<acc>_smem, op, BLOCK, tid_var=t),
        Cond(t == 0, Load(acc_final, <acc>_smem, (0,)), Write(out, (i,), acc_final)),
      ))

Trigger conditions (all must hold; otherwise the TileOp passes through
unchanged):

- The TileOp body is exactly one ``Enclosure`` whose body is exactly one
  ``Tile`` with one live axis.
- The ``Tile.body`` is exactly one reduce ``Loop`` followed by exactly
  one ``Write`` of the loop's accumulator.
- The reduce ``Loop`` body is one ``Load`` + one ``Accum`` (single-input
  reduction — multi-Accum / online-softmax fusion is a follow-up).
- ``Loop.axis.extent`` ≥ ``COOP_THRESHOLD`` (below this, per-thread
  serial wins on launch overhead).

Block size: ``BLOCK = 256`` matched to the host-side launch geometry.
The reduction axis extent need not be a multiple of ``BLOCK``; the
``StridedLoop`` walks ``[t, t+BLOCK, t+2*BLOCK, ...)`` per thread which
naturally tails off.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Accum, Axis, Cond, Load, Loop, Write
from deplodock.compiler.ir.tile.ir import (
    Enclosure,
    Smem,
    StridedLoop,
    Sync,
    Tile,
    TileOp,
    TreeHalve,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

BLOCK = 256
COOP_THRESHOLD = 128


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body = _maybe_rewrite_body(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_rewrite_body(body: tuple) -> tuple | None:
    """Walk top-level body for a single Enclosure → Tile → reduce shape;
    rewrite it cooperatively if the trigger conditions hold."""
    enclosures = [(idx, s) for idx, s in enumerate(body) if isinstance(s, Enclosure)]
    if len(enclosures) != 1:
        return None
    idx, encl = enclosures[0]

    # Must not have been rewritten already.
    if encl.block_axes:
        return None

    rewritten = _rewrite_enclosure(encl)
    if rewritten is None:
        return None
    return body[:idx] + (rewritten,) + body[idx + 1 :]


def _rewrite_enclosure(encl: Enclosure) -> Enclosure | None:
    if len(encl.body) != 1 or not isinstance(encl.body[0], Tile):
        return None
    tile: Tile = encl.body[0]

    # Single-live-axis (1D-reduction) only — matmul / multi-axis live deferred.
    if len(tile.live_axes) != 1 or len(encl.thread_axes) != 1:
        return None
    if tile.live_axes != encl.thread_axes:
        return None

    if len(tile.body) != 2:
        return None
    reduce_loop, write = tile.body
    if not isinstance(reduce_loop, Loop) or not isinstance(write, Write):
        return None

    # Reduce-Loop body must be exactly Load + Accum on the same name.
    if len(reduce_loop.body) != 2:
        return None
    load, accum = reduce_loop.body
    if not isinstance(load, Load) or not isinstance(accum, Accum):
        return None
    if accum.value != load.name:
        return None
    if write.value != accum.name:
        return None
    if int(reduce_loop.axis.extent) < COOP_THRESHOLD:
        return None

    # --- Construct cooperative shape ---
    i_axis = tile.live_axes[0]
    k_axis = reduce_loop.axis
    t_axis = Axis(name="t", extent=BLOCK)
    smem_name = f"{accum.name}_smem"
    final_name = f"{accum.name}_final"

    new_tile_body: tuple = (
        Smem(name=smem_name, extents=(BLOCK,)),
        StridedLoop(
            axis=k_axis,
            start=Var(t_axis.name),
            step=BLOCK,
            body=(load, accum),
        ),
        Write(output=smem_name, index=(Var(t_axis.name),), value=accum.name),
        Sync(),
        TreeHalve(buf=smem_name, op=accum.op, length=BLOCK, tid_var=t_axis.name),
        Cond(
            cond=BinaryExpr("==", Var(t_axis.name), Literal(0, "int")),
            body=(
                Load(name=final_name, input=smem_name, index=(Literal(0, "int"),)),
                Write(output=write.output, index=write.index, value=final_name),
            ),
            else_body=(),
        ),
    )

    new_tile = Tile(live_axes=tile.live_axes, extents=tile.extents, body=new_tile_body)
    return Enclosure(thread_axes=(t_axis,), block_axes=(i_axis,), body=(new_tile,))
