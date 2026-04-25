"""Lower each ``LoopOp`` node to a ``TileOp``.

Mechanical translation: outer free-Loop chain becomes
``Tile(thread_axes=...)``, leaves pass through unchanged. Loop-IR
``Loop`` is rewritten to Tile-IR ``BoundLoop`` carrying a ``BoundAxis``
whose ``bind`` defaults to ``BIND_SERIAL`` (every thread walks the axis
itself). Strategy passes flip the bind on select ``BoundLoop``s to
``BIND_BLOCK_STRIDED`` to express cooperative iteration.

**Outer free-Loop chain → ``Tile.thread_axes``**. After stripping
leading non-Loop stmts (scalar Loads) into the TileOp body prefix,
lowering walks the outer free-Loop chain and strips it into a
``Tile(thread_axes=...)`` (default: one thread per output point — the
correct shape for pointwise kernels and for reductions the cooperative
strategy chooses not to rewrite). The chain ends at: a level with
multiple sibling stmts, a Loop with ``Accum`` in its immediate body
(reduce — can't strip), or no Loop at all.

The node's id, inputs, and output tensor are preserved — only the op
changes.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Loop
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.tile.ir import BoundLoop, Stmt, Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, LoopOp):
        return None
    kname = _kernel_name_for(node.op, match.root_node_id)
    node.op = lower_naive(node.op, kname)
    return None


def lower_naive(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
    """Translate a ``LoopOp`` into a ``TileOp`` holding a logical ``Tile``.

    Steps:

    1. Pull leading non-Loop stmts (typically scalar Loads) off the LoopOp
       body — they sit at the start of ``TileOp.body``, above any Tile.
    2. Descend the outer free-Loop chain, collecting axes until the chain
       breaks (multi-stmt level, reduce Loop, or no more Loops).
    3. If any axes were collected, wrap the remaining inner body in a
       ``Tile(thread_axes=..., bind=BIND_THREAD)``. Otherwise, lower the
       inner body in place (single-thread serial — degenerate).

    Inner ``Loop``s are translated to
    ``BoundLoop(BoundAxis(axis, BIND_SERIAL))``. Strategy passes flip
    the bind on select BoundLoops later.
    """
    leading: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]

    output_axes, inner = _strip_outer_free_chain(rest)

    body: list[Stmt] = list(_lower_body(tuple(leading)))
    inner_lowered = tuple(_lower_body(inner))
    if output_axes:
        bound = tuple(BoundAxis(axis=ax, bind=BIND_THREAD) for ax in output_axes)
        body.append(Tile(axes=bound, body=inner_lowered))
    else:
        body.extend(inner_lowered)

    return TileOp(body=tuple(body), name=kernel_name)


def _strip_outer_free_chain(stmts: tuple[LoopStmt, ...]) -> tuple[tuple[Axis, ...], tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return ``(stripped_axes, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a Loop whose body contains an ``Accum`` at the immediate level (a
    reduce Loop — stripping it would lose the accumulator)."""
    axes: list[Axis] = []
    cur = stmts
    while len(cur) == 1 and isinstance(cur[0], Loop) and not any(isinstance(s, Accum) for s in cur[0].body):
        axes.append(cur[0].axis)
        cur = cur[0].body
    return tuple(axes), cur


def _lower_body(stmts: tuple[LoopStmt, ...]) -> list[Stmt]:
    """Translate Loop-IR stmts to Tile-IR. Loop → BoundLoop(bind=SERIAL);
    leaves pass through unchanged."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(BoundLoop(axis=BoundAxis(axis=s.axis, bind=BIND_SERIAL), body=tuple(_lower_body(s.body))))
        else:
            out.append(s)
    return out


def _kernel_name_for(loop: LoopOp, node_id: str) -> str:
    if any(isinstance(s, Accum) for s in loop):
        return f"k_{node_id}_reduce"
    return f"k_{node_id}_pointwise"
