"""Loop IR → Tile IR (naive — produces the high-level ``Block`` form).

Mechanical translation. Loop IR's leaves (``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond``) pass through unchanged.
Loop-IR ``Loop`` is rewritten to Tile-IR ``BoundLoop`` with
``bind=BIND_SERIAL`` — the pre-strategy default (every thread walks
the axis itself). Strategy passes flip bindings on select ``BoundLoop``s
to express cooperative / threaded walks.

**Outer free-Loop chain → ``Block.thread_axes``**. After stripping
leading non-Loop stmts (scalar Loads) into the TileOp body prefix,
lowering walks the outer free-Loop chain and strips it into a
``Block(thread_axes=...)`` (default: one thread per output point, the
correct shape for pointwise kernels and for reductions the cooperative
strategy chooses not to rewrite). Strategies that opt into cooperative
materialization move the axes from ``thread_axes`` to ``block_axes``.

The chain ends at: a level with multiple sibling stmts, a Loop with
``Accum`` in its immediate body (reduce — can't strip), or no Loop at
all.
"""

from __future__ import annotations

from deplodock.compiler.ir.loop import (
    Accum,
    Loop,
    LoopOp,
)
from deplodock.compiler.ir.loop import (
    Stmt as LoopStmt,
)
from deplodock.compiler.ir.tile.ir import (
    BIND_SERIAL,
    Axis,
    Block,
    BoundLoop,
    Stmt,
    TileOp,
)


def lower_naive(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
    """Translate a ``LoopOp`` into a ``TileOp`` holding a logical ``Block``.

    Steps:

    1. Pull leading non-Loop stmts (typically scalar Loads) off the LoopOp
       body — they sit at the start of ``TileOp.body``, above any Block.
    2. Descend the outer free-Loop chain, collecting axes until the chain
       breaks (multi-stmt level, reduce Loop, or no more Loops).
    3. If any axes were collected, wrap the remaining inner body in a
       ``Block(output_axes=..., output_bind=BIND_THREAD)``. Otherwise,
       lower the inner body in place (single-thread serial — degenerate).

    Inner ``Loop``s are translated to ``BoundLoop(bind=BIND_SERIAL)``.
    Strategy passes flip bindings later.
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
        body.append(Block(thread_axes=output_axes, block_axes=(), body=inner_lowered))
    else:
        body.extend(inner_lowered)

    return TileOp(body=tuple(body), name=kernel_name)


def _strip_outer_free_chain(stmts: tuple[LoopStmt, ...]) -> tuple[tuple[Axis, ...], tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return ``(stripped_axes, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a Loop whose body contains an ``Accum`` at the immediate level (a
    reduce Loop — stripping it would lose the accumulator).
    """
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
            out.append(BoundLoop(axis=s.axis, body=tuple(_lower_body(s.body)), bind=BIND_SERIAL))
        else:
            out.append(s)
    return out


__all__ = ["lower_naive"]
