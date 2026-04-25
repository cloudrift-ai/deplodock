"""Loop IR → Tile IR (naive — outer free axes → thread_axes).

Mechanical translation. Loop IR's leaves (``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond``) pass through unchanged —
Tile IR re-uses them directly. Loop IR's ``Loop`` is reused as-is in
Tile IR; reductions are detected structurally (a Loop is a reduce-Loop
iff its body contains an ``Accum``) by downstream passes / the renderer.

**Outer free-Loop chain → ``Enclosure(thread_axes=...)``**. After
stripping leading non-Loop stmts (scalar Loads) into the TileOp body
prefix, lowering walks the outer free-Loop chain and strips it into an
``Enclosure``. Each stripped axis becomes a ``thread_axes`` entry; the
inside of the deepest stripped Loop becomes the Enclosure body.

The chain ends at: a level with multiple sibling stmts, a Loop with
``Accum`` in its immediate body (reduce — can't strip), or no Loop at
all. ``block_axes`` stays empty here — a future strategy splits some
``thread_axes`` off into block-tile bindings.
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
    Axis,
    Enclosure,
    Stmt,
    TileOp,
)


def lower_naive(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
    """Translate a ``LoopOp`` into a ``TileOp`` with thread axes extracted.

    Steps:

    1. Pull leading non-Loop stmts (typically scalar Loads) off the LoopOp
       body — they sit at the start of ``TileOp.body``, above any Enclosure.
    2. Descend the outer free-Loop chain, collecting axes until the chain
       breaks (multi-stmt level, reduce Loop, or no more Loops).
    3. If any axes were collected, wrap the remaining inner body in an
       ``Enclosure(thread_axes=...)``. Otherwise, lower the inner body in
       place (single-thread serial — degenerate kernel).

    ``block_axes`` is empty here — a later strategy splits some axes off
    into block-tile bindings. Buffer parameters are not stored on TileOp;
    the renderer derives them from ``TileOp.inputs`` / ``TileOp.output_bufs``
    (computed from body Loads / Writes) and looks up shapes from the
    surrounding graph.
    """
    leading: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]

    thread_axes, inner = _strip_outer_free_chain(rest)

    body: list[Stmt] = list(_lower_body(tuple(leading)))
    inner_lowered = tuple(_lower_body(inner))
    if thread_axes:
        body.append(Enclosure(thread_axes=thread_axes, block_axes=(), body=inner_lowered))
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
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(Loop(axis=s.axis, body=tuple(_lower_body(s.body))))
        else:
            # Loop IR leaves pass through — Tile IR's Stmt union admits them.
            out.append(s)
    return out


__all__ = ["lower_naive"]
