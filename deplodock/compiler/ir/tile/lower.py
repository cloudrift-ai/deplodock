"""Loop IR → Tile IR (single-thread, naive).

Mechanical translation. Loop IR's leaves (``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum`` / ``Cond``) pass through unchanged —
Tile IR re-uses them directly. Loop IR's ``Loop`` becomes:

- ``Reduce(axis, body)`` when the body has any ``Accum`` (carries the
  per-tile ``extent`` slot for later tiling strategies).
- ``Loop(axis, body)`` (Loop IR's own ``Loop``, re-used here) for free
  iteration. A future ``Enclosure`` will replace this when an axis is
  bound to a thread / block / cooperative coord.

Top-level non-Loop stmts (typically scalar Loads with empty index) lift
into ``Kernel.prologue`` so they sit above the tid guard at render time.

The output Kernel has ``thread_axes == ()`` and ``block_axes == ()`` —
it's a fully-serial single-thread program. ``ExtractGlobalSchedule``
(step 4) strips the outer free-Loop chain into ``thread_axes``.
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
    Param,
    Reduce,
    Stmt,
    TileOp,
)


def lower_naive(loop_op: LoopOp, kernel_name: str, inputs: tuple[Param, ...], output: Param) -> TileOp:
    """Translate a ``LoopOp`` into a single-thread serial ``TileOp``.

    ``inputs`` and ``output`` populate ``TileOp.params`` (with
    ``Param.shape`` used by the renderer to row-major-flatten multi-dim
    ``Load`` / ``Write`` indices). Buffer identity is carried inline on
    Loop IR's ``Load.input`` and ``Write.output`` — lowering only needs
    to translate ``Loop`` → ``Reduce`` (when body has Accum) or pass
    through. ``TileOp.body`` is a single sequence; scalar Loads and other
    pre-Enclosure stmts sit before any ``Enclosure`` introduced later by
    ``ExtractGlobalSchedule``.
    """
    return TileOp(
        body=tuple(_lower_body(loop_op.body)),
        params=(*inputs, output),
        name=kernel_name,
    )


def _lower_body(stmts: tuple[LoopStmt, ...]) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(_lower_loop(s))
        else:
            # Loop IR leaves pass through — Tile IR's Stmt union admits them.
            out.append(s)
    return out


def _lower_loop(loop: Loop) -> Stmt:
    """Loop with Accum in body → ``Reduce``; otherwise pass through as Loop
    (free iteration). The body recurses so nested Loops also translate."""
    body = tuple(_lower_body(loop.body))
    if any(isinstance(s, Accum) for s in loop.body):
        return Reduce(axis=loop.axis, body=body)
    return Loop(axis=loop.axis, body=body)


__all__ = ["lower_naive"]
