"""Loop IR → Tile IR (single-thread, naive).

Mechanical translation: the body's leaves (``Load`` / ``Assign`` /
``Select`` / ``Write`` / ``Accum``) pass through unchanged — Tile IR
re-uses them directly. Only Loop IR's ``Loop`` needs translation: it
becomes a Tile IR ``FreeLoop`` (no Accum in body) or ``Reduce`` (Accums
present). Top-level non-Loop stmts (typically scalar Loads with empty
index) lift into ``Kernel.prologue`` so they sit above the tid guard at
render time.

The output Kernel has ``thread_axes == ()`` and ``block_axes == ()`` —
it's a fully-serial single-thread program. ``ExtractGlobalSchedule``
(step 4) strips the outer FreeLoop chain into ``thread_axes``.
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
    FreeLoop,
    Kernel,
    Param,
    Reduce,
    Stmt,
)


def lower_naive(loop_op: LoopOp, kernel_name: str, inputs: tuple[Param, ...], output: Param) -> Kernel:
    """Translate a ``LoopOp`` into a single-thread serial ``Kernel``.

    ``inputs`` and ``output`` populate ``Kernel.params`` (with ``Param.shape``
    used by the renderer to row-major-flatten multi-dim ``Load`` / ``Write``
    indices). Buffer identity is carried inline on Loop IR's ``Load.input``
    and ``Write.output`` — lowering only needs to translate ``Loop`` →
    ``FreeLoop`` / ``Reduce``.
    """
    pre: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        pre.append(rest[0])
        rest = rest[1:]

    return Kernel(
        name=kernel_name,
        params=(*inputs, output),
        body=tuple(_lower_body(rest)),
        prologue=tuple(_lower_body(tuple(pre))),
    )


def _lower_body(stmts: tuple[LoopStmt, ...]) -> list[Stmt]:
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop):
            out.append(_lower_loop(s))
        else:
            # Loop IR leaves (Load / Assign / Select / Write / Accum) pass
            # through — Tile IR's Stmt union admits them directly.
            out.append(s)
    return out


def _lower_loop(loop: Loop) -> Stmt:
    """Loop with Accum in body → ``Reduce``; otherwise ``FreeLoop``."""
    body = tuple(_lower_body(loop.body))
    if any(isinstance(s, Accum) for s in loop.body):
        return Reduce(axis=loop.axis, body=body)
    return FreeLoop(axis=loop.axis, body=body)


__all__ = ["lower_naive"]
