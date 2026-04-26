"""Loop-IR walker that drives ``Expr.simplify`` over a LoopOp body.

The per-Expr rewrite logic (constant folding, identity collapse, range-based
comparison folding) lives on each ``Expr`` subclass as ``simplify(ctx)``;
this module only walks the LoopOp body, threading a ``SimplifyCtx`` whose
``ranges`` accumulate axis intervals as the walker descends through ``Loop``
blocks. Body Loads, Selects, and Writes carry Expr fields and get their
contents simplified; Assigns and Accums carry only SSA names so they pass
through unchanged.

Re-exports ``Interval`` and ``SimplifyCtx`` from :mod:`ir.expr` for callers.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.expr import Expr, Interval, SimplifyCtx
from deplodock.compiler.ir.loop.ir import (
    Load,
    Loop,
    LoopOp,
    Select,
    SelectBranch,
    Write,
)
from deplodock.compiler.ir.loop.ir import Stmt as LoopStmt

__all__ = [
    "Interval",
    "SimplifyCtx",
    "simplify_body",
    "simplify_loop_op",
]


def _simplify_expr_tuple(xs: tuple[Expr, ...], ctx: SimplifyCtx) -> tuple[Expr, ...]:
    return tuple(e.simplify(ctx) for e in xs)


def _simplify_loop_stmt(stmt: LoopStmt, ctx: SimplifyCtx) -> LoopStmt:
    if isinstance(stmt, Loop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return Loop(stmt.axis, tuple(_simplify_loop_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, Select):
        return Select(stmt.name, tuple(SelectBranch(b.value, b.select.simplify(ctx)) for b in stmt.branches))
    if isinstance(stmt, Write):
        return Write(stmt.output, _simplify_expr_tuple(stmt.index, ctx), stmt.value)
    if isinstance(stmt, Load):
        return Load(stmt.name, stmt.input, _simplify_expr_tuple(stmt.index, ctx))
    # Assign / Accum carry only SSA names — no Expr field to simplify.
    return stmt


def simplify_body(body: tuple[LoopStmt, ...]) -> tuple[LoopStmt, ...]:
    """Simplify every Expr inside a LoopOp body. Seeds ``SimplifyCtx`` from
    Loop extents as the walker descends."""
    ctx = SimplifyCtx.empty()
    return tuple(_simplify_loop_stmt(s, ctx) for s in body)


def simplify_loop_op(op: LoopOp) -> LoopOp:
    """Apply ``simplify_body`` to a LoopOp's body, returning a new LoopOp."""
    return replace(op, body=simplify_body(op.body))
