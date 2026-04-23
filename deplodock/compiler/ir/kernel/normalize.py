"""Structural normalization passes for ``GpuKernel`` bodies.

Invoked from ``KernelOp.__post_init__`` so every constructed ``KernelOp`` —
including intermediate results produced by lowering — lands in a
canonical shape. Pure ``GpuKernel → GpuKernel`` transforms; idempotent.

Shares the generic ``simplify_expr`` / range analysis with ``ir/simplify.py``
— this module only carries the statement-level walker plus the
kernel-specific range seeding (``ForLoop`` bounds, provably-nonneg
thread-index compositions, ``IfStmt`` condition tightening).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.expr import (
    BinOp,
    Builtin,
    Cast,
    Literal,
    Var,
)
from deplodock.compiler.ir.kernel.ir import (
    ArrayAccess,  # noqa: F401  — re-used by simplify_expr via isinstance checks
    ArrayDecl,
    AugAssign,
    FieldAccess,  # noqa: F401
    ForLoop,
    GpuKernel,
    IfStmt,
    PragmaUnroll,
    RawCode,
    SyncThreads,
    VarAssign,
    VarDecl,
    VectorLoad,  # noqa: F401
)
from deplodock.compiler.ir.kernel.ir import Assign as GpuAssign
from deplodock.compiler.ir.kernel.ir import Stmt as GpuStmt
from deplodock.compiler.ir.simplify import Context, Interval, simplify_expr

__all__ = ["normalize_kernel"]


_INT_MAX = 2**31 - 1

_NONNEG_VAR_PREFIXES = ("threadIdx.", "blockIdx.", "blockDim.", "gridDim.")


def _for_bounds(stmt: ForLoop) -> Interval | None:
    """Return ``Interval(start, end-1)`` for a ForLoop with literal bounds."""
    if isinstance(stmt.start, Literal) and isinstance(stmt.end, Literal):
        try:
            lo = int(stmt.start.value)
            hi = int(stmt.end.value) - 1
            if lo <= hi:
                return Interval(lo, hi)
        except (TypeError, ValueError):
            return None
    return None


def _is_nonneg_composition(expr: object, ctx: Context) -> bool:
    """True when ``expr`` is provably ``>= 0``.

    Recognizes nonneg compositions of:
    - ``Builtin`` (the dedicated GPU-builtin node type, always nonneg).
    - ``Var`` with a CUDA thread-index name (``threadIdx.x``, ``blockIdx.y``,
      ``blockDim.z``, ``gridDim.x``, plus ``warpSize``) — the CUDA emitter
      renders these as plain Var nodes, not Builtin.
    - Nonneg ``Literal``s.
    - ``Var``s already known nonneg in ``ctx``.
    - ``+`` / ``*`` / ``/`` / ``%`` of nonneg operands (``-`` is rejected).

    Used by the ``VarDecl`` walker to push a lower bound of 0 onto variables
    initialized from thread-index expressions (``tid = bx*bdx + tx``); an
    enclosing ``IfStmt`` then tightens the upper bound.
    """
    if isinstance(expr, Literal):
        try:
            return expr.value >= 0
        except TypeError:
            return False
    if isinstance(expr, Builtin):
        return True
    if isinstance(expr, Var):
        if expr.name == "warpSize" or expr.name.startswith(_NONNEG_VAR_PREFIXES):
            return True
        r = ctx.ranges.get(expr.name)
        return r is not None and r.lo >= 0
    if isinstance(expr, BinOp):
        if expr.op in ("+", "*", "/", "//", "%"):
            return _is_nonneg_composition(expr.left, ctx) and _is_nonneg_composition(expr.right, ctx)
        return False
    if isinstance(expr, Cast):
        return _is_nonneg_composition(expr.expr, ctx)
    return False


def _tighten_from_cond(cond: object, ctx: Context) -> Context:
    """If ``cond`` is ``Var OP Literal`` (or reversed), intersect the Var's
    range in ``ctx`` with the implication. Returns a new Context (unchanged
    when the condition doesn't yield a single-Var tightening).
    """
    if not isinstance(cond, BinOp):
        return ctx
    left, right, op = cond.left, cond.right, cond.op

    if isinstance(left, Var) and isinstance(right, Literal) and isinstance(right.value, int):
        var, lit, flip = left.name, int(right.value), False
    elif isinstance(right, Var) and isinstance(left, Literal) and isinstance(left.value, int):
        var, lit, flip = right.name, int(left.value), True
    else:
        return ctx

    if flip:
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "=="}.get(op, op)

    cur = ctx.ranges.get(var, Interval(-_INT_MAX, _INT_MAX))
    if op == "<":
        new = Interval(cur.lo, min(cur.hi, lit - 1))
    elif op == "<=":
        new = Interval(cur.lo, min(cur.hi, lit))
    elif op == ">":
        new = Interval(max(cur.lo, lit + 1), cur.hi)
    elif op == ">=":
        new = Interval(max(cur.lo, lit), cur.hi)
    elif op == "==":
        new = Interval(max(cur.lo, lit), min(cur.hi, lit))
    else:
        return ctx
    if new.lo > new.hi:
        return ctx
    return ctx.extend(var, new)


def _simplify_gpu_stmt(stmt: GpuStmt, ctx: Context) -> tuple[GpuStmt, Context]:
    """Simplify one statement and return the ctx to use for *subsequent*
    siblings. A ``VarDecl`` with a provably-nonneg init publishes a lower
    bound of 0 for its variable; an ``IfStmt`` narrows Var ranges only
    inside its body (returned ctx is the caller's, unchanged)."""
    if isinstance(stmt, VarDecl):
        new_init = simplify_expr(stmt.init, ctx) if stmt.init is not None else None
        new_ctx = ctx
        if new_init is not None and _is_nonneg_composition(new_init, ctx):
            new_ctx = ctx.extend(stmt.name, Interval(0, _INT_MAX))
        return VarDecl(stmt.dtype, stmt.name, new_init), new_ctx  # type: ignore[arg-type]
    if isinstance(stmt, VarAssign):
        return VarAssign(stmt.name, simplify_expr(stmt.value, ctx)), ctx  # type: ignore[arg-type]
    if isinstance(stmt, AugAssign):
        return AugAssign(stmt.target, stmt.op, simplify_expr(stmt.value, ctx)), ctx  # type: ignore[arg-type]
    if isinstance(stmt, GpuAssign):
        new_target = simplify_expr(stmt.target, ctx)
        new_value = simplify_expr(stmt.value, ctx)
        return GpuAssign(new_target, new_value), ctx  # type: ignore[arg-type]
    if isinstance(stmt, ForLoop):
        bounds = _for_bounds(stmt)
        inner = ctx.extend(stmt.var, bounds) if bounds is not None else ctx
        new_start = simplify_expr(stmt.start, ctx)
        new_end = simplify_expr(stmt.end, ctx)
        new_step = simplify_expr(stmt.step, ctx) if stmt.step is not None else None
        new_body = _simplify_gpu_body(stmt.body, inner)
        return ForLoop(stmt.var, new_start, new_end, new_body, new_step), ctx  # type: ignore[arg-type]
    if isinstance(stmt, IfStmt):
        new_cond = simplify_expr(stmt.cond, ctx)
        body_ctx = _tighten_from_cond(new_cond, ctx)
        new_body = _simplify_gpu_body(stmt.body, body_ctx)
        new_else = _simplify_gpu_body(stmt.else_body, ctx) if stmt.else_body else None
        return IfStmt(new_cond, new_body, new_else), ctx  # type: ignore[arg-type]
    if isinstance(stmt, ArrayDecl):
        new_init = simplify_expr(stmt.init, ctx) if stmt.init is not None else None
        return ArrayDecl(stmt.dtype, stmt.name, stmt.dimensions, new_init), ctx  # type: ignore[arg-type]
    if isinstance(stmt, (SyncThreads, PragmaUnroll, RawCode)):
        return stmt, ctx
    return stmt, ctx


def _simplify_gpu_body(stmts: list[GpuStmt], ctx: Context) -> list[GpuStmt]:
    """Simplify a sequence of statements, threading Context so each statement
    sees VarDecl-induced range facts published by its predecessors."""
    out: list[GpuStmt] = []
    cur = ctx
    for s in stmts:
        new_s, cur = _simplify_gpu_stmt(s, cur)
        out.append(new_s)
    return out


def normalize_kernel(kernel: GpuKernel) -> GpuKernel:
    """Simplify every Expr inside a ``GpuKernel``'s statement tree.

    Seeds Context from enclosing ``ForLoop`` bounds (literal start/end
    only), ``VarDecl`` inits provably ≥ 0, and ``IfStmt`` condition
    tightening. Idempotent; running twice yields the same AST.
    """
    return replace(kernel, body=_simplify_gpu_body(kernel.body, Context.empty()))
