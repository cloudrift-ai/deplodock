"""Generic Expr / IR simplification — applied at every pipeline stage.

Single bottom-up pass over the shared ``Expr`` AST (``ir.expr``) plus the
GPU-specific extensions (``ir.kernel``). Rules:

- Constant folding: any ``BinaryExpr`` or ``CastExpr`` of ``Literal`` children folds
  via ``BinaryExpr.eval({})`` / ``CastExpr`` dispatch (Literal.eval ignores env).
- Algebraic identities: ``x+0``, ``x-0``, ``x*0``, ``x*1``, ``x/1``, ``x%1``,
  ``x-x``, ``x&&True``, ``x||False``, etc.
- TernaryExpr collapse: ``Literal(c) ? a : b`` → a or b; equal branches → branch.
- Range-based comparison folding: when an ``Axis``' extent or a ``ForLoop``'s
  static bounds prove a comparison result, collapse to ``Literal(0/1)``.
  Combined with TernaryExpr collapse this erases chained index clamps like
  ``(k0 > 2047 ? 2047 : k0) < 0 ? 0 : ...`` down to ``k0``.

Idempotent: running twice yields the same AST. Pure: ``Expr → Expr``.

Entry points:

- ``simplify_expr(e, ctx)`` — Expr rewriter; covers every shared Expr type
  (``Var | Literal | BinaryExpr | Builtin | FuncCallExpr | TernaryExpr | CastExpr``) plus
  the GPU extensions (``ArrayAccess | FieldAccess | VectorLoad``).
- ``simplify_loop_op(op)`` — walks a ``LoopOp``, seeding Context from its
  ``Axis`` extents, rewriting every Expr in ``Load.index`` / ``Select`` /
  ``Write`` / ``Accum.init``.

Kernel-IR walking lives in ``ir.kernel.normalize``; it re-uses
``Context`` / ``Interval`` / ``simplify_expr`` from here.

Range analysis (``infer_range``) tracks integer intervals only; ``Builtin``,
``FuncCallExpr``, ``CastExpr`` return ``None`` (unknown range → no comparison
folding through them, but surrounding arithmetic can still constant-fold).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.kernel.ir import ArrayAccess, FieldAccess, VectorLoad
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
    "Context",
    "simplify_expr",
    "infer_range",
    "simplify_body",
    "simplify_loop_op",
]


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Closed integer interval ``[lo, hi]``. Used for range-based folding."""

    lo: int
    hi: int


@dataclass
class Context:
    """Range info available at a given scope. Immutable per-call; callers
    build a fresh Context (or ``extend`` copy) when pushing into a nested
    scope, so the pass stays pure."""

    ranges: dict[str, Interval]

    @classmethod
    def empty(cls) -> Context:
        return cls({})

    def extend(self, name: str, interval: Interval) -> Context:
        return Context({**self.ranges, name: interval})


# ---------------------------------------------------------------------------
# Small predicates
# ---------------------------------------------------------------------------


def _is_literal(e: object) -> bool:
    return isinstance(e, Literal)


def _is_zero(e: object) -> bool:
    return isinstance(e, Literal) and e.value == 0


def _is_one(e: object) -> bool:
    return isinstance(e, Literal) and e.value == 1


def _is_truthy(e: object) -> bool:
    return isinstance(e, Literal) and bool(e.value)


def _is_falsy(e: object) -> bool:
    return isinstance(e, Literal) and not bool(e.value)


# ---------------------------------------------------------------------------
# Range analysis
# ---------------------------------------------------------------------------


def infer_range(expr: object, ctx: Context) -> Interval | None:
    """Conservative integer-range analysis over ``Expr``. Returns None when
    the range is unknown — callers must treat that as "no fold"."""
    if isinstance(expr, Literal):
        if isinstance(expr.value, int) and not isinstance(expr.value, bool):
            return Interval(expr.value, expr.value)
        return None
    if isinstance(expr, Var):
        return ctx.ranges.get(expr.name)
    if isinstance(expr, BinaryExpr):
        la = infer_range(expr.left, ctx)
        lb = infer_range(expr.right, ctx)
        if la is None or lb is None:
            # comparisons still have [0,1] range even if operands unknown
            if expr.op in ("<", "<=", ">", ">=", "==", "&&", "||"):
                return Interval(0, 1)
            return None
        op = expr.op
        if op == "+":
            return Interval(la.lo + lb.lo, la.hi + lb.hi)
        if op == "-":
            return Interval(la.lo - lb.hi, la.hi - lb.lo)
        if op == "*":
            prods = [la.lo * lb.lo, la.lo * lb.hi, la.hi * lb.lo, la.hi * lb.hi]
            return Interval(min(prods), max(prods))
        if op in ("/", "//"):
            if lb.lo == lb.hi and lb.lo > 0:
                return Interval(la.lo // lb.lo, la.hi // lb.lo)
            return None
        if op == "%":
            if lb.lo == lb.hi and lb.lo > 0:
                # assuming nonneg dividend, which holds for every loop index
                return Interval(0, lb.lo - 1)
            return None
        if op in ("<", "<=", ">", ">=", "==", "&&", "||"):
            return Interval(0, 1)
        return None
    # Builtin / FuncCallExpr / CastExpr / TernaryExpr: unknown
    return None


def _static_cmp(op: str, la: Interval, lb: Interval) -> bool | None:
    """Decide a comparison statically from operand ranges, or return None."""
    if op == "<":
        if la.hi < lb.lo:
            return True
        if la.lo >= lb.hi:
            return False
        return None
    if op == "<=":
        if la.hi <= lb.lo:
            return True
        if la.lo > lb.hi:
            return False
        return None
    if op == ">":
        if la.lo > lb.hi:
            return True
        if la.hi <= lb.lo:
            return False
        return None
    if op == ">=":
        if la.lo >= lb.hi:
            return True
        if la.hi < lb.lo:
            return False
        return None
    if op == "==":
        if la.hi < lb.lo or la.lo > lb.hi:
            return False
        if la.lo == la.hi == lb.lo == lb.hi:
            return True
        return None
    return None


# ---------------------------------------------------------------------------
# Expr simplification
# ---------------------------------------------------------------------------


def _make_int_literal(v: int) -> Literal:
    return Literal(int(v), "int")


def _fold_binop_literals(op: str, left: Literal, right: Literal) -> Literal:
    """Constant-fold a BinaryExpr whose children are both Literal. Preserves int
    dtype when both operands are int-typed and the result is integral."""
    folded = BinaryExpr(op, left, right).eval({})
    if isinstance(folded, bool):
        return _make_int_literal(1 if folded else 0)
    both_int = left.dtype == "int" and right.dtype == "int"
    if isinstance(folded, int) and both_int:
        return _make_int_literal(folded)
    return Literal(float(folded), "float")


def simplify_expr(expr: object, ctx: Context) -> object:
    """Bottom-up rewrite of an Expr (or GPU-extended Expr). Pure."""
    if isinstance(expr, (Var, Literal, Builtin)):
        return expr

    if isinstance(expr, BinaryExpr):
        left = simplify_expr(expr.left, ctx)
        right = simplify_expr(expr.right, ctx)
        op = expr.op

        if isinstance(left, Literal) and isinstance(right, Literal):
            return _fold_binop_literals(op, left, right)

        if op == "+":
            if _is_zero(left):
                return right
            if _is_zero(right):
                return left
        elif op == "-":
            if _is_zero(right):
                return left
            if left == right:
                return _make_int_literal(0)
        elif op == "*":
            if _is_zero(left) or _is_zero(right):
                return _make_int_literal(0)
            if _is_one(left):
                return right
            if _is_one(right):
                return left
        elif op in ("/", "//"):
            if _is_one(right):
                return left
            if _is_zero(left):
                return _make_int_literal(0)
        elif op == "%":
            if _is_one(right):
                return _make_int_literal(0)
            if _is_zero(left):
                return _make_int_literal(0)
        elif op == "&&":
            if _is_truthy(left):
                return right
            if _is_truthy(right):
                return left
            if _is_falsy(left) or _is_falsy(right):
                return _make_int_literal(0)
        elif op == "||":
            if _is_falsy(left):
                return right
            if _is_falsy(right):
                return left
            if _is_truthy(left) or _is_truthy(right):
                return _make_int_literal(1)
        elif op in ("<", "<=", ">", ">=", "=="):
            la = infer_range(left, ctx)
            lb = infer_range(right, ctx)
            if la is not None and lb is not None:
                decided = _static_cmp(op, la, lb)
                if decided is not None:
                    return _make_int_literal(1 if decided else 0)

        if left is expr.left and right is expr.right:
            return expr
        return BinaryExpr(op, left, right)

    if isinstance(expr, TernaryExpr):
        cond = simplify_expr(expr.cond, ctx)
        if _is_truthy(cond):
            return simplify_expr(expr.if_true, ctx)
        if _is_falsy(cond):
            return simplify_expr(expr.if_false, ctx)
        a = simplify_expr(expr.if_true, ctx)
        b = simplify_expr(expr.if_false, ctx)
        if a == b:
            return a
        if cond is expr.cond and a is expr.if_true and b is expr.if_false:
            return expr
        return TernaryExpr(cond, a, b)

    if isinstance(expr, CastExpr):
        inner = simplify_expr(expr.expr, ctx)
        if isinstance(inner, Literal) and expr.dtype == "int":
            return _make_int_literal(int(inner.value))
        if inner is expr.expr:
            return expr
        return CastExpr(expr.dtype, inner)

    if isinstance(expr, FuncCallExpr):
        new_args = [simplify_expr(a, ctx) for a in expr.args]
        if all(x is y for x, y in zip(new_args, expr.args, strict=True)):
            return expr
        return FuncCallExpr(expr.name, new_args)

    if isinstance(expr, ArrayAccess):
        new_index = simplify_expr(expr.index, ctx)
        if new_index is expr.index:
            return expr
        return ArrayAccess(expr.array, new_index)

    if isinstance(expr, FieldAccess):
        new_inner = simplify_expr(expr.expr, ctx)
        if new_inner is expr.expr:
            return expr
        return FieldAccess(new_inner, expr.field)

    if isinstance(expr, VectorLoad):
        new_index = simplify_expr(expr.index, ctx)
        if new_index is expr.index:
            return expr
        return VectorLoad(expr.array, new_index, expr.width)

    return expr


# ---------------------------------------------------------------------------
# Loop IR walker
# ---------------------------------------------------------------------------


def _simplify_expr_tuple(xs: tuple[Expr, ...], ctx: Context) -> tuple[Expr, ...]:
    return tuple(simplify_expr(e, ctx) for e in xs)  # type: ignore[misc]


def _simplify_loop_stmt(stmt: LoopStmt, ctx: Context) -> LoopStmt:
    if isinstance(stmt, Loop):
        inner = ctx.extend(stmt.axis.name, Interval(0, stmt.axis.extent - 1))
        return Loop(stmt.axis, tuple(_simplify_loop_stmt(s, inner) for s in stmt.body))
    if isinstance(stmt, Select):
        return Select(
            stmt.name,
            tuple(
                SelectBranch(b.value, simplify_expr(b.select, ctx))  # type: ignore[arg-type]
                for b in stmt.branches
            ),
        )
    if isinstance(stmt, Write):
        return Write(stmt.output, _simplify_expr_tuple(stmt.index, ctx), stmt.value)
    if isinstance(stmt, Load):
        return Load(stmt.name, stmt.source, _simplify_expr_tuple(stmt.index, ctx))
    # Assign / Accum carry only SSA names — no Expr field to simplify.
    return stmt


def simplify_body(body: tuple[LoopStmt, ...]) -> tuple[LoopStmt, ...]:
    """Simplify every Expr inside a LoopOp body. Seeds Context from Loop extents.

    Body Loads, Accums, Writes, Selects get simplified as the walker
    descends through Loop blocks, accumulating axis ranges.
    """
    ctx = Context.empty()
    return tuple(_simplify_loop_stmt(s, ctx) for s in body)


def simplify_loop_op(op: LoopOp) -> LoopOp:
    """Apply ``simplify_body`` to a LoopOp's body, returning a new LoopOp."""
    return replace(op, body=simplify_body(op.body))
