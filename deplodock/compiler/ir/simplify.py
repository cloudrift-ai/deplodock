"""Generic Expr / IR simplification — applied at every pipeline stage.

Single bottom-up pass over the shared ``Expr`` AST (``ir.expr``) plus the
GPU-specific extensions (``ir.kernel_ir``). Rules:

- Constant folding: any ``BinOp`` or ``Cast`` of ``Literal`` children folds
  via ``BinOp.eval({})`` / ``Cast`` dispatch (Literal.eval ignores env).
- Algebraic identities: ``x+0``, ``x-0``, ``x*0``, ``x*1``, ``x/1``, ``x%1``,
  ``x-x``, ``x&&True``, ``x||False``, etc.
- Ternary collapse: ``Literal(c) ? a : b`` → a or b; equal branches → branch.
- Range-based comparison folding: when an ``Axis``' extent or a ``ForLoop``'s
  static bounds prove a comparison result, collapse to ``Literal(0/1)``.
  Combined with Ternary collapse this erases chained index clamps like
  ``(k0 > 2047 ? 2047 : k0) < 0 ? 0 : ...`` down to ``k0``.

Idempotent: running twice yields the same AST. Pure: ``Expr → Expr``.

Entry points:

- ``simplify_expr(e, ctx)`` — Expr rewriter; covers every shared Expr type
  (``Var | Literal | BinOp | Builtin | FuncCall | Ternary | Cast``) plus
  the GPU extensions (``ArrayAccess | FieldAccess | VectorLoad``).
- ``simplify_loop_op(op)`` — walks a ``LoopOp``, seeding Context from its
  ``Axis`` extents, rewriting every Expr in ``Port.index`` / ``Select`` /
  ``Write`` / ``LocalBuffer.init``.
- ``simplify_kernel(k)`` — walks a ``GpuKernel``, seeding Context from
  enclosing ``ForLoop`` bounds (literal start/end only), rewriting every
  Expr inside statements.

Range analysis (``infer_range``) tracks integer intervals only; ``Builtin``,
``FuncCall``, ``Cast`` return ``None`` (unknown range → no comparison
folding through them, but surrounding arithmetic can still constant-fold).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from deplodock.compiler.ir.expr import (
    BinOp,
    Builtin,
    Cast,
    Expr,
    FuncCall,
    Literal,
    Ternary,
    Var,
)
from deplodock.compiler.ir.kernel_ir import (
    ArrayAccess,
    ArrayDecl,
    AugAssign,
    FieldAccess,
    ForLoop,
    GpuKernel,
    IfStmt,
    PragmaUnroll,
    RawCode,
    SyncThreads,
    VarAssign,
    VarDecl,
    VectorLoad,
)
from deplodock.compiler.ir.kernel_ir import Assign as GpuAssign
from deplodock.compiler.ir.kernel_ir import Stmt as GpuStmt
from deplodock.compiler.ir.loop_ir import (
    LocalBuffer,
    Loop,
    LoopOp,
    Port,
    Select,
    SelectBranch,
    Write,
)
from deplodock.compiler.ir.loop_ir import Stmt as LoopStmt

__all__ = [
    "Interval",
    "Context",
    "simplify_expr",
    "infer_range",
    "simplify_loop_op",
    "simplify_kernel",
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
    if isinstance(expr, BinOp):
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
    # Builtin / FuncCall / Cast / Ternary: unknown
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
    """Constant-fold a BinOp whose children are both Literal. Preserves int
    dtype when both operands are int-typed and the result is integral."""
    folded = BinOp(op, left, right).eval({})
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

    if isinstance(expr, BinOp):
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
        return BinOp(op, left, right)

    if isinstance(expr, Ternary):
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
        return Ternary(cond, a, b)

    if isinstance(expr, Cast):
        inner = simplify_expr(expr.expr, ctx)
        if isinstance(inner, Literal) and expr.dtype == "int":
            return _make_int_literal(int(inner.value))
        if inner is expr.expr:
            return expr
        return Cast(expr.dtype, inner)

    if isinstance(expr, FuncCall):
        new_args = [simplify_expr(a, ctx) for a in expr.args]
        if all(x is y for x, y in zip(new_args, expr.args, strict=True)):
            return expr
        return FuncCall(expr.name, new_args)

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
    # Assign / Update carry only SSA names — no Expr field to simplify
    return stmt


def simplify_loop_op(op: LoopOp) -> LoopOp:
    """Simplify every Expr inside a LoopOp. Seeds Context from axis extents."""
    ctx = Context.empty()
    # Axis names aren't in ctx at the LoopOp level — they only gain a range
    # once inside a Loop block. Port indices are read at the innermost point
    # of the iteration space, so we walk them per-scope too. But for the
    # top-level inputs/locals we rebuild with an empty Context first and
    # rely on the Loop walker to push axis ranges.
    new_inputs = tuple(Port(_simplify_expr_tuple(p.index, ctx)) for p in op.inputs)
    new_locals = tuple(
        LocalBuffer(
            name=lb.name,
            dtype=lb.dtype,
            init=(simplify_expr(lb.init, ctx) if lb.init is not None else None),  # type: ignore[arg-type]
            combine=lb.combine,
            shape=lb.shape,
            scope=lb.scope,
        )
        for lb in op.locals
    )
    # Body — push axis ranges as we descend into Loop blocks.
    new_body = tuple(_simplify_loop_stmt(s, ctx) for s in op.body)
    # Now walk inputs again with a Context that has every axis range, so
    # Port indices see their bounds and can fold comparison clamps.
    full_ctx = Context({a.name: Interval(0, a.extent - 1) for a in op.axes})
    new_inputs = tuple(Port(_simplify_expr_tuple(p.index, full_ctx)) for p in op.inputs)
    return replace(op, inputs=new_inputs, locals=new_locals, body=new_body)


# ---------------------------------------------------------------------------
# Kernel IR walker
# ---------------------------------------------------------------------------


_INT_MAX = 2**31 - 1


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


_NONNEG_VAR_PREFIXES = ("threadIdx.", "blockIdx.", "blockDim.", "gridDim.")


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


def simplify_kernel(kernel: GpuKernel) -> GpuKernel:
    """Simplify every Expr inside a GpuKernel's statement tree."""
    return replace(kernel, body=_simplify_gpu_body(kernel.body, Context.empty()))
