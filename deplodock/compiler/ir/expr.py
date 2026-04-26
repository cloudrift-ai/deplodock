"""Shared expression AST + coordinate-expression helpers.

Backend-agnostic expression sublanguage used by:

- ``IndexMapOp.coord_map`` (``ir.tensor.ir``): affine output→input coord maps.
- ``Mux.select`` / ``MuxBranch.select`` (``ir.loop``): coord predicates.
- Tile IR (``ir.tile``): array indices, loop bounds, ternary selects.

The ``_ExprOps`` mixin adds Python operator overloading so expressions can be
built as arithmetic (``Var("i") * 4 + Var("j")``) and comparisons
(``Var("i").lt(Var("n"))``). Each concrete node implements ``eval(env)`` for
evaluation against a name → value environment (scalars or ndarrays).

Each node also implements ``substitute(mapping)`` (replace named
``Var`` nodes with another Expr) and ``free_vars()`` (Var-name set);
the placeholder helpers (``placeholder``, ``is_placeholder``) build
on the same AST.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Operator overloading mixin
# ---------------------------------------------------------------------------


class _ExprOps:
    """Mixin that adds arithmetic and comparison operators to Expr nodes.

    Returns BinaryExpr nodes, enabling::

        Var("row") * Var("cols") + Var("j")   # → BinaryExpr("+", BinaryExpr("*", ...), ...)
        Var("i").lt(Var("n"))                   # → BinaryExpr("<", ...)
    """

    def __add__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("+", self, _coerce(other))

    def __radd__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("+", _coerce(other), self)

    def __sub__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("-", self, _coerce(other))

    def __rsub__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("-", _coerce(other), self)

    def __mul__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("*", self, _coerce(other))

    def __rmul__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("*", _coerce(other), self)

    def __truediv__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("/", self, _coerce(other))

    def __mod__(self, other: Expr) -> BinaryExpr:
        return BinaryExpr("%", self, _coerce(other))

    def __neg__(self) -> BinaryExpr:
        return BinaryExpr("-", Literal(0, "int"), self)

    def lt(self, other: Expr) -> BinaryExpr:
        """Less-than (``<``)."""
        return BinaryExpr("<", self, _coerce(other))

    def pretty(self) -> str:
        """Format this Expr as a compact, human-readable string.

        Default raises so any Expr subclass that forgot to override is
        surfaced loudly. Concrete subclasses override.
        """
        raise NotImplementedError(f"{type(self).__name__}.pretty not implemented")

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        """Replace ``Var(name)`` subterms with ``mapping[name]``.

        Walks the expression tree non-destructively. Used at lowering
        time (placeholder coords → kernel output coords) and at
        composition time (outer IndexMap placeholders → inner
        coord_map). Vars not present in ``mapping`` are left unchanged.
        Default raises; concrete subclasses override.
        """
        raise NotImplementedError(f"{type(self).__name__}.substitute not implemented")

    def free_vars(self) -> frozenset[str]:
        """Return the set of ``Var.name`` strings appearing anywhere in this Expr."""
        raise NotImplementedError(f"{type(self).__name__}.free_vars not implemented")

    def render(self, ctx, parent_prec: int = 0) -> str:  # type: ignore[override]
        """Emit a target-language (C / CUDA) expression string.

        Concrete subclasses override. The ``ctx`` is duck-typed against
        :class:`deplodock.compiler.ir.stmt.RenderCtx` (intrinsic /
        builtin spelling tables, optional shape map). ``parent_prec``
        drives parenthesization in nested ``BinaryExpr`` chains.
        """
        raise NotImplementedError(f"{type(self).__name__}.render not implemented")

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        """Bottom-up rewrite: constant-fold literals, drop algebraic
        identities (``x+0``, ``x*1``, ``x-x``, etc.), collapse static
        comparisons via ``ctx.ranges``. Pure; idempotent."""
        raise NotImplementedError(f"{type(self).__name__}.simplify not implemented")

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        """Conservative integer-range analysis. ``None`` when unknown.
        Subclasses override; default surfaces missing implementations."""
        return None


def _coerce(v: Expr | int | float) -> Expr:
    """Coerce Python int/float to Literal for operator overloading."""
    if isinstance(v, int):
        return Literal(v, "int")
    if isinstance(v, float):
        return Literal(v)
    return v


# ---------------------------------------------------------------------------
# Expression types
# ---------------------------------------------------------------------------


@dataclass
class Var(_ExprOps):
    """Variable reference."""

    name: str

    def eval(self, env: dict[str, object]) -> object:
        return env[self.name]

    def pretty(self) -> str:
        return self.name

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return mapping.get(self.name, self)

    def free_vars(self) -> frozenset[str]:
        return frozenset({self.name})

    def render(self, ctx, parent_prec: int = 0) -> str:
        return self.name

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        return self

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        return ctx.ranges.get(self.name)


@dataclass
class Literal(_ExprOps):
    """Numeric constant."""

    value: int | float
    dtype: str = "float"

    def eval(self, env: dict[str, object]) -> object:
        return self.value

    def pretty(self) -> str:
        return str(self.value)

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return self

    def free_vars(self) -> frozenset[str]:
        return frozenset()

    def render(self, ctx, parent_prec: int = 0) -> str:
        if isinstance(self.value, float) or self.dtype == "float":
            return _float_lit(float(self.value))
        v = int(self.value)
        return f"{v}LL" if abs(v) > 32768 else str(v)

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        return self

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return Interval(self.value, self.value)
        return None


@dataclass
class BinaryExpr(_ExprOps):
    """Binary operation.

    Evaluates ``left`` and ``right`` in ``env`` then applies the op.
    Values may be scalars or numpy ndarrays; arithmetic composes via
    numpy broadcasting. Integer floor division is used for both ``/``
    and ``//``. Logical ``&&`` / ``||`` fall back to ``np.logical_and``
    / ``np.logical_or`` when the operand is an ndarray (scalar bool
    coercion would raise).
    """

    op: str  # "+", "-", "*", "/", "//", "%", "<", "<=", ">", ">=", "==", "&&", "||"
    left: Expr
    right: Expr

    def eval(self, env: dict[str, object]) -> object:
        lv, rv = self.left.eval(env), self.right.eval(env)
        op = self.op
        if op == "+":
            return lv + rv
        if op == "-":
            return lv - rv
        if op == "*":
            return lv * rv
        if op in ("/", "//"):
            try:
                return int(lv) // int(rv)
            except TypeError:
                return lv // rv
        if op == "%":
            try:
                return int(lv) % int(rv)
            except TypeError:
                return lv % rv
        if op == "<":
            return lv < rv
        if op == "<=":
            return lv <= rv
        if op == ">":
            return lv > rv
        if op == ">=":
            return lv >= rv
        if op == "==":
            return lv == rv
        if op == "&&":
            try:
                return bool(lv) and bool(rv)
            except (TypeError, ValueError):
                return np.logical_and(lv, rv)
        if op == "||":
            try:
                return bool(lv) or bool(rv)
            except (TypeError, ValueError):
                return np.logical_or(lv, rv)
        raise ValueError(f"Unknown BinaryExpr: {op}")

    def pretty(self) -> str:
        return f"({self.left.pretty()} {self.op} {self.right.pretty()})"

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return BinaryExpr(self.op, self.left.substitute(mapping), self.right.substitute(mapping))

    def free_vars(self) -> frozenset[str]:
        return self.left.free_vars() | self.right.free_vars()

    def render(self, ctx, parent_prec: int = 0) -> str:
        prec = _PRECEDENCE.get(self.op, 10)
        left = self.left.render(ctx, prec)
        right = self.right.render(ctx, prec + 1)
        result = f"{left} {self.op} {right}"
        return f"({result})" if prec < parent_prec else result

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        left = self.left.simplify(ctx)
        right = self.right.simplify(ctx)
        op = self.op

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
            if _is_one(right) or _is_zero(left):
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
            la = left.range(ctx)
            lb = right.range(ctx)
            if la is not None and lb is not None:
                decided = _static_cmp(op, la, lb)
                if decided is not None:
                    return _make_int_literal(1 if decided else 0)

        if left is self.left and right is self.right:
            return self
        return BinaryExpr(op, left, right)

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        la = self.left.range(ctx)
        lb = self.right.range(ctx)
        op = self.op
        if la is None or lb is None:
            if op in ("<", "<=", ">", ">=", "==", "&&", "||"):
                return Interval(0, 1)
            return None
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


@dataclass
class Builtin(_ExprOps):
    """GPU built-in variable (threadIdx.x, blockIdx.y, blockDim.x, etc.).

    Not evaluable outside a kernel — GPU codegen substitutes these at emit
    time. Calling ``eval`` raises.
    """

    name: str

    def eval(self, env: dict[str, object]) -> object:
        raise NotImplementedError(f"Builtin {self.name!r} is GPU-only; cannot eval in numpy")

    def pretty(self) -> str:
        return self.name

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return self

    def free_vars(self) -> frozenset[str]:
        return frozenset()

    def render(self, ctx, parent_prec: int = 0) -> str:
        spelling = ctx.builtins.get(self.name) if ctx is not None else None
        if spelling is None:
            raise ValueError(f"render: Builtin {self.name!r} has no target spelling in ctx.builtins")
        return spelling

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        return self


@dataclass
class FuncCallExpr(_ExprOps):
    """Intrinsic / math function call.

    ``name`` is an ``ElementwiseImpl`` registry name (numpy-aligned: ``exp`` /
    ``tanh`` / ``maximum`` / ``reciprocal`` / …). ``FuncCallExpr.eval`` pulls
    the pre-bound callable off the registered ``ElementwiseImpl``; the CUDA
    emitter's ``_translate_intrinsic`` rewrites the same name to the
    ``f``-suffixed libm spelling at source-render time.
    """

    name: str
    args: list[Expr]

    def eval(self, env: dict[str, object]) -> object:
        from deplodock.compiler.ir.elementwise import ElementwiseImpl

        try:
            op = ElementwiseImpl(self.name)
        except ValueError as e:
            raise NotImplementedError(f"FuncCallExpr.eval: unknown intrinsic {self.name!r}") from e
        return op(*(a.eval(env) for a in self.args))

    def pretty(self) -> str:
        return f"{self.name}({', '.join(a.pretty() for a in self.args)})"

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return FuncCallExpr(self.name, [a.substitute(mapping) for a in self.args])

    def free_vars(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for a in self.args:
            out |= a.free_vars()
        return out

    def render(self, ctx, parent_prec: int = 0) -> str:
        spelling = ctx.intrinsics.get(self.name, self.name) if ctx is not None else self.name
        args = ", ".join(a.render(ctx) for a in self.args)
        return f"{spelling}({args})"

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        new_args = [a.simplify(ctx) for a in self.args]
        if all(x is y for x, y in zip(new_args, self.args, strict=True)):
            return self
        return FuncCallExpr(self.name, new_args)


@dataclass
class TernaryExpr(_ExprOps):
    """TernaryExpr expression: cond ? if_true : if_false.

    Uses Python's conditional: when ``cond`` evaluates to an ndarray,
    callers that want elementwise selection should use ``np.where``
    directly; ``TernaryExpr.eval`` only supports scalar ``cond``.
    """

    cond: Expr
    if_true: Expr
    if_false: Expr

    def eval(self, env: dict[str, object]) -> object:
        return self.if_true.eval(env) if self.cond.eval(env) else self.if_false.eval(env)

    def pretty(self) -> str:
        return f"({self.cond.pretty()} ? {self.if_true.pretty()} : {self.if_false.pretty()})"

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return TernaryExpr(self.cond.substitute(mapping), self.if_true.substitute(mapping), self.if_false.substitute(mapping))

    def free_vars(self) -> frozenset[str]:
        return self.cond.free_vars() | self.if_true.free_vars() | self.if_false.free_vars()

    def render(self, ctx, parent_prec: int = 0) -> str:
        c = self.cond.render(ctx)
        t = self.if_true.render(ctx)
        f = self.if_false.render(ctx)
        return f"(({c}) ? ({t}) : ({f}))"

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        cond = self.cond.simplify(ctx)
        if _is_truthy(cond):
            return self.if_true.simplify(ctx)
        if _is_falsy(cond):
            return self.if_false.simplify(ctx)
        a = self.if_true.simplify(ctx)
        b = self.if_false.simplify(ctx)
        if a == b:
            return a
        if cond is self.cond and a is self.if_true and b is self.if_false:
            return self
        return TernaryExpr(cond, a, b)


@dataclass
class CastExpr(_ExprOps):
    """Type cast of an inner expression to ``dtype`` (e.g. ``"int"``, ``"float"``)."""

    dtype: str
    expr: Expr

    def eval(self, env: dict[str, object]) -> object:
        v = self.expr.eval(env)
        if self.dtype == "int":
            return np.asarray(v).astype(np.int64) if hasattr(v, "__array__") else int(v)
        return v

    def pretty(self) -> str:
        return f"({self.dtype}){self.expr.pretty()}"

    def substitute(self, mapping: dict[str, Expr]) -> Expr:
        return CastExpr(self.dtype, self.expr.substitute(mapping))

    def free_vars(self) -> frozenset[str]:
        return self.expr.free_vars()

    def render(self, ctx, parent_prec: int = 0) -> str:
        return f"(({self.dtype})({self.expr.render(ctx)}))"

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        inner = self.expr.simplify(ctx)
        if isinstance(inner, Literal) and self.dtype == "int":
            return _make_int_literal(int(inner.value))
        if inner is self.expr:
            return self
        return CastExpr(self.dtype, inner)


Expr = Var | Literal | BinaryExpr | Builtin | FuncCallExpr | TernaryExpr | CastExpr


# ---------------------------------------------------------------------------
# Render helpers — shared C / CUDA literal formatting + operator precedence
# ---------------------------------------------------------------------------


_PRECEDENCE: dict[str, int] = {
    "||": 1,
    "&&": 2,
    "==": 3,
    "!=": 3,
    "<": 4,
    ">": 4,
    "<=": 4,
    ">=": 4,
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "%": 6,
}


def _float_lit(v: float) -> str:
    """C / CUDA float literal — always carries a decimal so ``0.0`` renders
    as ``0.0f`` not the invalid ``0f``, and large/scientific values keep
    their exponent."""
    s = repr(float(v))
    if "." not in s and "e" not in s and "E" not in s and "inf" not in s and "nan" not in s:
        s += ".0"
    return f"{s}f"


# ---------------------------------------------------------------------------
# Simplification context + helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Interval:
    """Closed integer interval ``[lo, hi]`` — used for static range analysis
    that drives comparison folding (``i < N`` where ``i ∈ [0, N-1]`` → ``1``)."""

    lo: int
    hi: int


@dataclass
class SimplifyCtx:
    """Range info available at a given scope. Immutable per-call; callers
    push into a nested scope by calling :meth:`extend` to get a fresh ctx
    (so the pass stays pure)."""

    ranges: dict[str, Interval]

    @classmethod
    def empty(cls) -> SimplifyCtx:
        return cls({})

    def extend(self, name: str, interval: Interval) -> SimplifyCtx:
        return SimplifyCtx({**self.ranges, name: interval})


def _make_int_literal(v: int) -> Literal:
    return Literal(int(v), "int")


def _is_zero(e: object) -> bool:
    return isinstance(e, Literal) and e.value == 0


def _is_one(e: object) -> bool:
    return isinstance(e, Literal) and e.value == 1


def _is_truthy(e: object) -> bool:
    return isinstance(e, Literal) and bool(e.value)


def _is_falsy(e: object) -> bool:
    return isinstance(e, Literal) and not bool(e.value)


def _fold_binop_literals(op: str, left: Literal, right: Literal) -> Literal:
    """Constant-fold a ``BinaryExpr`` whose children are both ``Literal``.
    Preserves int dtype when both operands are int and the result is integral."""
    folded = BinaryExpr(op, left, right).eval({})
    if isinstance(folded, bool):
        return _make_int_literal(1 if folded else 0)
    both_int = left.dtype == "int" and right.dtype == "int"
    if isinstance(folded, int) and both_int:
        return _make_int_literal(folded)
    return Literal(float(folded), "float")


def _static_cmp(op: str, la: Interval, lb: Interval) -> bool | None:
    """Decide a comparison statically from operand ranges, or return ``None``."""
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
# Coordinate-expression helpers for IndexMapOp
# ---------------------------------------------------------------------------
#
# ``IndexMapOp`` (in ``ir.tensor.ir``) describes layout-only ops (slice, cat,
# transpose, reshape, unsqueeze) by mapping output coordinates to input
# coordinates via the ``Expr`` AST above. Convention: an IndexMap's
# ``coord_map[i]`` is an ``Expr`` over placeholder variables
# ``Var("out_coord_0")``, ``Var("out_coord_1")``, ... — one per output
# dimension. At lowering time the placeholders are substituted with the
# ``Expr``s that the kernel uses for its actual output coordinates.


PLACEHOLDER_PREFIX = "out_coord_"


def placeholder(d: int) -> Var:
    """Return the placeholder variable for output coordinate axis ``d``."""
    return Var(f"{PLACEHOLDER_PREFIX}{d}")


def is_placeholder(expr: object, d: int | None = None) -> bool:
    """Check if ``expr`` is a placeholder ``Var``. If ``d`` is given, check that axis."""
    if not isinstance(expr, Var):
        return False
    if not expr.name.startswith(PLACEHOLDER_PREFIX):
        return False
    if d is None:
        return True
    return expr.name == f"{PLACEHOLDER_PREFIX}{d}"


# ---------------------------------------------------------------------------
# Free functions — numpy callables for intrinsics numpy itself doesn't expose
# under the same name. ``resolve_fn`` tries this module first (so our ``max``
# means elementwise maximum, not the reduction ``np.max``) then falls back to
# ``numpy``.
# ---------------------------------------------------------------------------
