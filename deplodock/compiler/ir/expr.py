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


def apply_binop(op: str, lv: object, rv: object) -> object:
    """Apply a BinOp to two already-evaluated values.

    Pure helper — no env coupling. Shared by ``BinOp.eval`` (numpy-aware
    evaluation) and ``ir.simplify`` (constant folding on Literal children).
    Values may be scalars or numpy ndarrays; arithmetic composes via numpy
    broadcasting. Integer floor division is used for both ``/`` and ``//``.
    """
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
            import numpy as np

            return np.logical_and(lv, rv)
    if op == "||":
        try:
            return bool(lv) or bool(rv)
        except (TypeError, ValueError):
            import numpy as np

            return np.logical_or(lv, rv)
    raise ValueError(f"Unknown BinOp: {op}")


# ---------------------------------------------------------------------------
# Expression types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
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
        # If this SSA name was bound by a literal-constant Load, inline the
        # literal value instead of emitting the variable name. ``render_body``
        # populates ``literal_ssa`` after scanning the body and skips the
        # corresponding ``Load`` decl.
        lit_map = getattr(ctx, "literal_ssa", None)
        if lit_map and self.name in lit_map:
            return _float_lit(lit_map[self.name])
        return self.name

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        return self

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        return ctx.ranges.get(self.name)


@dataclass(frozen=True)
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
            txt = _float_lit(float(self.value))
            # Float literals embedded in a non-default-dtype expression
            # must compose with the surrounding-typed operands; delegate
            # the wrapping decision to the target (e.g. wrap with
            # ``__float2half`` for fp16). Int literals (loop bounds /
            # shape coords) are unaffected — they only go through the
            # int branch below.
            if ctx is not None:
                wrap_dt = getattr(ctx, "literal_default_dtype", None)
                target = getattr(ctx, "target", None)
                if wrap_dt is not None and target is not None:
                    return target.literal(txt, wrap_dt)
            return txt
        v = int(self.value)
        return f"{v}LL" if abs(v) > 32768 else str(v)

    def simplify(self, ctx: SimplifyCtx) -> Expr:
        return self

    def range(self, ctx: SimplifyCtx) -> Interval | None:
        if isinstance(self.value, int) and not isinstance(self.value, bool):
            return Interval(self.value, self.value)
        return None


@dataclass(frozen=True)
class BinaryExpr(_ExprOps):
    """Binary operation.

    Evaluates ``left`` and ``right`` in ``env`` then applies the op.
    Values may be scalars or numpy ndarrays; arithmetic composes via
    numpy broadcasting. Integer floor division is used for both ``/``
    and ``//``. Logical ``&&`` / ``||`` fall back to ``np.logical_and``
    / ``np.logical_or`` when the operand is an ndarray (scalar bool
    coercion would raise).
    """

    op: str  # "+", "-", "*", "/", "//", "%", "<", "<=", ">", ">=", "==", "&&", "||", "^"
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
        if op == "^":
            return int(lv) ^ int(rv)
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
        # ``//`` is Python floor-div; C integer division ``/`` floors for the
        # non-negative operands shape arithmetic produces.
        c_op = "/" if self.op == "//" else self.op
        result = f"{left} {c_op} {right}"
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
            if isinstance(right, Literal) and right.dtype == "int" and isinstance(right.value, int) and right.value > 1:
                decomp = _div_mod_decompose(left, right.value, ctx)
                if decomp is not None:
                    return decomp[0]
            cancelled = _cancel_common_factors(op, left, right, ctx)
            if cancelled is not None:
                return cancelled
        elif op == "%":
            if _is_one(right) or _is_zero(left):
                return _make_int_literal(0)
            if isinstance(right, Literal) and right.dtype == "int" and isinstance(right.value, int) and right.value > 1:
                decomp = _div_mod_decompose(left, right.value, ctx)
                if decomp is not None:
                    return decomp[1]
            cancelled = _cancel_common_factors(op, left, right, ctx)
            if cancelled is not None:
                return cancelled
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


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class FuncCallExpr(_ExprOps):
    """Intrinsic / math function call.

    ``name`` is an ``ElementwiseImpl`` registry name (numpy-aligned: ``exp`` /
    ``tanh`` / ``maximum`` / ``reciprocal`` / …). ``FuncCallExpr.eval`` pulls
    the pre-bound callable off the registered ``ElementwiseImpl``; the CUDA
    emitter's ``_translate_intrinsic`` rewrites the same name to the
    ``f``-suffixed libm spelling at source-render time.
    """

    name: str
    args: tuple[Expr, ...]

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
        return FuncCallExpr(self.name, tuple(a.substitute(mapping) for a in self.args))

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
        new_args = tuple(a.simplify(ctx) for a in self.args)
        if all(x is y for x, y in zip(new_args, self.args, strict=True)):
            return self
        return FuncCallExpr(self.name, new_args)


@dataclass(frozen=True)
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


@dataclass(frozen=True)
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
    "^": 4,  # bitwise XOR — match relational so ``a ^ b + c`` always parens
    "&": 4,  # bitwise AND — same intent: ``a & b * c`` must paren as ``(a & b) * c``
    "|": 4,  # bitwise OR — symmetric with ``&`` / ``^``
    "+": 5,
    "-": 5,
    "*": 6,
    "/": 6,
    "//": 6,  # integer floor-div — renders as C ``/`` (truncation == floor for ≥0)
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


def _multiplicative_factors(expr: Expr) -> tuple[list[Expr], int]:
    """Flatten an ``Expr`` tree of ``*`` into ``([non-int factors], int coefficient)``.

    ``(a * 4) * b`` → ``([Var('a'), Var('b')], 4)``. Used by
    ``_cancel_common_factors`` to canonicalize both sides of ``//`` / ``%``
    before checking for cancellation.
    """
    if isinstance(expr, Literal) and expr.dtype == "int" and isinstance(expr.value, int):
        return [], expr.value
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        lf, lk = _multiplicative_factors(expr.left)
        rf, rk = _multiplicative_factors(expr.right)
        return lf + rf, lk * rk
    return [expr], 1


def _is_positive(expr: Expr, ctx: SimplifyCtx) -> bool:
    """Conservatively decide whether ``expr`` is provably ``>= 1``.

    Used to gate factor cancellation in ``a // b → ...`` — cancelling a
    shared factor is unsound if that factor could be zero. Shape vars are
    flagged via ``SimplifyCtx.ranges`` by ``Dim._simplify``."""
    if isinstance(expr, Literal) and isinstance(expr.value, int):
        return expr.value >= 1
    rng = expr.range(ctx)
    return rng is not None and rng.lo >= 1


def _rebuild_product(factors: list[Expr], k: int) -> Expr:
    """Inverse of ``_multiplicative_factors``: reassemble into an ``Expr``."""
    if k == 0:
        return _make_int_literal(0)
    if not factors:
        return _make_int_literal(k)
    out: Expr = factors[0]
    for f in factors[1:]:
        out = BinaryExpr("*", out, f)
    if k != 1:
        out = BinaryExpr("*", out, _make_int_literal(k))
    return out


def _cancel_common_factors(op: str, left: Expr, right: Expr, ctx: SimplifyCtx) -> Expr | None:
    """Try to simplify ``left op right`` (``op in {"/", "//", "%"}``) by
    cancelling positive multiplicative factors common to both sides.

    For ``(s * 128) // (s * 4)`` with ``s >= 1`` (from ``ctx.ranges``):
    cancels the ``s`` and gcds the integer parts → ``32``. Returns ``None``
    when no cancellation applies, letting the caller fall through to the
    default ``BinaryExpr(op, left, right)`` path.
    """
    import math  # noqa: PLC0415

    L_factors, L_k = _multiplicative_factors(left)
    R_factors, R_k = _multiplicative_factors(right)

    cancelled = False
    R_remaining = list(R_factors)
    L_remaining: list[Expr] = []
    common_factors: list[Expr] = []  # factors removed from BOTH sides (for the % identity)
    common_int = 1
    for f in L_factors:
        matched_idx = None
        for i, rf in enumerate(R_remaining):
            if f == rf and _is_positive(f, ctx):
                matched_idx = i
                break
        if matched_idx is not None:
            R_remaining.pop(matched_idx)
            cancelled = True
            common_factors.append(f)
        else:
            L_remaining.append(f)

    if L_k > 0 and R_k > 0:
        g = math.gcd(L_k, R_k)
        if g > 1:
            L_k //= g
            R_k //= g
            cancelled = True
            common_int = g

    if not cancelled:
        return None

    new_left = _rebuild_product(L_remaining, L_k)
    new_right = _rebuild_product(R_remaining, R_k)
    if op in ("/", "//"):
        # (c·x) / (c·y) == x / y — the common factor cancels outright.
        if _is_one(new_right):
            return new_left
        if _is_zero(new_left):
            return _make_int_literal(0)
    elif op == "%":
        if _is_one(new_right) or _is_zero(new_left):
            return _make_int_literal(0)
        # (c·x) % (c·y) == c · (x % y) — the common factor does NOT cancel out
        # of a modulo; it scales the remainder. Multiplying it back is the fix
        # for the old ``(f*8) % 64 → f % 8`` bug (should be ``8 * (f % 8)``).
        common = _rebuild_product(common_factors, common_int)
        mod = BinaryExpr("%", new_left, new_right)
        return mod if _is_one(common) else BinaryExpr("*", common, mod)
    return BinaryExpr(op, new_left, new_right)


def _div_mod_decompose(expr: Expr, n: int, ctx: SimplifyCtx) -> tuple[Expr, Expr] | None:
    """Try to express ``expr`` as ``n * q + r`` with ``0 <= r < n``, both
    proven from operand ranges. Used by ``BinaryExpr.simplify`` to fold
    ``expr / Lit(n)`` → ``q`` and ``expr % Lit(n)`` → ``r`` when the
    decomposition succeeds; returns ``None`` to signal "give up, keep
    the original ``/`` or ``%``".
    Cleans up collapsed-reshape indices like ``((X*N + Y) / N) % M``
    that compose-indexmaps emits, where range bounds make the div/mod
    pure strength-reduction.
    """
    if isinstance(expr, Literal) and expr.dtype == "int" and isinstance(expr.value, int) and expr.value >= 0:
        return (_make_int_literal(expr.value // n), _make_int_literal(expr.value % n))
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        for k_side, other_side in ((expr.right, expr.left), (expr.left, expr.right)):
            if not (isinstance(k_side, Literal) and k_side.dtype == "int" and isinstance(k_side.value, int)):
                continue
            if k_side.value <= 0 or k_side.value % n != 0:
                continue
            rng = other_side.range(ctx)
            if rng is not None and rng.lo >= 0:
                q_expr = BinaryExpr("*", other_side, _make_int_literal(k_side.value // n)).simplify(ctx)
                return (q_expr, _make_int_literal(0))
    if isinstance(expr, BinaryExpr) and expr.op == "+":
        L = _div_mod_decompose(expr.left, n, ctx)
        R = _div_mod_decompose(expr.right, n, ctx)
        if L is None or R is None:
            return None
        ql, rl = L
        qr, rr = R
        r_sum = BinaryExpr("+", rl, rr).simplify(ctx)
        rng = r_sum.range(ctx)
        if rng is None or rng.lo < 0 or rng.hi >= n:
            return None
        q = BinaryExpr("+", ql, qr).simplify(ctx)
        return (q, r_sum)
    rng = expr.range(ctx)
    if rng is not None and rng.lo >= 0 and rng.hi < n:
        return (_make_int_literal(0), expr)
    return None


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


def affine_form(expr: Expr, vars: frozenset[str] | set[str]) -> tuple[Expr, dict[str, int]] | None:
    """Decompose ``expr`` as ``anchor + sum(coeffs[v] * Var(v))`` over the given var set.

    Returns ``(anchor, coeffs)`` where ``anchor`` is the part of ``expr`` that's
    free of ``vars`` (with each ``Var(v)`` substituted to 0) and ``coeffs[v]`` is
    the integer coefficient of ``Var(v)``. Returns ``None`` when ``expr`` isn't
    affine in ``vars`` — i.e. it contains a non-additive use of one of those
    vars (``v / N``, ``v % N``, ``v * v``, ``v`` inside a ternary or call) or
    a non-literal-int coefficient.

    Vars not in ``vars`` are treated as opaque constants and accumulate into
    ``anchor`` unchanged. Missing vars in ``coeffs`` mean coefficient 0.
    """
    if not (expr.free_vars() & set(vars)):
        return expr, {}  # opaque w.r.t. ``vars`` — folds into the anchor
    if isinstance(expr, Var):
        return Literal(0, "int"), {expr.name: 1}  # expr.name in vars (else above)
    if isinstance(expr, BinaryExpr) and expr.op in ("+", "-"):
        left = affine_form(expr.left, vars)
        right = affine_form(expr.right, vars)
        if left is None or right is None:
            return None
        l_anchor, l_coeffs = left
        r_anchor, r_coeffs = right
        sign = 1 if expr.op == "+" else -1
        anchor = BinaryExpr(expr.op, l_anchor, r_anchor)
        coeffs = dict(l_coeffs)
        for v, c in r_coeffs.items():
            coeffs[v] = coeffs.get(v, 0) + sign * c
        return anchor, {v: c for v, c in coeffs.items() if c != 0}
    if isinstance(expr, BinaryExpr) and expr.op == "*":
        l_uses = expr.left.free_vars() & vars
        r_uses = expr.right.free_vars() & vars
        if l_uses and r_uses:
            return None  # var * var — non-affine
        side, mult = (expr.right, expr.left) if not l_uses else (expr.left, expr.right)
        if not (isinstance(mult, Literal) and isinstance(mult.value, int)):
            return None  # symbolic multiplier — would yield non-int coeff
        sub = affine_form(side, vars)
        if sub is None:
            return None
        sub_anchor, sub_coeffs = sub
        anchor = BinaryExpr("*", sub_anchor, mult) if not l_uses else BinaryExpr("*", mult, sub_anchor)
        return anchor, {v: c * mult.value for v, c in sub_coeffs.items()}
    if expr.free_vars() & set(vars):
        return None  # vars appear inside a non-affine context (div, mod, ternary, ...)
    return expr, {}


def index_set_size(exprs: tuple[Expr, ...], var_extents: dict[str, int]) -> int | None:
    """Upper bound on ``|{tuple(e(env) for e in exprs) : env in extents}|``.

    For each Expr, decompose to affine form over ``var_extents.keys()``. The
    per-dim projection size is bounded by ``1 + sum(|c_v| * (extent_v - 1))``
    over the var coefficients in that dim — this is tight when the affine
    form's coefficients align (rare false positive only across correlated
    dims). The total is the product over dims (over-counts when dims are
    correlated, fine as an upper bound). Returns ``None`` if any expr isn't
    affine in ``var_extents``.

    Used by staging to compute reuse: ``work / index_set_size`` over a
    Load's index gives the per-element fan-in, i.e., how many threads /
    iterations read the same source value.
    """
    var_set = frozenset(var_extents)
    total = 1
    for e in exprs:
        form = affine_form(e, var_set)
        if form is None:
            return None
        _, coeffs = form
        size = 1
        for v, c in coeffs.items():
            size += abs(c) * (var_extents[v] - 1)
        total *= size
    return total


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
