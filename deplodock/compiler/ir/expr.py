"""Shared expression AST + coordinate-expression helpers.

Backend-agnostic expression sublanguage used by:

- ``IndexMapOp.coord_map`` (``ir.tensor.ir``): affine output→input coord maps.
- ``Mux.select`` / ``MuxBranch.select`` (``ir.loop``): coord predicates.
- GPU IR (``ir.kernel``): array indices, loop bounds, ternary selects.

The ``_ExprOps`` mixin adds Python operator overloading so expressions can be
built as arithmetic (``Var("i") * 4 + Var("j")``) and comparisons
(``Var("i").lt(Var("n"))``). Each concrete node implements ``eval(env)`` for
evaluation against a name → value environment (scalars or ndarrays).

Coordinate helpers (``placeholder``, ``is_placeholder``, ``substitute``) are
pure operations over the expression AST.
"""

from __future__ import annotations

from collections.abc import Callable
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

    def ge(self, other: Expr) -> BinaryExpr:
        """Greater-or-equal (``>=``)."""
        return BinaryExpr(">=", self, _coerce(other))

    def eq(self, other: Expr) -> BinaryExpr:
        """Equal (``==``)."""
        return BinaryExpr("==", self, _coerce(other))

    def and_(self, other: Expr) -> BinaryExpr:
        """Logical AND (``&&``)."""
        return BinaryExpr("&&", self, _coerce(other))

    def or_(self, other: Expr) -> BinaryExpr:
        """Logical OR (``||``)."""
        return BinaryExpr("||", self, _coerce(other))


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


@dataclass
class Literal(_ExprOps):
    """Numeric constant."""

    value: int | float
    dtype: str = "float"

    def eval(self, env: dict[str, object]) -> object:
        return self.value


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


@dataclass
class Builtin(_ExprOps):
    """GPU built-in variable (threadIdx.x, blockIdx.y, blockDim.x, etc.).

    Not evaluable outside a kernel — GPU codegen substitutes these at emit
    time. Calling ``eval`` raises.
    """

    name: str

    def eval(self, env: dict[str, object]) -> object:
        raise NotImplementedError(f"Builtin {self.name!r} is GPU-only; cannot eval in numpy")


@dataclass
class FuncCallExpr(_ExprOps):
    """Intrinsic / math function call.

    ``name`` is an ``ExprOp`` registry name (numpy-aligned: ``exp`` /
    ``tanh`` / ``maximum`` / ``reciprocal`` / …). ``FuncCallExpr.eval`` pulls
    the pre-bound callable off the registered ``ExprOp``; the CUDA
    emitter's ``_translate_intrinsic`` rewrites the same name to the
    ``f``-suffixed libm spelling at source-render time.
    """

    name: str
    args: list[Expr]

    def eval(self, env: dict[str, object]) -> object:
        try:
            op = coerce_expr_op(self.name)
        except ValueError as e:
            raise NotImplementedError(f"FuncCallExpr.eval: unknown intrinsic {self.name!r}") from e
        return op.fn(*(a.eval(env) for a in self.args))


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


Expr = Var | Literal | BinaryExpr | Builtin | FuncCallExpr | TernaryExpr | CastExpr


def render(expr: Expr, formatter: Callable[[object], str | None] | None = None) -> str:
    """Format an ``Expr`` tree as a compact, human-readable string.

    ``formatter``: optional hook, called with each node before the default
    dispatch. Return a string to override rendering; return ``None`` to fall
    through to the default. Lets extensions (e.g. ``ir.kernel``'s GPU-specific
    node types plus no-paren C-style formatting) reuse this dispatch while
    overriding select nodes. The hook must recurse back through ``render``
    (passing itself) to preserve the override for nested nodes.
    """
    if formatter is not None:
        override = formatter(expr)
        if override is not None:
            return override
    if isinstance(expr, Var):
        return expr.name
    if isinstance(expr, Literal):
        return str(expr.value)
    if isinstance(expr, Builtin):
        return expr.name
    if isinstance(expr, BinaryExpr):
        return f"({render(expr.left, formatter)} {expr.op} {render(expr.right, formatter)})"
    if isinstance(expr, FuncCallExpr):
        return f"{expr.name}({', '.join(render(a, formatter) for a in expr.args)})"
    if isinstance(expr, TernaryExpr):
        return f"({render(expr.cond, formatter)} ? {render(expr.if_true, formatter)} : {render(expr.if_false, formatter)})"
    if isinstance(expr, CastExpr):
        return f"({expr.dtype}){render(expr.expr, formatter)}"
    return repr(expr)


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


def substitute(expr: Expr, mapping: dict[str, Expr]) -> Expr:
    """Replace ``Var(name)`` nodes in ``expr`` with ``mapping[name]``.

    Walks the expression tree non-destructively. Used at:
    - **Lowering time**: substitute placeholder coords with the kernel's
      actual output-coord expressions.
    - **Composition time**: substitute outer IndexMap's placeholders with
      the inner IndexMap's coord_map entries.

    Variables not present in ``mapping`` are left unchanged.
    """
    if isinstance(expr, Var):
        return mapping.get(expr.name, expr)
    if isinstance(expr, (Literal, Builtin)):
        return expr
    if isinstance(expr, BinaryExpr):
        return BinaryExpr(expr.op, substitute(expr.left, mapping), substitute(expr.right, mapping))
    if isinstance(expr, TernaryExpr):
        return TernaryExpr(
            substitute(expr.cond, mapping),
            substitute(expr.if_true, mapping),
            substitute(expr.if_false, mapping),
        )
    if isinstance(expr, FuncCallExpr):
        return FuncCallExpr(expr.name, [substitute(a, mapping) for a in expr.args])
    if isinstance(expr, CastExpr):
        return CastExpr(expr.dtype, substitute(expr.expr, mapping))
    return expr


def free_vars(expr: Expr) -> frozenset[str]:
    """Return the set of ``Var.name`` strings appearing anywhere in ``expr``.

    Used by analyses that need to know which axes an Expr depends on — e.g.
    the fusion splicer's live-axis restriction, which dedups σ entries that
    don't affect any reachable rewrite.
    """
    if isinstance(expr, Var):
        return frozenset({expr.name})
    if isinstance(expr, (Literal, Builtin)):
        return frozenset()
    if isinstance(expr, BinaryExpr):
        return free_vars(expr.left) | free_vars(expr.right)
    if isinstance(expr, TernaryExpr):
        return free_vars(expr.cond) | free_vars(expr.if_true) | free_vars(expr.if_false)
    if isinstance(expr, FuncCallExpr):
        out: frozenset[str] = frozenset()
        for a in expr.args:
            out |= free_vars(a)
        return out
    if isinstance(expr, CastExpr):
        return free_vars(expr.expr)
    return frozenset()


# ---------------------------------------------------------------------------
# Free functions — numpy callables for intrinsics numpy itself doesn't expose
# under the same name. ``resolve_fn`` tries this module first (so our ``max``
# means elementwise maximum, not the reduction ``np.max``) then falls back to
# ``numpy``.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ExprOp — hierarchy for elementwise / reduction ops
# ---------------------------------------------------------------------------
#
# ``ExprOp`` is the base. ``UnaryOp`` / ``BinaryOp`` subclasses fix the
# ``arity``. The op's ``name`` resolves to its numpy callable at
# construction — no registry, no singletons; ``coerce_expr_op(name)``
# just tries ``BinaryOp`` then ``UnaryOp``, with numpy's ufunc ``nin``
# picking the right arity. ``commutative`` and ``identity`` are computed
# properties that read from small class-level tables keyed by name.


class ExprOp:
    """Base class for named scalar ops (elementwise or reducer).

    Subclasses (``UnaryOp`` / ``BinaryOp``) fix the ``arity``. Construction
    resolves the numpy callable from the op's ``name`` — either via
    ``_NAME_TO_FN`` (for non-numpy intrinsics like ``rsqrt`` / ``relu``
    and fused frontend stubs like ``rms_norm``) or ``getattr(np, name)``
    for numpy-aligned names. Unknown names raise. ``commutative`` and
    ``identity`` are derived from class-level tables keyed by name —
    only ops in the tables get those properties set; everything else
    defaults to ``False`` / ``None``.
    """

    arity: int = 1

    # Commutative ops — binary combines where ``op(a, b) == op(b, a)``.
    _COMMUTATIVE: frozenset[str] = frozenset({"add", "multiply", "maximum", "minimum", "amax", "sum", "prod"})
    # Reducer neutral elements — only meaningful when the op is used as
    # an ``Accum`` combine or a ``ReduceOp``.
    _IDENTITY: dict[str, float] = {
        "add": 0.0,
        "sum": 0.0,
        "multiply": 1.0,
        "prod": 1.0,
        "maximum": -1e30,
        "amax": -1e30,
        "minimum": 1e30,
    }

    def __init__(self, name: str) -> None:
        fn = _NAME_TO_FN.get(name)
        if fn is None:
            fn = getattr(np, name, None)
        if fn is None:
            raise ValueError(f"unknown ExprOp name: {name!r} (not in numpy or ir.expr._NAME_TO_FN)")
        # Validate arity against numpy's ufunc metadata when available.
        nin = getattr(fn, "nin", None)
        if nin is not None and nin != self.arity:
            raise ValueError(f"arity mismatch for {name!r}: {type(self).__name__} expects {self.arity}, numpy callable has nin={nin}")
        self.name = name
        self.fn = fn

    @property
    def commutative(self) -> bool:
        return self.name in self._COMMUTATIVE

    @property
    def identity(self) -> float | None:
        return self._IDENTITY.get(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExprOp) and type(self) is type(other) and self.name == other.name

    def __hash__(self) -> int:
        return hash((type(self), self.name))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class UnaryOp(ExprOp):
    arity = 1


class BinaryOp(ExprOp):
    arity = 2


def _unimpl(name: str):
    """Stub callable for fused frontend ops (decomposed before eval)."""

    def _raise(*_args):
        raise NotImplementedError(f"ExprOp({name!r}) has no numpy implementation")

    return _raise


# Names whose callable isn't a plain ``getattr(np, name)``:
# - Non-numpy intrinsics (rsqrt / relu / sigmoid / silu, plus copy).
# - Fused frontend ops (rms_norm / softmax) — stubs; they're decomposed
#   before any eval or codegen touches ``.fn``, but the name must still
#   resolve so ``ExprOp(name)`` doesn't reject them.
# Every other op name matches a numpy attribute — ``ExprOp.__init__``
# falls through to ``getattr(np, name)`` for them.
_NAME_TO_FN: dict[str, object] = {
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "relu": lambda x: np.maximum(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "copy": lambda x: x,
    "rms_norm": _unimpl("rms_norm"),
    "softmax": _unimpl("softmax"),
}


def coerce_expr_op(v: str | ExprOp) -> ExprOp:
    """Accept a string name or ``ExprOp`` instance; return an ``ExprOp``.

    Tries ``BinaryOp(name)`` first, falls back to ``UnaryOp(name)``.
    numpy ufunc ``nin`` metadata makes the attempt self-selecting — binary
    ops like ``add`` (nin=2) pass the binary check, unary ops like ``exp``
    (nin=1) raise in ``BinaryOp.__init__`` and land in ``UnaryOp``. For
    stub callables without ``nin`` (the fused frontend names), binary wins
    by default; arity is informational only since those never evaluate.
    """
    if isinstance(v, ExprOp):
        return v
    try:
        return BinaryOp(v)
    except ValueError:
        pass
    return UnaryOp(v)
