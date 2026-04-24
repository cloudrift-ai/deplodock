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

    Returns BinOp nodes, enabling::

        Var("row") * Var("cols") + Var("j")   # → BinOp("+", BinOp("*", ...), ...)
        Var("i").lt(Var("n"))                   # → BinOp("<", ...)
    """

    def __add__(self, other: Expr) -> BinOp:
        return BinOp("+", self, _coerce(other))

    def __radd__(self, other: Expr) -> BinOp:
        return BinOp("+", _coerce(other), self)

    def __sub__(self, other: Expr) -> BinOp:
        return BinOp("-", self, _coerce(other))

    def __rsub__(self, other: Expr) -> BinOp:
        return BinOp("-", _coerce(other), self)

    def __mul__(self, other: Expr) -> BinOp:
        return BinOp("*", self, _coerce(other))

    def __rmul__(self, other: Expr) -> BinOp:
        return BinOp("*", _coerce(other), self)

    def __truediv__(self, other: Expr) -> BinOp:
        return BinOp("/", self, _coerce(other))

    def __mod__(self, other: Expr) -> BinOp:
        return BinOp("%", self, _coerce(other))

    def __neg__(self) -> BinOp:
        return BinOp("-", Literal(0, "int"), self)

    def lt(self, other: Expr) -> BinOp:
        """Less-than (``<``)."""
        return BinOp("<", self, _coerce(other))

    def ge(self, other: Expr) -> BinOp:
        """Greater-or-equal (``>=``)."""
        return BinOp(">=", self, _coerce(other))

    def eq(self, other: Expr) -> BinOp:
        """Equal (``==``)."""
        return BinOp("==", self, _coerce(other))

    def and_(self, other: Expr) -> BinOp:
        """Logical AND (``&&``)."""
        return BinOp("&&", self, _coerce(other))

    def or_(self, other: Expr) -> BinOp:
        """Logical OR (``||``)."""
        return BinOp("||", self, _coerce(other))


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
class BinOp(_ExprOps):
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
        raise ValueError(f"Unknown BinOp: {op}")


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
class FuncCall(_ExprOps):
    """Intrinsic / math function call.

    ``name`` is an ``ExprOp`` registry name (numpy-aligned: ``exp`` /
    ``tanh`` / ``maximum`` / ``reciprocal`` / …). ``FuncCall.eval`` pulls
    the pre-bound callable off the registered ``ExprOp``; the CUDA
    emitter's ``_translate_intrinsic`` rewrites the same name to the
    ``f``-suffixed libm spelling at source-render time.
    """

    name: str
    args: list[Expr]

    def eval(self, env: dict[str, object]) -> object:
        op = _EXPR_OP_REGISTRY.get(self.name)
        if op is None:
            raise NotImplementedError(f"FuncCall.eval: unknown intrinsic {self.name!r}")
        return op.fn(*(a.eval(env) for a in self.args))


@dataclass
class Ternary(_ExprOps):
    """Ternary expression: cond ? if_true : if_false.

    Uses Python's conditional: when ``cond`` evaluates to an ndarray,
    callers that want elementwise selection should use ``np.where``
    directly; ``Ternary.eval`` only supports scalar ``cond``.
    """

    cond: Expr
    if_true: Expr
    if_false: Expr

    def eval(self, env: dict[str, object]) -> object:
        return self.if_true.eval(env) if self.cond.eval(env) else self.if_false.eval(env)


@dataclass
class Cast(_ExprOps):
    """Type cast of an inner expression to ``dtype`` (e.g. ``"int"``, ``"float"``)."""

    dtype: str
    expr: Expr

    def eval(self, env: dict[str, object]) -> object:
        v = self.expr.eval(env)
        if self.dtype == "int":
            return np.asarray(v).astype(np.int64) if hasattr(v, "__array__") else int(v)
        return v


Expr = Var | Literal | BinOp | Builtin | FuncCall | Ternary | Cast


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
    if isinstance(expr, BinOp):
        return f"({render(expr.left, formatter)} {expr.op} {render(expr.right, formatter)})"
    if isinstance(expr, FuncCall):
        return f"{expr.name}({', '.join(render(a, formatter) for a in expr.args)})"
    if isinstance(expr, Ternary):
        return f"({render(expr.cond, formatter)} ? {render(expr.if_true, formatter)} : {render(expr.if_false, formatter)})"
    if isinstance(expr, Cast):
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
    if isinstance(expr, BinOp):
        return BinOp(expr.op, substitute(expr.left, mapping), substitute(expr.right, mapping))
    if isinstance(expr, Ternary):
        return Ternary(
            substitute(expr.cond, mapping),
            substitute(expr.if_true, mapping),
            substitute(expr.if_false, mapping),
        )
    if isinstance(expr, FuncCall):
        return FuncCall(expr.name, [substitute(a, mapping) for a in expr.args])
    if isinstance(expr, Cast):
        return Cast(expr.dtype, substitute(expr.expr, mapping))
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
    if isinstance(expr, BinOp):
        return free_vars(expr.left) | free_vars(expr.right)
    if isinstance(expr, Ternary):
        return free_vars(expr.cond) | free_vars(expr.if_true) | free_vars(expr.if_false)
    if isinstance(expr, FuncCall):
        out: frozenset[str] = frozenset()
        for a in expr.args:
            out |= free_vars(a)
        return out
    if isinstance(expr, Cast):
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
# ``ExprOp`` is the base. ``UnaryOp`` / ``BinaryOp`` subclasses fix
# ``arity`` at the class level; the op's ``name``, ``commutative``, and
# ``identity`` (reducer neutral) live on the instance. Canonical instances
# are singletons stored in ``_EXPR_OP_REGISTRY`` and looked up by name.
#
# Each op carries its numpy callable directly — no resolve_fn indirection.


class ExprOp:
    """Base class for named scalar ops (elementwise or reducer).

    Subclasses (``UnaryOp`` / ``BinaryOp``) fix the ``arity``. Instances
    carry ``name`` (used for codegen + serialization), ``fn`` (the numpy
    callable that implements the op — pre-bound at registration so no
    runtime lookup is needed), and reducer metadata (``commutative``,
    ``identity``; ``None`` when the op isn't a valid reducer). Equality
    compares by ``(type, name)`` so singletons round-trip through sets /
    dicts without surprises.
    """

    arity: int = 1

    def __init__(self, name: str, fn, commutative: bool = False, identity: float | None = None) -> None:
        self.name = name
        self.fn = fn
        self.commutative = commutative
        self.identity = identity

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


_EXPR_OP_REGISTRY: dict[str, ExprOp] = {}


def _reg(op: ExprOp) -> ExprOp:
    _EXPR_OP_REGISTRY[op.name] = op
    return op


# Canonical op instances. Each carries its numpy callable directly — no
# ``resolve_fn`` indirection. The four ops with no numpy equivalent
# (``rsqrt`` / ``relu`` / ``sigmoid`` / ``silu``) get a lambda; everything
# else pulls from numpy.
ADD = _reg(BinaryOp("add", np.add, commutative=True, identity=0.0))
SUBTRACT = _reg(BinaryOp("subtract", np.subtract))
MULTIPLY = _reg(BinaryOp("multiply", np.multiply, commutative=True, identity=1.0))
DIVIDE = _reg(BinaryOp("divide", np.divide))
MOD = _reg(BinaryOp("mod", np.mod))
MAXIMUM = _reg(BinaryOp("maximum", np.maximum, commutative=True, identity=-1e30))
MINIMUM = _reg(BinaryOp("minimum", np.minimum, commutative=True, identity=1e30))
# Torch-level reduction name (``amax``) kept distinct from ``maximum`` so
# the traced graph preserves the source op's spelling; semantics match.
AMAX = _reg(BinaryOp("amax", np.maximum, commutative=True, identity=-1e30))
POW = _reg(BinaryOp("pow", np.power))
# Reduction-only names (same shape as add / multiply but distinct spellings
# because tensor-IR ``ReduceOp`` uses ``sum`` / ``prod``).
SUM = _reg(BinaryOp("sum", np.add, commutative=True, identity=0.0))
PROD = _reg(BinaryOp("prod", np.multiply, commutative=True, identity=1.0))
# Unary math — numpy-backed
NEGATIVE = _reg(UnaryOp("negative", np.negative))
EXP = _reg(UnaryOp("exp", np.exp))
RECIPROCAL = _reg(UnaryOp("reciprocal", np.reciprocal))
TANH = _reg(UnaryOp("tanh", np.tanh))
ABS = _reg(UnaryOp("abs", np.abs))
FABS = _reg(UnaryOp("fabs", np.fabs))
SQRT = _reg(UnaryOp("sqrt", np.sqrt))
LOG = _reg(UnaryOp("log", np.log))
COPY = _reg(UnaryOp("copy", lambda x: x))
# Unary math — no numpy equivalent, so an inline lambda
RSQRT = _reg(UnaryOp("rsqrt", lambda x: 1.0 / np.sqrt(x)))
RELU = _reg(UnaryOp("relu", lambda x: np.maximum(0.0, x)))
SIGMOID = _reg(UnaryOp("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))))
SILU = _reg(UnaryOp("silu", lambda x: x / (1.0 + np.exp(-x))))


def get_expr_op(name: str) -> ExprOp:
    """Look up the canonical ``ExprOp`` singleton for ``name``; raise on unknown."""
    op = _EXPR_OP_REGISTRY.get(name)
    if op is None:
        raise ValueError(f"unknown ExprOp name: {name!r}")
    return op


def coerce_expr_op(v: str | ExprOp) -> ExprOp:
    """Accept a string name or ``ExprOp`` instance; return an ``ExprOp``.

    Used by ``ElementwiseOp`` / ``ReduceOp`` / ``Accum`` in ``__post_init__``
    to keep string-based construction working while storing structured
    metadata on the field. Names not in the registry mint a generic
    ``UnaryOp(name)`` — the fused frontend ops (``rms_norm`` / ``softmax``
    / …) pass through Tensor IR under such names and are decomposed
    before any eval/codegen cares about arity or identity.
    """
    if isinstance(v, ExprOp):
        return v
    op = _EXPR_OP_REGISTRY.get(v)
    if op is not None:
        return op

    # Unknown name — fused frontend ops like ``rms_norm`` / ``softmax`` pass
    # through Tensor IR under such names and are decomposed before any eval
    # or codegen cares. Give them a stub callable that raises if invoked.
    def _unimpl(*_args, _name=v):
        raise NotImplementedError(f"ExprOp({_name!r}) has no numpy implementation")

    return UnaryOp(v, _unimpl)
