"""Shared expression AST + coordinate-expression helpers.

Backend-agnostic expression sublanguage used by:

- ``IndexMapOp.coord_map`` (``ir.tensor``): affine output→input coord maps.
- ``Mux.select`` / ``MuxBranch.select`` (``ir.block``): coord predicates.
- Kernel IR (``ir.kernel``): array indices, loop bounds, ternary selects.

The ``_ExprOps`` mixin adds Python operator overloading so expressions can be
built as arithmetic (``Var("i") * 4 + Var("j")``) and comparisons
(``Var("i").lt(Var("n"))``).

Coordinate helpers live here because they are pure operations over the
expression AST — substitution, composition, and placeholder convention.
``compose_index_maps`` is the one exception: it constructs ``IndexMapOp`` /
``IndexSource`` from ``ir.tensor``, which it imports lazily to avoid a
tensor→expr→tensor cycle at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir.tensor import IndexMapOp


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
                import numpy as np

                return np.logical_and(lv, rv)
        if op == "||":
            try:
                return bool(lv) or bool(rv)
            except (TypeError, ValueError):
                import numpy as np

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


_FUNC_CALLS: dict[str, object] = {}
"""Registered math intrinsics for ``FuncCall.eval``. Populated lazily on first call."""


def _load_func_calls() -> dict[str, object]:
    """One-time load of numpy math intrinsics for FuncCall dispatch."""
    if _FUNC_CALLS:
        return _FUNC_CALLS
    import numpy as np

    _FUNC_CALLS.update(
        {
            "expf": np.exp,
            "exp": np.exp,
            "rsqrtf": lambda x: 1.0 / np.sqrt(x),
            "rsqrt": lambda x: 1.0 / np.sqrt(x),
            "tanhf": np.tanh,
            "tanh": np.tanh,
            "fabsf": np.abs,
            "fabs": np.abs,
            "fmaxf": np.maximum,
            "fmax": np.maximum,
            "fminf": np.minimum,
            "fmin": np.minimum,
            "powf": np.power,
            "pow": np.power,
        }
    )
    return _FUNC_CALLS


@dataclass
class FuncCall(_ExprOps):
    """Intrinsic / math function call (C-level: expf, fmaxf, etc.)."""

    name: str
    args: list[Expr]

    def eval(self, env: dict[str, object]) -> object:
        fn = _load_func_calls().get(self.name)
        if fn is None:
            raise NotImplementedError(f"FuncCall.eval: unknown intrinsic {self.name!r}")
        return fn(*(a.eval(env) for a in self.args))


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


Expr = Var | Literal | BinOp | Builtin | FuncCall | Ternary


# ---------------------------------------------------------------------------
# Coordinate-expression helpers for IndexMapOp
# ---------------------------------------------------------------------------
#
# ``IndexMapOp`` (in ``ir.tensor``) describes layout-only ops (slice, cat,
# transpose, reshape, unsqueeze) by mapping output coordinates to input
# coordinates via the ``Expr`` AST above. Convention: an IndexMap's
# ``coord_map[i]`` is an ``Expr`` over placeholder variables
# ``Var("out_coord_0")``, ``Var("out_coord_1")``, ... — one per output
# dimension. At lowering time the placeholders are substituted with the
# ``Expr``s that the kernel uses for its actual output coordinates. The
# same substitution machinery composes adjacent IndexMaps during
# optimization.


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
    return expr


def compose_index_maps(outer: IndexMapOp, inner: IndexMapOp) -> IndexMapOp:
    """Compose two adjacent IndexMapOps into one.

    Substitutes the outer's placeholder coords with the inner's coord_map.
    Both must be single-source (multi-source × multi-source composition is
    not supported — the optimization rule rejects that case).

    Returns a new ``IndexMapOp`` with:
    - ``out_shape`` from ``outer``
    - one source whose ``input_idx = inner.sources[0].input_idx``
    - ``coord_map`` = outer's coord_map composed with inner's coord_map
    - ``select`` = composed conjunction of outer's and inner's selects (if any)
    """
    from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource

    if len(outer.sources) != 1 or len(inner.sources) != 1:
        raise ValueError("compose_index_maps only supports single-source IndexMaps")

    outer_src = outer.sources[0]
    inner_src = inner.sources[0]

    # Mapping: outer's placeholder for axis i → outer's coord_map[i] applied to inner's coords.
    # We want the result expressed over the inner's input coords; walk the inner's coord_map,
    # substituting its placeholders with the outer's coord_map (which are themselves over the
    # merged op's placeholders).
    outer_to_inner_mapping = {placeholder(d).name: outer_src.coord_map[d] for d in range(len(outer_src.coord_map))}
    merged_coord_map = tuple(substitute(c, outer_to_inner_mapping) for c in inner_src.coord_map)

    merged_select = None
    if outer_src.select is not None and inner_src.select is not None:
        inner_select_under_outer = substitute(inner_src.select, outer_to_inner_mapping)
        merged_select = BinOp("&&", outer_src.select, inner_select_under_outer)
    elif outer_src.select is not None:
        merged_select = outer_src.select
    elif inner_src.select is not None:
        merged_select = substitute(inner_src.select, outer_to_inner_mapping)

    return IndexMapOp(
        out_shape=tuple(outer.out_shape),
        sources=(IndexSource(input_idx=inner_src.input_idx, coord_map=merged_coord_map, select=merged_select),),
    )
