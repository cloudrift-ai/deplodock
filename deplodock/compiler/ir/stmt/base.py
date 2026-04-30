"""``Stmt`` abstract base + render context + Expr-tree helpers.

The atom of every IR body. Concrete subclasses live in ``leaves`` (pure
compute) and ``blocks`` (control flow); body walkers + body-level
normalization passes live in ``visit`` and ``normalize``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.body import Body
    from deplodock.compiler.ir.stmt.leaves import Select

INDENT = "    "


# ---------------------------------------------------------------------------
# RenderCtx — target-tuned tables + walk state for ``Stmt.render`` / ``Expr.render``
# ---------------------------------------------------------------------------


@dataclass
class RenderCtx:
    """Per-render state. Targets pre-fill ``intrinsics`` / ``builtins`` with
    target-specific spellings (``"exp" → "expf"``, ``"thread_idx.x" →
    "threadIdx.x"``, ...). ``shapes`` maps every buffer to its declared
    shape so multi-dim ``Load`` / ``Write`` indices can be flattened
    row-major. ``explicit_inits`` carries the set of accumulator names
    whose init has been emitted by an enclosing ``Init`` Stmt — Loop's
    default per-Loop init is suppressed for those names.
    """

    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    explicit_inits: set[str] = field(default_factory=set)
    literal_constants: dict[str, float] = field(default_factory=dict)

    def child(self) -> RenderCtx:
        """Return a new ctx one indent level deeper, sharing all tables."""
        return RenderCtx(
            shapes=self.shapes,
            indent=self.indent + 1,
            intrinsics=self.intrinsics,
            builtins=self.builtins,
            explicit_inits=self.explicit_inits,
            literal_constants=self.literal_constants,
        )


def _pad(n: int) -> str:
    return "    " * n


def _axis_identity(a: Axis) -> Axis:
    """Default ``axis_fn`` for ``Loop.rewrite`` / ``StridedLoop.rewrite``."""
    return a


# ---------------------------------------------------------------------------
# Render helpers — translate elementwise op names to Expr trees, and flatten
# multi-dim coord tuples into row-major flat-index strings.
# ---------------------------------------------------------------------------


_BINARY_OP: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
    "mod": "%",
}


def op_to_expr(fn: str, inputs: list[Expr]) -> Expr:
    """Translate an elementwise op name to an ``Expr`` tree.

    Emits abstract intrinsic names (``"exp"``, ``"fmax"``, ``"fabs"``, ...)
    that targets translate to libm / CUDA spellings via
    ``RenderCtx.intrinsics`` at ``FuncCallExpr.render`` time.
    """
    if fn in _BINARY_OP:
        return BinaryExpr(_BINARY_OP[fn], inputs[0], inputs[1])
    if fn == "maximum":
        return FuncCallExpr("fmax", list(inputs))
    if fn == "minimum":
        return FuncCallExpr("fmin", list(inputs))
    if fn == "pow":
        return FuncCallExpr("pow", list(inputs))
    if fn == "negative":
        return BinaryExpr("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "reciprocal":
        return BinaryExpr("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCallExpr("fmax", [Literal(0.0, "float"), inputs[0]])
    if fn == "sigmoid":
        neg_x = BinaryExpr("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCallExpr("exp", [neg_x])
        return BinaryExpr("/", Literal(1.0, "float"), BinaryExpr("+", Literal(1.0, "float"), exp_neg))
    if fn in ("exp", "rsqrt", "tanh", "sqrt", "erf"):
        return FuncCallExpr(fn, list(inputs))
    if fn == "abs":
        return FuncCallExpr("fabs", list(inputs))
    raise NotImplementedError(f"render: elementwise fn={fn!r} not supported")


def select_to_ternary(s: Select) -> Expr:
    """Build a chained ternary from a ``Select``'s branch list."""
    branches = list(s.branches)
    result: Expr = Var(branches[-1].value)
    for b in reversed(branches[:-1]):
        result = TernaryExpr(cond=b.select, if_true=Var(b.value), if_false=result)
    return result


def render_index(buf: str, indices: tuple, ctx: RenderCtx) -> str:
    """Row-major flatten ``buf[i0][i1]...`` to a single C/CUDA expression.

    Builds the row-major sum as an ``Expr`` and runs ``simplify`` on it so
    constant-zero indices (typical of size-1 outer dims) drop out via the
    standard ``0 * x → 0`` / ``0 + y → y`` folds rather than emitting
    ``0 * stride`` terms in the output.
    """
    if len(indices) == 0:
        return "0"
    if len(indices) == 1:
        return indices[0].simplify(SimplifyCtx.empty()).render(ctx)
    shape = ctx.shapes.get(buf)
    if shape is None or len(shape) != len(indices):
        flat: Expr = indices[0]
        for i in indices[1:]:
            flat = BinaryExpr("+", flat, i)
        return flat.simplify(SimplifyCtx.empty()).render(ctx)
    flat = None
    for d, idx in enumerate(indices):
        stride = 1
        for k in range(d + 1, len(shape)):
            stride *= int(shape[k])
        term: Expr = idx if stride == 1 else BinaryExpr("*", idx, Literal(stride, "int"))
        flat = term if flat is None else BinaryExpr("+", flat, term)
    assert flat is not None
    return flat.simplify(SimplifyCtx.empty()).render(ctx)


# ---------------------------------------------------------------------------
# Stmt — abstract base
# ---------------------------------------------------------------------------


class Stmt:
    """Base class for IR body statements.

    Every concrete Stmt implements:

    - ``deps()`` — SSA names this stmt reads.
    - ``rewrite(rename_ssa, sigma)`` — return a copy with SSA names mapped
      through ``rename_ssa`` and Expr subterms σ-substituted.
    - ``nested()`` — child statement bodies for tree traversal (default:
      no children; block-structured stmts override).
    """

    def deps(self) -> tuple[str, ...]:
        """SSA names this stmt reads — its 'requirements'."""
        raise NotImplementedError

    def defines(self) -> tuple[str, ...]:
        """SSA names this stmt produces — its 'bindings'.

        Default: ``()`` (no SSA def). Name-bearing leaves (``Load``,
        ``Assign``, ``Accum``, ``Init``, ``Select``) override to return
        ``(self.name,)``. Block stmts (``Loop`` / ``StridedLoop`` /
        ``Tile`` / ``Cond``) inherit the default — their bodies define
        names, but the wrapper itself doesn't bind one. ``Write``
        also inherits the default since it writes to a buffer, not
        an SSA value.

        Together with :meth:`deps` this is the def-use surface that
        body-level dependency analyses query (without resorting to
        ``getattr(s, "name", None)`` patterns).
        """
        return ()

    def rewrite(
        self,
        rename_ssa: Callable[[str], str],
        sigma: Sigma = Sigma.IDENTITY,
        axis_fn: Callable[[Axis], Axis] = _axis_identity,
    ) -> Stmt:
        """Return a copy with every SSA name (binding + dep refs) mapped
        through ``rename_ssa``, every Expr subterm σ-substituted, and
        every axis on a ``Loop`` / ``StridedLoop`` mapped through
        ``axis_fn``. Subclasses without axes accept and ignore ``axis_fn``;
        Loop-like subclasses thread it through their bodies.

        ``rename_ssa`` is applied uniformly to the stmt's own name (if any)
        and to each name it reads. Callers typically provide a callable
        that defaults to identity (``lambda n: mapping.get(n, n)``) so only
        the names they care about are changed.
        """
        raise NotImplementedError

    def nested(self) -> tuple[Body, ...]:
        """Child statement bodies for tree traversal.

        Default: no children (leaf stmt). Block-structured stmts override
        to return their body tuple(s) — ``Loop`` returns ``(self.body,)``;
        ``Cond`` returns ``(self.body, self.else_body)``; ``Tile`` returns
        ``(self.body,)``.

        ``iter_body`` walks all IR layers via this single method — every
        node knows its own children, so the walker doesn't need to
        switch on type.
        """
        return ()

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        """Write-side counterpart to :meth:`nested`. Return a copy of this
        stmt with its child bodies replaced by ``bodies`` (positionally
        matching :meth:`nested`'s order).

        Default: leaves have no children, so ``bodies`` must be empty and
        ``self`` is returned unchanged. Block-structured stmts override
        to rebuild themselves from the new bodies. Used by ``Body.map``
        to recurse without an isinstance ladder over the block-stmt set.
        """
        assert not bodies, f"{type(self).__name__}.with_bodies: leaf stmt got {len(bodies)} bodies"
        return self

    def binds_axes(self) -> frozenset[str]:
        """Axes this stmt introduces into scope for its nested bodies.

        Default: ``frozenset()`` (no axis binding). ``Loop`` / ``StridedLoop``
        return ``{self.axis.name}``; ``Tile`` returns the axis names of every
        ``BoundAxis``; ``Cond`` keeps the default. Used by ``Body.fold``
        to thread the bound-axis set through the def-use walk without an
        isinstance ladder over the block-stmt set.
        """
        return frozenset()

    def pretty(self, indent: str = "") -> list[str]:
        """Render this stmt as a list of indented lines.

        Block-structured stmts recurse into their bodies via
        ``child.pretty(indent + INDENT)``; leaves return a single line.
        Subclasses override to control formatting; default surfaces the
        class name as a placeholder for any stmt that forgot to override.
        """
        return [f"{indent}<unrecognized {type(self).__name__}>"]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit indented C / CUDA source lines for this stmt.

        Block-structured stmts recurse via ``child.render(ctx.child())``;
        leaves return a single line. The ``ctx`` carries target-specific
        intrinsic / builtin tables, current indent, and per-buf shapes
        for index flattening. Subclasses override.
        """
        raise NotImplementedError(f"{type(self).__name__}.render not implemented")


def pretty_body(body: Body, indent: str = "") -> list[str]:
    """Flatten ``stmt.pretty(indent)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.pretty(indent))
    return out


def render_body(body: Body, ctx: RenderCtx) -> list[str]:
    """Flatten ``stmt.render(ctx)`` over a body sequence."""
    out: list[str] = []
    for s in body:
        out.extend(s.render(ctx))
    return out
