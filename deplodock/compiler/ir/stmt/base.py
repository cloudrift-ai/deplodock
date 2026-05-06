"""``Stmt`` abstract base + render context + Expr-tree helpers.

The atom of every IR body. Concrete subclasses live in ``leaves`` (pure
compute) and ``blocks`` (control flow); body walkers + body-level
normalization passes live in ``visit`` and ``normalize``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
    # Per-buffer byte offsets into a single ``extern __shared__`` pool
    # ``_smem_pool``. When non-empty, ``Smem.render`` emits a pointer
    # alias into the pool instead of a stand-alone ``__shared__`` array
    # — the only way to exceed the 48 KB static-smem cap.
    smem_dynamic_offsets: dict[str, int] = field(default_factory=dict)

    def child(self) -> RenderCtx:
        """Return a new ctx one indent level deeper, sharing all tables."""
        return RenderCtx(
            shapes=self.shapes,
            indent=self.indent + 1,
            intrinsics=self.intrinsics,
            builtins=self.builtins,
            explicit_inits=self.explicit_inits,
            literal_constants=self.literal_constants,
            smem_dynamic_offsets=self.smem_dynamic_offsets,
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

        Per-stmt logic lives in :mod:`.passes` (singledispatch over Stmt
        type + introspection walker for the Stage hierarchy). This method
        is a thin shim so existing call sites (``s.rewrite(...)``) keep
        working. Tile-IR Stmt registrations are loaded by importing
        ``deplodock.compiler.ir.tile.ir`` (which any caller passing a
        Tile-IR Stmt has done already).
        """
        from deplodock.compiler.ir.stmt.passes import rewrite  # noqa: PLC0415

        return rewrite(self, rename_ssa, sigma, axis_fn)

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

    def has_side_effects(self) -> bool:
        """True iff executing this stmt produces an externally observable
        effect (a buffer write). For compound stmts (Loop / StridedLoop /
        Tile / Cond), True iff any nested stmt does.

        Use this to gate transforms that change execution count
        (hoisting, loop interchange, predication): a side-effecting
        stmt run N times instead of M is observable, so it pins the
        enclosing iteration to its current scope.

        Note: ``Accum`` / ``Init`` are *not* side-effecting in this
        sense — they're scope-bound (their semantics depend on which
        Loop encloses them) but moving the *whole enclosing block* is
        safe. Hoisting passes that want to move a Loop containing an
        Accum need a separate scope-bound check on the leaf, not
        ``has_side_effects`` on the wrapper."""
        return any(c.has_side_effects() for sub in self.nested() for c in sub.iter())

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

    def exprs(self) -> Iterable[Expr]:
        """Direct Expr fields of this stmt (non-recursive into nested bodies).

        Default: ``()``. Concrete stmts override to surface their carried
        Expr trees: ``Load`` / ``Write`` yield ``self.index``; ``Select``
        yields each branch's predicate; ``Cond`` yields ``self.cond``;
        ``StridedLoop`` yields ``self.start`` and ``self.step`` (when an
        ``Expr``); Tile-IR ``Stage`` yields its source-index template.
        ``Loop`` / ``Tile`` keep the default — their Exprs live inside
        their bodies, which the fold's tree traversal reaches separately.

        The third slice of the per-stmt analysis surface alongside
        :meth:`deps` (SSA reads) and :meth:`nested` (child bodies). The
        fold callback uses ``exprs()`` to pull free Vars at each stmt
        for direct axis contributions, with ``free_vars()`` filtered by
        the ``bound`` set threaded through :meth:`binds_axes`.
        """
        return ()

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
    """Flatten ``stmt.render(ctx)`` over a body sequence.

    Vectorizes runs of N=4 (or N=2) consecutive ``Load`` stmts whose
    indices match in every higher dim and form ``e0, e0+1, ...`` on the
    last dim into a single ``ld.shared.v4`` (``LDS.128``) emit. Each lane
    receives 4 fp32 in a single instruction, and the warp-wide phase
    structure of an LDS.128 (4 phases of 8 lanes × 32 fp32) naturally
    maps to 32 distinct banks per phase — eliminating the 4-way conflict
    that plagues 4 separate scalar ``LDS.32`` reads at the same offsets.

    Buffer-side prerequisites (16-byte alignment, fp32 dtype) are
    enforced upstream — TMA-target ``Smem`` decls already get
    ``__align__(16)``, and the renderer only emits ``float`` typed
    Loads. The match is purely syntactic so degenerate cases (a Load
    whose ``input`` resolves to a scalar literal constant, or a run
    that doesn't form a contiguous index sequence) bypass cleanly.
    """
    out: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        for run_n in (4, 2):
            run = _vec_load_run(body, i, run_n, ctx)
            if run is not None:
                out.extend(run)
                i += run_n
                break
        else:
            out.extend(body[i].render(ctx))
            i += 1
    return out


def _vec_load_run(body: Body, start: int, n: int, ctx: RenderCtx) -> list[str] | None:
    """If ``body[start:start+n]`` matches the consecutive-Load pattern,
    return the rendered ``float<n>`` vector load + per-lane unpacks.
    Otherwise return ``None`` so :func:`render_body` falls back to the
    per-stmt path."""
    from deplodock.compiler.ir.stmt.leaves import Load  # local — avoid cycle

    if start + n > len(body):
        return None
    loads = body[start : start + n]
    if not all(isinstance(s, Load) for s in loads):
        return None
    if any(s.is_literal(ctx.literal_constants) for s in loads):
        return None
    inputs = {s.input for s in loads}
    if len(inputs) != 1:
        return None
    rank = len(loads[0].index)
    if rank == 0 or any(len(s.index) != rank for s in loads[1:]):
        return None
    higher = loads[0].index[:-1]
    for s in loads[1:]:
        if s.index[:-1] != higher:
            return None
    # Compare last-dim expressions via affine decomposition: same var
    # coefficients, anchor differing by exactly ``k``. ``simplify`` alone
    # doesn't fold ``(a*4 + k) - (a*4)`` to ``k`` (it only handles
    # literal-only arithmetic), so we extract the affine form and
    # subtract the anchors instead.
    from deplodock.compiler.ir.expr import affine_form  # local — keep base.py minimal

    inner_0 = loads[0].index[-1]
    free = inner_0.free_vars()
    for s in loads[1:]:
        free = free | s.index[-1].free_vars()
    af0 = affine_form(inner_0, free)
    if af0 is None:
        return None
    anchor_0, coeffs_0 = af0
    for k, s in enumerate(loads):
        if k == 0:
            continue
        af = affine_form(s.index[-1], free)
        if af is None:
            return None
        anchor_k, coeffs_k = af
        if coeffs_k != coeffs_0:
            return None
        diff = BinaryExpr("-", anchor_k, anchor_0).simplify(SimplifyCtx.empty())
        if not (isinstance(diff, Literal) and isinstance(diff.value, int) and diff.value == k):
            return None
    flat = render_index(loads[0].input, loads[0].index, ctx)
    pad = _pad(ctx.indent)
    vname = f"_v_{loads[0].name}"
    components = ("x", "y", "z", "w")[:n]
    out = [f"{pad}float{n} {vname} = *reinterpret_cast<const float{n}*>(&{loads[0].input}[{flat}]);"]
    out.extend(f"{pad}float {s.name} = {vname}.{c};" for s, c in zip(loads, components, strict=True))
    return out
