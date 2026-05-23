"""``Stmt`` abstract base + render context + Expr-tree helpers.

The atom of every IR body. Concrete subclasses live in ``leaves`` (pure
compute) and ``blocks`` (control flow); body walkers + body-level
normalization passes live in ``visit`` and ``normalize``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.body import Body
    from deplodock.compiler.ir.stmt.leaves import Select
    from deplodock.compiler.render_target import RenderTarget


def _default_render_target():
    """Lazy default: a :class:`CudaRenderTarget` instance. Used when
    ``RenderCtx`` is constructed without an explicit target — keeps the
    legacy "everything is CUDA" behavior for tests / golden output.

    Lazy to avoid importing the backend at IR-module load time."""
    from deplodock.compiler.backend.cuda.render_target import CudaRenderTarget  # noqa: PLC0415

    return CudaRenderTarget()


INDENT = "    "


# ---------------------------------------------------------------------------
# RenderCtx — target-tuned tables + walk state for ``Stmt.render`` / ``Expr.render``
# ---------------------------------------------------------------------------


@dataclass
class RenderCtx:
    """Per-render state. ``target`` is the :class:`RenderTarget` that
    owns every target-specific C spelling decision (type names,
    conversion intrinsics, per-dtype op spellings, native-op coverage,
    vector load shapes). Everything else here is generic walk state.

    ``intrinsics`` / ``builtins`` keep the legacy abstract→spelling
    indirection used by ``FuncCallExpr.render`` / ``Builtin.render``
    for symbols that don't vary by dtype (``thread_idx.x`` →
    ``threadIdx.x``, etc.). Dtype-aware lookups go through
    ``ctx.target.intrinsic(...)`` instead.

    ``shapes`` maps every buffer to its declared shape so multi-dim
    ``Load`` / ``Write`` indices can be flattened row-major.
    ``explicit_inits`` carries the set of accumulator names whose init
    has been emitted by an enclosing ``Init`` Stmt — Loop's default
    per-Loop init is suppressed for those names.
    """

    target: RenderTarget = field(default_factory=_default_render_target)
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    indent: int = 1
    intrinsics: dict[str, str] = field(default_factory=dict)
    builtins: dict[str, str] = field(default_factory=dict)
    explicit_inits: set[str] = field(default_factory=set)
    literal_constants: dict[str, float] = field(default_factory=dict)
    # SSA names whose defining Load came from a ``literal_constants``
    # input — populated by ``render_body`` after scanning the body, and
    # consumed by ``Var.render`` to inline the float at use sites instead
    # of emitting a named ``float in0 = 0.044f;`` decl.
    literal_ssa: dict[str, float] = field(default_factory=dict)
    # Per-buffer byte offsets into a single ``extern __shared__`` pool
    # ``_smem_pool``. When non-empty, ``Smem.render`` emits a pointer
    # alias into the pool instead of a stand-alone ``__shared__`` array
    # — the only way to exceed the 48 KB static-smem cap.
    smem_dynamic_offsets: dict[str, int] = field(default_factory=dict)
    # Per-buffer canonical dtype tokens (``"f32"`` / ``"f16"``) for every
    # global-buffer name (kernel inputs + outputs). ``Load`` declares its
    # SSA-name local in the source buffer's C type so values flow at
    # buffer dtype end-to-end where possible; ``Write`` inserts the
    # target's conversion intrinsic only when the value's dtype disagrees
    # with the destination buffer's dtype. Missing entries default to
    # ``"f32"`` so legacy bodies render unchanged.
    buffer_dtypes: dict[str, str] = field(default_factory=dict)
    # Per-SSA-name canonical dtype tokens, populated as ``render_body``
    # walks. ``Load`` writes the source buffer's dtype; ``Assign`` writes
    # ``promote(args)``. Consumed by downstream ``Assign`` / ``Write`` to
    # decide native-vs-promote-fallback and to insert conversions.
    ssa_dtypes: dict[str, str] = field(default_factory=dict)
    # Render-time hint to ``Literal.render``: when set to a non-default
    # dtype, float literals render via ``target.literal(text, dtype)``
    # so they compose with non-default-dtype operands. Set transiently
    # by ``Assign.render`` around the native expression render.
    literal_default_dtype: str | None = None
    # True inside a ``GridTile``'s render scope. Set by ``GridTile.render``
    # when descending into its body so a nested ``ThreadTile.render`` picks
    # the cooperative-decode form (threadIdx → axes) rather than the
    # standalone pointwise form (linear tid + bounds guard).
    inside_grid_tile: bool = False

    def child(self) -> RenderCtx:
        """Return a new ctx one indent level deeper, sharing all tables."""
        return RenderCtx(
            target=self.target,
            shapes=self.shapes,
            indent=self.indent + 1,
            intrinsics=self.intrinsics,
            builtins=self.builtins,
            explicit_inits=self.explicit_inits,
            literal_constants=self.literal_constants,
            literal_ssa=self.literal_ssa,
            smem_dynamic_offsets=self.smem_dynamic_offsets,
            buffer_dtypes=self.buffer_dtypes,
            ssa_dtypes=self.ssa_dtypes,
            literal_default_dtype=self.literal_default_dtype,
            inside_grid_tile=self.inside_grid_tile,
        )

    # ---- Convenience wrappers over ``self.target``. These exist so the
    # render methods read ``ctx.type_name(dt)`` instead of pulling the
    # target out by hand; they also default ``None`` dtype to F32 so the
    # call sites don't repeat that boilerplate.

    def type_name(self, dtype) -> str:
        """C type spelling for a local declaration. Accepts a
        :class:`DataType`, a canonical-name string, or ``None`` (treated
        as F32)."""
        return self.target.type_name(_canonical_dtype_name(dtype))

    def identity_literal(self, identity: float, dtype) -> str:
        """Render an accumulator's identity (0, 1, -inf, ...) as a C
        literal in ``dtype``, wrapping with the target's dtype cast if
        needed (e.g. ``__float2half(0.0f)`` for fp16). Accepts a
        :class:`DataType`, a canonical-name string, or ``None`` (F32)."""
        from deplodock.compiler.ir.expr import _float_lit  # noqa: PLC0415

        return self.target.literal(_float_lit(float(identity)), _canonical_dtype_name(dtype))


def _canonical_dtype_name(dtype) -> str:
    """Normalize ``DataType | str | None`` to the canonical dtype token."""
    if dtype is None:
        return "f32"
    if isinstance(dtype, str):
        return dtype
    return dtype.name


def dtype_promote(op_name: str, arg_dtypes: list[str]) -> str:
    """Promote an elementwise op's arg dtypes to a single result dtype.

    Mirrors the inline rule in ``Assign.render``: the result is the
    common dtype iff every arg agrees; any disagreement promotes to f32.
    Today's IR only meaningfully encounters f16/f32 mixes; the rule is
    "all f16 → f16, otherwise f32." ``op_name`` is accepted for future
    op-specific overrides (e.g. ``relu`` would still emit f16 even with
    no f16 args), but is unused today.

    Lifted out of ``Assign.render`` so the ``001_stamp_types`` pass can
    reuse the same rule when stamping ``Assign.dtype`` on the IR.
    """
    if arg_dtypes and all(d == "f16" for d in arg_dtypes):
        return "f16"
    return "f32"


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
        """SSA names this stmt reads — its 'requirements'.

        Default: ``()`` (no SSA deps). Mirrors :meth:`defines` so stmts
        with no SSA inputs (kernel-IR primitives like ``Sync`` /
        ``CpAsyncCommit`` / ``MbarrierInit``) don't need to override.
        """
        return ()

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

    def external_reads(self) -> tuple[str, ...]:
        """External-buffer names this stmt reads from. Default: ``()``.

        Overrides: ``Load`` returns ``(self.input,)``; ``CpAsyncCopy``
        returns ``(self.src,)``; ``TmaDescriptor`` returns
        ``(self.src_buf,)``; ``Stage`` returns the source-Load inputs
        from its body. The reads of any name covered by some stmt's
        :meth:`local_decls` (smem / staged buffers) get filtered out at
        :class:`BodyOp` aggregation time, so leaves don't need to know
        about the surrounding declarations."""
        return ()

    def external_writes(self) -> tuple[str, ...]:
        """External-buffer names this stmt writes to. Default: ``()``.

        Overrides: ``Write`` returns ``(self.output,)``. Smem writes
        (``CpAsyncCopy`` / ``TmaLoad``) are local — they target a name
        covered by some :class:`Smem` decl's :meth:`local_decls` — and
        get filtered out at aggregation rather than fighting to suppress
        them at the leaf."""
        return ()

    def local_decls(self) -> tuple[str, ...]:
        """Buffer names this stmt declares as kernel-local — the body's
        external-buffer aggregator filters reads / writes naming these
        names out of its inputs / outputs (they live inside the kernel,
        not on the signature). Default: ``()``.

        Overrides: ``Smem`` returns ``(self.name,)`` (kernel-IR shared
        buffers); ``Stage`` returns ``(self.name,)`` (tile-IR staged
        buffers — materialized into an ``Smem`` later)."""
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

    Detection of vectorizable Load runs (``LDS.128`` / ``__half2``) is
    no longer done here — the dedicated Kernel-IR pass
    ``003_vectorize_loads`` widens those runs into one :class:`Load`
    with ``extra_names`` populated before render. This function's only
    pre-walk responsibility is registering literal-constant Loads so
    their ``Var(name)`` uses inline as float literals instead of
    materializing a named local.
    """
    from deplodock.compiler.ir.stmt.leaves import Load  # local — avoid cycle

    # Pre-pass: register every literal-constant Load's SSA name in the ctx
    # so subsequent ``Var(name)`` references render as the literal value.
    # The Load itself is then skipped — the loop IR pretty printer behaves
    # the same way (``multiply(in4, 0.044)`` instead of an explicit
    # ``in0 = load mul_1_c1[0]; multiply(in4, in0)`` chain).
    if ctx.literal_constants:
        new_map = dict(ctx.literal_ssa)
        changed = False
        for s in body:
            # Literal-const Loads are always scalar (the vectorize pass
            # excludes literal-const buffers from its widening logic).
            if isinstance(s, Load) and s.is_scalar and s.is_literal(ctx.literal_constants):
                new_map[s.name] = ctx.literal_constants[s.input]
                changed = True
        if changed:
            ctx = replace(ctx, literal_ssa=new_map)

    out: list[str] = []
    for s in body:
        if isinstance(s, Load) and s.is_scalar and s.name in ctx.literal_ssa and s.is_literal(ctx.literal_constants):
            continue
        out.extend(s.render(ctx))
    return out
