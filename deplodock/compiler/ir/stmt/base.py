"""``Stmt`` abstract base + render context + Expr-tree helpers.

The atom of every IR body. Concrete subclasses live in ``leaves`` (pure
compute) and ``blocks`` (control flow); body walkers + body-level
normalization passes live in ``visit`` and ``normalize``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, CastExpr, Expr, FuncCallExpr, Literal, SimplifyCtx, TernaryExpr, Var
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
    # Per-Write coordination metadata derived from the escape-analysis
    # helper. Populated once at ``render_kernelop`` entry; consumed by
    # ``Write.render`` to decide atomicAdd vs plain store and whether to
    # wrap the store in a single-thread broadcast guard. Keyed by
    # ``id(write)`` because ``Write.index`` may hold ``BinaryExpr`` nodes
    # that aren't hashable. Empty when no helper analysis was run (legacy
    # callers / unit-test fixtures that build a RenderCtx directly).
    atomic_writes: dict[int, frozenset[str]] = field(default_factory=dict)
    broadcast_writes: dict[int, frozenset[str]] = field(default_factory=dict)

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
            atomic_writes=self.atomic_writes,
            broadcast_writes=self.broadcast_writes,
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

    Lifted out of ``Assign.render`` so the ``030_stamp_types`` pass can
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
        return FuncCallExpr("fmax", tuple(inputs))
    if fn == "minimum":
        return FuncCallExpr("fmin", tuple(inputs))
    if fn == "pow":
        return FuncCallExpr("pow", tuple(inputs))
    if fn == "negative":
        return BinaryExpr("-", Literal(0.0, "float"), inputs[0])
    if fn == "copy":
        return inputs[0]
    if fn == "reciprocal":
        return BinaryExpr("/", Literal(1.0, "float"), inputs[0])
    if fn == "relu":
        return FuncCallExpr("fmax", (Literal(0.0, "float"), inputs[0]))
    if fn == "sigmoid":
        neg_x = BinaryExpr("-", Literal(0.0, "float"), inputs[0])
        exp_neg = FuncCallExpr("exp", (neg_x,))
        return BinaryExpr("/", Literal(1.0, "float"), BinaryExpr("+", Literal(1.0, "float"), exp_neg))
    if fn in ("exp", "rsqrt", "tanh", "sqrt", "erf"):
        return FuncCallExpr(fn, tuple(inputs))
    if fn == "abs":
        return FuncCallExpr("fabs", tuple(inputs))
    raise NotImplementedError(f"render: elementwise fn={fn!r} not supported")


def render_merge_program(program, state_names, ctx: RenderCtx, pad: str | None = None, *, dtype=None) -> list[str]:
    """Render a monoid ``merge`` / ``combine_states`` program (a tuple of
    ``Assign``) at ``dtype`` (a :class:`DataType`; ``None`` → fp32). An ``Assign``
    whose target is in ``state_names`` is a reassignment of an already-declared
    carried value (``name = …;``); every other ``Assign`` declares a local temp
    (``<ty> t = …;``). Statement order is load-bearing — a state update must follow
    every read of that state's old value (the carrier builder guarantees it).

    Shared by ``Monoid.render`` (the streaming step, fp32) and the kernel-IR
    cross-thread combine primitives ``WarpShuffle`` / ``TreeHalve`` (the
    state-merges-state step, at the carrier's dtype — fp32 for monoids, the
    accumulator dtype for a degenerate scalar reduce), so all spell the carrier's
    algebra identically."""
    if pad is None:
        pad = _pad(ctx.indent)
    dt = "f32" if dtype is None else dtype.name
    ty = ctx.type_name(dt)
    sset = set(state_names)
    out: list[str] = []
    for a in program:
        # The merge runs at ``dt`` — cast any arg with a different dtype (e.g. a raw
        # ``__half`` value loaded into the partial, as in fp16 flash's ``p · v``) so
        # the operator isn't ambiguous. The cast is a no-op for matching args.
        arg_exprs = [Var(x) if ctx.ssa_dtypes.get(x, dt) == dt else CastExpr(ty, Var(x)) for x in a.args]
        rhs = op_to_expr(a.op.name, arg_exprs).render(ctx)
        if a.name in sset:
            out.append(f"{pad}{a.name} = {rhs};")  # reassign carried state
        else:
            ctx.ssa_dtypes[a.name] = dt
            out.append(f"{pad}{ty} {a.name} = {rhs};")  # local temp
    return out


def select_to_ternary(s: Select) -> Expr:
    """Build a chained ternary from a ``Select``'s branch list.

    Each branch value is cast to ``float`` to match the ``float`` result
    ``Select.render`` declares. Without it, a branch list mixing ``__half``
    SSA values (a raw smem/gmem load) with ``float`` ones (a computed value)
    makes the C++ conditional operator's common type ambiguous
    (``cond ? float : __half`` — each converts to the other), which nvcc
    rejects. The casts are no-ops when a value is already ``float``.
    """
    branches = list(s.branches)
    result: Expr = CastExpr("float", Var(branches[-1].value))
    for b in reversed(branches[:-1]):
        result = TernaryExpr(cond=b.select, if_true=CastExpr("float", Var(b.value)), if_false=result)
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
        stride: Expr = Literal(1, "int")
        for k in range(d + 1, len(shape)):
            stride = BinaryExpr("*", stride, _to_expr(shape[k]))
        stride = stride.simplify(SimplifyCtx.empty())
        term: Expr = idx if _is_one(stride) else BinaryExpr("*", idx, stride)
        flat = term if flat is None else BinaryExpr("+", flat, term)
    assert flat is not None
    return flat.simplify(SimplifyCtx.empty()).render(ctx)


def _to_expr(d) -> Expr:
    """Pull the ``Expr`` out of a shape element: ``Dim`` exposes its expr
    directly; bare ``int`` becomes ``Literal``. ``Tensor.shape`` is always
    ``tuple[Dim, ...]``, so the ``int`` branch only catches the rare
    bare-int shape that slipped past coercion."""
    if isinstance(d, Dim):
        return d.expr
    if isinstance(d, int):
        return Literal(d, "int")
    raise TypeError(f"_to_expr: unexpected shape element {d!r}")


def _is_one(e: Expr) -> bool:
    return isinstance(e, Literal) and e.value == 1


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
        working.
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


class ReduceCarrier(Stmt):
    """A loop-carried reduce accumulator — the shared structural role behind
    ``Accum`` (scalar monoid fold), ``Mma`` (tensor-core ``c += a @ b``), and
    the general monoid ``Monoid`` (e.g. flash attention's online softmax).

    A ``Loop`` / ``StridedLoop`` / serial tile is a *reduce* loop iff its
    immediate body holds a ``ReduceCarrier`` — that is the one predicate the
    trait-agnostic machinery (``is_reduce``, axis threading, topo-sort)
    keys off, so a new carrier kind is recognized everywhere by subclassing
    this rather than being threaded through an ``isinstance(s, (Accum, Mma))``
    ladder at every site.

    Rules that *do* care about the combine's algebra don't switch on the
    carrier type either — they read the algebraic traits (``associative`` /
    ``commutative`` / ``has_identity``) the carrier exposes directly to decide
    whether a reduction may be split / reordered (split-K, cooperative
    tree-combine). The traits live on the carrier itself: ``Accum`` forwards to
    its scalar ``op``; ``Mma`` reports the additive-fold constants; ``Monoid``
    reports `associative` / `has_identity` `True` (it is a monoid) with a
    per-instance `commutative` flag. No separate combine object is reified —
    three booleans don't earn one, and an ``Mma`` has no scalar op to point at
    (its accumulation merely *is* additive).

    Two methods name the carrier's reduce surface, distinct from the generic
    SSA def-use surface (:meth:`deps` / :meth:`defines`):

    - :meth:`carried_names` — the accumulator name(s) read-and-written across
      the reduce axis (the carried read is *implicit*, so it is absent from
      :meth:`deps`).
    - :meth:`partial_deps` — the SSA names read to form this iteration's
      contribution, excluding the carried accumulator. Defaults to
      :meth:`deps`, which ``Accum`` / ``Mma`` already define to exclude it.
    """

    @property
    def associative(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__}.associative")

    @property
    def commutative(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__}.commutative")

    @property
    def has_identity(self) -> bool:
        raise NotImplementedError(f"{type(self).__name__}.has_identity")

    def carried_names(self) -> tuple[str, ...]:
        raise NotImplementedError(f"{type(self).__name__}.carried_names")

    def partial_deps(self) -> tuple[str, ...]:
        return self.deps()

    def combine_partials(self) -> tuple:
        """The operator that folds two FULLY-reduced partial states — the ONE
        recombine a decomposition move (split-K / split-KV / cooperative-tree)
        reads, independent of how it is realized (atomic / warp-shuffle / smem
        tree / mma). Returned **as data**: a tuple of :class:`Assign` steps that
        read the carried state plus a second state operand named by
        :meth:`combine_operands` and reassign the merged state.

        ``Monoid`` returns its authored ``combine_states``; the additive carriers
        (``Accum`` scalar fold, ``Mma`` fragment add) return a one-``Assign``
        op-fold. This is the carrier-side half of
        ``plans/algebra-licensed-decomposition-moves.md`` (phase 1)."""
        raise NotImplementedError(f"{type(self).__name__}.combine_partials")

    def combine_operands(self) -> tuple[str, ...]:
        """The SSA names of the SECOND state operand :meth:`combine_partials`
        folds in (the first is the carried state itself). ``Monoid``'s is its
        ``state_b`` (``"<s>__o"`` by default); the additive carriers name theirs
        ``"<carried>__o"`` to match."""
        raise NotImplementedError(f"{type(self).__name__}.combine_operands")


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
    ``050_vectorize_loads`` widens those runs into one :class:`Load`
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
