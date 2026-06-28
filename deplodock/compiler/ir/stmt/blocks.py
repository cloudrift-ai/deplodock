"""Block-structured ``Stmt`` subclasses — ``Loop``, ``StridedLoop``, ``Cond``.

Each carries a child body (or two, for ``Cond``) and overrides
``Stmt.nested`` so :func:`iter_body` can recurse uniformly. Tile-axis
decode helpers (``_render_grid_axis_decode``, ``_render_thread_axis_decode``,
``_body_uses_lane_warp``) live alongside and were consumed by the typed
tile flavors of the (now demolished) tile IR.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32 as _F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt.algebra import Monoid
from deplodock.compiler.ir.stmt.base import INDENT, RenderCtx, Stmt, _pad, pretty_body, render_body
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Mma

# The loop-carried reduce accumulators — a Loop is a *reduce* loop iff its immediate
# body holds one of these (the predicate `is_reduce` keys off, see below).
_CARRIERS = (Accum, Mma, Monoid)


def _source_suffix(axis: Axis) -> str:
    """Render ``" (of <source.name>)"`` when ``axis`` was carved out of a parent.

    Returns empty for top-level axes (``source_axis is None``) or self-referential
    sources (``source is axis``). Surfaces the partition-planner's split parentage
    in IR dumps without affecting structural keys.
    """
    src = axis.source_axis
    if src is None or src is axis or src.name == axis.name:
        return ""
    return f" (of {src.name})"


@dataclass(frozen=True)
class Loop(Stmt):
    """Explicit iteration block — one loop over an axis.

    ``body`` executes ``axis.extent`` times, once per axis value. Reduce-
    kind Loops fold any ``Accum`` statements in their body into the named
    accumulator (one sweep over the axis per accumulator). Free-kind
    Loops run in parallel with no folding.

    SSA scoping: ``Assign`` / ``Select`` names defined inside ``body`` are
    scoped to that body — invisible to statements outside the Loop. Only
    ``Accum`` targets cross the Loop boundary, carrying the finalized
    reduced value.

    Used by Loop IR for general iteration; reused by Kernel IR for
    serial (post-materialization) loops inside cooperative blocks.

    ``unroll=True`` annotates the loop for ``#pragma unroll`` at render
    time. Set by scheduling passes (``mark_unroll``); has no
    effect on the IR's iteration semantics.
    """

    axis: Axis
    body: Body
    unroll: bool = False

    def __post_init__(self) -> None:
        # Coerce so ``Loop(body=tuple_value)`` keeps working without
        # forcing every construction site to wrap explicitly.
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return Loop(axis=self.axis, body=body, unroll=self.unroll)

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

    @property
    def is_reduce(self) -> bool:
        """A loop is a reduce-loop iff its immediate body contains a carrier
        (``Accum``, its tensor-core form ``Mma``, or the general ``Monoid``)."""
        return any(isinstance(s, _CARRIERS) for s in self.body)

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}{_source_suffix(self.axis)}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:

        pad = _pad(ctx.indent)
        out: list[str] = []
        # Per-Loop ``<dtype> <carrier> = identity;`` for each distinct carried fold in
        # the immediate body — the seed rides on the fold, derived here (no explicit
        # ``Init``). An ``Accum``'s seed is its ``op.identity`` at its ``dtype`` (so
        # fp32-over-fp16 declares ``float acc = 0.0f;``, a fp16 ``max`` declares
        # ``__half acc = __float2half(0.0f);``). A ``Monoid`` carrier (the flat-``Map``
        # fallback, where the carrier renders standalone rather than as lifted bare
        # ``Accum``\\ s) seeds each state component from ``seed_identities()`` at fp32 (the
        # merge runs fp32). A nested fold re-declares per enclosing iteration —
        # scope-local shadowing — so a same-named outer carrier is harmless.
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
                ctx.ssa_dtypes[s.name] = (s.dtype or _F32).name
            elif isinstance(s, Monoid):
                # The carrier's seed is its monoid identity — ``State.identity`` (the neutral
                # element), a property of the carried algebra, not the twist's combine spelling.
                for name, ident in zip(s.state.names, s.state.identity, strict=True):
                    if name in seen:
                        continue
                    seen.add(name)
                    out.append(f"{pad}{ctx.type_name(_F32)} {name} = {ctx.identity_literal(ident.value, _F32)};")
                    ctx.ssa_dtypes[name] = "f32"
        var = self.axis.name
        extent = _extent_c(self.axis, ctx)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        out.extend(render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


def _body_uses_lane_warp(body: Body) -> bool:
    """True iff the body needs the ``lane`` / ``warp`` helper vars.

    The materializer emits ``lane`` / ``warp`` use sites alongside:
    - ``WarpShuffle`` and ``TreeHalve`` (the per-warp partial gate
      ``if (lane == 0)``, ``TreeHalve(..., tid_var="warp")``).
    - ``SetMaxNReg`` (only emitted by the warp-specialized materializer
      path, which wraps in ``Cond(warp < P, ...)`` — the Cond predicate
      references ``Var("warp")`` directly so the helper must be in scope).

    Checking for these three primitives covers every kernel that
    references either helper.
    """
    # Local import — kernel-IR primitives sit in a downstream module.
    from deplodock.compiler.ir.kernel.ir import SetMaxNReg, TreeHalve, WarpShuffle

    return bool(body.iter_of_type(WarpShuffle, TreeHalve, SetMaxNReg))


def _extent_c(ax: Axis, ctx: RenderCtx) -> str:
    """Render one axis extent as a C expression: literal int for static
    ``Dim``, the symbolic name for ``Dim('seq_len')``. ``Dim.__str__``
    already returns the bare ``value`` for those, so a plain ``str`` is
    enough. A composite extent (the ceil-div block axis of a hint-driven
    masked tile, ``(seq_len+31)//32``) must go through the C expr renderer
    so ``//`` becomes ``/`` — ``str`` would leak the Python spelling."""
    ext = ax.extent
    if ext.is_static or isinstance(ext.expr, Var):
        return str(ext)
    # Parenthesize: the extent lands as a ``%`` / ``/`` operand in the decode
    # (``idx % (seq_len+31)/32`` would otherwise mis-associate).
    return f"({ext.expr.render(ctx)})"


def _stride_c(axes, ctx: RenderCtx) -> str:
    """Build a C expression for the product of axis extents — used as the
    stride divisor in the grid-axis decode. ``1`` if ``axes`` is empty."""
    if not axes:
        return "1"
    return " * ".join(_extent_c(a, ctx) for a in axes)


def _render_grid_axis_decode(axes: tuple[Axis, ...], idx_expr: str, ctx: RenderCtx) -> list[str]:
    """Decode ``idx_expr`` (``blockIdx.x`` or ``threadIdx.x``) into per-axis ints."""
    pad = _pad(ctx.indent)
    if not axes:
        return []
    if len(axes) == 1:
        return [f"{pad}int {axes[0].name} = {idx_expr};"]
    decoded: list[str] = []
    stride_axes: list[Axis] = []  # accumulated INNER axes (per-step product is their extents)
    for ax in reversed(axes):
        extent = _extent_c(ax, ctx)
        if not stride_axes:
            decoded.append(f"int {ax.name} = {idx_expr} % {extent};")
        else:
            decoded.append(f"int {ax.name} = ({idx_expr} / ({_stride_c(stride_axes, ctx)})) % {extent};")
        stride_axes.append(ax)
    outer = axes[0]
    outer_stride = _stride_c(list(axes[1:]), ctx)
    decoded[-1] = f"int {outer.name} = {idx_expr} / ({outer_stride});"
    return [pad + line for line in reversed(decoded)]


def _render_swizzled_grid_decode(axes: tuple[Axis, ...], idx_expr: str, group_m: int, ctx: RenderCtx) -> list[str]:
    """Emit a Triton-canonical CTA-swizzle decode of ``idx_expr`` for a
    matmul-shape grid (``axes`` ending in ``(M_b, N_b)``, optionally
    preceded by ``K_s``).

    Consecutive CTA IDs walk down M in groups of ``group_m`` before
    stepping N, so each row-group of CTAs shares A's row tile in L2.
    The runtime ``gsize_m = min(group_m, num_m - first_m)`` clamp makes
    a non-divisor ``num_m`` collapse to standard decode on the tail
    group, and any matmul with ``num_m <= group_m`` (tiny / tall-skinny)
    is a runtime no-op.

    Falls back to the row-major decode when there are fewer than two
    axes — the swizzle pass should have skipped, but the renderer stays
    self-contained.
    """
    if len(axes) < 2:
        return _render_grid_axis_decode(axes, idx_expr, ctx)
    pad = _pad(ctx.indent)
    m_axis, n_axis = axes[-2], axes[-1]
    outer_axes = axes[:-2]  # K_s when SPLITK>1, otherwise empty
    num_m = _extent_c(m_axis, ctx)
    num_n = _extent_c(n_axis, ctx)
    lines: list[str] = ["// CTA swizzle: walk GROUP_M tiles down M before stepping N (L2 A-row reuse)."]
    if outer_axes:
        # Peel the outer block axes (K_s etc.) off the linear CTA ID first
        # so the swizzle only re-decodes the M_b/N_b tail.
        outer_stride = " * ".join((num_m, num_n))  # M_b * N_b
        # Emit outer decodes against (idx_expr / outer_stride) using the
        # standard row-major helper, then re-decode the (m_b, n_b) tail
        # below with the swizzle.
        outer_quotient = f"({idx_expr} / ({outer_stride}))"
        for ax_line in _render_grid_axis_decode(outer_axes, outer_quotient, ctx):
            lines.append(ax_line.lstrip())
        lines.append(f"int bid = {idx_expr} % ({outer_stride});")
    else:
        lines.append(f"int bid = {idx_expr};")
    lines.append(f"int num_m = {num_m};")
    lines.append(f"int num_n = {num_n};")
    lines.append(f"int gsz = {group_m} * num_n;")
    lines.append("int gid = bid / gsz;")
    lines.append(f"int first_m = gid * {group_m};")
    lines.append(f"int gsize_m = ({group_m} < num_m - first_m) ? {group_m} : (num_m - first_m);")
    lines.append(f"int {m_axis.name} = first_m + ((bid % gsz) % gsize_m);")
    lines.append(f"int {n_axis.name} = (bid % gsz) / gsize_m;")
    return [pad + ln for ln in lines]


def _render_thread_axis_decode(axes: tuple[Axis, ...], ctx: RenderCtx) -> list[str]:
    """Emit ``int <axis> = (tid / stride) % extent;`` per axis."""
    pad = _pad(ctx.indent)
    decoded: list[str] = []
    stride_axes: list[Axis] = []
    for ax in reversed(axes):
        extent = _extent_c(ax, ctx)
        if not stride_axes:
            decoded.append(f"int {ax.name} = tid % {extent};")
        else:
            decoded.append(f"int {ax.name} = (tid / ({_stride_c(stride_axes, ctx)})) % {extent};")
        stride_axes.append(ax)
    if len(axes) == 1:
        decoded = [f"int {axes[0].name} = tid;"]
    else:
        outer = axes[0]
        outer_stride = _stride_c(list(axes[1:]), ctx)
        decoded[-1] = f"int {outer.name} = tid / ({outer_stride});"
    return [pad + line for line in reversed(decoded)]


@dataclass(frozen=True)
class StridedLoop(Stmt):
    """Strided iteration: ``for (axis = start; axis < axis.extent; axis += step)``.

    Cooperative variant of ``Loop`` — used at Tile IR to express "threads
    of the CUDA block stride through this axis" (typical
    ``start = Var('t'), step = BLOCK_SIZE``). The body uses the original
    axis Var directly; the strided iteration shape is encoded by the
    loop construct itself rather than via affine indexing in the body.

    Reduction detection mirrors ``Loop``: a ``StridedLoop`` is a
    reduce-loop iff its body contains an ``Accum``."""

    axis: Axis
    start: Expr
    step: Expr
    body: Body
    unroll: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

    def deps(self) -> tuple[str, ...]:
        return ()

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return StridedLoop(axis=self.axis, start=self.start, step=self.step, body=body, unroll=self.unroll)

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

    def exprs(self) -> tuple[Expr, ...]:
        return (self.start, self.step) if isinstance(self.step, Expr) else (self.start,)

    @property
    def is_reduce(self) -> bool:
        """A strided loop is a reduce-loop iff its immediate body contains a carrier
        (``Accum``, its tensor-core form ``Mma``, or the general ``Monoid``)."""
        return any(isinstance(s, _CARRIERS) for s in self.body)

    def pretty(self, indent: str = "") -> list[str]:
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}for {self.axis.name} in {start}..{self.axis.extent}:{step}{_source_suffix(self.axis)}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``for (int axis = start; axis < extent; axis += step)`` with the
        same per-Loop accumulator-init prelude as ``Loop.render``."""

        pad = _pad(ctx.indent)
        out: list[str] = []
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
                ctx.ssa_dtypes[s.name] = (s.dtype or _F32).name
        var = self.axis.name
        start_str = self.start.render(ctx)
        step_str = self.step.render(ctx) if isinstance(self.step, Expr) else str(self.step)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {self.axis.extent.as_static()}; {var} += {step_str}) {{")
        inner = ctx.child()
        out.extend(render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class Cond(Stmt):
    """Conditional block — ``if (cond) { body } [else { else_body }]``.

    ``cond`` is an ``Expr`` over axis Vars and previously-defined SSA
    names; ``body`` and ``else_body`` are stmt sequences executed when
    the predicate evaluates true / false respectively. ``else_body``
    empty means a bare ``if``.

    SSA scoping mirrors ``Loop``: names defined inside either body are
    scoped to that body, except ``Accum`` targets which cross the boundary
    with their finalized value (matching Loop semantics).

    ``deps`` are the SSA names referenced inside ``cond`` — the splicer /
    dataflow analyses need them to thread the predicate's reads through.
    Names referenced inside ``body`` / ``else_body`` are the body stmts'
    own deps; the recursive walker picks them up.
    """

    cond: Expr
    body: Body
    else_body: Body = ()

    def __post_init__(self) -> None:

        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))
        if not isinstance(self.else_body, Body):
            object.__setattr__(self, "else_body", Body(self.else_body))

    def deps(self) -> tuple[str, ...]:
        return tuple(self.cond.free_vars())

    def nested(self) -> tuple[Body, ...]:
        return (self.body, self.else_body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        body, else_body = bodies
        return Cond(cond=self.cond, body=body, else_body=else_body)

    def exprs(self) -> tuple[Expr, ...]:
        return (self.cond,)

    def pretty(self, indent: str = "") -> list[str]:
        lines = [f"{indent}if ({self.cond.pretty()}):", *pretty_body(self.body, indent + INDENT)]
        if self.else_body:
            lines.append(f"{indent}else:")
            lines.extend(pretty_body(self.else_body, indent + INDENT))
        return lines

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        cond = self.cond.render(ctx)
        inner = ctx.child()
        body = render_body(self.body, inner)
        out = [f"{pad}if ({cond}) {{", *body, f"{pad}}}"]
        if self.else_body:
            out[-1] = f"{pad}}} else {{"
            out.extend(render_body(self.else_body, inner))
            out.append(f"{pad}}}")
        return out
