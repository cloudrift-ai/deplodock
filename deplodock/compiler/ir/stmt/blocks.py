"""Block-structured ``Stmt`` subclasses — ``Loop``, ``StridedLoop``, ``Cond``.

Each carries a child body (or two, for ``Cond``) and overrides
``Stmt.nested`` so :func:`iter_body` can recurse uniformly. Tile-axis
decode helpers (``_render_grid_axis_decode``, ``_render_thread_axis_decode``,
``_body_uses_lane_warp``) live alongside and are consumed by the typed
tile flavors in :mod:`deplodock.compiler.ir.tile.ir`.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32 as _F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt.base import INDENT, RenderCtx, Stmt, _pad, pretty_body, render_body
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum


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
        """A loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}{_source_suffix(self.axis)}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:

        pad = _pad(ctx.indent)
        out: list[str] = []
        # Per-Loop ``<dtype> <acc> = identity;`` for each distinct Accum in
        # the immediate body — suppressed when an enclosing Init already
        # declared it. dtype comes from the Accum's optional ``dtype`` field
        # (defaults to fp32) so fp32-over-fp16 accumulators declare as
        # ``float acc = 0.0f;`` while a fp16-typed Accum declares as
        # ``__half acc = __float2half(0.0f);``.
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}{ctx.type_name(s.dtype)} {s.name} = {ctx.identity_literal(identity, s.dtype)};")
                ctx.ssa_dtypes[s.name] = (s.dtype or _F32).name
        var = self.axis.name
        extent = int(self.axis.extent)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        out.extend(render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


def _body_uses_lane_warp(body: Body) -> bool:
    """True iff the body needs the ``lane`` / ``warp`` helper vars.

    The materializer emits ``lane`` / ``warp`` use sites alongside
    ``WarpShuffle`` and ``TreeHalve`` (the per-warp partial gate
    ``if (lane == 0)``, ``TreeHalve(..., tid_var="warp")``), so checking
    for those two primitives covers every kernel that references
    either helper.
    """
    # Local import — kernel-IR primitives sit in a downstream module.
    from deplodock.compiler.ir.kernel.ir import TreeHalve, WarpShuffle

    return bool(body.iter_of_type(WarpShuffle, TreeHalve))


def _render_grid_axis_decode(axes: tuple[Axis, ...], idx_expr: str, ctx: RenderCtx) -> list[str]:
    """Decode ``idx_expr`` (``blockIdx.x`` or ``threadIdx.x``) into per-axis ints."""
    pad = _pad(ctx.indent)
    if not axes:
        return []
    if len(axes) == 1:
        return [f"{pad}int {axes[0].name} = {idx_expr};"]
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = {idx_expr} % {extent};")
        else:
            decoded.append(f"int {ax.name} = ({idx_expr} / {stride}) % {extent};")
        stride *= extent
    outer = axes[0]
    outer_stride = 1
    for ax in axes[1:]:
        outer_stride *= int(ax.extent)
    decoded[-1] = f"int {outer.name} = {idx_expr} / {outer_stride};"
    return [pad + line for line in reversed(decoded)]


def _render_thread_axis_decode(axes: tuple[Axis, ...], ctx: RenderCtx) -> list[str]:
    """Emit ``int <axis> = (tid / stride) % extent;`` per axis."""
    pad = _pad(ctx.indent)
    decoded: list[str] = []
    stride = 1
    for ax in reversed(axes):
        extent = int(ax.extent)
        if stride == 1:
            decoded.append(f"int {ax.name} = tid % {extent};")
        else:
            decoded.append(f"int {ax.name} = (tid / {stride}) % {extent};")
        stride *= extent
    if len(axes) == 1:
        decoded = [f"int {axes[0].name} = tid;"]
    else:
        outer = axes[0]
        outer_stride = 1
        for ax in axes[1:]:
            outer_stride *= int(ax.extent)
        decoded[-1] = f"int {outer.name} = tid / {outer_stride};"
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
        """A strided loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

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
                if s.name in ctx.explicit_inits:
                    continue
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
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {int(self.axis.extent)}; {var} += {step_str}) {{")
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
