"""Block-structured ``Stmt`` subclasses — ``Loop``, ``Tile``, ``StridedLoop``, ``Cond``.

Each carries a child body (or two, for ``Cond``) and overrides
``Stmt.nested`` so :func:`iter_body` can recurse uniformly. Tile-axis
decode helpers used by ``Tile.render`` live alongside.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Expr, _float_lit
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt.base import INDENT, RenderCtx, Stmt, _axis_identity, _pad, pretty_body
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum


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
    time. Set by scheduling passes (``unroll_small_loops``); has no
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

    @property
    def is_reduce(self) -> bool:
        """A loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        """Recursive rewrite: rebuild ``body`` with each child's ``rewrite``.
        ``axis`` is mapped through ``axis_fn`` (default identity)."""
        return Loop(
            axis=axis_fn(self.axis),
            body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body),
            unroll=self.unroll,
        )

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        unroll = " unroll" if self.unroll else ""
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}:  # {kind}{unroll}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        pad = _pad(ctx.indent)
        out: list[str] = []
        # Per-Loop ``float <acc> = identity;`` for each distinct Accum in the
        # immediate body — suppressed when an enclosing Init already declared it.
        seen: set[str] = set()
        for s in self.body:
            if isinstance(s, Accum) and s.name not in seen:
                seen.add(s.name)
                if s.name in ctx.explicit_inits:
                    continue
                identity = s.op.identity
                if identity is None:
                    raise ValueError(f"Accum {s.name!r} op {s.op.name!r} has no identity")
                out.append(f"{pad}float {s.name} = {_float_lit(float(identity))};")
        var = self.axis.name
        extent = int(self.axis.extent)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        for s in self.body:
            out.extend(s.render(inner))
        out.append(f"{pad}}}")
        return out


@dataclass
class Tile(Stmt):
    """Axis-bound scope wrapper — one CUDA-kernel scope.

    Carries ``axes: tuple[BoundAxis, ...]`` (launch geometry —
    ``BIND_THREAD`` and ``BIND_BLOCK`` axes) plus a body of statements.
    Used at both Tile IR (with Tile-IR-specific stmts like ``Stage`` /
    ``Combine`` in the body) and Kernel IR (with hardware primitives
    like ``Smem`` / ``Sync`` / ``TreeHalve`` after materialization).

    Materialization rewrites the body content but preserves the
    wrapper — same axes, same type, just different body shape.

    ``thread_axes`` / ``block_axes`` are convenience properties that
    project ``axes`` by binding kind — render and launch geometry use
    them.
    """

    axes: tuple[BoundAxis, ...]
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body(self.body)

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return Tile(axes=self.axes, body=body)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        new_axes = tuple(BoundAxis(axis=axis_fn(ba.axis), bind=ba.bind) for ba in self.axes)
        return Tile(axes=new_axes, body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body))

    @property
    def thread_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_THREAD)

    @property
    def block_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes if ba.bind == BIND_BLOCK)

    @property
    def all_axes(self) -> tuple[Axis, ...]:
        return tuple(ba.axis for ba in self.axes)

    def pretty(self, indent: str = "") -> list[str]:
        axes = ", ".join(f"{ba.axis.name}:{ba.axis.extent}={ba.bind}" for ba in self.axes) or "-"
        return [f"{indent}Tile(axes=({axes})):", *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """CUDA block / thread axis decode + body emission.

        Two forms:

        - **Cooperative (``block_axes`` populated):** one CUDA block per
          ``block_axes`` slot, ``thread_axes`` index threads inside the
          block. Decodes ``blockIdx.x`` and ``threadIdx.x`` directly.
        - **Linear (``block_axes`` empty):** flatten all ``thread_axes``
          into one linear ``tid``; bounds-guard against the product of
          extents.
        """
        pad = _pad(ctx.indent)
        inner = ctx.child()
        if self.block_axes:
            out = [f"{pad}{{"]
            out.extend(_render_grid_axis_decode(self.block_axes, "blockIdx.x", inner))
            out.extend(_render_grid_axis_decode(self.thread_axes, "threadIdx.x", inner))
            for s in self.body:
                out.extend(s.render(inner))
            out.append(f"{pad}}}")
            return out

        n_threads = 1
        for ax in self.thread_axes:
            n_threads *= int(ax.extent)
        out = [
            f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
            f"{pad}if (tid < {n_threads}) {{",
        ]
        out.extend(_render_thread_axis_decode(self.thread_axes, inner))
        for s in self.body:
            out.extend(s.render(inner))
        out.append(f"{pad}}}")
        return out


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

    @property
    def is_reduce(self) -> bool:
        """A strided loop is a reduce-loop iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return StridedLoop(
            axis=axis_fn(self.axis),
            start=sigma.apply(self.start),
            step=sigma.apply(self.step) if isinstance(self.step, Expr) else self.step,
            body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body),
            unroll=self.unroll,
        )

    def pretty(self, indent: str = "") -> list[str]:
        kind = "reduce" if self.is_reduce else "free"
        unroll = " unroll" if self.unroll else ""
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}StridedLoop({self.axis.name} = {start}; < {self.axis.extent}; += {step}):  # {kind}{unroll}"
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
                out.append(f"{pad}float {s.name} = {_float_lit(float(identity))};")
        var = self.axis.name
        start_str = self.start.render(ctx)
        step_str = self.step.render(ctx) if isinstance(self.step, Expr) else str(self.step)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {int(self.axis.extent)}; {var} += {step_str}) {{")
        inner = ctx.child()
        for s in self.body:
            out.extend(s.render(inner))
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

    def rewrite(
        self, rename_ssa: Callable[[str], str], sigma: Sigma = Sigma.IDENTITY, axis_fn: Callable[[Axis], Axis] = _axis_identity
    ) -> Stmt:
        return Cond(
            cond=sigma.apply(self.cond),
            body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.body),
            else_body=tuple(s.rewrite(rename_ssa, sigma, axis_fn) for s in self.else_body),
        )

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
        body: list[str] = []
        for s in self.body:
            body.extend(s.render(inner))
        out = [f"{pad}if ({cond}) {{", *body, f"{pad}}}"]
        if self.else_body:
            out[-1] = f"{pad}}} else {{"
            for s in self.else_body:
                out.extend(s.render(inner))
            out.append(f"{pad}}}")
        return out
