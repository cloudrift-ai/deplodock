"""Tile IR — schedule decisions as structural Stmts (wrap-body Stage).

Tile IR sits between Loop IR (math) and Kernel IR (fully-scheduled
kernel form). Its job is to encode the *logical* compute plus the
*scheduling decisions* — without committing to hardware primitives.
Materialization (``passes/lowering/kernel``) consumes Tile IR and
produces Kernel IR.

Pipeline shape::

    Loop IR ──launch_geometry──▶ Tile IR (logical compute, default bindings)
                     ──[strategy passes]──▶ Tile IR (annotated)
                     ──materialize_tile──▶ Kernel IR
                     ──render_kernelop──▶ CUDA source

**Wrap-body Stage:** every ``Stage`` is a block-structured Stmt whose
``body`` is the *consumer* subtree that uses the staged smem buffers.
The producer (cooperative Load+Write per source) is synthesized at
materialize time from ``Stage.sources``. Smem lifetime is structural
(decl-to-end-of-Stage.body, not implicit-end-of-block).

**Sources.** Each Stage carries one or more ``Source`` entries; each
Source maps one gmem buffer into one smem slab with its own cache axes
and origin. Multi-source stages (e.g. A + B in a matmul reduce) load
both behind a single sync boundary. Stages with genuinely different
consumer scopes nest instead of multi-sourcing.

**Transport subclasses** (``Stage``, ``BufferedStage``,
``AsyncBufferedStage``, ``TmaBufferedStage``) encode transport policy:
sync cooperative load, ring-buffered sync, cp.async, TMA box-copy.
``pipeline_depth > 1`` on the async / TMA flavors marks a stage for
temporal pipelining (prologue/main/epilogue), expanded by
``070_pipeline_stages`` before materialization.

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` / ``Cond`` come straight from ``ir.loop`` — buf
names are strings so they're directly renderable.

**Tile launch-geometry.** ``Tile.thread_axes`` / ``Tile.block_axes``:
which output axes are bound to thread coords vs CUDA block coords.
Pointwise has ``thread_axes`` populated and ``block_axes`` empty (one
thread per output element). Cooperative reductions have ``block_axes``
populated and ``thread_axes`` empty; the cooperative thread axis is
synthesized at materialization.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Literal as _Lit

from deplodock.compiler.dtype import DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    Expr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)
from deplodock.compiler.ir.stmt import (
    INDENT,
    Accum,
    Assign,
    Body,
    Cond,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Write,
    pretty_body,
)

# `render_body` is the per-Stmt body renderer used by the new tile flavors'
# render methods. Local import below to keep top-of-file imports tidy.
from deplodock.compiler.ir.stmt import render_body as _render_body  # noqa: E402
from deplodock.compiler.ir.stmt.base import RenderCtx, _pad
from deplodock.compiler.ir.stmt.blocks import _body_uses_lane_warp, _render_grid_axis_decode, _render_thread_axis_decode
from deplodock.compiler.ir.stmt.ir import BodyOp

SerialKind = _Lit["plain", "stage_inner", "serial_outer", "pipeline"]


# ---------------------------------------------------------------------------
# AsyncWait — explicit wait carrier for pipelined schedules
# ---------------------------------------------------------------------------
#
# Sync-style async / TMA stages (``pipeline_depth == 1``) get an
# implicit wait at their wrap boundary, emitted by ``_emit_stage`` /
# ``emit_tma_stage`` in the materializer. Pipelined stages
# (``pipeline_depth > 1``) need explicit waits at non-default schedule
# positions: ``070_pipeline_stages`` emits ``AsyncWait``
# Stmts between the issue and consume halves of each steady-state K_o
# iteration (and at the epilogue drain). The materializer's
# ``emit_async_wait`` closure lowers them to ``CpAsyncWait(group=keep)``
# for cp.async, or ``MbarrierWait(mbar, phase, slot)`` for TMA.


@dataclass
class AsyncWait(Stmt):
    """Explicit wait carrier for pipelined async / TMA schedules.

    Sync-style stages (``pipeline_depth == 1``) don't need this — the
    materializer emits an implicit wait at the wrap boundary. Pipelined
    stages do: ``070_pipeline_stages`` peels the steady
    state into issue-now / wait-for-prev / consume-prev, with explicit
    ``AsyncWait`` carrying the schedule:

    - ``keep`` — cp.async ``wait_group`` argument (number of commits to
      leave in flight). ``keep = 1`` in the steady-state body leaves
      the just-issued chunk in flight while waiting for the older one;
      ``keep = 0`` in the epilogue drains every outstanding commit.
    - ``phase`` / ``slot`` — TMA mbarrier-test phase + ring slot for
      the consumer-side ``MbarrierWait``. ``phase = (K_o / bc) % 2``
      tracks how many times the slot has been reused; ``slot = K_o % bc``
      picks the ring slot to wait on.
    """

    keep: int = 0
    phase: Expr | None = None
    slot: Expr | None = None

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        if self.phase is not None:
            out = (*out, self.phase)
        if self.slot is not None:
            out = (*out, self.slot)
        return out

    def pretty(self, indent: str = "") -> list[str]:
        extra = ""
        if self.phase is not None:
            extra += f", phase={self.phase.pretty()}"
        if self.slot is not None:
            extra += f", slot={self.slot.pretty()}"
        return [f"{indent}AsyncWait(keep={self.keep}{extra})"]


# ---------------------------------------------------------------------------
# Stage primitives: Source + CacheDim + AffineAddressing/TemplateAddressing
# ---------------------------------------------------------------------------

# Bytes per stored element in smem. fp32-only assumption — fp16 paths
# over-count by 2x (soft latent bug; see project_tile_ir_fp32_only memory).
BYTES_PER_ELEM = 4


@dataclass(frozen=True)
class AffineAddressing:
    """Affine slab addressing: each cache axis ``i``'s decoded coord is
    *added* to source dim ``dims[i]``.

    ``source_index[d] = origin[d] + decoded_coord(dims[i] == d)``.

    Common case (matmul, RMSNorm, softmax). Materialize reconstructs
    addresses without symbolic substitution.
    """

    dims: tuple[int, ...]


@dataclass(frozen=True)
class TemplateAddressing:
    """Non-affine slab addressing: the consumer Load's original index
    kept verbatim with cache-axis Vars left symbolic. Materialize
    Sigma-substitutes cache-axis Vars → iter-decoded coords.

    Used for collapsed-reshape views (``/``, ``%``) and any case where
    the affine ``origin + decoded`` reconstruction fails. Length ==
    source-buffer rank.
    """

    exprs: tuple[Expr, ...]


@dataclass(frozen=True)
class CacheDim:
    """One cache (smem) axis + which source-buffer dim it covers.

    ``axis`` carries the cache axis identity (its ``source_axis``
    back-pointer, set by 010_stage_inputs at construction time, identifies
    which original output axis this cache dim corresponds to — used by
    downstream passes for per-source-axis grouping).

    ``source_dim`` is the index into the source buffer's shape that this
    cache axis decodes into (i.e. the dim the cache var is added to).
    """

    axis: Axis
    source_dim: int


@dataclass(frozen=True)
class Source:
    """One gmem operand staged into one smem slab.

    Carries everything needed to materialize the cooperative producer:

    - ``name`` — smem buffer name visible to consumer Loads.
    - ``buf`` — gmem source buffer name (the input).
    - ``cache_dims`` — per-cache-axis source-dim mapping; defines the
      slab layout in smem and how cache vars decode into source dims.
    - ``origin`` — per-source-dim CTA-uniform anchor. The cooperative
      load reads ``buf[origin[d] + cache_var(d)]`` (affine) or
      ``buf[template_index[d]]`` (template).
    - ``pad`` — per-cache-axis bank-conflict-breaking pad. Empty = no
      pad. Padding affects smem allocation, not the cooperative-load
      iteration extent.
    - ``addressing`` — affine when every cache axis appears coef-1 in
      exactly one source dim; template when the consumer's original
      Load was a collapsed-reshape and ``origin + decoded`` can't
      reconstruct it. ``cache_dims`` carries the affine mapping (which
      source_dim each cache axis maps to); ``template_index`` carries
      the verbatim source-dim Exprs for the template case.
    - ``dtype`` — source buffer's element dtype. Stamped by
      ``001_stamp_types`` from ``graph.nodes[buf].output.dtype`` so smem
      allocation (``smem_bytes`` / ``alloc_extents``) and downstream
      materialization can read it off the IR without reaching for the
      matcher-populated graph node. ``None`` keeps legacy fp32-assuming
      behavior for tests that construct Source by hand.
    """

    name: str
    buf: str
    cache_dims: tuple[CacheDim, ...]
    origin: tuple[Expr, ...]
    pad: tuple[int, ...] = ()
    template_index: tuple[Expr, ...] | None = None
    dtype: DataType | None = None

    @property
    def cache_axes(self) -> tuple[Axis, ...]:
        return tuple(cd.axis for cd in self.cache_dims)

    @property
    def addressing(self) -> AffineAddressing | TemplateAddressing:
        if self.template_index is not None:
            return TemplateAddressing(exprs=self.template_index)
        return AffineAddressing(dims=tuple(cd.source_dim for cd in self.cache_dims))

    @property
    def alloc_extents(self) -> tuple[int, ...]:
        """Per-cache-axis smem allocation extent: cache extent + pad."""
        extents = tuple(int(ax.extent) for ax in self.cache_axes)
        if not self.pad:
            return extents
        return tuple(e + p for e, p in zip(extents, self.pad, strict=True))

    @property
    def smem_bytes(self) -> int:
        """Bytes of dynamic shared memory this Source allocates (single-slot).

        Uses ``self.dtype.nbytes`` when ``001_stamp_types`` has populated it;
        falls back to the legacy fp32-assuming ``BYTES_PER_ELEM`` constant
        otherwise so handwritten test fixtures without dtype continue to work.
        """
        n = self.dtype.nbytes if self.dtype is not None else BYTES_PER_ELEM
        for e in self.alloc_extents:
            n *= e
        return n

    def with_pad(self, pad: tuple[int, ...]) -> Source:
        return replace(self, pad=pad)


def trivial_stage_body(
    name: str,
    buf: str,
    origin: tuple[Expr, ...],
    axes: tuple[Axis, ...],
    addressing: AffineAddressing | TemplateAddressing,
) -> Body:
    """**Deprecated** — kept for import compatibility during stage-wrap-body refactor.

    Pre-refactor: built the canonical ``Load + Write`` cooperative-load body
    for a Stage. Post-refactor: producer body is reconstructed at materialize
    time from ``Source`` entries; no caller should need this helper. Phase C
    bucket 12 (swizzle split) removes the last reference.
    """
    cache_index = tuple(Var(ax.name) for ax in axes)
    if isinstance(addressing, AffineAddressing):
        decoded: dict[int, Expr] = dict(zip(addressing.dims, cache_index, strict=True))
        src_index = tuple(o if d not in decoded else o + decoded[d] for d, o in enumerate(origin))
    else:
        src_index = addressing.exprs
    load_name = f"{name}__src"
    return Body(
        (
            Load(name=load_name, input=buf, index=src_index),
            Write(output=name, index=cache_index, value=load_name),
        )
    )


def _source_pretty(src: Source) -> str:
    """Legacy single-line source description — kept for debugging / dump
    output. New consumer-facing pretty-print uses ``_source_decl_line``
    which formats each source as ``shared name[...] = buf[...]`` at the
    Stage's indent.
    """
    cache = ", ".join(f"{ax.name}:{ax.extent}@{cd.source_dim}" for ax, cd in zip(src.cache_axes, src.cache_dims, strict=True))
    origin = ", ".join(e.pretty() for e in src.origin)
    pad = f" pad=({', '.join(str(p) for p in src.pad)})" if src.pad and any(src.pad) else ""
    tpl = ""
    if src.template_index is not None:
        tpl = " template=[" + ", ".join(e.pretty() for e in src.template_index) + "]"
    return f"{src.name}<-{src.buf}(origin=({origin}), slab=({cache})){pad}{tpl}"


def _source_decl_line(src: Source) -> str:
    """Render one ``Source`` as ``shared <name>[<cache_axes>] = <buf>[<source_index>];``.

    Cache axes show their extents (``a5:64, a3:16``). The source index
    prefers the literal ``template_index`` when set (preserves explicit
    stride math like ``a3*16 + a6``); otherwise reconstructs from
    ``origin + decoded`` per affine addressing semantics.

    Trailing ``pad`` and stage-flavor suffixes are NOT appended here — the
    Stage subclasses prepend / postfix those at the call site.
    """
    cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in src.cache_axes)
    if src.template_index is not None:
        idx = ", ".join(e.pretty() for e in src.template_index)
    else:
        decoded: dict[int, str] = {}
        for ax, cd in zip(src.cache_axes, src.cache_dims, strict=True):
            existing = decoded.get(cd.source_dim, "")
            decoded[cd.source_dim] = (existing + " + " if existing else "") + ax.name
        parts: list[str] = []
        for d, origin_expr in enumerate(src.origin):
            o = origin_expr.pretty()
            if d in decoded:
                parts.append(f"{o} + {decoded[d]}")
            else:
                parts.append(o)
        idx = ", ".join(parts)
    pad = f" pad=({', '.join(str(p) for p in src.pad)})" if src.pad and any(src.pad) else ""
    return f"shared {src.name}[{cache}] = {src.buf}[{idx}]{pad}"


# ---------------------------------------------------------------------------
# Tile-flavor pretty-print helper — bracket-on-right style
# ---------------------------------------------------------------------------
#
# Every tile flavor renders its axes as Python-style ``for X in 0..N:``
# lines (one per axis, progressively indented like a regular loop nest),
# with a vertical-pipe-and-corner bracket on the right margin grouping
# the axes belonging to one tile. The tile's label (``GridTile``,
# ``serial_outer``, ``reduce stage_inner``, etc.) sits on the closing
# corner.
#
# Example::
#
#     for a0 in 0..8:                  │
#         for a1 in 0..1:              └ GridTile
#             for a2 in 0..16:         │
#                 for a3 in 0..16:     └ ThreadTile
#                     for a4 in 0..64: └ reduce stage_inner
#                         <body>
#
# Single-axis tiles render as one line with ``└ <label>`` directly.


_BRACKET_PAD = 2  # spaces between for-text and the right-margin bracket


def _render_tile_bracket(
    indent: str,
    for_lines: list[str],
    label: str,
    body: Body,
) -> list[str]:
    """Render a tile flavor's ``for`` lines with a right-margin bracket
    grouping them, then recurse into the body at the post-innermost indent.

    ``for_lines`` are the bare ``for ... :`` strings (one per axis),
    rendered without indentation; this helper adds progressive indent.
    ``indent`` is the indent prefix for the outermost ``for``. ``label``
    is the tile-flavor annotation (``"GridTile"``, ``"reduce stage_inner"``,
    etc.) that lands on the closing corner.
    """
    # Each successive for-line indents one more level (Python loop-nest
    # convention). Compute the absolute indent per line.
    lines: list[tuple[str, str]] = []  # (indent_prefix, for_text)
    cur = indent
    for text in for_lines:
        lines.append((cur, text))
        cur = cur + INDENT
    # Right-margin column: pad to the longest (indent+text) of this group.
    max_w = max(len(ind) + len(text) for ind, text in lines)
    margin_col = max_w + _BRACKET_PAD

    out: list[str] = []
    for i, (ind, text) in enumerate(lines):
        line = ind + text
        pad = " " * (margin_col - len(line))
        is_last = i == len(lines) - 1
        bracket = f"└ {label}" if is_last else "│"
        out.append(line + pad + bracket)
    # Body lives inside the innermost ``for``, at one more indent level.
    out.extend(pretty_body(body, cur))
    return out


# ---------------------------------------------------------------------------
# Tile flavors — typed parallel / serial scoping wrappers
# ---------------------------------------------------------------------------
#
# Each tile flavor's *type* encodes its binding decision (block-grid /
# threadIdx / register / serial / strided). Together with the wrap-body
# ``Stage`` family, these are the only block-structured Stmts allowed
# inside a ``TileOp.body`` post-``001_launch_geometry``. ``Loop`` /
# ``StridedLoop`` / ``Tile`` survive in Loop IR (``LoopOp.body``) and as
# transient inputs to ``001_launch_geometry``, but downstream Tile-IR
# passes and Tile→Kernel materialization only see the new flavors.
#
# Shape contract (mirrors ``Stage``'s wrap-body):
#
# - ``ParallelTile`` subclasses (``GridTile`` / ``ThreadTile`` /
#   ``RegisterTile``) carry ``axes: tuple[Axis, ...]`` + ``body: Body``.
#   The body executes once per coord tuple; coords are implicit from the
#   binding (``blockIdx`` / ``threadIdx`` / per-thread register cell).
# - ``SerialTileBase`` subclasses (``SerialTile`` / ``StridedTile``)
#   carry ``axis: Axis`` + ``body: Body`` and run sequentially. Reduce
#   semantics are derived: ``is_reduce`` iff the body contains ``Accum``.


@dataclass
class ParallelTile(Stmt):
    """Abstract base for tile flavors that bind a parallel axis tuple.

    Subclasses pick a parallel coord (``blockIdx`` / ``threadIdx`` /
    register file) for the body to be executed under. Coord decode happens
    at materialize time; the tile itself only carries the axes + body.
    """

    axes: tuple[Axis, ...]
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body(self.body)

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return replace(self, body=body)

    def binds_axes(self) -> frozenset[str]:
        return frozenset(ax.name for ax in self.axes)

    def deps(self) -> tuple[str, ...]:
        return ()

    def _pretty_axes(self) -> str:
        return ", ".join(f"{ax.name}:{ax.extent}" for ax in self.axes) or "-"

    def _pretty_label(self) -> str:
        """Right-margin bracket label. Subclasses override to append
        flavor-specific metadata if any."""
        return type(self).__name__.lower().replace("tile", "")

    def pretty(self, indent: str = "") -> list[str]:
        if not self.axes:
            # Degenerate empty-axis tile (shouldn't normally happen) — just
            # render the label as a one-line header so the body still nests.
            head = f"{indent}{self._pretty_label()}"
            return [head, *pretty_body(self.body, indent + INDENT)]
        for_lines = [f"for {ax.name} in 0..{ax.extent}" for ax in self.axes]
        return _render_tile_bracket(indent, for_lines, self._pretty_label(), self.body)


@dataclass
class GridTile(ParallelTile):
    """CTA-grid parallel tile. Axes lift to ``blockIdx`` (row-major).

    Replaces ``Tile`` with ``BIND_BLOCK`` axes. Split-K is derived at
    codegen time from ``escape_analysis.atomic_axes`` (Write index vs
    enclosing block axes) — no per-tile metadata required.
    """

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return GridTile(axes=self.axes, body=body)

    def _pretty_label(self) -> str:
        return "grid"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit ``blockIdx.x`` axis decode + body. The inner ``ThreadTile``
        renders its threadIdx decode under ``ctx.inside_grid_tile=True``,
        so no per-CTA bounds guard is needed at this level."""
        out = list(_render_grid_axis_decode(self.axes, "blockIdx.x", ctx))
        inner_ctx = replace(ctx, inside_grid_tile=True)
        out.extend(_render_body(self.body, inner_ctx))
        return out


@dataclass
class ThreadTile(ParallelTile):
    """Thread-parallel tile. Axes lift to ``threadIdx`` (row-major flatten).

    Replaces ``Tile`` with ``BIND_THREAD`` axes. Cooperative-K
    cooperativity is derived at materialize / render time from
    ``Accum.axes ∩ ThreadTile.axes`` — see
    ``ir/tile/escape_analysis.py``.
    """

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return ThreadTile(axes=self.axes, body=body)

    def _pretty_label(self) -> str:
        return "thread"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Two render forms picked by ``ctx.inside_grid_tile``.

        - **Cooperative** (inside ``GridTile``): emit ``threadIdx.x`` axis
          decode + optional ``lane`` / ``warp`` helper decls + body. No
          extra brace level — the surrounding ``__global__`` provides one.
        - **Standalone** (pointwise — no enclosing ``GridTile``): flatten
          all axes into a linear ``tid``; bounds-guard against the product
          of extents.
        """
        pad = _pad(ctx.indent)
        if ctx.inside_grid_tile:
            out = list(_render_grid_axis_decode(self.axes, "threadIdx.x", ctx))
            if _body_uses_lane_warp(self.body):
                out.append(f"{pad}int lane = threadIdx.x & 31;")
                out.append(f"{pad}int warp = threadIdx.x >> 5;")
            out.extend(_render_body(self.body, ctx))
            return out

        inner = ctx.child()
        n_threads = 1
        for ax in self.axes:
            n_threads *= int(ax.extent)
        out = [
            f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
            f"{pad}if (tid < {n_threads}) {{",
        ]
        out.extend(_render_thread_axis_decode(self.axes, inner))
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass
class RegisterTile(ParallelTile):
    """Per-thread register-cell tile. Body replicated F× per axis by 006a.

    Replaces ``Loop(role=REGISTER)``. The ``axes`` tuple carries one or
    more register axes (typically M_r / N_r for matmul); the planner
    chooses the extents (``FM`` / ``FN`` knobs). After the 006a
    register-tile pass runs, every ``RegisterTile`` is consumed: the
    body is fully unrolled, SSA names get per-cell suffixes, and the
    ``RegisterTile`` wrapper disappears.
    """

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            "RegisterTile must be consumed by 006a_register_tile_planned before render — "
            f"reached render with axes={tuple(ax.name for ax in self.axes)!r}"
        )


@dataclass
class SerialTileBase(Stmt):
    """Abstract base for serial-iteration tile flavors. One axis, one body."""

    axis: Axis
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body(self.body)

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def binds_axes(self) -> frozenset[str]:
        return frozenset({self.axis.name})

    def deps(self) -> tuple[str, ...]:
        return ()

    @property
    def is_reduce(self) -> bool:
        """A serial tile is a reduce iff its immediate body contains an ``Accum``."""
        return any(isinstance(s, Accum) for s in self.body)


@dataclass
class SerialTile(SerialTileBase):
    """Sequential iteration over ``axis``. Replaces ``Loop``.

    ``kind`` carries the planner's structural intent:

    - ``"plain"``: ordinary serial loop (no special role).
    - ``"serial_outer"``: outer chunked-K loop driving slab refresh
      (today's ``Role.SERIAL_OUTER``). Targeted by ``030_use_ring_buffers``
      / ``015_pipeline_k_outer``.
    - ``"stage_inner"``: inner reduce loop inside a ``Stage``'s wrapped
      body (today's ``Role.STAGE_INNER``). Slab-axis marker for
      ``010_stage_inputs``.
    - ``"pipeline"``: serial outer loop marked for temporal pipelining
      by ``015_pipeline_k_outer``.

    ``unroll=True`` annotates the loop for ``#pragma unroll`` at render
    time. Set by ``080_mark_unroll``; has no effect on iteration semantics.
    """

    kind: SerialKind = "plain"
    unroll: bool = False

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return SerialTile(axis=self.axis, body=body, kind=self.kind, unroll=self.unroll)

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}for {self.axis.name} in 0..{self.axis.extent}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """Per-Loop accumulator-init prelude (same as ``Loop.render``) +
        ``for (int axis = 0; axis < extent; axis++) { body }``."""
        from deplodock.compiler.dtype import F32 as _F32  # noqa: PLC0415

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
        extent = int(self.axis.extent)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass
class StridedTile(SerialTileBase):
    """Strided serial iteration: ``for (axis = start; axis < extent; axis += step)``.

    Replaces ``StridedLoop``. Cooperative thread-stride iteration when a
    surrounding ``ThreadTile`` axis covers the stride (typical
    ``start = Var('tid'), step = BLOCK_SIZE``). Reduce semantics derive
    from body content like ``SerialTile``.
    """

    start: Expr = field(default_factory=lambda: Literal(0, "int"))
    step: Expr = field(default_factory=lambda: Literal(1, "int"))
    unroll: bool = False

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return StridedTile(axis=self.axis, body=body, start=self.start, step=self.step, unroll=self.unroll)

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = (self.start,)
        if isinstance(self.step, Expr):
            out = (*out, self.step)
        return out

    def pretty(self, indent: str = "") -> list[str]:
        start = self.start.pretty()
        step = self.step.pretty() if isinstance(self.step, Expr) else self.step
        head = f"{indent}for {self.axis.name} in {start}..{self.axis.extent}:{step}"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def render(self, ctx: RenderCtx) -> list[str]:
        """``for (int axis = start; axis < extent; axis += step)`` with the
        same per-Loop accumulator-init prelude as ``SerialTile.render``."""
        from deplodock.compiler.dtype import F32 as _F32  # noqa: PLC0415

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
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


# ---------------------------------------------------------------------------
# Stage hierarchy — wrap-body
# ---------------------------------------------------------------------------


@dataclass
class Stage(Stmt):
    """Wrap-body cooperative stage. ``body`` is the CONSUMER subtree.

    Materializes to: ``Sync`` + cooperative ``Load+Write`` per source +
    ``Sync`` + ``<materialized body>``.

    The leading ``Sync`` ensures prev-iteration compute finishes reading
    before this iteration's slab is overwritten; the trailing ``Sync``
    makes the freshly-loaded slab visible CTA-wide before the consumer
    reads it.

    Subclasses (``BufferedStage``, ``AsyncBufferedStage``,
    ``TmaBufferedStage``) override the producer transport — ring-buffered
    sync slabs, cp.async copy, or TMA box copy via mbarrier respectively.
    """

    sources: tuple[Source, ...]
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            self.body = Body.coerce(self.body)
        if not self.sources:
            raise ValueError(f"{type(self).__name__}: requires at least one Source")

    def nested(self) -> tuple[Body, ...]:
        return (self.body,)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return replace(self, body=body)

    def deps(self) -> tuple[str, ...]:
        return ()

    def external_reads(self) -> tuple[str, ...]:
        return tuple(s.buf for s in self.sources)

    def local_decls(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sources)

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        for s in self.sources:
            out = (*out, *s.origin)
            if s.template_index is not None:
                out = (*out, *s.template_index)
        return out

    @property
    def smem_bytes(self) -> int:
        return sum(s.smem_bytes for s in self.sources)

    def replace_sources(self, sources: tuple[Source, ...]) -> Stage:
        return replace(self, sources=sources)

    def pretty(self, indent: str = "") -> list[str]:
        """Per-source ``shared name[...] = buf[...];`` decls at ``indent``,
        followed by the consumer body at the SAME indent (no extra nest).

        Subclasses prepend a flavor prefix (``buffered`` / ``async`` /
        ``tma`` / ``compute``) and may append a metadata suffix via
        :meth:`_pretty_prefix` / :meth:`_pretty_suffix`.
        """
        prefix = self._pretty_prefix()
        suffix = self._pretty_suffix()
        prefix = f"{prefix} " if prefix else ""
        decls = [f"{indent}{prefix}{_source_decl_line(s)}{suffix}" for s in self.sources]
        return decls + list(pretty_body(self.body, indent))

    def _pretty_prefix(self) -> str:
        """Optional prefix (``buffered`` / ``async`` / ``tma`` / ``compute``)
        prepended to each ``shared`` decl line. Defaults to empty for plain
        ``Stage``."""
        return ""

    def _pretty_suffix(self) -> str:
        """Optional suffix (``  # depth=N`` / ``  # swizzle=B128``) appended
        after each ``shared`` decl. Defaults to empty."""
        return ""


@dataclass
class BufferedStage(Stage):
    """Stage with ``buffer_count`` rotating smem slabs selected by ``phase``.

    Sync transport (cooperative ``Load + Write``); the leading
    ``__syncthreads`` between prev-compute and next-load is dropped
    because consecutive iterations write to different physical slabs.

    ``buffer_count >= 2`` is enforced. ``phase`` is required and
    typically ``Var(K_outer_axis) % buffer_count``.
    """

    buffer_count: int = field(default=2, kw_only=True)
    phase: Expr = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.buffer_count < 2:
            raise ValueError(f"BufferedStage: buffer_count must be >= 2, got {self.buffer_count}")

    @property
    def smem_bytes(self) -> int:
        return super().smem_bytes * self.buffer_count

    def exprs(self) -> tuple[Expr, ...]:
        return (*super().exprs(), self.phase)

    def _pretty_prefix(self) -> str:
        return f"buffered[{self.buffer_count}@{self.phase.pretty()}]"


@dataclass
class AsyncBufferedStage(BufferedStage):
    """Buffered stage transported via ``cp.async``.

    Materialize emits ``Smem`` + cooperative ``CpAsyncCopy`` +
    ``CpAsyncCommit`` per source; the wait is implicit at the wrap
    boundary (cp.async wait_group(0) + Sync). Requires sm_80+.

    ``pipeline_depth == 1`` ⇒ simple wait-at-boundary lowering.
    ``pipeline_depth > 1`` ⇒ the enclosing K-outer Loop is software-
    pipelined: a prologue issues the first depth-1 chunks, the main
    Loop overlaps issue+wait+compute, the epilogue drains. Expansion
    happens in ``070_pipeline_stages`` before materialize.
    """

    pipeline_depth: int = field(default=1, kw_only=True)

    def _pretty_prefix(self) -> str:
        depth = f" depth={self.pipeline_depth}" if self.pipeline_depth > 1 else ""
        return f"async[{self.buffer_count}@{self.phase.pretty()}{depth}]"


class SwizzleMode(enum.Enum):
    """TMA shared-memory swizzle pattern.

    Picked by the lowering pass from inner-dim byte stride; consumed by
    the backend's ``cuTensorMapEncodeTiled`` call. ``NONE`` is the
    interim default until MMA-side swizzle support lands.
    """

    NONE = "NONE"
    B32 = "B32"
    B64 = "B64"
    B128 = "B128"


@dataclass
class TmaBufferedStage(BufferedStage):
    """Buffered stage transported via ``cp.async.bulk.tensor`` (TMA).

    Materialize emits a ``TmaDescriptor`` (host-side ``CUtensorMap``,
    hoisted to kernel prologue), an ``MbarrierInit``, and at the stage
    site a ``Cond(tid==0, [MbarrierArriveExpectTx, TmaLoad])`` — one
    elected thread issues the box copy. The mbarrier wait is implicit
    at the wrap boundary; the trailing ``Sync`` is omitted because
    mbarrier arrival already provides CTA-wide visibility.

    Requires sm_90+ and ``--gpu-architecture=sm_90a``. Eligible only for
    ``AffineAddressing`` with the inner source dim contiguous and 16 B
    aligned. ``pad`` must be empty (TMA box writes contiguous rows;
    bank-pad would misalign).

    ``pipeline_depth`` mirrors ``AsyncBufferedStage``.
    """

    pipeline_depth: int = field(default=1, kw_only=True)
    swizzle: SwizzleMode = field(default=SwizzleMode.NONE, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        for s in self.sources:
            if s.pad and any(s.pad):
                raise ValueError(f"TmaBufferedStage: source {s.name!r} pad must be empty, got {s.pad!r}")

    def _pretty_prefix(self) -> str:
        depth = f" depth={self.pipeline_depth}" if self.pipeline_depth > 1 else ""
        sw = "" if self.swizzle == SwizzleMode.NONE else f" swizzle={self.swizzle.value}"
        return f"tma[{self.buffer_count}@{self.phase.pretty()}{depth}{sw}]"


@dataclass
class ComputeStage(Stage):
    """Hoisted invariant compute Stage.

    Distinguished from a transport Stage by ``compute``: a per-thread
    cooperative compute body that runs once per stage activation,
    reading from sibling-stage smem (via ``Source`` entries whose ``buf``
    names a sibling Stage's smem buffer) and writing into this Stage's
    smem allocation.

    No gmem transport — sources' ``buf`` names target smem, not gmem.
    ``external_reads`` returns empty so 015 / 013 / 011 don't classify
    compute stages as gmem-transport eligibility candidates.

    Optional ``buffer_count`` / ``phase`` mirror ``BufferedStage`` so 010
    can promote a ``ComputeStage`` to a ring-buffered output. ``buffer_count
    = 1`` (default) means single-slot; ``>= 2`` requires ``phase``.
    """

    compute: Body = field(default_factory=lambda: Body(()), kw_only=True)
    buffer_count: int = field(default=1, kw_only=True)
    phase: Expr | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.compute, Body):
            self.compute = Body.coerce(self.compute)
        if self.buffer_count < 1:
            raise ValueError(f"ComputeStage: buffer_count must be >= 1, got {self.buffer_count}")
        if self.buffer_count > 1 and self.phase is None:
            raise ValueError("ComputeStage: phase required when buffer_count > 1")

    def nested(self) -> tuple[Body, ...]:
        return (self.compute, self.body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (compute, body) = bodies
        return replace(self, compute=compute, body=body)

    def external_reads(self) -> tuple[str, ...]:
        # Sibling-smem reads aren't external buffer dependencies — the
        # producers live in the same Tile scope.
        return ()

    @property
    def smem_bytes(self) -> int:
        return super().smem_bytes * self.buffer_count

    def exprs(self) -> tuple[Expr, ...]:
        out = super().exprs()
        if self.phase is not None:
            out = (*out, self.phase)
        return out

    def _pretty_prefix(self) -> str:
        if self.buffer_count > 1 and self.phase is not None:
            return f"compute[{self.buffer_count}@{self.phase.pretty()}]"
        return "compute"

    def _cooperative_label(self) -> str:
        """Right-margin label on the synthesized cooperative for-nest.
        Carries the buffer/phase annotation when ring-buffered so the
        prefix-less decl line doesn't have to."""
        if self.buffer_count > 1 and self.phase is not None:
            return f"cooperative[{self.buffer_count}@{self.phase.pretty()}]"
        return "cooperative"

    def pretty(self, indent: str = "") -> list[str]:
        """``shared <name>[<cache_axes>]`` per output Source, then a
        synthesized ``for <ax> in 0..<extent>`` nest labeled ``cooperative``
        wrapping the producer body, then the native consumer subtree at the
        same indent. The ``compute`` nature reads from the body below —
        no leading ``compute`` prefix is needed.

        ComputeStage carries TWO bodies: ``compute`` runs once per
        activation to populate the output slabs from sibling-Stage smem;
        ``body`` is the regular consumer subtree. The producer's source
        is the ComputeStage's own smem, so the RHS that ``_source_decl_line``
        would print (``<self> = <self>[...]``) is suppressed here.
        """
        out: list[str] = []
        for s in self.sources:
            cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in s.cache_axes)
            out.append(f"{indent}shared {s.name}[{cache}]")
        # Synthesize a cooperative for-nest over the first source's cache
        # axes — every Source in the ComputeStage shares the same cache
        # axes by construction (007b builds a single fused output Source
        # spanning the cone's cache axes).
        cache_axes = self.sources[0].cache_axes
        for_lines = [f"for {ax.name} in 0..{ax.extent}" for ax in cache_axes]
        out.extend(_render_tile_bracket(indent + INDENT, for_lines, self._cooperative_label(), self.compute))
        out.extend(pretty_body(self.body, indent + INDENT))
        return out


# ---------------------------------------------------------------------------
# Top-level: TileOp
# ---------------------------------------------------------------------------


@dataclass
class TileOp(BodyOp):
    """One GPU kernel as a Tile IR program — pre-materialization.

    :class:`BodyOp` subclass parallel to ``LoopOp``: lives as a graph
    node, carries a body of Tile IR statements plus a kernel name.
    Materialization turns a ``TileOp`` into a ``KernelOp``.
    """

    def __post_init__(self) -> None:
        from deplodock.compiler.ir.stmt import normalize_body

        coerced = Body.coerce(self.body)
        normalized = normalize_body(coerced, hoist=False)
        self.body = normalized if isinstance(normalized, Body) else Body(normalized)
        n_tiles = sum(1 for s in self.body if isinstance(s, (GridTile, ThreadTile)))
        if n_tiles > 1:
            raise ValueError(f"TileOp.body must contain at most one outer GridTile/ThreadTile, got {n_tiles}")
        self._seed_io_placeholders()

    def _launch_geometry(self) -> tuple[tuple[Axis, ...], tuple[Axis, ...]]:
        """``(block_axes, thread_axes)`` for the outermost tile flavor.

        Returns ``((), ())`` if no ``GridTile``/``ThreadTile`` is present
        (e.g. a degenerate body). For ``GridTile`` wrapping a ``ThreadTile``,
        the block axes come from the GridTile and thread axes from the
        inner ThreadTile. For a standalone ``ThreadTile`` (pointwise), the
        block set is empty.
        """
        for s in self.body:
            if isinstance(s, GridTile):
                block_axes = s.axes
                for child in s.body:
                    if isinstance(child, ThreadTile):
                        return block_axes, child.axes
                return block_axes, ()
            if isinstance(s, ThreadTile):
                return (), s.axes
        return (), ()

    def validate(self, ctx) -> bool:
        """Reject post-register-tile variants whose launch geometry would
        exceed device limits (threads-per-CTA and dynamic smem).

        Pre-register-tile TileOps skip the THREAD check; the smem check
        runs whenever Stages are present.
        """
        from math import prod  # noqa: PLC0415

        # Dedupe by Source name: pipelining
        # (070_pipeline_stages) replicates an
        # ``AsyncBufferedStage`` for prologue + steady-state issue, both
        # writing into the same smem buffer (same name, same allocation).
        # Counting them independently would double-charge the budget and
        # silently reject pipelined variants on smem-tight kernels.
        per_source: dict[str, int] = {}
        for s in self.body.iter():
            if isinstance(s, Stage):
                for src in s.sources:
                    # ``s.smem_bytes`` includes the buffer_count factor; the
                    # per-source share is the source's slab × buffer_count.
                    buf_count = getattr(s, "buffer_count", 1)
                    per_source[src.name] = src.smem_bytes * buf_count
        staged = sum(per_source.values())
        if staged > ctx.max_dynamic_smem:
            return False

        if "FM" not in self.knobs:
            return True
        _, thread_axes = self._launch_geometry()
        if not thread_axes:
            return True
        threads = prod(int(ax.extent) for ax in thread_axes)
        return threads <= ctx.max_threads_per_cta

    def score(self, ctx) -> float:  # noqa: ARG002 — ctx reserved for cc-specific tuning
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.ir.tile.ir import Stage as _Stage  # noqa: PLC0415

        target_threads = 256
        target_ctas = 256
        score = 0.0
        block_axes, thread_axes = self._launch_geometry()
        if not thread_axes and not block_axes:
            return 0.0

        thread_extents = [int(ax.extent) for ax in thread_axes]
        block_extents = [int(ax.extent) for ax in block_axes]
        if not thread_extents:
            return 0.0

        threads = prod(thread_extents)
        ctas = prod(block_extents) if block_extents else 1

        if "FM" in self.knobs:
            final_threads = threads
            cells = max(1, int(self.knobs.get("FM", 1)) * int(self.knobs.get("FN", 1)))
        elif threads >= 1024:
            final_threads = target_threads
            cells = max(1, threads // target_threads)
        else:
            final_threads = threads
            cells = 1

        if final_threads < 32:
            score -= 2.0
        elif final_threads > 1024:
            score -= 2.0
        else:
            distance = abs(final_threads - target_threads)
            multiplier = 2.0 if (cells < 16 and "FM" in self.knobs) else 1.0
            score -= min(distance / target_threads * multiplier, 2.0)

        if cells == 1:
            score -= 1.0
        elif cells > 64:
            score -= min((cells - 64) / 64.0, 1.0)

        if ctas < target_ctas:
            score -= (target_ctas - ctas) / target_ctas
        elif ctas <= 2048:
            score += 0.5
        else:
            score -= min((ctas - 2048) / 4096.0, 2.5)

        splitk = int(self.knobs.get("SPLITK", 1))
        # SPLITK > 4 pays a real atomicAdd contention cost on top of the
        # cross-CTA reduction tax; SPLITK=2/4 is usually enough.
        if splitk > 4:
            score -= min((splitk - 4) / 4.0, 1.0)

        if any(isinstance(s, _Stage) for s in self.body.iter()):
            score += 1.0
        if "FM" in self.knobs:
            score += 1.0

        score += self._coalescing_bonus(thread_axes)

        return score

    def _coalescing_bonus(self, thread_axes) -> float:
        """Reward tile shapes whose thread axes align with the innermost
        output-write dimension (the coalesced-stride axis).

        For each top-level ``Write`` in the body, look at the **last**
        index expression — that's the inner-stride dim of the output
        buffer. Count how many thread-axis Vars appear free in it, sum
        their extents, and turn that into a bonus capped at +1.0.

        Why this matters: empirically, two variants with identical
        ``cells × threads × ctas`` can differ 2x in measured latency
        purely because one parks its threads along the M (outer-stride)
        output axis and the other parks them along N (inner-stride).
        The N-major variant gets coalesced gmem loads on the matmul B
        operand and coalesced store on the output; the M-major one
        strides B by ``N`` per thread. The four base score terms
        (threads, cells, ctas, splitk) can't distinguish them — this
        bonus does.
        """
        from deplodock.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        if not thread_axes:
            return 0.0
        thread_names = {ax.name for ax in thread_axes}
        best_inner_extent = 0
        for stmt in self.body.iter():
            if not isinstance(stmt, Write) or not stmt.index:
                continue
            inner_vars = set(stmt.index[-1].free_vars())
            matched = inner_vars & thread_names
            if not matched:
                continue
            extent_in_inner = sum(int(ax.extent) for ax in thread_axes if ax.name in matched)
            if extent_in_inner > best_inner_extent:
                best_inner_extent = extent_in_inner
        if best_inner_extent <= 0:
            return 0.0
        # Bonus saturates at warp-size alignment: an inner-dim thread
        # extent of ≥ 32 already gives full coalescing; anything past
        # that has no marginal coalescing gain. Cap the reward so the
        # term doesn't dominate the threads/cells/ctas signals.
        warp = 32
        return min(best_inner_extent / warp, 1.0)


# ---------------------------------------------------------------------------
# Cooperative thread-block size — number of threads per CUDA block when a
# Tile uses BIND_THREAD axes from a cooperative strategy.
# ---------------------------------------------------------------------------

from deplodock.compiler.tuning import cooperative_block_size as _coop_block_size  # noqa: E402

BLOCK_SIZE = _coop_block_size()


__all__ = [
    # Shared expressions (re-exported for convenience)
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow (re-exported)
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    "StridedLoop",
    # Tile-IR statements — typed tile flavor hierarchy
    "ParallelTile",
    "GridTile",
    "ThreadTile",
    "RegisterTile",
    "SerialTileBase",
    "SerialTile",
    "StridedTile",
    "SerialKind",
    "Stage",
    "BufferedStage",
    "AsyncBufferedStage",
    "TmaBufferedStage",
    "ComputeStage",
    "SwizzleMode",
    "AffineAddressing",
    "TemplateAddressing",
    "CacheDim",
    "Source",
    "AsyncWait",
    "trivial_stage_body",  # deprecated stub during refactor
    "BYTES_PER_ELEM",
    "Stmt",
    # Top-level
    "TileOp",
    # Scheduling constants
    "BLOCK_SIZE",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]

# Register Tile-IR stmts with the shared rewrite/simplify dispatch.
from deplodock.compiler.ir.tile import passes as _passes  # noqa: E402, F401
