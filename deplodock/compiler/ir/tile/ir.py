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
``080_pipeline_stages`` before materialization.

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
from deplodock.compiler.ir.stmt.blocks import (
    _body_uses_lane_warp,
    _render_grid_axis_decode,
    _render_swizzled_grid_decode,
    _render_thread_axis_decode,
)
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
# positions: ``080_pipeline_stages`` emits ``AsyncWait``
# Stmts between the issue and consume halves of each steady-state K_o
# iteration (and at the epilogue drain). The materializer's
# ``emit_async_wait`` closure lowers them to ``CpAsyncWait(group=keep)``
# for cp.async, or ``MbarrierWait(mbar, phase, slot)`` for TMA.


@dataclass(frozen=True)
class AsyncWait(Stmt):
    """Explicit wait carrier for pipelined async / TMA schedules.

    Sync-style stages (``pipeline_depth == 1``) don't need this — the
    materializer emits an implicit wait at the wrap boundary. Pipelined
    stages do: ``080_pipeline_stages`` peels the steady
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

    The trailing CTA-fence ``Sync`` after the materializer's
    ``MbarrierWait`` / ``CpAsyncWait`` defaults to ``__syncthreads()``.
    Inside a WS consumer subtree it routes to a named ``bar.sync N, M``
    instead — the materializer derives the named-barrier params from
    the enclosing ``WarpSpecialize`` context, not from fields on this
    Stmt. (``__syncthreads()`` is CUDA UB on the warp-divergent
    producer/consumer branch.)
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
# WarpSpecialize — producer/consumer split for TMA-pipelined kernels
# ---------------------------------------------------------------------------
#
# Tile-IR marker that the materializer (``100_materialize_tile``) lowers
# into the full mbarrier handshake: empty-mbarrier ring (``Smem`` +
# per-slot ``MbarrierInit``), per-K_o ``MbarrierWait`` / ``MbarrierArrive``
# pairs, named ``bar.sync`` consumer fences, ``SetMaxNReg`` register
# budget redistribution, and the producer/consumer ``Cond`` wrapper.
#
# The pass that emits this (``085_warp_specialize``) keeps all Tile-IR
# vocabulary — no ``from deplodock.compiler.ir.kernel.ir import …``.
# Companion to 080's ``AsyncWait``: where ``AsyncWait`` lets the pass
# declare "wait for an async chunk, materializer picks the primitive",
# ``WarpSpecialize`` lets the pass declare "split this ThreadTile body
# into producer and consumer roles, materializer wires the rest".


@dataclass(frozen=True)
class WarpSpecialize(Stmt):
    """Producer/consumer warp split inside a TMA-pipelined kernel.

    Fields:

    - ``producer_body`` — stmts run by producer warp(s) (TMA-issue
      ``StageBundle`` scaffolding inside ``SerialTile(serial_outer)``).
    - ``consumer_body`` — stmts run by consumer warps (``AsyncWait`` +
      reduce loop + output ``Write``). Indices reference the **original**
      thread-axis names directly — no σ-shift. The materializer emits the
      consumer-relative ``threadIdx.x - n_producer_threads`` decode at
      the head of the consumer branch (see :class:`ThreadTile.tid_offset`).
    - ``ring_depth`` — empty-mbarrier slot count (== TMA buffer_count).
    - ``n_producer_threads`` — number of threads in the producer warp(s).
      Today only ``32`` (one producer warp) is emitted; ``SetMaxNReg``
      accounting and the ``Cond(role < n_producer_warps, …)`` predicate
      both derive from this.
    - ``consumer_thread_axes`` — axes describing the consumer-side
      per-thread coord structure (the original ``ThreadTile.axes`` the
      input kernel carried). The materializer feeds these into a nested
      ``ThreadTile(tid_offset=n_producer_threads, …)`` so consumer
      threads see ``threadIdx.x - n_producer_threads`` decoded back into
      these axis names. ``()`` is the legacy / pre-refactor shape, kept
      for back-compat with any caller that doesn't track the axes yet —
      the new materializer arm raises if it's empty when expected.

    The K_o axis the WS pass aligned scheduling against is identified by
    the materializer structurally — the (single) ``SerialTile(serial_outer)``
    in each branch. Carrying the axis *name* would break under
    ``normalize_body``'s canonical rename pass (which renames Axis but
    can't see a plain string field).

    Other quantities (``role`` predicate, ``n_consumer_threads``,
    slot/phase exprs, mbar name) are derivable from these fields plus
    the enclosing ``WarpTile.axes`` (the role axis) and are reconstructed
    at materialize time.

    Nested ``WarpSpecialize`` is rejected at construction.
    """

    producer_body: Body
    consumer_body: Body
    ring_depth: int
    n_producer_threads: int
    consumer_thread_axes: tuple[Axis, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.producer_body, Body):
            object.__setattr__(self, "producer_body", Body.coerce(self.producer_body))
        if not isinstance(self.consumer_body, Body):
            object.__setattr__(self, "consumer_body", Body.coerce(self.consumer_body))
        if self.ring_depth < 1:
            raise ValueError(f"WarpSpecialize: ring_depth must be >= 1, got {self.ring_depth}")
        if self.n_producer_threads < 1:
            raise ValueError(f"WarpSpecialize: n_producer_threads must be >= 1, got {self.n_producer_threads}")
        if not isinstance(self.consumer_thread_axes, tuple):
            object.__setattr__(self, "consumer_thread_axes", tuple(self.consumer_thread_axes))
        # Nesting check — a WS inside another WS would require the
        # materializer to track a stack of ws_consumer contexts; today
        # 085 guards via the WS knob on the parent TileOp, but assert at
        # the IR level so other rules can't sneak one in.
        for body in (self.producer_body, self.consumer_body):
            for s in body.iter():
                if isinstance(s, WarpSpecialize):
                    raise ValueError("WarpSpecialize cannot nest")

    def nested(self) -> tuple[Body, ...]:
        return (self.producer_body, self.consumer_body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if len(bodies) != 2:
            raise ValueError(f"WarpSpecialize.with_bodies: expected 2 bodies, got {len(bodies)}")
        producer_body, consumer_body = bodies
        return replace(self, producer_body=producer_body, consumer_body=consumer_body)

    def deps(self) -> tuple[str, ...]:
        return ()

    def exprs(self) -> tuple[Expr, ...]:
        return ()

    def pretty(self, indent: str = "") -> list[str]:
        head = f"{indent}warp_specialize(ring={self.ring_depth}, n_prod={self.n_producer_threads}):"
        producer_lines = [f"{indent}{INDENT}producer:", *pretty_body(self.producer_body, indent + INDENT * 2)]
        consumer_lines = [f"{indent}{INDENT}consumer:", *pretty_body(self.consumer_body, indent + INDENT * 2)]
        return [head, *producer_lines, *consumer_lines]


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

    ``block`` is a per-cache-dim structural multiplier. Default ``()``
    means "all-1s, every cache var contributes coef-1 to its source dim"
    — byte-identical to pre-M2 semantics. A non-trivial ``block`` (set
    by ``020_stage_inputs._classify`` when the σ literal coefficient on
    a cache var is > 1, e.g. the MMA ``atom_M`` / ``atom_K`` factor)
    grows the slab and producer iteration range by ``block[i]`` per
    cache dim while keeping the consumer cache vars at their original
    extent. ``affine_decode_per_dim`` (M4) folds ``block[j]`` into the
    composite stride so the per-source-dim index reconstruction matches
    the planner's σ output.
    """

    dims: tuple[int, ...]
    block: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.block:
            return
        if len(self.block) != len(self.dims):
            raise ValueError(f"AffineAddressing.block length {len(self.block)} != dims length {len(self.dims)}")
        for i, b in enumerate(self.block):
            if not isinstance(b, int) or b < 1:
                raise ValueError(f"AffineAddressing.block[{i}] must be int >= 1, got {b!r}")

    def source_index(
        self,
        cache_axes: tuple[Axis, ...],
        coord_for: dict[str, Expr],
        origin: tuple[Expr, ...],
    ) -> tuple[Expr, ...]:
        """Build the per-source-dim index expression ``origin[d] + decoded[d]``
        for the affine reconstruction.

        Thin wrapper around :func:`affine_decode_per_dim` that threads
        ``self.block`` through. Returns a tuple of length ``len(origin)``
        with one Expr per source dim. Source dims not swept by any cache
        axis carry only the origin term. This is the single source of
        truth for ``_stage_expand`` (cooperative producer),
        ``025_unify_sibling_stages._reconstruct_global_index``
        (revert-to-gmem), and M5's ``005_lower_atom_tile`` (MMA fragment
        load): each calls it with the appropriate ``coord_for`` mapping.
        """
        decoded = affine_decode_per_dim(cache_axes, self.dims, coord_for, block=self.block)
        return tuple((origin_d + decoded[d]) if d in decoded else origin_d for d, origin_d in enumerate(origin))


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
    back-pointer, set by 020_stage_inputs at construction time, identifies
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
      ``buf[addressing.exprs[d]]`` (template).
    - ``pad`` — per-cache-axis bank-conflict-breaking pad. Empty = no
      pad. Padding affects smem allocation, not the cooperative-load
      iteration extent.
    - ``addressing`` — stored ``AffineAddressing | TemplateAddressing``.
      Affine when every cache axis appears coef-1 (``block=()``) or
      coef-block (e.g. atom-strided MMA) in exactly one source dim;
      template when the consumer's original Load was a collapsed-reshape
      and ``origin + decoded`` can't reconstruct it. Defaults to
      ``AffineAddressing(dims=tuple(cd.source_dim for cd in cache_dims))``
      when omitted, so legacy construction sites that didn't specify
      addressing keep their semantics. Pre-M2 this was a derived
      property of ``cache_dims`` + ``template_index``; the refactor
      collapses both addressing-mode payloads into the addressing object
      so ``Source`` stays focused on slab identity / gmem anchor.
    - ``dtype`` — source buffer's element dtype. Stamped by
      ``030_stamp_types`` from ``graph.nodes[buf].output.dtype`` so smem
      allocation (``smem_bytes`` / ``alloc_extents``) and downstream
      materialization can read it off the IR without reaching for the
      matcher-populated graph node. ``None`` keeps legacy fp32-assuming
      behavior for tests that construct Source by hand.
    - ``gmem_extents`` — the gmem buffer's per-source-dim static shape,
      stamped by ``021_hoist_staged_loads_above_mask`` ONLY when this
      Source's cooperative load is hoisted above a masked-tile boundary
      ``Cond``. A masked output axis tiles past the real extent (e.g. N=256
      tiled at 192 → the second tile spans [192, 384)), so the cooperative
      gmem read overruns the buffer for the overhang columns. When set,
      ``_stage_expand.emit_stage`` clamps each ``source_index`` dim to
      ``[0, gmem_extents[d])`` so the producer never reads OOB (the
      overhang slab slots get a clamped duplicate value, harmless because
      the masked output cells they feed are never written). ``None`` (the
      default, clean-divisor tiles) skips the clamp — no perf cost on the
      common path.
    """

    name: str
    buf: str
    cache_dims: tuple[CacheDim, ...]
    origin: tuple[Expr, ...]
    pad: tuple[int, ...] = ()
    addressing: AffineAddressing | TemplateAddressing | None = None
    dtype: DataType | None = None
    gmem_extents: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # Default addressing: affine with dims pulled off cache_dims. The
        # field is typed Optional only so the default sentinel can be
        # ``None`` (a tuple-of-int default would require a frozen factory
        # for the dims, which is awkward). Frozen-set via object.__setattr__.
        if self.addressing is None:
            object.__setattr__(
                self,
                "addressing",
                AffineAddressing(dims=tuple(cd.source_dim for cd in self.cache_dims)),
            )

    @property
    def cache_axes(self) -> tuple[Axis, ...]:
        return tuple(cd.axis for cd in self.cache_dims)

    @property
    def alloc_extents(self) -> tuple[int, ...]:
        """Per-cache-axis smem allocation extent: cache extent × block + pad.

        For affine addressing with empty ``block``, returns the bare
        cache extents (pre-M2 behavior). For affine with non-trivial
        ``block`` (M3+ stamps it on atom-strided σ), each extent is
        multiplied by ``block[i]`` so the slab holds the full per-cell
        micro-tile. Pad is added last so a future MMA-friendly swizzle
        could request padded slabs without re-deriving the block math.
        """
        extents = tuple(ax.extent.as_static() for ax in self.cache_axes)
        block: tuple[int, ...] = ()
        if isinstance(self.addressing, AffineAddressing):
            block = self.addressing.block
        if block:
            extents = tuple(e * b for e, b in zip(extents, block, strict=True))
        if not self.pad:
            return extents
        return tuple(e + p for e, p in zip(extents, self.pad, strict=True))

    @property
    def smem_bytes(self) -> int:
        """Bytes of dynamic shared memory this Source allocates (single-slot).

        Uses ``self.dtype.nbytes`` when ``030_stamp_types`` has populated it;
        falls back to the legacy fp32-assuming ``BYTES_PER_ELEM`` constant
        otherwise so handwritten test fixtures without dtype continue to work.
        """
        n = self.dtype.nbytes if self.dtype is not None else BYTES_PER_ELEM
        for e in self.alloc_extents:
            n *= e
        return n

    def with_pad(self, pad: tuple[int, ...]) -> Source:
        return replace(self, pad=pad)


def affine_decode_per_dim(
    cache_axes: tuple[Axis, ...],
    dims: tuple[int, ...],
    coord_for: dict[str, Expr],
    block: tuple[int, ...] = (),
) -> dict[int, Expr]:
    """Reconstruct the per-source-dim coord contribution from a set of
    cache axes that map to those source dims.

    For each source dim ``d``, the axes mapping to ``d`` form a composite
    in most-significant-first order: ``ax_0·(e_1·b_1·e_2·b_2·…·b_0) + … + ax_{k-1}·b_{k-1}``
    where ``e_i`` is ``cache_axes[i].extent`` and ``b_i`` is ``block[i]``
    (defaulting to 1 when ``block=()``). Each axis's coord (an Expr from
    ``coord_for[ax.name]``) is scaled by the product of ``e_j · b_j`` for
    the subsequent cache axes that ALSO map to dim ``d``, times its own
    ``b_i``, then summed per dim.

    Single-axis-per-dim with ``block=()`` collapses to a no-op
    (``stride = 1``, coord added verbatim). Multi-axis-per-dim (matmul
    N-side ``BN_thread × FN_register`` collapse) gets the composite
    stride that mirrors the original ``load.index[d]`` shape.
    Non-trivial ``block`` (e.g. WMMA ``(1, atom_M, 1, atom_K)``) folds
    each axis's atom multiplier into its own stride — the slab is sized
    ``extent · block`` per axis, and the per-axis decode reads from
    ``cache_var · block · stride_of_inner_axes`` so the σ output of an
    atom-strided gmem Load round-trips through smem-stage and back.

    The previous shape — ``dict(zip(dims, coord_for))`` — silently
    OVERWROTE the entry when two cache axes shared a dim, keeping only
    the last axis's coord and producing wrong gmem addresses on every
    consumer that reconstructed source indices (``_stage_expand``,
    ``025_unify_sibling_stages._reconstruct_global_index``,
    ``_source_decl_line``). Centralising the math here keeps the
    composite-stride formula consistent across all three sites.
    """
    out: dict[int, Expr] = {}
    use_block = bool(block)
    for i, (ax, d) in enumerate(zip(cache_axes, dims, strict=True)):
        stride = 1
        for j in range(i + 1, len(cache_axes)):
            if dims[j] == d:
                inner_factor = block[j] if use_block else 1
                stride *= cache_axes[j].extent.as_static() * inner_factor
        if use_block:
            stride *= block[i]
        term: Expr = coord_for[ax.name] if stride == 1 else BinaryExpr("*", coord_for[ax.name], Literal(stride, "int"))
        out[d] = term if d not in out else BinaryExpr("+", out[d], term)
    return out


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
        coord_for = {ax.name: cache_index[i] for i, ax in enumerate(axes)}
        src_index = addressing.source_index(axes, coord_for, origin)
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
    if isinstance(src.addressing, TemplateAddressing):
        tpl = " template=[" + ", ".join(e.pretty() for e in src.addressing.exprs) + "]"
    return f"{src.name}<-{src.buf}(origin=({origin}), slab=({cache})){pad}{tpl}"


def _source_decl_line(src: Source) -> str:
    """Render one ``Source`` as ``shared <name>[<cache_axes>] = <buf>[<source_index>];``.

    Cache axes show their extents (``a5:64, a3:16``). The source index
    prefers the literal ``TemplateAddressing.exprs`` when set (preserves
    explicit stride math like ``a3*16 + a6``); otherwise reconstructs
    from ``origin + decoded`` per affine addressing semantics.

    Trailing ``pad`` and stage-flavor suffixes are NOT appended here — the
    Stage subclasses prepend / postfix those at the call site.
    """
    cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in src.cache_axes)
    if isinstance(src.addressing, TemplateAddressing):
        idx = ", ".join(e.pretty() for e in src.addressing.exprs)
    else:
        # Composite-stride decode via ``AffineAddressing.source_index``:
        # for multi-axis-per-source-dim, the i-th axis carries the
        # product of subsequent same-dim ``extent · block`` as its
        # stride. Single-axis-per-dim with ``block=()`` collapses to
        # stride 1 = bare ``ax.name``.
        coord_for = {ax.name: Var(ax.name) for ax in src.cache_axes}
        full_index = src.addressing.source_index(src.cache_axes, coord_for, src.origin)
        idx = ", ".join(e.pretty() for e in full_index)
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
# threadIdx / warp-id / register / serial / strided). Together with the
# wrap-body ``Stage`` family, these are the only block-structured Stmts
# allowed inside a ``TileOp.body`` post-``001_launch_geometry``. ``Loop`` /
# ``StridedLoop`` / ``Tile`` survive in Loop IR (``LoopOp.body``) and as
# transient inputs to ``001_launch_geometry``, but downstream Tile-IR
# passes and Tile→Kernel materialization only see the new flavors.
#
# Shape contract (mirrors ``Stage``'s wrap-body):
#
# - ``ParallelTile`` subclasses (``GridTile`` / ``ThreadTile`` /
#   ``WarpTile`` / ``RegisterTile``) carry ``axes: tuple[Axis, ...]`` +
#   ``body: Body``. The body executes once per coord tuple; coords are
#   implicit from the binding (``blockIdx`` / ``threadIdx`` /
#   ``threadIdx.x / 32`` / per-thread register cell). ``ThreadTile`` and
#   ``WarpTile`` are mutually exclusive inside one ``TileOp.body``
#   (``TileOp.__post_init__`` rejects mixes) — both bind ``threadIdx``.
# - ``SerialTileBase`` subclasses (``SerialTile`` / ``StridedTile``)
#   carry ``axis: Axis`` + ``body: Body`` and run sequentially. Reduce
#   semantics are derived: ``is_reduce`` iff the body contains ``Accum``.


@dataclass(frozen=True)
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
            object.__setattr__(self, "body", Body(self.body))

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


@dataclass(frozen=True)
class GridTile(ParallelTile):
    """CTA-grid parallel tile. Axes lift to ``blockIdx`` (row-major).

    Replaces ``Tile`` with ``BIND_BLOCK`` axes. Split-K is derived at
    codegen time from ``escape_analysis.atomic_axes`` (Write index vs
    enclosing block axes) — no per-tile metadata required.

    ``swizzle_group_m`` selects an L2-friendly CTA-ID remap for matmul-shape
    grids (axes ending in ``(M_b, N_b)``): consecutive CTAs walk down M
    in groups of ``swizzle_group_m`` before stepping N, so a row-group of
    CTAs shares A's row tile in L2 (Triton/CUTLASS/cuBLAS convention).
    ``swizzle_group_m == 1`` (the default) keeps the row-major decode and
    is a structural no-op; the swizzled path is stamped by
    ``tile/025_swizzle_blocks.py`` on matmul-shape GridTiles. The field
    feeds ``_pretty_label`` so the structural digest tracks it.
    """

    swizzle_group_m: int = 1

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return GridTile(axes=self.axes, body=body, swizzle_group_m=self.swizzle_group_m)

    def _pretty_label(self) -> str:
        if self.swizzle_group_m == 1:
            return "grid"
        return f"grid swizzle_M={self.swizzle_group_m}"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Emit ``blockIdx.x`` axis decode + body. The inner ``ThreadTile``
        renders its threadIdx decode under ``ctx.inside_grid_tile=True``,
        so no per-CTA bounds guard is needed at this level."""
        if self.swizzle_group_m != 1:
            out = list(_render_swizzled_grid_decode(self.axes, "blockIdx.x", self.swizzle_group_m, ctx))
        else:
            out = list(_render_grid_axis_decode(self.axes, "blockIdx.x", ctx))
        inner_ctx = replace(ctx, inside_grid_tile=True)
        out.extend(_render_body(self.body, inner_ctx))
        return out


@dataclass(frozen=True)
class ThreadTile(ParallelTile):
    """Thread-parallel tile. Axes lift to ``threadIdx`` (row-major flatten).

    Replaces ``Tile`` with ``BIND_THREAD`` axes. Cooperative-K
    cooperativity is derived at materialize / render time from
    ``Accum.axes ∩ ThreadTile.axes`` — see
    ``ir/tile/escape_analysis.py``.

    ``tid_offset`` (default ``0``) shifts the linear thread index the
    cooperative-form decode is computed against — the per-axis decls
    use ``(threadIdx.x - tid_offset)`` instead of plain ``threadIdx.x``.
    Non-zero values are emitted by the warp-specialize materializer to
    drop a ``ThreadTile(consumer_thread_axes, tid_offset=n_producer_threads, …)``
    inside the consumer ``Cond.else_body``, so the original consumer-side
    thread axes decode against a consumer-relative tid in ``[0,
    n_consumer_threads)``. The field carries no semantic meaning outside
    that materializer-emitted nesting; planner-emitted ``ThreadTile``s
    keep the default ``0``.
    """

    tid_offset: int = 0

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return ThreadTile(axes=self.axes, body=body, tid_offset=self.tid_offset)

    def _pretty_label(self) -> str:
        if self.tid_offset:
            return f"thread offset={self.tid_offset}"
        return "thread"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Two render forms picked by ``ctx.inside_grid_tile``.

        - **Cooperative** (inside ``GridTile``): emit ``threadIdx.x``
          axis decode (optionally offset by ``tid_offset`` — used by the
          warp-specialize consumer arm) + optional ``lane`` / ``warp``
          helper decls + body. No extra brace level — the surrounding
          ``__global__`` provides one.
        - **Standalone** (pointwise — no enclosing ``GridTile``): flatten
          all axes into a linear ``tid``; bounds-guard against the product
          of extents.
        """
        pad = _pad(ctx.indent)
        if ctx.inside_grid_tile:
            idx_expr = "threadIdx.x" if self.tid_offset == 0 else f"(threadIdx.x - {self.tid_offset})"
            out = list(_render_grid_axis_decode(self.axes, idx_expr, ctx))
            if _body_uses_lane_warp(self.body):
                out.append(f"{pad}int lane = threadIdx.x & 31;")
                out.append(f"{pad}int warp = threadIdx.x >> 5;")
            out.extend(_render_body(self.body, ctx))
            return out

        if self.tid_offset:
            raise NotImplementedError("standalone ThreadTile with non-zero tid_offset not supported")
        inner = ctx.child()
        n_threads = 1
        for ax in self.axes:
            n_threads *= ax.extent.as_static()
        out = [
            f"{pad}long long tid = blockIdx.x * blockDim.x + threadIdx.x;",
            f"{pad}if (tid < {n_threads}) {{",
        ]
        out.extend(_render_thread_axis_decode(self.axes, inner))
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
class RegisterTile(ParallelTile):
    """Per-thread register-cell tile. Body replicated F× per axis by
    ``010_split_register_axes``.

    Replaces ``Loop(role=REGISTER)``. The ``axes`` tuple carries one or
    more register axes (typically M_r / N_r for matmul); the planner
    chooses the extents (``FM`` / ``FN`` knobs). After the
    ``010_split_register_axes`` pass runs, every ``RegisterTile`` is
    consumed: the body is fully unrolled, SSA names get per-cell
    suffixes, and the ``RegisterTile`` wrapper disappears.
    """

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            "RegisterTile must be consumed by 006a_register_tile_planned before render — "
            f"reached render with axes={tuple(ax.name for ax in self.axes)!r}"
        )


@dataclass(frozen=True)
class AtomTile(ParallelTile):
    """Hardware-atomic-cell tile — one coord = one MMA fragment cell.

    Marker for the per-cell hardware-atomic extent on a matmul-reduce kernel
    (see ``plans/mma-fragment-factorization.md``). The axes carry the cell
    shape (e.g. ``(M=16, N=16)`` for ``wmma_m16n16k16_f16``); the body inside
    is the per-cell compute the materializer will replace with a fragment
    instruction chain (``MmaFragment`` decls + ``MmaLoad`` + ``MmaSync``).

    The MMA cell materializer (``kernel/010_split_register_axes`` MMA arm)
    consumes ``AtomTile`` structurally — its presence is the "this kernel
    factorizes through tensor cores" signal, complementing the
    ``ATOM_KIND`` knob on the enclosing ``TileOp``. Scalar matmul kernels
    never emit an ``AtomTile`` (the absence of the tier is the absence of
    the atom — see :class:`ScalarTileParams` in
    ``passes/lowering/tile/_enumeration``).

    Render is intentionally unimplemented: every ``AtomTile`` must be
    consumed before kernel render, mirroring ``RegisterTile``'s contract.
    """

    def render(self, ctx: RenderCtx) -> list[str]:
        raise NotImplementedError(
            "AtomTile must be consumed by the MMA materializer "
            "(kernel/010_split_register_axes MMA arm) before render — "
            f"reached render with axes={tuple(ax.name for ax in self.axes)!r}"
        )


@dataclass(frozen=True)
class WarpTile(ParallelTile):
    """Warp-parallel tile — one coord tuple = one warp (32 lanes).

    The body executes once per warp coord; the 32 lanes inside the warp
    execute it collectively (cooperative MMA, lane-aware shuffles, etc.).
    Materialization rules / consumers — MMA fragment factorization
    (``plans/mma-fragment-factorization.md``), warp-specialized TMA
    pipelining refactor — emit ``WarpTile`` *inside* an outer
    ``GridTile`` to bind warps to a CTA. The cooperative form is the
    only one supported in v1; a standalone top-level ``WarpTile``
    (pointwise-style "one warp per output cell") has no consumer today.

    Rendering binds ``warp_id = threadIdx.x / 32`` (the row-major decode
    over the warp axes uses ``warp_id`` as the linear index), and
    unconditionally exposes ``lane = threadIdx.x & 31`` — the body
    presumes a lane is available (that's the entire reason a warp coord
    exists). Launch-bounds wiring (``_launch_bounds_for`` /
    ``_launch_geometry``) multiplies the warp-axis product by 32.

    Mutual exclusion with ``ThreadTile`` inside one ``TileOp.body`` is
    enforced by ``TileOp.__post_init__`` — both bind ``threadIdx`` and
    mixing would re-bind the same coord at two scopes.
    """

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        (body,) = bodies
        return WarpTile(axes=self.axes, body=body)

    def _pretty_label(self) -> str:
        return "warp"

    def render(self, ctx: RenderCtx) -> list[str]:
        """Cooperative form (inside ``GridTile``): ``warp_id`` decl + row-
        major warp-axis decode against ``warp_id`` + unconditional ``lane``
        decl + body. Standalone (no enclosing ``GridTile``) is not
        supported in v1 — pointwise kernels use ``ThreadTile`` and a
        top-level ``WarpTile`` has no consumer yet.
        """
        if not ctx.inside_grid_tile:
            raise NotImplementedError("WarpTile outside GridTile not supported in v1")
        pad = _pad(ctx.indent)
        out: list[str] = [f"{pad}int warp_id = threadIdx.x / 32;"]
        out.extend(_render_grid_axis_decode(self.axes, "warp_id", ctx))
        out.append(f"{pad}int lane = threadIdx.x & 31;")
        out.extend(_render_body(self.body, ctx))
        return out


@dataclass(frozen=True)
class SerialTileBase(Stmt):
    """Abstract base for serial-iteration tile flavors. One axis, one body."""

    axis: Axis
    body: Body

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body(self.body))

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


@dataclass(frozen=True)
class SerialTile(SerialTileBase):
    """Sequential iteration over ``axis``. Replaces ``Loop``.

    ``kind`` carries the planner's structural intent:

    - ``"plain"``: ordinary serial loop (no special role).
    - ``"serial_outer"``: outer chunked-K loop driving slab refresh
      (today's ``Role.SERIAL_OUTER``). Targeted by ``040_use_ring_buffers``
      / ``015_pipeline_k_outer``.
    - ``"stage_inner"``: inner reduce loop inside a ``Stage``'s wrapped
      body (today's ``Role.STAGE_INNER``). Slab-axis marker for
      ``020_stage_inputs``.
    - ``"pipeline"``: serial outer loop marked for temporal pipelining
      by ``015_pipeline_k_outer``.

    ``unroll=True`` annotates the loop for ``#pragma unroll`` at render
    time. Set by ``090_mark_unroll``; has no effect on iteration semantics.
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
        # ``Dim.__str__`` returns the bare value (literal for static, symbolic
        # name for ``Dim('seq_len')``) so both static and symbolic SerialTile
        # extents render correctly.
        extent = str(self.axis.extent)
        if self.unroll:
            out.append(f"{pad}#pragma unroll")
        out.append(f"{pad}for (int {var} = 0; {var} < {extent}; {var}++) {{")
        inner = ctx.child()
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


@dataclass(frozen=True)
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
        out.append(f"{pad}for (int {var} = {start_str}; {var} < {self.axis.extent.as_static()}; {var} += {step_str}) {{")
        inner = ctx.child()
        out.extend(_render_body(self.body, inner))
        out.append(f"{pad}}}")
        return out


# ---------------------------------------------------------------------------
# Stage + StageBundle — single-policy bundles of cooperative stages
# ---------------------------------------------------------------------------
#
# Replaces the wrap-body Stage hierarchy. There is now ONE ``Stage``
# class carrying ``sources`` plus an optional ``compute`` template body
# (the producer-side cooperative compute, previously specific to
# ``ComputeStage``). Staging policy — sync transport, ring-buffered,
# cp.async, TMA — lives on the enclosing ``StageBundle`` instead of
# per-Stage subclasses. A bundle is single-policy: all member stages
# share the same transport. K_o-dependency partitioning produces
# separate bundles with different policies rather than mixed bundles.
#
# Mapping from old hierarchy:
#   Stage(sources, body)                                 → StageBundle(stages=(Stage(sources),), body, SYNC)
#   BufferedStage(sources, body, buffer_count, phase)    → StageBundle(stages=(Stage(sources),), body, BUFFERED, buffer_count, phase)
#   AsyncBufferedStage(sources, body, ..., depth)        → StageBundle(stages=(Stage(sources),), body, ASYNC, ..., pipeline_depth)
#   TmaBufferedStage(sources, body, ..., depth, swizzle) → StageBundle(stages=(Stage(sources),), body, TMA, ..., pipeline_depth, swizzle)
#   ComputeStage(sources, body, compute, ...)            → StageBundle(stages=(Stage(sources, compute=...),), body, BUFFERED|SYNC, ...)
#
# Stage retains NO body — the bundle owns the consumer scope.
# Stage's ``compute`` field is non-None iff it's a hoisted-invariant
# compute stage (sibling-smem → own-smem producer template); None for
# plain transport stages.


class StagePolicy(enum.Enum):
    """Transport policy for a ``StageBundle`` — applied uniformly to
    every member ``Stage``.

    - ``SYNC``      — cooperative ``Load + Write`` + ``__syncthreads``
                      barrier. No ring-buffering.
    - ``BUFFERED``  — sync transport into ``buffer_count >= 2`` rotating
                      slabs selected by ``phase``. Drops the leading
                      pre-load sync since consecutive iterations write
                      to different physical slabs.
    - ``ASYNC``     — ring-buffered ``cp.async``; requires sm_80+.
                      Implicit wait at wrap boundary when
                      ``pipeline_depth == 1``; explicit
                      ``AsyncWait`` peeling when ``> 1``.
    - ``TMA``       — ring-buffered ``cp.async.bulk.tensor`` (TMA box
                      copy); requires sm_90+. Mbarrier-based completion.
    """

    SYNC = "sync"
    BUFFERED = "buffered"
    ASYNC = "async"
    TMA = "tma"


class SwizzleMode(enum.Enum):
    """TMA shared-memory swizzle pattern.

    Picked by the lowering pass from inner-dim byte stride; consumed by
    the backend's ``cuTensorMapEncodeTiled`` call. Only meaningful when
    ``StageBundle.policy == StagePolicy.TMA``.
    """

    NONE = "NONE"
    B32 = "B32"
    B64 = "B64"
    B128 = "B128"


@dataclass(frozen=True)
class Stage(Stmt):
    """Cooperative-staging unit: one or more ``Source`` entries plus an
    optional cooperative ``compute`` template.

    A bare transport stage has ``compute=None``; the producer-side IR
    is synthesized at materialize time from the sources (cooperative
    ``Load + Write`` per source).

    A *hoisted-invariant compute* stage has ``compute != None`` — a
    per-thread cooperative body that reads from sibling-stage smem (via
    Sources whose ``buf`` names a sibling Stage's smem buffer) and
    writes into this Stage's smem allocation. ``external_reads()``
    returns empty for compute stages (sibling smem isn't external).

    The consumer scope is owned by the enclosing :class:`StageBundle`;
    Stage itself carries no consumer ``body`` field. Transport policy
    (sync / buffered / async / TMA) and policy-specific knobs
    (``buffer_count`` / ``phase`` / ``pipeline_depth`` / ``swizzle``)
    live on the bundle.

    A Stage that lives outside a bundle is invalid; constructors that
    accept Stages always wrap them in a StageBundle.
    """

    sources: tuple[Source, ...]
    compute: Body | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not self.sources:
            raise ValueError("Stage: requires at least one Source")
        if self.compute is not None and not isinstance(self.compute, Body):
            object.__setattr__(self, "compute", Body.coerce(self.compute))

    def nested(self) -> tuple[Body, ...]:
        return (self.compute,) if self.compute is not None else ()

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if self.compute is None:
            if bodies:
                raise ValueError("Stage(compute=None).with_bodies: expected 0 bodies")
            return self
        (compute,) = bodies
        return replace(self, compute=compute)

    def deps(self) -> tuple[str, ...]:
        return ()

    def external_reads(self) -> tuple[str, ...]:
        # Compute stages read from sibling-stage smem (same Tile scope),
        # not external gmem — return empty so the staging-eligibility
        # passes don't treat them as gmem-transport candidates.
        if self.compute is not None:
            return ()
        return tuple(s.buf for s in self.sources)

    def local_decls(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.sources)

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        for s in self.sources:
            out = (*out, *s.origin)
            if isinstance(s.addressing, TemplateAddressing):
                out = (*out, *s.addressing.exprs)
        return out

    @property
    def smem_bytes(self) -> int:
        return sum(s.smem_bytes for s in self.sources)

    def replace_sources(self, sources: tuple[Source, ...]) -> Stage:
        return replace(self, sources=sources)

    def pretty(self, indent: str = "") -> list[str]:
        """Render per-source ``shared name[...] = buf[...];`` decl lines.
        When ``compute`` is present, follow with a synthesized
        ``cooperative`` for-nest over the first source's cache axes
        wrapping the producer body.
        """
        out: list[str] = []
        if self.compute is not None:
            for s in self.sources:
                cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in s.cache_axes)
                out.append(f"{indent}shared {s.name}[{cache}]")
            cache_axes = self.sources[0].cache_axes
            for_lines = [f"for {ax.name} in 0..{ax.extent}" for ax in cache_axes]
            out.extend(_render_tile_bracket(indent + INDENT, for_lines, "cooperative", self.compute))
        else:
            out = [f"{indent}{_source_decl_line(s)}" for s in self.sources]
        return out


@dataclass(frozen=True)
class StageBundle(Stmt):
    """Single-policy group of cooperative ``Stage`` members sharing one
    consumer ``body``.

    The bundle owns the consumer scope (``body``) and the staging
    policy (``policy`` plus policy-specific fields ``buffer_count`` /
    ``phase`` / ``pipeline_depth`` / ``swizzle``). Each member Stage
    contributes its ``sources`` (and optional ``compute`` template).

    Invariants:

    - ``stages`` is non-empty.
    - All members share the bundle's policy; mixed-policy bundles are
      illegal. K_o-dep partition (010_use_ring_buffers) emits separate
      bundles with different policies rather than mixing.
    - Member ORDER == issue order. Passes looking for a specific stage
      MUST locate it by name/source, never by position.
    - ``buffer_count >= 2`` requires ``phase``; allowed only for
      ``BUFFERED`` / ``ASYNC`` / ``TMA`` policies. ``SYNC`` has
      ``buffer_count == 1``, ``phase is None``.
    - ``pipeline_depth > 1`` allowed only for ``ASYNC`` / ``TMA``.
    - ``swizzle != NONE`` allowed only for ``TMA``.
    - ``TMA`` policy: every Source on every member must have empty
      ``pad`` (TMA writes contiguous rows; bank-pad would misalign).

    Iteration / recursion:

    - ``nested()`` returns ``(Body(self.stages), self.body)`` — the
      stages tuple is exposed as a synthetic Body so generic walkers
      (``Body.iter`` / ``Body.map`` / ``Body.fold``) traverse into
      individual members without special-casing StageBundle. Each
      member's own ``nested()`` (the optional ``compute`` template) is
      then reached via the standard descent.
    """

    stages: tuple[Stage, ...]
    body: Body
    policy: StagePolicy = StagePolicy.SYNC
    buffer_count: int = 1
    phase: Expr | None = None
    pipeline_depth: int = 1
    swizzle: SwizzleMode = SwizzleMode.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.body, Body):
            object.__setattr__(self, "body", Body.coerce(self.body))
        if not self.stages:
            raise ValueError("StageBundle: requires at least one Stage")
        for s in self.stages:
            if not isinstance(s, Stage):
                raise TypeError(f"StageBundle: member must be a Stage, got {type(s).__name__}")
        if self.policy == StagePolicy.SYNC:
            if self.buffer_count != 1:
                raise ValueError(f"StageBundle SYNC: buffer_count must be 1, got {self.buffer_count}")
            if self.phase is not None:
                raise ValueError("StageBundle SYNC: phase must be None")
            if self.pipeline_depth != 1:
                raise ValueError(f"StageBundle SYNC: pipeline_depth must be 1, got {self.pipeline_depth}")
        else:
            if self.buffer_count < 1:
                raise ValueError(f"StageBundle: buffer_count must be >= 1, got {self.buffer_count}")
            if self.buffer_count >= 2 and self.phase is None:
                raise ValueError(f"StageBundle {self.policy.value}: phase required when buffer_count >= 2")
        if self.pipeline_depth != 1 and self.policy not in (StagePolicy.ASYNC, StagePolicy.TMA):
            raise ValueError(f"StageBundle: pipeline_depth > 1 requires ASYNC or TMA policy, got {self.policy.value}")
        if self.swizzle != SwizzleMode.NONE and self.policy != StagePolicy.TMA:
            raise ValueError(f"StageBundle: non-NONE swizzle requires TMA policy, got {self.policy.value}")
        if self.policy == StagePolicy.TMA:
            for stage in self.stages:
                for src in stage.sources:
                    if src.pad and any(src.pad):
                        raise ValueError(f"StageBundle TMA: source {src.name!r} pad must be empty, got {src.pad!r}")

    def nested(self) -> tuple[Body, ...]:
        return (Body(self.stages), self.body)

    def with_bodies(self, bodies: tuple[Body, ...]) -> Stmt:
        if len(bodies) != 2:
            raise ValueError(f"StageBundle.with_bodies: expected 2 bodies (stages, body), got {len(bodies)}")
        stages_body, body = bodies
        return replace(self, stages=tuple(stages_body), body=body)

    def deps(self) -> tuple[str, ...]:
        return ()

    def external_reads(self) -> tuple[str, ...]:
        out: tuple[str, ...] = ()
        for s in self.stages:
            out = (*out, *s.external_reads())
        return out

    def local_decls(self) -> tuple[str, ...]:
        out: tuple[str, ...] = ()
        for s in self.stages:
            out = (*out, *s.local_decls())
        return out

    def exprs(self) -> tuple[Expr, ...]:
        out: tuple[Expr, ...] = ()
        for s in self.stages:
            out = (*out, *s.exprs())
        if self.phase is not None:
            out = (*out, self.phase)
        return out

    @property
    def smem_bytes(self) -> int:
        per_slab = sum(s.smem_bytes for s in self.stages)
        return per_slab * max(self.buffer_count, 1)

    def _policy_label(self) -> str:
        """Render a compact ``policy[buffer_count@phase depth=N swizzle=X]``
        label used in the bundle header line."""
        if self.policy == StagePolicy.SYNC:
            return "sync"
        parts: list[str] = [f"{self.policy.value}[{self.buffer_count}"]
        if self.phase is not None:
            parts.append(f"@{self.phase.pretty()}")
        if self.pipeline_depth > 1:
            parts.append(f" depth={self.pipeline_depth}")
        if self.swizzle != SwizzleMode.NONE:
            parts.append(f" swizzle={self.swizzle.value}")
        parts.append("]")
        return "".join(parts)

    def pretty(self, indent: str = "") -> list[str]:
        """Render as ``bundle <policy>:`` header with member stages and
        the consumer body indented beneath."""
        inner = indent + INDENT
        out: list[str] = [f"{indent}bundle {self._policy_label()}:"]
        for s in self.stages:
            out.extend(s.pretty(inner))
        out.extend(pretty_body(self.body, inner))
        return out


# ---------------------------------------------------------------------------
# Top-level: TileOp
# ---------------------------------------------------------------------------


def score_tile_geometry(
    *,
    thread_extents: list[int],
    block_extents: list[int],
    knobs: dict,
    has_stages: bool,
    coalescing_inner_extent: int,
    smem_cap_bytes: int | None = None,
    n_staged_inputs: int = 2,
    compute_capability: tuple[int, int] | None = None,
) -> float:
    """Pure-formula scoring shared by :meth:`TileOp.score` and the partition
    planner. Takes the launch-geometry summary the score depends on (already
    resolved to static ints, symbolic axes substituted as 1) plus the knob
    bundle, so the planner can score un-materialized ``TileParams`` against
    the same yardstick a fully-built ``TileOp`` would use.

    ``coalescing_inner_extent`` is the summed extent of thread axes that
    appear in any Write's inner-stride dim (0 when none align).

    Cell budget knee is at 128 (the NVRTC per-thread cell cap and a sane
    upper bound for fp32 register-tile matmul where 255 regs/thread fits
    ~100 accumulators + working state). The earlier knee at 64 mis-ranked
    matmul shapes whose optimal tile sits in the register-bound 80–110
    cells/thread regime — at 2048³ the article's golden tile (``BM=8 BN=32
    FM=26 FN=4`` — 104 cells, 275 us) was 30× faster than the 32-cell
    greedy default the old prior picked.

    ``smem_cap_bytes`` is accepted for forward compatibility with planners
    that want to plumb the live device's smem budget for tile-fit checks,
    but isn't read by the formula today; callers may pass ``None``."""
    from math import prod  # noqa: PLC0415

    # Warp-tier MMA on TMA-capable HW (sm_90+) hits its perf sweet spot at
    # 4 warps × 128 threads with the mma chain saturating the tensor cores
    # while TMA off-warps the gmem loads. The pre-2026 ``target_threads =
    # 256`` was tuned for scalar matmul + gmem-direct WMMA where larger
    # warp counts amortized cooperative-load instructions; with TMA those
    # loads cost one mbarrier handshake from one issuer thread, so 256
    # threads buys nothing and competes for tensor-core throughput.
    # Empirical sweep at 2048² fp16 on sm_120 (RTX 5090): top 5 TMA-
    # firing variants all sit at threads=128; the 256-thread variants
    # land at 106+ µs vs 84 µs for the 128-thread golden tile.
    tma_capable_warp = "ATOM_KIND" in knobs and isinstance(compute_capability, tuple) and compute_capability >= (9, 0)
    target_threads = 128 if tma_capable_warp else 256
    target_ctas = 128
    score = 0.0
    if not thread_extents:
        return 0.0
    threads = prod(thread_extents)
    ctas = prod(block_extents) if block_extents else 1

    if "FM" in knobs:
        final_threads = threads
        cells = max(1, int(knobs.get("FM", 1)) * int(knobs.get("FN", 1)))
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
        multiplier = 2.0 if (cells < 16 and "FM" in knobs) else 1.0
        score -= min(distance / target_threads * multiplier, 2.0)

    if cells == 1:
        score -= 1.0
    elif "ATOM_KIND" in knobs:
        # Tensor-core MMA (warp-tier WMMA path): the per-lane register
        # budget is dominated by accumulator fragments (one ~4-reg/lane
        # frag per FM·FN cell, half acc on consumer / fp32 acc on
        # datacenter) plus the operand frags loaded per K-iter. On
        # sm_8x / sm_9x / sm_12x with the 255-reg/lane cap, ``cells ≈
        # 16`` is the 2-blocks/SM occupancy knee; past ~24 the per-lane
        # reg footprint spills to local memory (measured at 173 regs/
        # thread on a 2048² fp16 sweep — see ``plans/mma-warp-scoring.md``
        # and the warp-tier section of ``compiler/pipeline/ARCHITECTURE.md``).
        # The pre-2026 cells-up-to-128 reward (tuned for scalar f32
        # matmul where 104-cell golden tiles win) pushed the MMA picker
        # to FM=1 FN=32, 3.0× slower than cuBLAS instead of 0.91×.
        sweet = 16
        score += 0.5 - min(abs(cells - sweet) / sweet, 0.5)
        # Square per-warp tile reuses operands equally — each A frag
        # used ``FN`` times per K-iter, each B frag used ``FM`` times.
        # Skewed shapes (FM=1, FN=N) waste one operand's reuse and
        # inflate the per-K-iter fragment-load count linearly.
        fm = max(1, int(knobs.get("FM", 1)))
        fn = max(1, int(knobs.get("FN", 1)))
        aspect = abs(fm - fn) / max(fm, fn)  # 0 = square, 1 = fully skewed
        score -= aspect * 0.25
    elif cells <= 128:
        # Reward register-tile budget gradually up to 128 cells (NVRTC cap +
        # fp32 register-tile sweet spot). Without this gradient every
        # cells-in-[16,128] variant tied at the same MCTS score and the
        # search burned patience on equally-scored siblings instead of
        # descending into the register-bound regime where matmul wins live.
        # Capped at +0.5 — small enough that ctas/coalescing/threads stay
        # the dominant signals on tile-shape choice, but big enough to break
        # ties toward the cells=104 golden region (vs the cells=32 default
        # the original priors picked at 2048³, 30× slower).
        score += (cells / 128.0) * 0.5
    else:
        # Past the NVRTC + fp32-register-tile knee, register pressure causes
        # spill or compile slowdown; penalize cap-0.5.
        score -= min((cells - 128) / 128.0, 0.5)

    if ctas < target_ctas:
        score -= (target_ctas - ctas) / target_ctas
    elif ctas <= 2048:
        score += 0.5
    else:
        score -= min((ctas - 2048) / 4096.0, 2.5)

    splitk = int(knobs.get("SPLITK", 1))
    if splitk > 1:
        # Cross-CTA reduce via atomic-add costs gmem RMW (the partials need to
        # round-trip through L2 between CTAs); pure-matmul kernels almost
        # always prefer SPLITK=1 unless they're starved for CTAs. The earlier
        # gate at SPLITK > 4 missed this — at 2048³ a (FM=20 FN=6 SPLITK=2)
        # variant outscored the golden's SPLITK=1 by 0.5 (the extra CTAs lifted
        # it into the "ctas ≥ target" bonus band), so greedy picked the
        # atomic-cost variant and ran ~3.5× slower.
        score -= 0.4 * min(splitk - 1, 4)

    if has_stages:
        score += 1.0
    if "FM" in knobs:
        score += 1.0

    if coalescing_inner_extent > 0:
        score += min(coalescing_inner_extent / 32, 1.0)

    # Smem-fit signal (matmul only — "FM" in knobs gates the matmul path).
    # Estimate the per-iter SYNC slab size from the planner's knob shape and
    # penalize macros that won't fit a double-buffered ring (BUFFER_COUNT=2).
    # Without that ring, the K-loop transport (cp.async, TMA) won't promote
    # and the kernel runs SYNC + no overlap — typically 5-20× slower than
    # the double-buffered variant. At 2048³ on sm_120 (99 KB cap) the prior
    # greedy picked a 256×32 stripe whose 96 KB SYNC slab grew to 192 KB at
    # BUFFER_COUNT=2 (skipped), running at 8.1 ms; the BUFFER_COUNT-fit-aware greedy
    # picks a 128×256 / 256×128 macro that double-buffers and runs ~360 us.
    # ``BYTES_PER_ELEM=4`` matches the slab bookkeeping the tile-IR uses
    # today (fp32 assumption; fp16 over-counts by 2× but the signal is still
    # in the right direction).
    if "FM" in knobs and smem_cap_bytes is not None:
        macro_m = int(knobs.get("BM", 1)) * int(knobs.get("FM", 1))
        macro_n = int(knobs.get("BN", 1)) * int(knobs.get("FN", 1))
        bk = int(knobs.get("BK", 1))
        if bk > 0 and macro_m > 0 and macro_n > 0:
            # Bare ``(macro_m + macro_n) * BK * 4 B`` is the per-slot slab
            # for an A+B matmul. Three things bloat the actual materialized
            # smem past this estimate:
            #   1. ``PAD_SMEM`` row padding to dodge bank conflicts (~30 %
            #      at BK=16 with wide N);
            #   2. Multi-bundle ring — prologue StageBundle + K_o StageBundle
            #      share storage but the materializer over-allocates the
            #      worst-case slot;
            #   3. Extra-input fusion — SDPA-with-RoPE / SDPA-with-mask /
            #      gated MLP kernels stage 3-4 source tensors instead of 2,
            #      so the slab scales with ``n_staged_inputs``.
            # Empirically the validated ``smem_bytes()`` is ~1.7× the naive
            # doubled-slab at 2 inputs, and ~3× at 4 inputs. Scale the
            # estimate accordingly. The penalty has to outweigh the +1.0
            # coalesce + +0.5 ctas bonus a wide-N stripe earns; otherwise the
            # planner picks that stripe and the materializer's validate trips.
            # See ``test_run_code_sdpa_tinyllama_per_head`` (2-input QK^T
            # at 105 KB) and ``test_full_self_attn_tinyllama_seq512`` (4-input
            # masked QK^T at 166 KB) for the two regimes.
            base_slab = (macro_m + macro_n) * bk * BYTES_PER_ELEM
            # ``base_slab`` already counts two staged buffers (A's
            # ``macro_m × BK`` + B's ``BK × macro_n``); each extra staged
            # source (gated MLP's three matmul operands, SDPA-with-RoPE,
            # …) typically lands at ~the larger of A / B in shape, so an
            # extra 0.75× per buffer beyond the second matches the actual
            # ``smem_bytes()`` within 5 % on the tinyllama gated-MLP fused
            # ``mul_8`` (102 KB actual vs 105 KB predicted at n=3).
            extra = max(0, n_staged_inputs - 2)
            input_scale = 1.0 + 0.75 * extra
            # ``smem_bytes()`` doesn't include the slot-header (~64 B per
            # source per buffer for TMA mbarriers, swizzle bookkeeping, etc.)
            # or the slab-rounded alignment to TMA's 128 B box-stride. On the
            # 99 KB sm_120 cap that overhead is ~3 KB on a 2-source matmul, so
            # treat the cap as 97 % of the live value to keep the score honest.
            effective_cap = smem_cap_bytes - smem_cap_bytes // 32  # ≈ 97%
            # Safety multiplier is aspect-conditional: PAD_SMEM padding adds
            # ~1 cell per row, so wide stripes (macro_n=512 on macro_m=4 at
            # SDPA QK^T) inflate the actual ``smem_bytes()`` ~1.6× past the
            # base slab, but well-balanced macros (golden's 208×128 at 2048³
            # matmul fp32, aspect 1.6) inflate only ~1.1×. Empirically pick
            # ×1.7 for aspect > 4, ×1.1 otherwise — without this split the
            # tighter 1.7× kills the balanced golden tile (which actually fits
            # at 92 KB < 99 KB cap) while the looser 1.1× lets the wide-N
            # stripe through (actual 105 KB fails the cap).
            aspect = max(macro_m, macro_n) / max(1, min(macro_m, macro_n))
            safety_num, safety_den = (17, 10) if aspect > 4 else (11, 10)
            slab_bytes = int(base_slab * input_scale * safety_num // safety_den)
            if slab_bytes > effective_cap:
                score -= 4.0
            elif 2 * slab_bytes > effective_cap:
                score -= 2.5

    # TMA-eligibility bonus (matmul on sm_90+). ``050_use_tma`` promotes the
    # K-outer BUFFERED bundle to TMA when every source's inner box extent is
    # ≥128 B aligned; for a 2-input matmul that means ``BK·sizeof(elem) ≥ 128``
    # (A's K-inner, stride-1) AND ``BN·FN·sizeof(elem) ≥ 128`` (B's N-inner,
    # stride-1). Without this bonus the prior is BK-independent so MCTS would
    # measure the (BM, BN, FM, FN, BK=16) leaf first (insertion order tie-
    # break) and exhaust patience before ever benching the (BK=32) sibling —
    # which is the very leaf TMA fires on, and which at 2048³ fp32 runs ~290 us
    # vs ~347 us cp.async (-O3) and ~371 us vs ~3.5 ms (-O1). +1.0 is large
    # enough to outweigh the +0.5 cells gradient between two cell-budget
    # neighbours, so the BK=32 sibling outranks its BK=16 cousin at scoring
    # time and MCTS visits it first.
    if "FM" in knobs and isinstance(compute_capability, tuple) and compute_capability >= (9, 0):
        bk = int(knobs.get("BK", 1))
        fn = int(knobs.get("FN", 1))
        # The N-side inner stride in elements differs by tier: scalar
        # carries ``BN`` (per-thread N tile) explicitly; warp-tier MMA
        # has no ``BN`` knob — its equivalent is ``WN · FN · atom_N``.
        # Defaulting to 1 (the pre-2026 shape) under-counted the warp
        # tier by ``WN · atom_N`` (typically 32-128×) and made the
        # bonus fire for ``FN=32`` but not ``FN=4`` even though both
        # comfortably clear the 128 B TMA alignment — a 1.0 spurious
        # advantage that pushed the picker to the FM=1 FN=32 spill
        # corner (see ``plans/mma-warp-scoring.md`` for the 2048² fp16
        # sweep).
        if "ATOM_KIND" in knobs:
            from deplodock.compiler.pipeline.passes.lowering.tile._atom import atom_spec  # noqa: PLC0415

            _atom_n = atom_spec(str(knobs["ATOM_KIND"])).shape[1]
            n_stride_elems = int(knobs.get("WN", 1)) * fn * _atom_n
            # Warp-tier MMA: the actual TMA-pipelined band on sm_90+ /
            # sm_120 is ``BK ≤ 4`` (BK=2 is the empirical sweet spot —
            # 84 µs at 2048² fp16 on RTX 5090; BK ≥ 8 falls back to
            # SYNC because the 040_use_ring_buffers smem budget rejects
            # buffer_count >= 2 promotion at that slab size). The pre-
            # 2026 condition ``bk * BYTES_PER_ELEM >= 128`` was the
            # legacy K-side alignment check copied from the scalar
            # tier; it gated the bonus *out* of every TMA-firing warp
            # variant (BK=2 → 8 B << 128 B) while firing for the
            # gmem-direct BK=64 baseline that loses to TMA by ~24 %.
            # Fire bonus when N-side TMA-aligned AND BK is in the
            # TMA-firing band.
            if n_stride_elems * BYTES_PER_ELEM >= 128 and bk <= 4:
                score += 1.0
        else:
            n_stride_elems = int(knobs.get("BN", 1)) * fn
            # Scalar tier: legacy condition kept verbatim. ``bk *
            # BYTES_PER_ELEM >= 128`` corresponds to the K-side per-
            # warp slab fitting the TMA descriptor's 128 B inner-row
            # alignment — meaningful for the scalar matmul path where
            # the K dim is the cooperative-load inner extent.
            if bk * BYTES_PER_ELEM >= 128 and n_stride_elems * BYTES_PER_ELEM >= 128:
                score += 1.0

    return score


def _count_loop_input_buffers(shape) -> int:
    """Number of distinct ``Load.input`` references in the kernel's outer
    body — a planner-time stand-in for "how many source tensors will be
    staged into smem". Used by :meth:`TileOp.lazy_score` to scale the
    matmul slab estimate in :func:`score_tile_geometry`. The plain matmul
    has 2 (A + B); SDPA-with-RoPE / SDPA-with-mask kernels fuse 3-6
    sources into the matmul-with-prologue path.

    Walks ``outer_m`` (nests ``outer_n``), every ``extra_outer`` sibling
    (the multi-reduce shape stashes max / sum loops here), and the
    prologue stmts (softmax max / sum / reciprocal Loads). Constants
    (broadcast scalars like a softmax scale or mask-fill value) inflate
    the count but read from gmem the same way at score time, so counting
    them too is conservative — scaling the slab a bit too much only
    nudges the planner toward smaller tiles, which is the safe side.
    """
    inputs: set[str] = set()

    def _walk(body) -> None:
        for stmt in body.iter():
            if isinstance(stmt, Load):
                inputs.add(stmt.input)

    if shape.outer_m is not None:
        _walk(shape.outer_m.body)
    else:
        _walk(shape.outer_n.body)
    for loop in shape.extra_outer:
        _walk(loop.body)
    for stmt in shape.prologue:
        if isinstance(stmt, Load):
            inputs.add(stmt.input)
    return len(inputs)


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
        n_tiles = sum(1 for s in self.body if isinstance(s, (GridTile, ThreadTile, WarpTile)))
        if n_tiles > 1:
            raise ValueError(f"TileOp.body must contain at most one outer GridTile/ThreadTile/WarpTile, got {n_tiles}")
        # ThreadTile and WarpTile both bind threadIdx; mixing them inside one
        # body re-binds the same coord at two scopes. The outer-tile check
        # above already catches them at top level; this catches the
        # cooperative form (GridTile wrapping a ThreadTile and a WarpTile
        # sibling, or one of them nesting inside the other).
        has_thread = any(isinstance(s, ThreadTile) for s in self.body.iter())
        has_warp = any(isinstance(s, WarpTile) for s in self.body.iter())
        if has_thread and has_warp:
            raise ValueError("TileOp.body cannot contain both a ThreadTile and a WarpTile (both bind threadIdx)")
        self._seed_io_placeholders()

    def _launch_geometry(self) -> tuple[tuple[Axis, ...], tuple[Axis, ...]]:
        """``(block_axes, thread_axes)`` for the outermost tile flavor.

        Returns ``((), ())`` if no ``GridTile``/``ThreadTile`` is present
        (e.g. a degenerate body, or a warp-cooperative body whose inner
        tile is a ``WarpTile`` rather than a ``ThreadTile`` — see
        :meth:`_warp_axes`). For ``GridTile`` wrapping a ``ThreadTile``,
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

    def _warp_axes(self) -> tuple[Axis, ...]:
        """Inner-``WarpTile`` axes, or ``()`` if no ``WarpTile`` is present.

        Companion accessor to :meth:`_launch_geometry`. Warp-cooperative
        bodies (``GridTile > WarpTile > …``, today emitted only by the
        MMA-fragment-factorization consumer plan) bind 32 lanes per warp
        coord — callers that compute per-CTA thread budgets must add
        ``prod(_warp_axes().extent) * 32`` on top of the (typically empty)
        thread-axes product. ``ThreadTile`` and ``WarpTile`` are mutually
        exclusive inside one body (``__post_init__`` rejects mixes), so
        either this returns ``()`` or :meth:`_launch_geometry`'s
        ``thread_axes`` does.
        """
        for s in self.body:
            if isinstance(s, GridTile):
                for child in s.body:
                    if isinstance(child, WarpTile):
                        return child.axes
                return ()
            if isinstance(s, WarpTile):
                return s.axes
        return ()

    def validate(self, ctx) -> bool:
        """Reject post-register-tile variants whose launch geometry would
        exceed device limits (threads-per-CTA and dynamic smem).

        Pre-register-tile TileOps skip the THREAD check; the smem check
        runs whenever Stages are present.
        """
        from math import prod  # noqa: PLC0415

        # Dedupe by Source name: pipelining
        # (080_pipeline_stages) replicates an
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
        threads = prod((ax.extent.as_static() if ax.extent.is_static else 1) for ax in thread_axes)
        return threads <= ctx.max_threads_per_cta

    def score(self, ctx) -> float:
        cap = getattr(ctx, "max_dynamic_smem", None) if ctx is not None else None
        cc = getattr(ctx, "compute_capability", None) if ctx is not None else None
        cache = self.__dict__.setdefault("_score_by_cap", {})
        key = (cap, cc)
        if key in cache:
            return cache[key]
        val = self._compute_score(cap, cc)
        cache[key] = val
        return val

    def _compute_score(self, smem_cap: int | None, compute_capability: tuple[int, int] | None = None) -> float:
        """Score formula — keyed on the live geometry and the ctx-supplied
        ``smem_cap`` (so the macro-tile fit signal in :func:`score_tile_geometry`
        gets a real cap to compare against). Memoized by cap on ``self`` so
        repeated MCTS sibling scoring stays cheap — without that cache,
        ``Candidate.score`` re-walks every body Stmt for coalescing analysis
        on every visit; on Qwen3 0.6B with ~150 TileOps in the graph that
        was the dominant tune-mode cost. Per-cap caching is sound because
        :class:`Context` is frozen and the score is a pure function of
        ``(self, cap)``.
        """
        from deplodock.compiler.ir.tile.ir import Stage as _Stage  # noqa: PLC0415

        block_axes, thread_axes = self._launch_geometry()
        if not thread_axes and not block_axes:
            return 0.0
        thread_extents = [ax.extent.as_static() if ax.extent.is_static else 1 for ax in thread_axes]
        block_extents = [ax.extent.as_static() if ax.extent.is_static else 1 for ax in block_axes]
        if not thread_extents:
            return 0.0
        has_stages = any(isinstance(s, _Stage) for s in self.body.iter())
        # Match the planner-time count by walking the materialized body's
        # Load.input set — so the smem-fit signal in score_tile_geometry sees
        # the same ``n_staged_inputs`` here as it did in ``lazy_score`` for
        # the pre-materialization version. Without this, fused-prologue or
        # multi-input matmul kernels score differently between the two paths
        # and ``test_lazy_score_matches_tile_op_score`` mismatches.
        n_staged = len({s.input for s in self.body.iter() if isinstance(s, Load)})
        if n_staged == 0:
            n_staged = 2
        return score_tile_geometry(
            thread_extents=thread_extents,
            block_extents=block_extents,
            knobs=self.knobs,
            has_stages=has_stages,
            coalescing_inner_extent=self._coalescing_inner_extent(thread_axes),
            smem_cap_bytes=smem_cap,
            n_staged_inputs=n_staged,
            compute_capability=compute_capability,
        )

    @staticmethod
    def lazy_score(ctx, *, knobs=None, shapes=None, params=None) -> float | None:  # noqa: ARG004 — knobs reserved for symmetry with Op.lazy_score
        """Lazy scorer (see :meth:`Op.lazy_score`). Returns ``None`` unless
        called with both ``shapes`` (a :class:`KernelShape` from the
        partition planner) and ``params`` (a :class:`TileParams` variant —
        :class:`ScalarTileParams` or :class:`WarpTileParams`). Computes the
        same value :meth:`score` would on the materialized TileOp, but
        without running ``_build_split_body`` or ``TileOp.__post_init__`` —
        lets the planner rank dozens of variants per LoopOp in microseconds
        instead of milliseconds each.

        Launch geometry the planner will produce (mirrors ``_wrap_tower``):
            - scalar tier: block axes ``[K_s?, M_b?, N_b]`` extent
              ``[SPLITK, ceil(E_M/(BM·FM)), ceil(E_N/(BN·FN))]``; thread axes
              ``[K_c?, M_t?, N_t]`` extent ``[BR, BM, BN]``.
            - warp tier (MMA): block axes ``[K_s?, M_b?, N_b]`` extent
              ``[SPLITK, ceil(E_M/(WM·FM·atom_m)), ceil(E_N/(WN·FN·atom_n))]``;
              "thread" axes (warps × lanes) extent
              ``[WN·32, WM]`` so ``prod() = wn·wm·32`` matches the per-CTA
              thread count the materializer sets up.

        Symbolic outer M/N axes contribute extent 1 to the block-extent
        product (matches what :meth:`score` does with non-static
        ``Axis.extent``).
        """
        if shapes is None or params is None:
            return None
        from deplodock.compiler.pipeline.passes.lowering.tile._atom import atom_shape  # noqa: PLC0415
        from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import WarpTileParams  # noqa: PLC0415

        shape = shapes
        p = params

        m_symbolic = shape.outer_m is not None and not shape.outer_m.axis.extent.is_static
        n_symbolic = not shape.outer_n.axis.extent.is_static

        is_warp = isinstance(p, WarpTileParams)
        if is_warp:
            atom_m, atom_n, _ = atom_shape(p.atom_kind)
            per_m = p.wm * p.fm * atom_m
            per_n = p.wn * p.fn * atom_n
            # "Thread axes" for the warp tier = warps × lanes. Use a
            # 2-element decomposition that products to wn·wm·32 so the
            # score_tile_geometry occupancy / coalescing math still works.
            thread_extents: list[int] = [p.wn * 32, p.wm]
        else:
            per_m = p.bm * p.fm
            per_n = p.bn * p.fn
            thread_extents = []
            if p.br > 1:
                thread_extents.append(p.br)
            if shape.outer_m is not None:
                thread_extents.append(p.bm)
            thread_extents.append(p.bn)

        # ceil-div, not floor — when ``per_m > E_M`` (or ``per_n > E_N``) the
        # materializer still emits 1 CTA (with a masked Cond), and when
        # ``per_m`` doesn't divide ``E_M`` it emits ⌈E_M / per_m⌉ CTAs with the
        # last one masked. Mirror that here so the post-materialization
        # ``_compute_score`` (which reads the GridTile's actual axis extents)
        # sees the same block_extents as ``lazy_score`` does pre-materialize.
        block_extents: list[int] = []
        if p.splitk > 1:
            block_extents.append(p.splitk)
        if shape.outer_m is not None:
            if m_symbolic:
                block_extents.append(1)
            else:
                E_M = shape.outer_m.axis.extent.as_static()
                block_extents.append(max(1, (E_M + per_m - 1) // per_m))
        if n_symbolic:
            block_extents.append(1)
        else:
            E_N = shape.outer_n.axis.extent.as_static()
            block_extents.append(max(1, (E_N + per_n - 1) // per_n))

        # Planner always stamps FM/FN — score keys off ``"FM" in knobs``.
        # BM/BN/BK go in too so the macro-aspect + smem-fit signals see the
        # planner's tile shape (post-materialization ``_cached_score`` reads
        # ``self.knobs`` which already carries these; without them here the
        # lazy_score path would silently fall back to BM=BN=1 defaults).
        if is_warp:
            # ATOM_KIND / WM / WN feed the warp-tier branch of
            # ``score_tile_geometry``: the cells-sweet-spot prior keys off
            # ``ATOM_KIND in knobs``, the aspect-ratio prior reads FM/FN
            # (already here), and the TMA-bonus's N-stride calc uses
            # ``WN · FN · atom_N``. Omitting them silently routed the warp
            # tier through the scalar branch and over-counted the TMA
            # bonus for skewed shapes — exactly the FM=1 FN=32 regression
            # ``plans/mma-warp-scoring.md`` chases.
            score_knobs = {
                "FM": p.fm,
                "FN": p.fn,
                "SPLITK": p.splitk,
                "BK": p.bk,
                "WM": p.wm,
                "WN": p.wn,
                "ATOM_KIND": p.atom_kind,
            }
        else:
            score_knobs = {
                "FM": p.fm,
                "FN": p.fn,
                "SPLITK": p.splitk,
                "BM": p.bm,
                "BN": p.bn,
                "BK": p.bk,
            }

        # Stages are added by later passes (020_stage_inputs); the planner's
        # output never contains them at this point.
        has_stages = False

        coalescing = TileOp._coalescing_inner_extent_from_writes(shape, p)

        smem_cap = getattr(ctx, "max_dynamic_smem", None) if ctx is not None else None
        cc = getattr(ctx, "compute_capability", None) if ctx is not None else None
        # Count distinct input buffers the loop body reads — score_tile_geometry's
        # slab estimate is per-A+B (2 inputs); kernels that fuse more (SDPA-with-
        # mask + RoPE = 4-6 inputs; gated MLP = 3 inputs) need the slab estimate
        # bumped to match the per-iter smem footprint.
        n_staged = _count_loop_input_buffers(shape)

        return score_tile_geometry(
            thread_extents=thread_extents,
            block_extents=block_extents,
            knobs=score_knobs,
            has_stages=has_stages,
            coalescing_inner_extent=coalescing,
            smem_cap_bytes=smem_cap,
            n_staged_inputs=n_staged,
            compute_capability=cc,
        )

    @staticmethod
    def _coalescing_inner_extent_from_writes(shape, params) -> int:
        """Mirror of :meth:`_coalescing_inner_extent` computed against the
        un-rewritten LoopOp body. σ_outer keeps the outer M/N axis names
        on the eventual thread tiles (M_t inherits ``outer_m.axis.name``,
        N_t inherits ``outer_n.axis.name``, K_c inherits ``k_loop.axis.name``),
        so the inner-stride free-var match works the same way before and
        after body construction.

        Size-1 thread axes are skipped — ``normalize_body`` inlines them
        out of the materialized body, so they don't appear in the post-
        materialization ``thread_axes`` ``_coalescing_inner_extent``
        iterates. Mirroring that here keeps the lazy score consistent
        with ``TileOp.score`` on size-1 corners (e.g. BN=1 matmul rows).
        """
        from deplodock.compiler.ir.stmt.leaves import Write  # noqa: PLC0415
        from deplodock.compiler.pipeline.passes.lowering.tile._atom import atom_shape  # noqa: PLC0415
        from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import WarpTileParams  # noqa: PLC0415

        thread_extent: dict[str, int] = {}
        if isinstance(params, WarpTileParams):
            # Warp-tier: the "thread" axes are warps × lanes. Treat the
            # WN × 32 contribution as the inner-stride coalescer along N,
            # mirroring the scalar BN role.
            atom_m, atom_n, _ = atom_shape(params.atom_kind)
            if shape.outer_m is not None and params.wm > 1:
                thread_extent[shape.outer_m.axis.name] = params.wm * atom_m
            n_extent = params.wn * atom_n
            if n_extent > 1:
                thread_extent[shape.outer_n.axis.name] = n_extent
        else:
            if params.br > 1 and shape.k_loop is not None:
                thread_extent[shape.k_loop.axis.name] = params.br
            if shape.outer_m is not None and params.bm > 1:
                thread_extent[shape.outer_m.axis.name] = params.bm
            if params.bn > 1:
                thread_extent[shape.outer_n.axis.name] = params.bn
        if not thread_extent:
            return 0

        best = 0
        for stmt in shape.outer_n.body.iter():
            if not isinstance(stmt, Write) or not stmt.index:
                continue
            inner_vars = set(stmt.index[-1].free_vars())
            matched = inner_vars & thread_extent.keys()
            if not matched:
                continue
            extent = sum(thread_extent[name] for name in matched)
            if extent > best:
                best = extent
        return best

    def _coalescing_inner_extent(self, thread_axes) -> int:
        """Sum the extents of thread axes that appear in the inner-stride
        dim of any top-level ``Write`` in the body. Feeds the coalescing
        term in :func:`score_tile_geometry`.

        Why this matters: two variants with identical ``cells × threads
        × ctas`` can differ 2x in measured latency purely because one
        parks its threads along the M (outer-stride) output axis and
        the other along N (inner-stride). The N-major variant coalesces
        gmem loads + the output store; the M-major one strides by N
        per thread. The four base score terms can't distinguish them.
        """
        from deplodock.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        if not thread_axes:
            return 0
        thread_names = {ax.name for ax in thread_axes}
        best = 0
        for stmt in self.body.iter():
            if not isinstance(stmt, Write) or not stmt.index:
                continue
            inner_vars = set(stmt.index[-1].free_vars())
            matched = inner_vars & thread_names
            if not matched:
                continue
            extent = sum((ax.extent.as_static() if ax.extent.is_static else 1) for ax in thread_axes if ax.name in matched)
            if extent > best:
                best = extent
        return best


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
    "WarpTile",
    "AtomTile",
    "SerialTileBase",
    "SerialTile",
    "StridedTile",
    "SerialKind",
    "Stage",
    "StageBundle",
    "StagePolicy",
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
    "score_tile_geometry",
    # Scheduling constants
    "BLOCK_SIZE",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]

# Register Tile-IR stmts with the shared rewrite/simplify dispatch.
from deplodock.compiler.ir.tile import passes as _passes  # noqa: E402, F401
