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
``015_lower_pipelined_async_stage`` before materialization.

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

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
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
    Tile,
    Write,
    pretty_body,
)
from deplodock.compiler.ir.stmt.ir import BodyOp

# ---------------------------------------------------------------------------
# Deprecated AsyncWait stub (stage-wrap-body refactor)
# ---------------------------------------------------------------------------
#
# AsyncWait is preserved as a stub for import compatibility during the
# stage-wrap-body refactor. The async/TMA pipelining passes that used to
# emit AsyncWait are stubbed in Phase B and rewritten in Phase C buckets
# 10/11/12 to fold wait semantics into the wrapping stage's transport
# policy (pipeline_depth on AsyncBufferedStage / TmaBufferedStage). No
# new passes should emit AsyncWait; the class deletion happens once
# Phase C's pipelining + TMA + cp.async buckets close.


@dataclass
class AsyncWait(Stmt):
    """**Deprecated** — kept for import compatibility during stage-wrap-body refactor.

    Pre-refactor: synchronization with previously-issued ``AsyncBufferedStage``
    / ``TmaBufferedStage`` loads. ``keep`` = cp.async wait_group argument;
    ``phase`` / ``slot`` = TMA mbarrier phase + ring slot.

    Post-refactor: wait semantics fold into the wrapping stage's transport
    policy. Phase C bucket 10 (pipelining) / 11 (cp.async) / 12 (TMA)
    delete the last emission sites and this class.
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
# Cross-thread combine (Tile-IR-specific Stmt for cooperative reductions)
# ---------------------------------------------------------------------------


@dataclass
class Combine(Stmt):
    """Cross-thread reduction of an ``Accum`` target.

    Placed immediately after a cooperative reduce loop (``StridedLoop``
    whose ``Accum`` produced ``name``). Materialization emits the
    cross-thread combine — smem tree-halve today; warp-shuffle / atomic
    in the future.

    ``op`` is a redundant copy of the matching ``Accum.op`` — kept as a
    cross-check; if the strategy constructs a Combine with the wrong op
    relative to the matching Accum, validation surfaces the bug.
    """

    name: str
    op: ElementwiseImpl

    def deps(self) -> tuple[str, ...]:
        return (self.name,)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}Combine({self.name}, op={self.op.name})"]


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
    back-pointer, set by 002_stage_inputs at construction time, identifies
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
    """

    name: str
    buf: str
    cache_dims: tuple[CacheDim, ...]
    origin: tuple[Expr, ...]
    pad: tuple[int, ...] = ()
    template_index: tuple[Expr, ...] | None = None

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
        """Bytes of dynamic shared memory this Source allocates (single-slot)."""
        n = BYTES_PER_ELEM
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
    cache = ", ".join(f"{ax.name}:{ax.extent}@{cd.source_dim}" for ax, cd in zip(src.cache_axes, src.cache_dims, strict=True))
    origin = ", ".join(e.pretty() for e in src.origin)
    pad = f" pad=({', '.join(str(p) for p in src.pad)})" if src.pad and any(src.pad) else ""
    tpl = ""
    if src.template_index is not None:
        tpl = " template=[" + ", ".join(e.pretty() for e in src.template_index) + "]"
    return f"{src.name}<-{src.buf}(origin=({origin}), slab=({cache})){pad}{tpl}"


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
        sources_pretty = "; ".join(_source_pretty(s) for s in self.sources)
        head = f"{indent}{type(self).__name__}([{sources_pretty}]){self._pretty_extra()}:"
        return [head, *pretty_body(self.body, indent + INDENT)]

    def _pretty_extra(self) -> str:
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

    def _pretty_extra(self) -> str:
        return f" buffers={self.buffer_count}@{self.phase.pretty()}"


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
    happens in ``015_lower_pipelined_async_stage`` before materialize.
    """

    pipeline_depth: int = field(default=1, kw_only=True)

    def _pretty_extra(self) -> str:
        depth = f" depth={self.pipeline_depth}" if self.pipeline_depth > 1 else ""
        return f"{super()._pretty_extra()} async{depth}"


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

    def _pretty_extra(self) -> str:
        depth = f" depth={self.pipeline_depth}" if self.pipeline_depth > 1 else ""
        sw = "" if self.swizzle == SwizzleMode.NONE else f" swizzle={self.swizzle.value}"
        return f"{super()._pretty_extra()} tma{depth}{sw}"


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

    def _pretty_extra(self) -> str:
        if self.buffer_count > 1 and self.phase is not None:
            return f" compute buffers={self.buffer_count}@{self.phase.pretty()}"
        return " compute"


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
        n_tiles = sum(1 for s in self.body if isinstance(s, Tile))
        if n_tiles > 1:
            raise ValueError(f"TileOp.body must contain at most one Tile, got {n_tiles}")
        self._seed_io_placeholders()

    def validate(self, ctx) -> bool:
        """Reject post-register-tile variants whose launch geometry would
        exceed device limits (threads-per-CTA and dynamic smem).

        Pre-register-tile TileOps skip the THREAD check; the smem check
        runs whenever Stages are present.
        """
        from math import prod  # noqa: PLC0415

        staged = sum(s.smem_bytes for s in self.body.iter() if isinstance(s, Stage))
        if staged > ctx.max_dynamic_smem:
            return False

        if "FM" not in self.knobs:
            return True
        tile = next((s for s in self.body.iter() if isinstance(s, Tile)), None)
        if tile is None:
            return True
        thread_extents = [int(ba.axis.extent) for ba in tile.axes if ba.bind == BIND_THREAD]
        if not thread_extents:
            return True
        threads = prod(thread_extents)
        return threads <= ctx.max_threads_per_cta

    def score(self, ctx) -> float:  # noqa: ARG002 — ctx reserved for cc-specific tuning
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.ir.tile.ir import Stage as _Stage  # noqa: PLC0415

        target_threads = 256
        target_ctas = 256
        score = 0.0
        tile = next((s for s in self.body.iter() if isinstance(s, Tile)), None)
        if tile is None:
            return 0.0

        thread_extents = [int(ba.axis.extent) for ba in tile.axes if ba.bind == BIND_THREAD]
        block_extents = [int(ba.axis.extent) for ba in tile.axes if ba.bind == BIND_BLOCK]
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
        if splitk > 8:
            score -= min((splitk - 8) / 8.0, 1.0)

        if any(isinstance(s, _Stage) for s in self.body.iter()):
            score += 1.0
        if "FM" in self.knobs:
            score += 1.0

        return score


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
    # Tile-IR statements
    "Tile",
    "Combine",
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
    "AsyncWait",  # deprecated stub during refactor
    "trivial_stage_body",  # deprecated stub during refactor
    "BYTES_PER_ELEM",
    # Bindings
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
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
