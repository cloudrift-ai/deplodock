"""Tile IR — schedule decisions as structural Stmts.

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

**Leaf compute reuses Loop IR.** ``Load`` / ``Assign`` / ``Select`` /
``Write`` / ``Accum`` / ``Cond`` come straight from ``ir.loop`` — buf
names are strings so they're directly renderable.

**Scheduling decisions live where they naturally belong**:

- ``Tile.thread_axes`` / ``Tile.block_axes`` — same shape as
  ``Tile``: which output axes are bound to thread coords vs CUDA
  block coords. Pointwise has ``thread_axes`` populated and
  ``block_axes`` empty (one thread per output element). Cooperative
  reductions have ``block_axes`` populated and ``thread_axes`` empty;
  the cooperative thread axis is synthesized at materialization.
- Loop constructs in the Tile-IR body are ``Loop`` (serial) and
  ``StridedLoop`` (cooperative — threads stride through the axis).
  Both are shared with Loop-IR / Kernel-IR via ``ir.stmt``.
- ``Combine`` — cross-thread collapse of an Accum target; sibling
  Stmt because it's buffer/accumulator-scoped, not axis-local.

The compute body is ``Loop`` / ``StridedLoop`` / ``Accum`` / ``Load`` /
``Assign`` / ``Write`` — a straight iteration tree.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

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
# Schedule-bearing Stmts
# ---------------------------------------------------------------------------
#
# Scheduling decisions are expressed via ``BoundAxis.bind`` values on
# ``Tile.axes`` (``BIND_THREAD`` / ``BIND_BLOCK`` for launch geometry)
# and via the choice of body loop construct (``Loop`` for serial,
# ``StridedLoop`` for cooperative striding).


# ``Tile`` is shared infrastructure — defined in ``ir/stmt.py`` and
# re-exported here. Used at Tile IR (with Stage / Combine in the body)
# and at Kernel IR (with Smem / Sync / TreeHalve after materialization).


# Tile-IR loop constructs are ``Loop`` (serial) and ``StridedLoop``
# (cooperative — threads of the block stride through the axis). Both
# come from ``ir.stmt`` directly; Tile IR doesn't add a wrapper.


@dataclass
class AsyncWait(Stmt):
    """Synchronize with previously-issued ``AsyncBufferedStage`` /
    ``TmaBufferedStage`` loads.

    ``keep`` is the number of most-recently-issued cp.async groups that
    may *remain* in flight after the wait — i.e. the PTX
    ``cp.async.wait_group N`` argument. ``keep=0`` drains every
    outstanding group (epilogue / synchronous-style use). ``keep=K``
    where K equals the number of ``AsyncBufferedStage`` issued per
    pipelined iteration leaves the just-issued chunk in flight while
    older chunks complete (steady-state body of a software-pipelined
    K-outer loop).

    ``phase`` is the consumer-side ring-buffer phase for TMA waits:
    when set, materialization emits ``MbarrierWait(mbar, phase)`` for
    each pending TMA mbar, where ``phase`` is the EXPRESSION the consumer
    is about to read (e.g. ``K_outer % 2`` inside the pipelined main
    loop, ``(n_chunks - 1) % 2`` for the epilogue's tail wait). Without
    this, the materializer would have to guess from issuance-time phase
    expressions, which leak loop-axis Vars across scopes. ``None`` means
    "cp.async wait" — falls back to ``CpAsyncWait(keep) + Sync()``.

    Carrying ``keep`` explicitly (rather than re-deriving it from the
    surrounding stage count) keeps the wait correct after structural
    rewrites such as unrolling a 1-iteration steady-state loop.
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


# Bytes per stored element in smem. Today's pipeline emits fp32 only;
# Stage carries no dtype. If multi-dtype support lands, move this onto
# Stage as a per-instance field and update callers.
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


def trivial_stage_body(
    name: str,
    buf: str,
    origin: tuple[Expr, ...],
    axes: tuple[Axis, ...],
    addressing: AffineAddressing | TemplateAddressing,
) -> Body:
    """Build the canonical ``Load + Write`` body for a cooperative-load
    Stage. Source-side index is reconstructed from ``origin + cache_var``
    pairs (affine) or kept verbatim (template); the Write target is the
    Stage's smem name with cache-local Var coords.

    Single source of truth for body synthesis: ``Stage`` callers that don't
    fuse anything into the producer use this helper directly; the
    ``primary_load`` / ``origin`` / ``addressing`` properties round-trip
    through it."""
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


@dataclass
class Stage(Stmt):
    """Operand-cache declaration — stage a contiguous slab of a gmem buffer
    into a named local buffer for reuse across the surrounding ``Tile``
    body.

    Synchronous, single-slot form. Materialization emits a leading
    ``Sync`` (so iter-N compute can finish reading before iter-N+1
    overwrites smem) plus a cooperative ``Load+Write`` pair and a
    trailing ``Sync`` so the freshly-loaded smem is visible to all
    threads in the CTA.

    Carried state:

    - ``name`` — staged buffer's SSA-like identifier. Subsequent
      ``Load(input=name, index=cache-local)`` reads in the surrounding
      Tile body refer to it directly.
    - ``axes`` — cache axes (smem layout, in this order).
    - ``body`` — cooperative-load program. Today this is the trivial
      ``Load(input=src_buf, index=...); Write(output=name, ...)`` pair
      built by :func:`trivial_stage_body`; future producer-side fusion
      will inline elementwise compute between the Load and Write.
    - ``pad`` — bank-conflict-breaking padding (per cache axis).

    The legacy slab-geometry fields (``buf``, ``origin``, ``addressing``)
    are now derived from ``body``'s primary Load via properties:

    - ``buf = body[0].input``
    - ``origin = body[0].index`` with cache-axis Vars substituted to 0
    - ``addressing`` = ``AffineAddressing`` iff every cache axis appears
      coef-1 in exactly one source dim, else ``TemplateAddressing``

    Transport is encoded structurally via subclass:

    - ``Stage`` (this class) — synchronous cooperative ``Load+Write+Sync``.
    - ``BufferedStage`` — N rotating smem slabs, sync transport.
    - ``AsyncBufferedStage`` — cp.async transport, caller-owned ``AsyncWait``.
    - ``TmaBufferedStage`` — TMA box copy issued by one elected thread,
      mbarrier-synchronized via ``AsyncWait``.
    """

    name: str
    axes: tuple[Axis, ...]
    body: Body
    # Per-cache-axis extra extent added to the smem allocation (not to the
    # cooperative-load extent). Empty tuple = no padding. Used by the bank-
    # conflict pass to break stride-aliased smem layouts: padding dim ``d``
    # by ``+1`` shifts every higher-stride row by one float, eliminating
    # 32-way bank conflicts that arise when adjacent threads stride through
    # power-of-2 multiples of bank width.
    pad: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Coerce tuple inputs (e.g. from rewrite/with_bodies that build a
        # plain tuple comprehension) into the Body subclass so downstream
        # walkers can call ``.map`` / ``.iter`` on Stage's nested body.
        if not isinstance(self.body, Body):
            self.body = Body.coerce(self.body)
        if len(self.body) < 2:
            raise ValueError(f"Stage {self.name!r}: body must contain at least one Load and one Write, got {len(self.body)} stmts")

    # Stage stays opaque to generic body walkers (no ``nested()`` /
    # ``with_bodies()`` override). Exposing the cooperative-load Loads to
    # ``008_register_tile``'s ``Body.map``-based F-axis replication would
    # corrupt their semantics — cache-axis Vars in the body are
    # smem-local, not values that participate in F-axis register tiling.
    # σ-substitution through the body is handled explicitly by the
    # Stage-specific ``rewrite`` / ``simplify`` handlers in
    # ``deplodock/compiler/ir/tile/passes.py``.

    def deps(self) -> tuple[str, ...]:
        return ()

    def external_reads(self) -> tuple[str, ...]:
        return tuple(s.input for s in self.source_loads)

    def local_decls(self) -> tuple[str, ...]:
        return (self.name,)

    # ------------------------------------------------------------------
    # Derived slab-geometry views (legacy ``buf`` / ``origin`` /
    # ``addressing`` field accessors, re-expressed as body-driven
    # properties so the body is the single source of truth).
    # ------------------------------------------------------------------

    @property
    def primary_load(self) -> Load:
        """The producer-side Load that anchors this Stage's slab.

        Today every Stage body has exactly one Load (trivial cooperative
        load); raises if a future fused body has more than one source
        Load. Callers needing all source Loads should use
        :attr:`source_loads`."""
        loads = tuple(s for s in self.body if isinstance(s, Load))
        if len(loads) != 1:
            raise ValueError(
                f"Stage {self.name!r}: primary_load undefined — body has {len(loads)} Loads "
                "(multi-source fused stages must use ``source_loads``)."
            )
        return loads[0]

    @property
    def source_loads(self) -> tuple[Load, ...]:
        """Every Load in ``body``, in body order. Trivial body returns a
        1-tuple; fused bodies (future) may return more."""
        return tuple(s for s in self.body if isinstance(s, Load))

    @property
    def buf(self) -> str:
        """Source gmem buffer name. ``body[0].input`` for the trivial /
        single-source case."""
        return self.primary_load.input

    @property
    def origin(self) -> tuple[Expr, ...]:
        """Per-source-dim CTA-uniform anchor: ``primary_load.index`` with
        every cache-axis Var substituted to 0 and simplified."""
        from deplodock.compiler.ir.expr import SimplifyCtx  # noqa: PLC0415
        from deplodock.compiler.ir.sigma import Sigma  # noqa: PLC0415

        sigma = Sigma({ax.name: Literal(0, "int") for ax in self.axes})
        ctx = SimplifyCtx.empty()
        return tuple(sigma.reduce(e, ctx) for e in self.primary_load.index)

    @property
    def addressing(self) -> AffineAddressing | TemplateAddressing:
        """Affine if every cache axis appears coef-1 in exactly one source
        dim of ``primary_load.index``, else template (verbatim index)."""
        from deplodock.compiler.ir.expr import SimplifyCtx  # noqa: PLC0415
        from deplodock.compiler.ir.sigma import Sigma  # noqa: PLC0415

        index = self.primary_load.index
        var_to_dim: dict[str, int] = {}
        for ax in self.axes:
            dims = [d for d, e in enumerate(index) if ax.name in e.free_vars()]
            if len(dims) != 1:
                return TemplateAddressing(exprs=index)
            var_to_dim[ax.name] = dims[0]

        ctx = SimplifyCtx.empty()
        origin = self.origin
        for ax in self.axes:
            d = var_to_dim[ax.name]
            one_sigma = Sigma({a.name: (Literal(1, "int") if a.name == ax.name else Literal(0, "int")) for a in self.axes})
            actual = one_sigma.reduce(index[d], ctx)
            expected = (origin[d] + Literal(1, "int")).simplify(ctx)
            if actual.pretty() != expected.pretty():
                return TemplateAddressing(exprs=index)

        return AffineAddressing(dims=tuple(var_to_dim[ax.name] for ax in self.axes))

    @property
    def alloc_extents(self) -> tuple[int, ...]:
        """Per-cache-axis smem allocation extent: cache extent + ``pad``.

        The cooperative load iterates over the unpadded extents; the
        smem allocation reserves the padded ones. Used by every smem-
        sizing check in the tile-lowering chain (007 admission, 012
        bank-pad budget, 013 double-buffer budget, materialize
        ``Smem.extents``)."""
        extents = tuple(int(ax.extent) for ax in self.axes)
        if not self.pad:
            return extents
        return tuple(e + p for e, p in zip(extents, self.pad, strict=True))

    @property
    def smem_bytes(self) -> int:
        """Bytes of dynamic shared memory this Stage allocates.

        ``product(alloc_extents) * BYTES_PER_ELEM`` × ``buffer_count``
        (the latter only for ``BufferedStage`` subtypes). Assumes
        ``BYTES_PER_ELEM == 4`` (fp32) — matches today's kernel
        codegen, which has no per-Stage dtype field."""
        n = BYTES_PER_ELEM
        for e in self.alloc_extents:
            n *= e
        return n

    def exprs(self) -> tuple[Expr, ...]:
        # Exprs visible to generic Expr walkers (axis-dep analysis in 008,
        # σ-substitution targets in 015). Iterate every source Load's index
        # so fused stages expose all gmem references; trivial stages
        # reduce to ``(*origin,)`` (one Load, AffineAddressing).
        out: tuple[Expr, ...] = ()
        for src in self.source_loads:
            out = (*out, *src.index)
        return out

    def pretty(self, indent: str = "") -> list[str]:
        if len(self.source_loads) > 1:
            srcs = ", ".join(src.input for src in self.source_loads)
            cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in self.axes)
            pad = f" pad=({', '.join(str(p) for p in self.pad)})" if self.pad and any(self.pad) else ""
            head = f"{indent}{self.name} = {type(self).__name__}(fuse[{srcs}], cache=({cache})){pad}{self._pretty_extra()}:"
            return [head, *pretty_body(self.body, indent + INDENT)]
        origin = ", ".join(e.pretty() for e in self.origin)
        if isinstance(self.addressing, AffineAddressing):
            slab = ", ".join(f"{ax.name}:{ax.extent}@{d}" for ax, d in zip(self.axes, self.addressing.dims, strict=True))
        else:
            cache = ", ".join(f"{ax.name}:{ax.extent}" for ax in self.axes)
            tpl = ", ".join(e.pretty() for e in self.addressing.exprs)
            slab = f"{cache} template=[{tpl}]"
        pad = f" pad=({', '.join(str(p) for p in self.pad)})" if self.pad and any(self.pad) else ""
        return [f"{indent}{self.name} = {type(self).__name__}({self.buf}, origin=({origin}), slab=({slab})){pad}{self._pretty_extra()}"]

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
            raise ValueError(f"BufferedStage {self.name!r}: buffer_count must be >= 2, got {self.buffer_count}")

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
    ``CpAsyncCommit`` only — *no* implicit wait, *no* ``Sync``. The
    caller must dominate every consumer with an ``AsyncWait`` Stmt.
    Requires sm_80+ (gated by ``013_async_copy``).
    """

    def _pretty_extra(self) -> str:
        return f"{super()._pretty_extra()} async"


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
    elected thread issues the box copy. Pairs with ``AsyncWait`` which
    lowers to ``MbarrierWait(phase)`` (no trailing ``Sync`` — mbarrier
    arrival already provides CTA-wide visibility).

    Requires sm_90+ and ``--gpu-architecture=sm_90a`` (gated by
    ``011_tma_copy``). Eligible only for ``AffineAddressing`` with the
    inner source dim contiguous and 16 B aligned.
    """

    swizzle: SwizzleMode = field(default=SwizzleMode.NONE, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        # TMA box copies write rows back-to-back at the cache extent;
        # bank-conflict ``+1`` padding (set by ``014_pad_smem`` for
        # cp.async / sync stages) would put body Loads' padded stride out
        # of step with the unpadded box write. The pad pass already skips
        # ``TmaBufferedStage`` — this assertion catches any future caller
        # that constructs a TMA stage with stale pad.
        if self.pad and any(self.pad):
            raise ValueError(f"TmaBufferedStage {self.name!r}: pad must be empty, got {self.pad!r}")

    def _pretty_extra(self) -> str:
        sw = "" if self.swizzle == SwizzleMode.NONE else f" swizzle={self.swizzle.value}"
        return f"{super()._pretty_extra()} tma{sw}"


@dataclass
class ComputeStage(Stage):
    """Compute Stage — hoisted invariant compute sitting between transport
    Stages and a reduce body.

    Distinguished from a canonical (transport) ``Stage`` by what its body
    Loads read: a transport Stage's body Loads target the gmem source
    buffer; a ``ComputeStage``'s body Loads target *sibling* Stage smem
    buffers and the body runs an elementwise chain whose result is
    written into ``ComputeStage``'s own smem buffer.

    Produced by the hoist-invariant-compute path: when a reduce-body
    cone depends only on cache-axes of a producer-Stage group, the cone
    Assigns are lifted into a ``ComputeStage`` that allocates its own
    smem and reads from the sibling transports. Downstream:

    - 010 keeps the canonical transport-Stage promotion path and
      handles a ``ComputeStage`` separately (no gmem to ring-buffer; the
      ``BUFFER_COMPUTE`` knob optionally ring-buffers the *output*
      buffer so reduce reads pipeline across K_outer).
    - 011 / 013 skip ``ComputeStage`` (no gmem transport to narrow).
    - 015 places ``ComputeStage`` between the wait and the K_inner
      reduce in the steady-state body — exactly the slot the cone needs.
    - materialize_tile hoists the smem allocation to kernel scope and
      emits the body as a per-thread cooperative compute (no
      ``CpAsyncCopy`` / ``TmaLoad`` — body Loads target sibling smem).

    Optional ``buffer_count`` / ``phase`` mirror ``BufferedStage`` so 010
    can promote a ``ComputeStage`` to a ring-buffered output without a
    separate ``BufferedComputeStage`` class. ``buffer_count = 1``
    (default) means single-slot; ``>= 2`` requires ``phase``.
    """

    buffer_count: int = field(default=1, kw_only=True)
    phase: Expr | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.buffer_count < 1:
            raise ValueError(f"ComputeStage {self.name!r}: buffer_count must be >= 1, got {self.buffer_count}")
        if self.buffer_count > 1 and self.phase is None:
            raise ValueError(f"ComputeStage {self.name!r}: phase required when buffer_count > 1")

    # Sibling-smem reads aren't external buffer dependencies — the
    # producers live in the same Tile scope. Returning empty here keeps
    # 015's TMA/async eligibility checks and the gmem-buffer
    # enumeration in tuning.py from misclassifying ComputeStage's body
    # Loads as gmem.
    def external_reads(self) -> tuple[str, ...]:
        return ()

    @property
    def primary_load(self) -> Load:
        raise ValueError(
            f"ComputeStage {self.name!r}: primary_load undefined — compute stages read from "
            "multiple sibling smem buffers. Use ``source_loads`` for the body Loads."
        )

    @property
    def buf(self) -> str:
        raise ValueError(f"ComputeStage {self.name!r}: buf undefined — compute stages have no single gmem source.")

    @property
    def origin(self) -> tuple[Expr, ...]:
        raise ValueError(f"ComputeStage {self.name!r}: origin undefined — compute stages have no gmem slab anchor.")

    @property
    def addressing(self) -> AffineAddressing | TemplateAddressing:
        raise ValueError(f"ComputeStage {self.name!r}: addressing undefined — compute stages have no gmem source dim.")

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

        # Body is a tuple subclass; coerce so ``Op(body=tuple_value)``
        # keeps working without forcing wrapping at every rule's
        # rewrite site.
        coerced = Body.coerce(self.body)
        normalized = normalize_body(coerced, hoist=False)
        self.body = normalized if isinstance(normalized, Body) else Body(normalized)
        n_tiles = sum(1 for s in self.body if isinstance(s, Tile))
        if n_tiles > 1:
            raise ValueError(f"TileOp.body must contain at most one Tile, got {n_tiles}")
        self._seed_io_placeholders()

    def validate(self, ctx) -> bool:
        """Reject post-register-tile variants whose launch geometry would
        exceed device limits. Two checks:

        - **threads ≤ ``ctx.max_threads_per_cta``** (1024 on every CUDA
          capability we support). Without this, F-fork candidates like
          ``(FM=1, FN=2)`` on a ``BM=64, BN=128`` tile spawn a CTA of
          64×64=4096 threads — the driver rejects it but the engine has
          already burned compile + bench wall-budget on it.
        - **staged smem footprint ≤ ``ctx.max_dynamic_smem``**.
          Sums every ``Stage.smem_bytes`` in the body (including nested
          Stages inside reduce loops). The check uses raw stage bytes —
          no upfront slack for the upcoming ``010_double_buffer`` /
          ``014_pad_smem`` overhead. Those passes' decisions are part of
          the autotune surface; when their multipliers push a variant
          over the cap, ``KernelOp.validate`` is the second-line gate.
          Without this tile-stage check, three-pass naive softmax +
          ``@V`` at large head_dim/seq stages mul+mask in three separate
          smem regions whose total exceeds the device cap and crashes
          the launch with ``CUDA_ERROR_INVALID_VALUE``.

        Pre-register-tile TileOps (the post-blockify state, where THREAD
        extents are ``BM, BN`` and the register-tile rule will still
        divide by ``(FM, FN)``) skip the THREAD check — otherwise we'd
        reject every healthy pre-tile candidate. The smem check runs at
        every stage where Stages are present."""
        from math import prod  # noqa: PLC0415

        # Staged-smem check — runs whether or not register_tile fired,
        # since 007_stage_inputs can add Stages independently.
        staged = sum(s.smem_bytes for s in self.body.iter() if isinstance(s, Stage))
        if staged > ctx.max_dynamic_smem:
            return False

        # THREAD-count check — only after register_tile committed.
        # ``FM`` presence in knobs is the post-008 marker.
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
        """Autotune prior over the failure modes we've observed:

        - Sub-warp launch (threads < 32) → ``-2.0``. A 4-thread CTA wastes
          ~7/8 of every warp instruction.
        - Single cell per thread → ``-1.0``. No work per thread, every
          load goes to global memory (no register reuse), memory-bound.
        - Massive unroll (cells/thread > 64) → graduated penalty up to
          ``-1.0``. NVRTC compile time explodes on the unrolled body.
        - Under-filled launch (ctas < ``_TARGET_CTAS``) → linear penalty
          up to ``-1.0`` at 1 CTA, scaled by ``(target − ctas) / target``.
          The dominant failure mode on transformer-prefill GEMMs — a
          ``(1,128) @ (128,2048)`` matmul at ``FM·FN = 32`` lands ~32
          CTAs, ~10% of a 1-wave fill on a 100-SM GPU, and cuBLAS beats
          us 4× until SPLITK kicks more CTAs into the launch.
        - Well-filled launch (``_TARGET_CTAS`` ≤ ctas ≤ 2048) → ``+0.5``.
          Modest positive credit so SPLITK variants that lift an
          under-filled launch into the sweet spot can beat their
          SPLITK=1 sibling on prior alone (before any measurement).
        - CTA count above ``2048`` → graduated penalty up to ``-2.5``.
          A typical sm_120-class GPU has ~150 SMs running ~2 concurrent
          CTAs each; 2 k CTAs is ~7 waves, beyond which the command
          processor spends most of its time scheduling tiny waves and
          atomic fan-in on output writebacks serializes hard (see the
          ``BM=16, BN=16`` failure on ``(M=32, K=3584, N=18944)`` —
          2 k+ CTAs, 1 cell/thread, kernel runs >3 s).
        - Distance from 256 threads/CTA → up to ``-1.0`` baseline,
          doubled to ``-2.0`` when post-register-tile ``cells < 16``
          (thin per-thread compute can't amortize the launch + atomic
          overhead at a wide launch).
        - High SPLITK (``> 8``) → graduated penalty up to ``-1.0``.
          Cross-CTA L2 contention per output cell scales with SPLITK
          (each CTA in the split races on the same address). The earlier
          ``atomic_writes × threads × ctas`` formula counted each cell of
          the unrolled register tile as a separate atomic op, then hit
          the threshold on every realistic transformer matmul because
          ``M·N ≥ 256k`` always — anti-correlated with measured perf
          (penalised SPLITK=4 / 37 µs winner harder than SPLITK=1 /
          752 µs loser on q_proj s128). The SPLITK-only formula matches
          empirical L2 contention without firing on healthy splits.
        - Stages (smem staging) → ``+1.0``. Walks the body recursively
          (Stage stmts live inside the loop nest post-staging, not at
          the Tile's direct children).
        - Register tile fired (``FM * FN > 1``) → ``+1.0``.

        Pre-register-tile TileOps are scored on a *predicted* final thread
        count (assuming 008 will pick F to target ~256 threads when
        possible). Post-register-tile TileOps use the actual ``FM``,
        ``FN`` from knobs.
        """
        from math import prod  # noqa: PLC0415

        from deplodock.compiler.ir.tile.ir import Stage as _Stage  # noqa: PLC0415

        target_threads = 256
        # ~1 wave on a 100-SM GPU at 2-4 concurrent CTAs/SM. Used as the
        # boundary between under-filled (penalty) and well-filled (bonus)
        # launches.
        target_ctas = 256
        score = 0.0
        # ``self.body.iter()`` is recursive — by the time this TileOp
        # is scored, ``chunk_matmul_k`` etc. may have wrapped the Tile
        # in a K-outer ``Loop``, so the top-level iter doesn't see it.
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
            # Post-008: knobs carry the actual cell shape.
            final_threads = threads
            cells = max(1, int(self.knobs.get("FM", 1)) * int(self.knobs.get("FN", 1)))
        elif threads >= 1024:
            # Pre-008 but large enough that 008 will likely divide F to
            # target ~256 threads.
            final_threads = target_threads
            cells = max(1, threads // target_threads)
        else:
            # Pre-008 small tile — 008's heuristic gate will likely skip,
            # so the kernel emits 1 cell per thread.
            final_threads = threads
            cells = 1

        # Thread count penalty. Slope is gentle by default (``/target``),
        # but doubles when ``cells < 16`` post-register-tile — a wide
        # launch with thin per-thread compute can't amortize the launch
        # + atomic-fanin overhead. Observed on
        # ``matmul_add (1,128,2048)x(2048,2048)+r``: (threads=512,
        # cells=32) ran in 37 us while (threads=512, cells=8) hung the
        # bench. The conjunctive penalty distinguishes them.
        if final_threads < 32:
            score -= 2.0
        elif final_threads > 1024:
            score -= 2.0
        else:
            distance = abs(final_threads - target_threads)
            multiplier = 2.0 if (cells < 16 and "FM" in self.knobs) else 1.0
            score -= min(distance / target_threads * multiplier, 2.0)

        # Cells-per-thread penalty.
        if cells == 1:
            score -= 1.0
        elif cells > 64:
            score -= min((cells - 64) / 64.0, 1.0)

        # CTA count: under-fill BAD (the dominant failure mode), well-fill
        # GOOD (modest bonus), over-fill BAD again. The previous formula
        # only had a flat ``-0.5`` at ``ctas < 32`` plus a tail penalty
        # past 2 k; that left a huge under-filled-launch dead zone (32 ≤
        # ctas < 256) where the heuristic was indifferent between filling
        # the GPU and not — and on transformer-prefill GEMMs that's
        # exactly the regime where SPLITK matters.
        if ctas < target_ctas:
            score -= (target_ctas - ctas) / target_ctas
        elif ctas <= 2048:
            score += 0.5
        else:
            score -= min((ctas - 2048) / 4096.0, 2.5)

        # SPLITK contention penalty. Cross-CTA split-K races ``SPLITK``
        # CTAs on each output cell's L2 line; the previous formula was
        # ``atomic_writes × threads × ctas`` which counted each cell of
        # the FM·FN-unrolled register tile as a separate atomic and hit
        # the threshold on every realistic matmul (M·N ≥ 256 k always).
        # SPLITK ≤ 8 is fine on sm_120; past that L2 contention rises
        # super-linearly.
        splitk = int(self.knobs.get("SPLITK", 1))
        if splitk > 8:
            score -= min((splitk - 8) / 8.0, 1.0)

        # Bonuses. Stage stmts get added by ``007_stage_inputs`` deeper
        # inside the body (under the loop nest), not at ``tile.body``'s
        # direct children — walk the body recursively to find them.
        if any(isinstance(s, _Stage) for s in self.body.iter()):
            score += 1.0
        if "FM" in self.knobs:
            score += 1.0

        return score


# ---------------------------------------------------------------------------
# Tree walk — shared with Loop IR (drives off ``Stmt.nested``)
# ---------------------------------------------------------------------------


# Cooperative thread-block size — number of threads per CUDA block when a
# Tile uses BIND_THREAD axes from a cooperative strategy. Lives at this
# layer because cooperative strategies (cooperative-reduce, blockify)
# already commit to "this many threads cooperate" when they choose axis
# binds and tile sizes; materialization just consumes the choice.
# Overridable via ``DEPLODOCK_BN`` for sweeps (shared with the matmul
# path — mutually exclusive per kernel).
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
    "AsyncWait",
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

# Register Tile-IR stmts with the shared rewrite/simplify dispatch (Stage
# subtree via introspection, AsyncWait + Combine via dedicated handlers).
# Imported here — after class definitions — so the tile→stmt→tile cycle
# resolves cleanly: ``stmt.passes`` doesn't know about Tile-IR types, and
# ``tile.passes`` re-imports back into this module for the concrete classes.
from deplodock.compiler.ir.tile import passes as _passes  # noqa: E402, F401
