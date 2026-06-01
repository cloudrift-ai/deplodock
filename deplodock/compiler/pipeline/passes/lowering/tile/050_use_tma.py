"""Promote BUFFERED / ASYNC StageBundle to TMA on sm_90+.

For each ``StageBundle`` with ``policy in {BUFFERED, ASYNC}`` (i.e.
ring-buffered cooperative-load OR cp.async) inside a
``SerialTile(serial_outer)``, switch ``policy`` to ``TMA`` (keeping
``buffer_count`` / ``phase`` / ``pipeline_depth``; the *bundle*
``swizzle`` stays NONE) when every member ``Stage`` is TMA-eligible.
On the mma.sync atom (``_is_mma_sync_kernel``) the per-``Source``
swizzle mode is stamped instead (``_stamp_source_swizzle``: A=B64 /
B=B128 from the inner-row byte span) — the bundle stays NONE because A
and B share it but need distinct modes; the materializer reads
``src.swizzle`` into each TmaDescriptor (S2 of
plans/mma-sync-smem-swizzle.md). Running on
BUFFERED directly (the post-``040_use_ring_buffers`` state) means the
rule fires before ``060_use_async_copy`` would promote to ASYNC —
otherwise the file ordering (050 < 060) leaves 050 staring at SYNC /
BUFFERED bundles with nothing to promote, since the cursor only
restarts the rule scan on Graph splices not Op rebinds.

Eligibility (per ``Source`` after multi-source split):

- ``ctx.compute_capability >= (9, 0)`` (Hopper+).
- The source uses ``AffineAddressing`` (template addressing is a
  collapsed-reshape view that ``cuTensorMapEncodeTiled`` can't describe).
- ``addressing.dims`` is a strictly-increasing permutation; gap source
  dims (not swept by any cache axis) must be extent-1 singletons.
- Box-inner and source-inner extents both 16 B-aligned, with source
  inner at least 2× the box inner.
- Source rank ≤ 5 (TMA descriptor limit).

A multi-source ``Stage`` (e.g. matmul A+B emitted by
``020_stage_inputs`` as a single ``Stage(sources=(A, B))``) is **split**
into N single-source ``Stage``s under the same ``StageBundle`` before
the eligibility check. The downstream materializer (``100_materialize_tile``
+ ``_tma_groups``) already handles N single-source stages: each source
gets its own ``MbarrierArriveExpectTx + TmaLoad`` pair issued from a
distinct elected thread (``issuer_tid 0, 1, …``), all arriving against
the same group mbarrier whose ``arrive_count`` equals the source count.
This is equivalent to the article's 1-thread-issues-both pattern via
hardware mbarrier semantics: tx-bytes from N arrives sum to the total,
and per-source ``cp.async.bulk.tensor`` completions add to that total.
Hoisted-compute Stages (``Stage.compute is not None``) can't split —
their producer body is shared across sources — and stay on cp.async.

The pass is **all-or-nothing per tile**: if any ASYNC bundle in the
tile body is ineligible, leave the whole tile on cp.async (avoids
mixed-mode pipeline deadlock).

Idempotence: TMA-policy bundles are left alone.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    SerialTile,
    Source,
    Stage,
    StageBundle,
    StagePolicy,
    SwizzleMode,
    TileOp,
    pick_swizzle_atom,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
# NVIDIA's recommended TMA-destination alignment for box copies. TMA's hardware
# minimum is 16 B, but ``100_materialize_tile`` pads each ring slot's inner
# extent up to 128 B to keep ``cp.async.bulk.tensor`` from issuing misaligned
# stores under double-buffering. That row-pad only round-trips correctly when
# the consumer's smem reads use the padded stride — for TMA, the box is written
# contiguous (no inter-row gap), so a smaller-than-128 B inner extent would
# silently corrupt reads. Filter those out at the eligibility stage so the
# pad never fires on TMA bundles.
_TMA_ALIGN_BYTES = 128

# Policies this rule will promote to TMA. SYNC bundles (no ring buffer)
# are excluded — the materializer's TMA emit path assumes a buffer-count
# >= 1 phase dim on the smem slab; promoting SYNC would break that.
_PROMOTABLE = frozenset({StagePolicy.BUFFERED, StagePolicy.ASYNC})

# TMA knob — hints ``(True, False)`` so the *first* candidate is True
# (the preferred policy on Hopper+). On sm < 9.0 the rule hands ``(False,)``
# alone to narrow, so the False candidate wins without the user ever pinning.
# When ``DEPLODOCK_TMA=1`` is set, narrow returns ``(True,)`` even on
# unsupported arch (a pin is authoritative — the user knows what they're
# doing); the downstream eligibility checks then raise ``ValueError`` rather
# than the silent ``RuleSkipped`` fallback that used to mask the article's
# matmul A+B case. ``DEPLODOCK_TMA=0`` skips the pass so
# ``060_use_async_copy`` promotes BUFFERED → ASYNC — useful when A/B-benching
# cp.async vs TMA on the same shape.
TMA = Knob(
    "TMA",
    KnobType.BOOL,
    hints=(True, False),
    help="Promote BUFFERED/ASYNC bundles to TMA. 1 = force (hard-fail on ineligibility), 0 = skip pass.",
)

# TMA hardware-swizzle opt-in. Hints ``(False, True)`` so the default (first
# candidate) is off — the swizzle only fires when a stage's inner box-dim byte
# size matches a swizzle width AND the user pins ``DEPLODOCK_TMA_SWIZZLE=1``.
# The SwizzleMode selection (currently stubbed to NONE) lives in this rule, so
# the knob is declared here even though no live code reads it yet.
TMA_SWIZZLE = Knob(
    "TMA_SWIZZLE",
    KnobType.BOOL,
    hints=(False, True),
    help="Opt in to TMA hardware-swizzle modes (SWIZZLE_{128,64,32}B). Off by default.",
)


def rewrite(ctx: Context, match: Match, root: Node) -> TileOp | None:
    # Arch-gated default: only Hopper+ offers TMA at all. Hand narrow the
    # full ``(True, False)`` hint tuple on supported arch, ``(False,)`` alone
    # otherwise — the first remaining candidate wins by priority order.
    candidates = TMA.hints if ctx.compute_capability >= _MIN_CAPABILITY else (False,)
    use_tma = TMA.narrow(candidates)[0]
    pinned = config.knob_raw(TMA.name) is not None

    if not use_tma:
        if pinned:
            raise RuleSkipped("TMA=0 pinned")
        if ctx.compute_capability < _MIN_CAPABILITY:
            raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")
        raise RuleSkipped("TMA defaulted off")

    def _fail(msg: str) -> None:
        """Raise ``ValueError`` when explicitly pinned on, ``RuleSkipped`` otherwise."""
        if pinned:
            raise ValueError(f"DEPLODOCK_TMA=1 but TMA cannot fire: {msg}")
        raise RuleSkipped(msg)

    # TMA descriptors bake the source shape statically into the cuTensorMap; bail
    # out cleanly if any input shape carries a symbolic dim.
    for nid, node in match.graph.nodes.items():
        for d in node.output.shape:
            if not d.is_static:
                _fail(f"TMA requires static shapes; node {nid!r} has symbolic dim {d!r}")
    src_shapes = {nid: tuple(d.as_static() for d in node.output.shape) for nid, node in match.graph.nodes.items()}

    body = root.op.body
    # All-or-nothing per tile (see module docstring): if any
    # promotable-policy bundle is not TMA-eligible, leave the whole
    # tile on cp.async — 060_use_async_copy promotes the BUFFERED
    # leftovers to ASYNC. Mixed TMA + cp.async inside one pipelined
    # K-loop deadlocks the mbarrier scheme, so we'd rather skip TMA
    # for the whole tile than partially promote.
    for s in body.iter():
        if not isinstance(s, StageBundle):
            continue
        if s.policy == StagePolicy.TMA:
            continue
        if s.policy in _PROMOTABLE and not _bundle_eligible(s, src_shapes):
            _fail(
                f"{s.policy.name} bundle on {list(s.local_decls())!r} not TMA-eligible; "
                "leaving the whole tile on cp.async (avoids mixed-mode pipeline deadlock)"
            )

    # Per-source swizzle (S2 of plans/mma-sync-smem-swizzle.md) is stamped
    # only on mma.sync kernels — their explicit-ldmatrix consumer can read a
    # swizzled slab; WMMA's opaque ``load_matrix_sync`` cannot, so its sources
    # stay NONE. The atom kind rides on the TileOp knobs.
    swizzle = _is_mma_sync_kernel(root.op.knobs.get("ATOM_KIND"))
    # Per-buffer element byte width from the graph. The tile-stage ``Source``
    # has no ``dtype`` yet (``030_stamp_types`` is a kernel pass that runs AFTER
    # this), so ``src.dtype`` would fall back to the fp32 ``BYTES_PER_ELEM`` and
    # mis-pick the swizzle atom (a 32-elem fp16 inner = 64 B → B64, but read as
    # 4 B/elem = 128 B → wrongly B128, whose box the descriptor can't satisfy →
    # TMA deadlock). Read the true element size off the gmem node here.
    dtype_bytes = {nid: node.output.dtype.nbytes for nid, node in match.graph.nodes.items()}
    new_body, changed = _walk(body, swizzle=swizzle, dtype_bytes=dtype_bytes)
    if not changed:
        _fail("no BUFFERED/ASYNC StageBundle inside SerialTile(serial_outer) eligible for TMA")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _is_mma_sync_kernel(atom_kind: str | None) -> bool:
    """True iff ``atom_kind`` names an ``instruction="mma_sync"`` AtomSpec."""
    if not atom_kind:
        return False
    from deplodock.compiler.pipeline.passes.lowering.tile._atom import ATOM_REGISTRY, atom_spec  # noqa: PLC0415

    return atom_kind in ATOM_REGISTRY and atom_spec(atom_kind).instruction == "mma_sync"


def _walk(body: Body, *, swizzle: bool = False, dtype_bytes: dict[str, int] | None = None) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            new_kouter_body, sub = _promote_in_kouter(s.body, swizzle=swizzle, dtype_bytes=dtype_bytes)
            if sub:
                s = SerialTile(axis=s.axis, body=new_kouter_body, kind=s.kind, unroll=s.unroll)
                changed = True
            out.append(s)
            continue
        nested = s.nested()
        if nested:
            new_bodies = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b, swizzle=swizzle, dtype_bytes=dtype_bytes)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _promote_in_kouter(body: Body, *, swizzle: bool = False, dtype_bytes: dict[str, int] | None = None) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, StageBundle) and s.policy in _PROMOTABLE:
            out.append(_promote(s, swizzle=swizzle, dtype_bytes=dtype_bytes))
            changed = True
        else:
            out.append(s)
    return Body(tuple(out)), changed


def _split_multi_source_stages(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    """Replace each multi-source transport Stage with N single-source Stages.

    ``020_stage_inputs`` packs every load-eligible buffer of a Tile into one
    ``Stage(sources=(A, B, …))``. TMA's per-source ``cp.async.bulk.tensor``
    needs one elected thread + one ``arrive_expect_tx`` per source descriptor,
    not per Stage — splitting here lets the existing N-stages-per-bundle
    materializer ( ``100_materialize_tile`` + ``_tma_groups.issuer_tid``)
    distribute the arrives across distinct elected threads without any
    materializer changes.

    Hoisted-compute Stages (``compute is not None``) are passed through
    intact — their producer body reads from sibling-stage smem and can't be
    split per source.
    """
    out: list[Stage] = []
    for stage in stages:
        if stage.compute is not None or len(stage.sources) <= 1:
            out.append(stage)
            continue
        for src in stage.sources:
            out.append(replace(stage, sources=(src,)))
    return tuple(out)


def _promote(bundle: StageBundle, *, swizzle: bool = False, dtype_bytes: dict[str, int] | None = None) -> StageBundle:
    stages = _split_multi_source_stages(bundle.stages)
    if swizzle:
        stages = tuple(_stamp_source_swizzle(stage, dtype_bytes or {}) for stage in stages)
    return StageBundle(
        stages=stages,
        body=bundle.body,
        policy=StagePolicy.TMA,
        buffer_count=bundle.buffer_count,
        phase=bundle.phase,
        pipeline_depth=bundle.pipeline_depth,
        # Bundle-level swizzle stays NONE — the mode is per-Source (A=B64,
        # B=B128) because A and B share this bundle (S2 of
        # plans/mma-sync-smem-swizzle.md). The materializer reads
        # ``src.swizzle`` into each TmaDescriptor.
        swizzle=SwizzleMode.NONE,
    )


def _source_inner_elems(src: Source) -> int:
    """Collapsed inner-row element span of ``src``'s slab — the product of
    ``(cache_extent × block)`` over every cache axis mapping to the innermost
    *source* dim.

    The inner source dim is ``max(dims)`` — the highest (contiguous, fastest-
    varying) source-buffer dim swept by any cache axis — NOT ``dims[-1]``. The
    cache axes are stored in slab-layout order, which for the mma.sync A
    operand puts the K (contiguous) axis FIRST and the M axes last, so
    ``dims = (1, 0, 0)`` and ``dims[-1]`` would pick the M dim (128 elems →
    wrongly B128) instead of K (32 elems → B64). Keying on ``max(dims)``
    matches the materializer's ``box[-1] = full_box[max(kept)]`` so S2's mode
    pick agrees with S1's box reshape (a disagreement makes the descriptor's
    swizzle mode claim a width its box doesn't have → TMA copy deadlock)."""
    addressing = src.addressing
    assert isinstance(addressing, AffineAddressing)
    dims = addressing.dims
    block = addressing.block
    inner_dim = max(dims)
    inner = 1
    for i, (d, ax) in enumerate(zip(dims, src.cache_axes, strict=True)):
        if d == inner_dim:
            b = block[i] if block else 1
            inner *= ax.extent.as_static() * b
    return inner


def _stamp_source_swizzle(stage: Stage, dtype_bytes: dict[str, int]) -> Stage:
    """Stamp each source's :class:`SwizzleMode` from its inner-row byte span.

    ``dtype_bytes`` maps gmem buffer name → element byte width (from the graph
    node, since the tile-stage ``Source.dtype`` isn't populated yet). Only
    affine single-source transport stages can swizzle (the box reshape needs
    ``AffineAddressing``); compute / template sources are left NONE."""
    if stage.compute is not None:
        return stage
    new_sources = []
    for src in stage.sources:
        if not isinstance(src.addressing, AffineAddressing):
            new_sources.append(src)
            continue
        elem_bytes = dtype_bytes.get(src.buf, BYTES_PER_ELEM)
        _, mode = pick_swizzle_atom(_source_inner_elems(src), elem_bytes)
        new_sources.append(replace(src, swizzle=mode))
    return stage.replace_sources(tuple(new_sources))


def _bundle_eligible(bundle: StageBundle, src_shapes: dict[str, tuple[int, ...]]) -> bool:
    """A bundle is TMA-eligible iff every (post-split) member Stage is."""
    for stage in _split_multi_source_stages(bundle.stages):
        if not _stage_eligible(stage, src_shapes):
            return False
    return True


def _stage_eligible(stage: Stage, src_shapes: dict[str, tuple[int, ...]]) -> bool:
    # Hoisted-compute stages are never TMA-eligible (their producer body
    # reads from sibling-stage smem, not gmem). The multi-source case has
    # already been split by ``_split_multi_source_stages`` before this
    # check fires — we only see single-source transport Stages here.
    if stage.compute is not None or len(stage.sources) != 1:
        return False
    (src,) = stage.sources
    if not isinstance(src.addressing, AffineAddressing):
        return False
    cache_axes = src.cache_axes
    if not cache_axes or len(cache_axes) > _MAX_RANK:
        return False
    src_shape = src_shapes.get(src.buf)
    if not src_shape or len(src_shape) > _MAX_RANK or len(src.origin) > _MAX_RANK:
        return False
    dims = src.addressing.dims
    # Allow consecutive duplicates: cache axes ``(a_thread, a_reg)`` mapping to
    # the same source dim ``d`` represent a collapse — the materializer's
    # ``box_per_dim[d] *= ax.extent`` already produces the correct contiguous
    # TMA box (e.g. dims ``(0, 1, 1)`` with extents ``(32, 32, 4)`` → box
    # ``(32, 128)``). The legacy strictly-unique check rejected this and forced
    # FN>1 / FM>1 matmul tiles onto cp.async even when the composite was
    # exactly the natural 2D box TMA describes.
    if not dims or list(dims) != sorted(dims):
        return False
    src_rank = len(src_shape)
    if dims[-1] != src_rank - 1:
        return False
    # Gap source dims (not swept by any cache axis) must be extent-1
    # singletons — the materializer drops those from the descriptor.
    dims_set = set(dims)
    for d in range(dims[0], src_rank):
        if d not in dims_set and src_shape[d] != 1:
            return False
    # Inner extent for alignment checks is the COLLAPSED box width at the
    # last source dim — product of the (cache_extent × block) of every cache
    # axis mapping to ``dims[-1]``. Mirrors the materializer's ``box_per_dim``
    # collapse so a tile with cache ``(a3=32, a5=4)`` both mapping to dim 1
    # gets the right 128-cell inner extent (vs the legacy
    # ``cache_axes[-1].extent`` = 4 that always failed the 128 B alignment
    # gate for collapse cases).
    #
    # The per-axis ``AffineAddressing.block`` multiplier encodes per-cell
    # strides (e.g. ``atom_n = 16`` for square WMMA on the N side) — without
    # it, warp-tier MMA slabs whose cache axes are warp/cell-granular
    # (``WN × FN`` per the inner dim) report ``inner_extent = WN·FN`` of
    # ~4-8 elements when the actual slab inner width is
    # ``WN·FN·atom_n`` of 64-128 elements. Pre-fix the warp tier was
    # silently TMA-ineligible at every shape; the gmem-direct WMMA was the
    # only path the picker could land on.
    inner_dim = dims[-1]
    block = src.addressing.block
    inner_extent = 1
    for i, (d, ax) in enumerate(zip(dims, cache_axes, strict=True)):
        if d == inner_dim:
            b = block[i] if block else 1
            inner_extent *= ax.extent.as_static() * b
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    if src_inner < inner_extent * 2:
        return False
    return True
