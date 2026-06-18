"""Promote BUFFERED / ASYNC StageBundle to TMA on sm_90+.

For each ``StageBundle`` with ``policy in {BUFFERED, ASYNC}`` (i.e.
ring-buffered cooperative-load OR cp.async) inside a
``SerialTile(serial_outer)``, switch ``policy`` to ``TMA`` (keeping
``buffer_count`` / ``phase`` / ``pipeline_depth``) when every member
``Stage`` is TMA-eligible. On the mma.sync path (detected structurally —
the body carries an ``AtomTile``) the per-``Source`` swizzle mode is
stamped (``_stamp_source_swizzle``: A=B64 / B=B128 from the inner-row
byte span). Swizzle is per-Source, not bundle-level, because A and B
share a bundle but need distinct modes; the materializer reads
``src.swizzle`` into each TmaDescriptor. Running on
BUFFERED directly (the post-``040_use_ring_buffers`` state) means the
rule fires before ``060_use_async_copy`` would promote to ASYNC —
otherwise the file ordering (050 < 060) leaves 050 staring at SYNC /
BUFFERED bundles with nothing to promote, since the cursor only
restarts the rule scan on Graph splices not Op rebinds.

Eligibility (per ``Source``):

- ``ctx.compute_capability >= (9, 0)`` (Hopper+).
- The source uses ``AffineAddressing`` (template addressing is a
  collapsed-reshape view that ``cuTensorMapEncodeTiled`` can't describe).
- ``addressing.dims`` is a strictly-increasing permutation; gap source
  dims (not swept by any cache axis) must be extent-1 singletons.
- Box-inner and source-inner extents both 16 B-aligned, with source
  inner at least 2× the box inner.
- Source rank ≤ 5 (TMA descriptor limit).
- Every collapsed per-dim box extent ≤ 256 (``cuTensorMapEncodeTiled``'s
  ``boxDim`` limit — violations only surface at launch, as CUresult=1).
- The source's innermost (fastest-varying) gmem dim is **static**. A
  symbolic OUTER/middle dim (dynamic ``seq_len``, the M-masked case) IS
  allowed: the descriptor's ``globalDim`` is encoded per launch from the
  runtime input shape (``program._prebuild_descriptors``) and TMA's hardware
  OOB handling zero-fills the masked overhang past the runtime extent. A
  symbolic *innermost* dim is rejected UNLESS it is a provable 16 B-aligned
  multiple (``_inner_stride_aligned`` — the demoted B/A cones pad their symbolic
  inner to a 64-multiple so the stride stays aligned at any ``seq_len``); a bare
  symbolic inner (an unpadded input) still stays on cp.async. Masked-K
  (symbolic-reduce) sources ALSO reach TMA now: the reduce overhang must read 0,
  which TMA's hardware OOB zero-fill delivers on the middle-K B operand (V),
  allocated at the real ``seq_len`` so its descriptor globalDim is ``seq_len`` —
  binding every overhang product to 0. ``040_use_ring_buffers`` rings a masked-K
  bundle only when ``tile_reaches_tma`` confirms this whole tile is TMA-eligible,
  so a masked-K bundle is never stranded on cp.async (it stays SYNC + ternary
  otherwise).

Additionally (per tile): the ``serial_outer`` K loop holding the bundles
must not be nested inside a serial loop with trip count > 1 — the
materializer initializes the ring mbarriers once at kernel entry, so a
re-entered pipeline starts from stale slot parities and deadlocks (see
``_reenters_pipeline``).

A multi-source bundle (e.g. matmul A+B emitted by ``020_stage_inputs`` as
``StageBundle(sources=(A, B))``) needs no splitting: the materializer
(``100_materialize_tile`` + ``_tma_groups``) emits one descriptor per
``Source`` directly — each source gets its own ``MbarrierArriveExpectTx +
TmaLoad`` pair issued from a distinct elected thread (``issuer_tid 0, 1,
…``), all arriving against the same group mbarrier whose ``arrive_count``
equals the source count. This is equivalent to the article's
1-thread-issues-both pattern via hardware mbarrier semantics: tx-bytes
from N arrives sum to the total, and per-source ``cp.async.bulk.tensor``
completions add to that total. Hoisted-compute bundles
(``StageBundle.compute is not None``) are emitted ``SYNC`` and never reach
this rule (the compute phase blocks TMA).

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
from deplodock.compiler.ir.expr import BinaryExpr, Literal
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AtomTile,
    SerialTile,
    SerialTileBase,
    Source,
    StageBundle,
    StagePolicy,
    TileOp,
    pick_swizzle_atom,
)
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

_MIN_CAPABILITY = (9, 0)
_MAX_RANK = 5
# Hardware limit of ``cuTensorMapEncodeTiled``: each ``boxDim[i]`` must be
# <= 256 elements. Violations are launch-time failures (CUresult=1), so the
# eligibility gate filters them while declining can still fall back to
# cp.async.
_TMA_MAX_BOX_DIM = 256
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

# cuTensorMapEncodeTiled's hard requirement: every global stride above the
# innermost dim must be a multiple of 16 bytes. The above-inner stride equals
# the inner extent in elements, so a symbolic inner extent is encode-safe iff
# it is a provable multiple of ``16 / elem_bytes`` elements.
_TMA_STRIDE_ALIGN_BYTES = 16


def _inner_stride_aligned(dim, elem_bytes: int) -> bool:  # noqa: ANN001
    """True when a *symbolic* inner extent ``dim`` is a provable constant multiple
    ``c`` whose byte width ``c * elem_bytes`` is a multiple of 16 — so the
    above-inner gmem stride stays 16 B-aligned at every runtime value and
    ``cuTensorMapEncodeTiled`` accepts the descriptor. Recognizes the padded
    ``round_up(N, P)`` Dim that ``_split_demoted._pad_inner_for_tma`` emits, whose
    top-level expr is ``X * Literal(P)`` (or the mirrored ``Literal(P) * X``). A
    bare symbolic dim (no constant factor) is unaligned and stays on cp.async."""
    expr = dim.expr
    if not (isinstance(expr, BinaryExpr) and expr.op == "*"):
        return False
    factor = None
    if isinstance(expr.right, Literal) and isinstance(expr.right.value, int):
        factor = expr.right.value
    elif isinstance(expr.left, Literal) and isinstance(expr.left.value, int):
        factor = expr.left.value
    return factor is not None and (factor * elem_bytes) % _TMA_STRIDE_ALIGN_BYTES == 0


def _shape_for_tma(node) -> tuple[int, ...] | None:  # noqa: ANN001
    """The node's output shape as static ints for the compile-time box checks:
    static dims as-is, symbolic dims at their Dim hint, ``None`` (cannot size the
    TMA box) for a symbolic dim with no hint."""
    dims: list[int] = []
    for d in node.output.shape:
        if d.is_static:
            dims.append(d.as_static())
        elif d.hint is not None:
            dims.append(int(d.hint))
        else:
            return None
    return tuple(dims)


def tile_reaches_tma(body: Body, graph_nodes: dict, ctx: Context) -> bool:
    """Pure predicate: would ``rewrite`` promote this tile body to TMA?

    Mirrors ``rewrite``'s all-or-nothing gate WITHOUT mutating — used by
    ``040_use_ring_buffers`` to decide whether to ring a masked-K bundle. A
    ringed masked-K is only correct if it reaches TMA (whose hardware OOB
    zero-fill replaces the SYNC ternary that cp.async / a synchronous
    double-buffer can't express), so 040 must not ring one this predicate
    rejects. The bundles here are still SYNC (040 runs before the promotion),
    but ``_source_eligible`` is policy-agnostic — it checks the Source's
    addressing / shape / alignment, which the ring doesn't change. The
    ``strict_slot_align`` ring-slot check keys off the post-ring
    ``buffer_count`` (the swizzle-stamped mma slabs masked-K rides never set
    it, so the SYNC ``buffer_count == 1`` reads identically)."""
    if ctx.compute_capability < _MIN_CAPABILITY:
        return False
    src_shapes: dict[str, tuple[int, ...]] = {}
    for nid, node in graph_nodes.items():
        shp = _shape_for_tma(node)
        if shp is None:
            return False
        src_shapes[nid] = shp
    inner_symbolic_bufs = {
        nid
        for nid, node in graph_nodes.items()
        if node.output.shape
        and not node.output.shape[-1].is_static
        and not _inner_stride_aligned(node.output.shape[-1], node.output.dtype.nbytes)
    }
    if _reenters_pipeline(body):
        return False
    swizzle = any(isinstance(s, AtomTile) for s in body.iter())
    dtype_bytes = {nid: node.output.dtype.nbytes for nid, node in graph_nodes.items()}
    saw_bundle = False
    for s in body.iter():
        if not isinstance(s, StageBundle) or s.policy == StagePolicy.TMA:
            continue
        saw_bundle = True
        strict_slot_align = not swizzle and s.buffer_count > 1
        if not _bundle_eligible(s, src_shapes, inner_symbolic_bufs, dtype_bytes, strict_slot_align):
            return False
    return saw_bundle


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
    off=False,
)


def rewrite(ctx: Context, match: Match, root: Node) -> TileOp | None:
    # Idempotence: the decision is recorded as the TMA knob (every path stamps
    # it now), so a re-scan of the rebound op skips here.
    if TMA.name in root.op.knobs:
        raise RuleSkipped("TMA already decided (idempotence via knob)")
    # Arch-gated default: only Hopper+ offers TMA at all. Hand narrow the
    # full ``(True, False)`` hint tuple on supported arch, ``(False,)`` alone
    # otherwise — the first remaining candidate wins by priority order.
    candidates = TMA.hints if ctx.compute_capability >= _MIN_CAPABILITY else (False,)
    use_tma = TMA.narrow(candidates)[0]
    pinned = config.knob_raw(TMA.name) is not None

    def _off() -> TileOp:
        """Record the declined decision: TMA=False, body unchanged. The realized
        config keeps a uniform knob set, and 060_use_async_copy still promotes
        the BUFFERED leftovers to cp.async."""
        return TileOp(body=root.op.body, name=root.op.name, knobs={**root.op.knobs, TMA.name: False})

    if not use_tma:
        return _off()

    def _decline(msg: str) -> TileOp:
        """TMA wanted but impossible: hard-fail when explicitly pinned on (the
        user forced it on infeasible ground), else record TMA=False."""
        if pinned:
            raise ValueError(f"DEPLODOCK_TMA=1 but TMA cannot fire: {msg}")
        return _off()

    # The TMA descriptor's globalDim is encoded per launch from the *runtime*
    # input array shape (``program._prebuild_descriptors`` reads ``arr.shape``),
    # not baked at compile time — so a symbolic source dim (dynamic ``seq_len``)
    # is fine for the descriptor. The box extents stay the static hint tile, so
    # the compile-time eligibility checks (box<=256, alignment, gap-dim) run
    # against the Dim *hint*; the launch uses the real extent and TMA's hardware
    # OOB handling zero-fills the masked overhang. A symbolic dim with no hint
    # can't size the box — decline.
    src_shapes: dict[str, tuple[int, ...]] = {}
    for nid, node in match.graph.nodes.items():
        shp = _shape_for_tma(node)
        if shp is None:
            return _decline(f"node {nid!r} has a symbolic dim with no hint — cannot size the TMA box")
        src_shapes[nid] = shp
    # Buffers whose innermost (fastest-varying) dim is symbolic can't be TMA
    # *sources*: every outer dim's global stride is ``∏(inner extents)·elem_size``,
    # which ``cuTensorMapEncodeTiled`` requires 16-byte-aligned, and a symbolic
    # innermost dim takes runtime values (31, 700) whose stride isn't 16-aligned
    # → ``CUresult=1`` at encode. A symbolic OUTER/middle dim (the common dynamic
    # ``seq_len``, M-masked) is safe: its stride is a product of static inner
    # extents and globalDim along it is unconstrained (TMA zero-fills the box
    # overhang past the runtime extent). Only the actually-staged sources are
    # checked (``_source_eligible``) — an in-kernel intermediate with a symbolic
    # inner dim (e.g. the SDPA scores in the o_proj kernel) is never staged, so
    # it must not veto the matmul operands' TMA path. N-masked inner sources stay
    # on cp.async.
    # A symbolic inner dim is fine for TMA when its extent is provably a multiple
    # of the 16 B alignment unit — the above-inner gmem stride is then 16 B-aligned
    # at every runtime value, so ``cuTensorMapEncodeTiled`` accepts it. The demoted
    # B-operand split pads its symbolic N inner up to ``_TMA_INNER_PAD`` exactly so
    # this holds (``_split_demoted._pad_inner_for_tma``); the padded overhang columns
    # are store-masked, so the descriptor can read them as ordinary in-bounds data.
    inner_symbolic_bufs = {
        nid
        for nid, node in match.graph.nodes.items()
        if node.output.shape
        and not node.output.shape[-1].is_static
        and not _inner_stride_aligned(node.output.shape[-1], node.output.dtype.nbytes)
    }

    body = root.op.body
    # Masked-K (symbolic-reduce) sources are no longer pinned off TMA. The reduce
    # overhang must read ZERO for the mma accumulation; TMA's hardware OOB
    # zero-fill provides exactly that on the *middle*-K B operand (V), whose
    # buffer is allocated at the real ``seq_len`` so its descriptor globalDim is
    # ``seq_len`` and coords past it zero-fill — binding the reduce overhang
    # product to 0 regardless of the A operand's (padded) overhang (which the
    # zero-init-reused scratch keeps finite, so ``finite × 0 = 0``). ``040`` only
    # rings a masked-K bundle when this whole tile is TMA-eligible (shared
    # ``tma_eligible.tile_reaches_tma``), so a ringed masked-K always reaches TMA
    # here — never a cp.async / synchronous-double-buffer state that can't zero the
    # overhang. An ineligible masked-K stays SYNC (``040`` declines the ring) and
    # keeps its ``_stage_expand`` ternary zero-fill.
    # A TMA ring pipeline must start from freshly-initialized mbarriers: the
    # materializer emits ``MbarrierInit`` once at kernel entry, and the
    # pipeline's parity schedule (steady-state wait at ``(k / RING) % 2``,
    # drain waits hardcoded to parity 0) assumes every slot begins at phase 0.
    # When the ``serial_outer`` K loop is itself nested inside a serial loop
    # with trip count > 1 (e.g. the FM register-cell loop of a fused
    # norm+matmul kernel, where the per-row norm reduction forces the K
    # pipeline under the cell loop), the second iteration re-enters the
    # pipeline with stale slot parities (K-tiles % RING != 0 leaves the slots
    # at mixed phase) and a parity wait eventually blocks on a phase that
    # never completes — a deterministic device hang (Qwen3-Embedding layer 0
    # ``k_linear_mean_reduce`` at FM=2 RING=3, 6 HungKernelErrors per tune).
    # Decline TMA for the whole tile; the cp.async path (commit/wait_prior —
    # no cross-iteration phase state) handles re-entry correctly.
    if _reenters_pipeline(body):
        return _decline(
            "serial_outer pipeline is nested inside a serial loop with trip count > 1; "
            "the once-initialized mbarrier ring would be re-entered at stale phase parity (device hang)"
        )
    # Per-source swizzle is stamped only on mma.sync kernels — their
    # explicit-ldmatrix consumer reads the swizzled slab with a matching
    # per-lane XOR; scalar kernels don't stage for ldmatrix, so their sources
    # stay NONE. Detect the tensor-core path structurally (the body carries an
    # ``AtomTile``, created by 010_partition_loops and not lowered until
    # kernel/005), rather than off the ``ATOM_KIND`` knob — the knob is a
    # tuning shadow, the AtomTile is the semantic source.
    swizzle = any(isinstance(s, AtomTile) for s in body.iter())
    # Per-buffer element byte width from the graph. The tile-stage ``Source``
    # has no ``dtype`` yet (``030_stamp_types`` is a kernel pass that runs AFTER
    # this), so ``src.dtype`` would fall back to the fp32 ``BYTES_PER_ELEM`` and
    # mis-pick the swizzle atom (a 32-elem fp16 inner = 64 B → B64, but read as
    # 4 B/elem = 128 B → wrongly B128, whose box the descriptor can't satisfy →
    # TMA deadlock). Read the true element size off the gmem node here.
    dtype_bytes = {nid: node.output.dtype.nbytes for nid, node in match.graph.nodes.items()}

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
        # A double-buffered (``buffer_count > 1``) NONE-swizzle bundle must keep
        # every ring slot 128 B-aligned; size that smem-slot check off the TRUE
        # element width (``strict_slot_align``) so a sub-128 B fp16 slab declines
        # to cp.async instead of emitting a misaligned ``cp.async.bulk.tensor``
        # (the #244 ``k_linear_mean_reduce`` wedge — the materializer's slot pad
        # uses the fp32 constant and under-pads fp16). Swizzled (mma) slabs align
        # via their swizzle atom and single-slot bundles sit at the aligned base,
        # so both keep the lenient fp32-constant check.
        strict_slot_align = not swizzle and s.buffer_count > 1
        if s.policy in _PROMOTABLE and not _bundle_eligible(s, src_shapes, inner_symbolic_bufs, dtype_bytes, strict_slot_align):
            return _decline(
                f"{s.policy.name} bundle on {list(s.local_decls())!r} not TMA-eligible; "
                "leaving the whole tile on cp.async (avoids mixed-mode pipeline deadlock)"
            )

    new_body, changed = _walk(body, swizzle=swizzle, dtype_bytes=dtype_bytes)
    already_tma = any(isinstance(s, StageBundle) and s.policy == StagePolicy.TMA for s in body.iter())
    if not changed and not already_tma:
        return _decline("no BUFFERED/ASYNC StageBundle inside SerialTile(serial_outer) eligible for TMA")
    return TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, TMA.name: True})


def _reenters_pipeline(body: Body, *, in_serial: bool = False) -> bool:
    """True if a promotable ``StageBundle``'s ``serial_outer`` K loop is nested
    inside a serial loop (``SerialTile`` / ``StridedTile``) with trip count > 1
    — i.e. the ring pipeline would run more than once over mbarriers the
    materializer initializes only once at kernel entry. A symbolic enclosing
    extent is treated as > 1 (conservative). Extent-1 cell loops (FM/FN = 1)
    stay eligible — the pipeline runs exactly once."""
    for s in body:
        if (
            in_serial
            and isinstance(s, SerialTile)
            and s.kind == "serial_outer"
            and any(isinstance(x, StageBundle) and x.policy in _PROMOTABLE for x in s.body)
        ):
            return True
        child_serial = in_serial
        if isinstance(s, SerialTileBase):
            ext = s.axis.extent
            if not ext.is_static or ext.as_static() > 1:
                child_serial = True
        if any(_reenters_pipeline(b, in_serial=child_serial) for b in s.nested()):
            return True
    return False


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


def _promote(bundle: StageBundle, *, swizzle: bool = False, dtype_bytes: dict[str, int] | None = None) -> StageBundle:
    # TMA emits one descriptor + one elected-thread ``arrive_expect_tx`` per
    # source directly (the materializer + ``_tma_groups.issuer_tid`` key per
    # ``Source.name``), so a multi-source bundle needs no splitting — the box
    # copies fan out across distinct elected threads from the source list.
    #
    # Swizzle is per-Source (A=B64, B=B128 — distinct modes on sources sharing
    # one bundle), stamped onto each Source here; the materializer reads
    # ``src.swizzle`` into each TmaDescriptor. There is no bundle-level mode.
    sources = bundle.sources
    if swizzle:
        sources = tuple(_stamp_source_swizzle(src, dtype_bytes or {}) for src in sources)
    return StageBundle(
        sources=sources,
        body=bundle.body,
        compute=bundle.compute,
        policy=StagePolicy.TMA,
        buffer_count=bundle.buffer_count,
        phase=bundle.phase,
        pipeline_depth=bundle.pipeline_depth,
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
    matches the materializer's ``box[-1] = full_box[max(kept)]`` so this mode
    pick agrees with the materializer's box reshape (a disagreement makes the
    descriptor's swizzle mode claim a width its box doesn't have → TMA copy
    deadlock)."""
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


def _stamp_source_swizzle(src: Source, dtype_bytes: dict[str, int]) -> Source:
    """Stamp one source's :class:`SwizzleMode` from its inner-row byte span.

    ``dtype_bytes`` maps gmem buffer name → element byte width (from the graph
    node, since the tile-stage ``Source.dtype`` isn't populated yet). Only
    affine transport sources can swizzle (the box reshape needs
    ``AffineAddressing``); template sources are left NONE."""
    if not isinstance(src.addressing, AffineAddressing):
        return src
    elem_bytes = dtype_bytes.get(src.buf, BYTES_PER_ELEM)
    _, mode = pick_swizzle_atom(_source_inner_elems(src), elem_bytes)
    return replace(src, swizzle=mode)


def _bundle_eligible(
    bundle: StageBundle,
    src_shapes: dict[str, tuple[int, ...]],
    inner_symbolic_bufs: set[str] = frozenset(),
    dtype_bytes: dict[str, int] | None = None,
    strict_slot_align: bool = False,
) -> bool:
    """A bundle is TMA-eligible iff every Source is."""
    return all(_source_eligible(src, src_shapes, inner_symbolic_bufs, dtype_bytes, strict_slot_align) for src in bundle.sources)


def _source_eligible(
    src: Source,
    src_shapes: dict[str, tuple[int, ...]],
    inner_symbolic_bufs: set[str] = frozenset(),
    dtype_bytes: dict[str, int] | None = None,
    strict_slot_align: bool = False,
) -> bool:
    if not isinstance(src.addressing, AffineAddressing):
        return False
    # A symbolic innermost gmem dim breaks the 16-byte global-stride alignment
    # ``cuTensorMapEncodeTiled`` requires at runtime (see the gate). Stay on cp.async.
    if src.buf in inner_symbolic_bufs:
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
    # COLLAPSED box extent per source dim — product of the (cache_extent ×
    # block) of every cache axis mapping to that dim. Mirrors the
    # materializer's ``box_per_dim`` collapse so a tile with cache
    # ``(a3=32, a5=4)`` both mapping to dim 1 gets the right 128-cell extent
    # (vs the legacy ``cache_axes[-1].extent`` = 4 that always failed the
    # 128 B alignment gate for collapse cases).
    #
    # The per-axis ``AffineAddressing.block`` multiplier encodes per-cell
    # strides (e.g. ``atom_n = 8`` on the N side of the m16n8k16 atom) —
    # without it, warp-tier MMA slabs whose cache axes are warp/cell-granular
    # (``WN × FN`` per the inner dim) report ``inner_extent = WN·FN`` of
    # ~4-8 elements when the actual slab inner width is
    # ``WN·FN·atom_n`` of 64-128 elements. Pre-fix the warp tier was
    # silently TMA-ineligible at every shape, so the picker could only fall
    # back to a slower non-TMA staging path.
    block = src.addressing.block
    box_per_dim: dict[int, int] = {}
    for i, (d, ax) in enumerate(zip(dims, cache_axes, strict=True)):
        b = block[i] if block else 1
        box_per_dim[d] = box_per_dim.get(d, 1) * ax.extent.as_static() * b
    # ``cuTensorMapEncodeTiled`` enforces ``boxDim[i] <= 256``; an oversized
    # collapsed box compiles fine and only fails at LAUNCH (the descriptor
    # embeds the device pointer) with ``CUDA_ERROR_INVALID_VALUE`` — so it
    # must be filtered here, where declining falls back to cp.async. Hit by
    # the scalar register-tile matmul at ``BM·FM > 256`` (e.g. the
    # Qwen3-Embedding down_proj tuned up to BM·FM = 768; 16 launch-time
    # bench_fails per tune). The swizzle-atom reshape later *splits* the
    # inner dim smaller, so this cap is conservative there — acceptable: the
    # autotune space tops out at exactly 256 per dim, and the fallback is
    # cp.async, not a crash.
    if any(ext > _TMA_MAX_BOX_DIM for ext in box_per_dim.values()):
        return False
    # Inner extent for alignment checks is the collapsed box width at the
    # last source dim.
    inner_extent = box_per_dim[dims[-1]]
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    # Ring-slot 128 B alignment, at the TRUE element width. A double-buffered
    # NONE-swizzle bundle (``strict_slot_align`` — set in the rewrite loop) keeps
    # ``buffer_count`` rotating slots at ``slot * inner_box`` byte offsets from a
    # 128 B-aligned base; every slot stays aligned only if the whole per-slot box
    # footprint is a 128 B multiple. The materializer's slot pad sizes that
    # threshold off the fp32 ``BYTES_PER_ELEM``, so a pure-reduction fp16 slab
    # whose box is a single 32-elem axis (64 B) reads as already-aligned, stays
    # UNPADDED, and the second ring slot lands at a 64 B offset →
    # ``cp.async.bulk.tensor`` faults with ``CUDA_ERROR_MISALIGNED_ADDRESS`` (the
    # #244 ``k_linear_mean_reduce`` wedge). Decline to cp.async (no 128 B rule)
    # for that case. A matmul slab whose box collapses several axes (e.g.
    # ``BK·BN·FN``) is already a 128 B multiple and stays on TMA; swizzled (mma)
    # slabs align via their swizzle atom and single-slot bundles sit at the
    # aligned base, so neither sets ``strict_slot_align``.
    if strict_slot_align:
        slot_elems = 1
        for ext in box_per_dim.values():
            slot_elems *= ext
        elem_bytes = (dtype_bytes or {}).get(src.buf, BYTES_PER_ELEM)
        if (slot_elems * elem_bytes) % _TMA_ALIGN_BYTES != 0:
            return False
    if src_inner < inner_extent * 2:
        return False
    return True
