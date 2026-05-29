"""Promote BUFFERED / ASYNC StageBundle to TMA on sm_90+.

For each ``StageBundle`` with ``policy in {BUFFERED, ASYNC}`` (i.e.
ring-buffered cooperative-load OR cp.async) inside a
``SerialTile(serial_outer)``, switch ``policy`` to ``TMA`` (keeping
``buffer_count`` / ``phase`` / ``pipeline_depth``; ``swizzle`` stays
NONE) when every member ``Stage`` is TMA-eligible. Running on
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
    Stage,
    StageBundle,
    StagePolicy,
    SwizzleMode,
    TileOp,
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

# USE_TMA knob — hints ``(True, False)`` so the *first* candidate is True
# (the preferred policy on Hopper+). On sm < 9.0 the rule hands ``(False,)``
# alone to narrow, so the False candidate wins without the user ever pinning.
# When ``DEPLODOCK_USE_TMA=1`` is set, narrow returns ``(True,)`` even on
# unsupported arch (a pin is authoritative — the user knows what they're
# doing); the downstream eligibility checks then raise ``ValueError`` rather
# than the silent ``RuleSkipped`` fallback that used to mask the article's
# matmul A+B case. ``DEPLODOCK_USE_TMA=0`` skips the pass so
# ``060_use_async_copy`` promotes BUFFERED → ASYNC — useful when A/B-benching
# cp.async vs TMA on the same shape.
USE_TMA = Knob(
    "USE_TMA",
    KnobType.BOOL,
    hints=(True, False),
    help="Promote BUFFERED/ASYNC bundles to TMA. 1 = force (hard-fail on ineligibility), 0 = skip pass.",
)


def rewrite(ctx: Context, match: Match, root: Node) -> TileOp | None:
    # Arch-gated default: only Hopper+ offers TMA at all. Hand narrow the
    # full ``(True, False)`` hint tuple on supported arch, ``(False,)`` alone
    # otherwise — the first remaining candidate wins by priority order.
    candidates = USE_TMA.hints if ctx.compute_capability >= _MIN_CAPABILITY else (False,)
    use_tma = USE_TMA.narrow(candidates)[0]
    pinned = config.knob_raw(USE_TMA.name) is not None

    if not use_tma:
        if pinned:
            raise RuleSkipped("USE_TMA=0 pinned")
        if ctx.compute_capability < _MIN_CAPABILITY:
            raise RuleSkipped(f"TMA requires compute capability >= {_MIN_CAPABILITY}, got {ctx.compute_capability}")
        raise RuleSkipped("USE_TMA defaulted off")

    def _fail(msg: str) -> None:
        """Raise ``ValueError`` when explicitly pinned on, ``RuleSkipped`` otherwise."""
        if pinned:
            raise ValueError(f"DEPLODOCK_USE_TMA=1 but TMA cannot fire: {msg}")
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

    new_body, changed = _walk(body)
    if not changed:
        _fail("no BUFFERED/ASYNC StageBundle inside SerialTile(serial_outer) eligible for TMA")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            new_kouter_body, sub = _promote_in_kouter(s.body)
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
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _promote_in_kouter(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, StageBundle) and s.policy in _PROMOTABLE:
            out.append(_promote(s))
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


def _promote(bundle: StageBundle) -> StageBundle:
    return StageBundle(
        stages=_split_multi_source_stages(bundle.stages),
        body=bundle.body,
        policy=StagePolicy.TMA,
        buffer_count=bundle.buffer_count,
        phase=bundle.phase,
        pipeline_depth=bundle.pipeline_depth,
        swizzle=SwizzleMode.NONE,
    )


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
    if not dims or len(set(dims)) != len(dims) or list(dims) != sorted(dims):
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
    inner_extent = cache_axes[-1].extent.as_static()
    if (inner_extent * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    src_inner = src_shape[-1]
    if (src_inner * BYTES_PER_ELEM) % _TMA_ALIGN_BYTES != 0:
        return False
    if src_inner < inner_extent * 2:
        return False
    return True
