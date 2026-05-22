"""Detect smem bank conflicts in body Loads of staged buffers and apply
``+1`` padding to one cache-axis extent to break the conflict.

Problem
=======

After ``002_stage_inputs`` lays out a slab in shared memory, body
Loads read it using thread-decoded coords. When the slab's per-thread-
axis stride is a multiple of 32 floats (the smem bank count × 4 bytes
/ 4 bytes per float), every thread in a warp hits the same bank — a
32-way conflict that serializes 32 LDS instructions into 32 single-
bank reads.

Concretely for a W slab with shape ``(8, 4, 64, 2)`` and strides
``(512, 128, 2, 1)``: within a warp ``(a3, a4)`` cycles through
``[0,8)×[0,4)``, addresses span ``a3*512 + a4*128``, mod 32 = 0 for
every thread. 32-way conflict.

Fix
===

Add ``+1`` padding to one cache-axis extent. The smem allocation grows
by one row, every higher-stride row shifts by one float, and the per-
thread address mod 32 changes from a uniform 0 to a sequence that
touches all 32 banks.

Analysis
========

Per-lane bank distribution comes from the shared kernel
:func:`deplodock.compiler.diagnostics.bank_conflicts.lane_bank_distribution`,
which enumerates the 32 warp lanes, decodes each lane's thread-axis
values per ``materialize_tile``'s flatten scheme, evaluates the Load
index against the smem strides (folded with any pad), and tallies
distinct addresses per bank. Block axes and outer loop axes are
auto-zero-bound — bank distribution is invariant to additive
warp-uniform offsets, so the choice of value doesn't change ``max_way``.

The pass:

1. For each ``Stage`` in the Tile body, find body Loads reading it.
2. Compute the worst-case ``max_way`` across all body Loads at the
   current (un-padded) extents.
3. If any Load can't be evaluated (rank mismatch with ``Stage.axes``),
   conservatively skip the Stage.
4. Try ``+1`` padding combinations (1-dim, then 2-dim) within
   ``_MAX_PADDED_SLAB_BYTES``; pick the first that drives every
   Load's max-way conflict to 1. Fall back to the best partial fix.
"""

from __future__ import annotations

import logging
from dataclasses import replace as dc_replace

from deplodock.compiler.diagnostics.bank_conflicts import lane_bank_distribution
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, BufferedStage, Stage, TileOp, TmaBufferedStage
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    loads_reading,
    single_tile,
)

logger = logging.getLogger(__name__)

PATTERN = [Pattern("root", TileOp)]

# Per-Stage padded-slab budget. The static-smem cap on consumer Ada/Hopper
# is 48 KB. Stages get double-buffered downstream (×2) and typical matmul
# kernels have ≥ 2 sibling Stages, so per-Stage pre-DB budget is roughly
# cap / 4 → 12 KB. We give a slightly larger headroom (20 KB) so single-
# large-Stage kernels still benefit, and accept that a matmul with two
# near-equal Stages may miss the perfect-fix candidate and fall back to
# the partial-fix pad.
_MAX_PADDED_SLAB_BYTES = 20 * 1024


def rewrite(root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("no Stage benefited from padding")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    if not tile.thread_axes:
        raise RuleSkipped("Tile has no THREAD axes — no bank-conflict layout to pad")

    new_tile_body = _process_body(tile.body, tile)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no Stage has a fixable bank conflict within slab budget")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_body(body: Body, tile: Tile) -> Body:
    """Walk body and any free-Loop scopes; pad each Stage that has a
    fixable conflict. Returns the new body (or original if no changes)."""
    new_body: list[Stmt] = list(body)
    changed = False
    for i, s in enumerate(body):
        # Skip TmaBufferedStage: TMA box copies write rows back-to-back at
        # the cache extent and use hardware swizzling for bank avoidance,
        # so ``+1`` padding would mis-align body Loads' stride with the
        # box write. The TmaBufferedStage class also asserts ``pad`` is
        # empty.
        if isinstance(s, Stage) and not isinstance(s, TmaBufferedStage) and not (s.pad and any(s.pad)):
            loads = loads_reading(body, s.name)
            if not loads:
                continue
            updated = _try_fix(s, loads, tile)
            if updated is not None:
                new_body[i] = updated
                changed = True
        elif isinstance(s, Loop):
            inner = _process_body(s.body, tile)
            if inner is not s.body and inner != s.body:
                new_body[i] = dc_replace(s, body=inner)
                changed = True
    return tuple(new_body) if changed else body


def _max_conflict(loads: list[Load], extents: tuple[int, ...], leading_phase: bool, tile: Tile) -> int | None:
    """Worst-case ``max_way`` across body ``loads`` of one Stage at the
    given hypothetical ``extents``. Returns ``None`` if any Load can't
    be evaluated (rank mismatch / unbound non-thread var that isn't
    auto-zero-bindable) — caller skips conservatively."""
    worst = 1
    for ld in loads:
        cache_idx = ld.index[1:] if leading_phase else ld.index
        if len(cache_idx) != len(extents):
            return None
        dist = lane_bank_distribution(tuple(cache_idx), extents, tile.thread_axes)
        if dist is None:
            return None
        worst = max(worst, dist.max_way)
    return worst


def _try_fix(stage: Stage, loads: list[Load], tile: Tile) -> Stage | None:
    n = len(stage.axes)
    base_extents = tuple(int(ax.extent) for ax in stage.axes)
    leading_phase = isinstance(stage, BufferedStage)

    base_conflict = _max_conflict(loads, base_extents, leading_phase, tile)
    if base_conflict is None:
        return None
    if base_conflict <= 1:
        return None

    # Try +1 padding combinations, smallest-pad-first. Single-dim pads
    # (n options) before pair-dim pads (n*(n-1)/2 options). Stop at the
    # first conflict-free configuration.
    no_pad = (0,) * n
    candidates: list[tuple[int, ...]] = []
    # 1-dim: prefer innermost-first (the natural granularity for bank-
    # conflict breakage)... BUT skip the innermost dim when its extent
    # is ≤ vec-load width (4 for fp32 float4 / 8 for fp16 half8). The
    # materializer emits a vectorized load over consecutive innermost-
    # dim cells in those cases, and a +1 pad on that dim breaks the
    # 16-byte alignment of the float4. Pad an outer dim instead.
    _VEC_INNERMOST_THRESHOLD = 8  # covers fp32 float4 + fp16 half8
    skip_innermost = n >= 1 and base_extents[n - 1] <= _VEC_INNERMOST_THRESHOLD
    inner_range = range(n - 1, -1, -1) if not skip_innermost else range(n - 2, -1, -1)
    for dim in inner_range:
        pad = [0] * n
        pad[dim] = 1
        candidates.append(tuple(pad))
    # 2-dim: every unordered pair — skip pairs that include the
    # innermost vec-load dim for the same alignment reason.
    for d1 in range(n):
        for d2 in range(d1 + 1, n):
            if skip_innermost and (d1 == n - 1 or d2 == n - 1):
                continue
            pad = [0] * n
            pad[d1] = 1
            pad[d2] = 1
            candidates.append(tuple(pad))

    best_pad = no_pad
    best_c = base_conflict
    for pad in candidates:
        padded = tuple(e + p for e, p in zip(base_extents, pad, strict=True))
        slab_bytes = BYTES_PER_ELEM
        for e in padded:
            slab_bytes *= e
        if slab_bytes > _MAX_PADDED_SLAB_BYTES:
            continue
        c = _max_conflict(loads, padded, leading_phase, tile)
        if c is None:
            continue
        if c <= 1:
            n_pad_dims = sum(1 for p in pad if p)
            logger.debug(
                "Stage %s: %d-way bank conflict → 1-way after %d-dim pad %s",
                stage.name,
                base_conflict,
                n_pad_dims,
                pad,
            )
            return dc_replace(stage, pad=pad)
        if c < best_c:
            best_c = c
            best_pad = pad

    if best_c < base_conflict:
        logger.debug(
            "Stage %s: %d-way bank conflict; partial fix → %d-way with pad %s (no perfect fix found)",
            stage.name,
            base_conflict,
            best_c,
            best_pad,
        )
        return dc_replace(stage, pad=best_pad)
    logger.debug("Stage %s: %d-way bank conflict; no padding fix found", stage.name, base_conflict)
    return None
