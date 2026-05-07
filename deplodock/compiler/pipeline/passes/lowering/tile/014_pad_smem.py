"""Detect smem bank conflicts in body Loads of staged buffers and apply
``+1`` padding to one cache-axis extent to break the conflict.

Problem
=======

After ``007_stage_inputs`` lays out a slab in shared memory, body
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

Analysis (closed-form, no per-thread evaluator)
================================================

For an affine Load index, decompose each dim into
``anchor + sum_v(coeff_v * Var(v))`` over the thread-axis vars (via
:func:`affine_form`). The flat smem address for a Load is
``sum_d (idx[d] * stride[d])``. Substituting the decomposition:

    flat(tid) = warp_const + sum_v (S_v * tid_v(tid))

where ``S_v = sum_d coeff_v_d * stride[d]`` is the per-thread-axis
contribution to the flat address. ``warp_const`` collects the anchor
parts plus everything that doesn't reference a thread-axis Var — it
shifts every thread's address by the same amount and so contributes a
constant offset to *all* banks; the bank distribution it produces is
independent of ``warp_const``.

So bank conflict counting reduces to: enumerate ``tid ∈ [0, 32)``,
decode ``(tid_v)`` per :func:`materialize_tile`'s tid-flatten scheme,
compute ``flat_v = sum_v S_v * tid_v``, distribute by ``flat_v % 32``,
count distinct addresses per bank. Broadcasts (multiple tids → same
``flat_v``) don't count as conflicts.

The pass:

1. For each ``Stage`` in the Tile body, find body Loads reading it.
2. Compute per-axis ``S_v`` once per Load via :func:`affine_form`.
3. If any Load is non-affine in the thread-axis vars, conservatively
   skip the Stage (we can't bound its conflict without the analyzer).
4. Try ``+1`` padding combinations (1-dim, then 2-dim) within
   ``_MAX_PADDED_SLAB_BYTES``; pick the first that drives every
   Load's max-way conflict to 1. Fall back to the best partial fix.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import BYTES_PER_ELEM, BufferedStage, Stage, TileOp, TmaBufferedStage
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    load_thread_axis_coeffs,
    loads_reading,
    max_bank_conflict,
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


def rewrite(graph: Graph, root: Node) -> Graph | None:
    # Diagnostic gate for visualizing pipeline behavior with this pass
    # disabled (e.g. smem-conflict before/after diffs).
    if os.environ.get("DEPLODOCK_DISABLE_PAD_SMEM") == "1":
        raise RuleSkipped("disabled via DEPLODOCK_DISABLE_PAD_SMEM=1")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)

    thread_axes = tuple(ba.axis for ba in tile.axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise RuleSkipped("Tile has no THREAD axes — no bank-conflict layout to pad")

    new_tile_body = _process_body(tile.body, thread_axes)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no Stage has a fixable bank conflict within slab budget")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_body(body: Body, thread_axes: tuple[Axis, ...]) -> Body:
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
            updated = _try_fix(s, loads, thread_axes)
            if updated is not None:
                new_body[i] = updated
                changed = True
        elif isinstance(s, Loop):
            inner = _process_body(s.body, thread_axes)
            if inner is not s.body and inner != s.body:
                new_body[i] = dc_replace(s, body=inner)
                changed = True
    return tuple(new_body) if changed else body


def _try_fix(stage: Stage, loads, thread_axes: tuple[Axis, ...]) -> Stage | None:
    n = len(stage.axes)
    base_extents = tuple(int(ax.extent) for ax in stage.axes)

    # Body Loads on a ``BufferedStage`` carry an extra leading phase index
    # added by ``010_double_buffer``; strip it for the analysis. Phase is
    # uniform across threads, so it doesn't affect bank distribution.
    per_load_coeffs = load_thread_axis_coeffs(
        loads,
        n,
        thread_axes,
        leading_phase_dim=isinstance(stage, BufferedStage),
    )
    if per_load_coeffs is None:
        return None

    no_pad = (0,) * n
    base_conflict = max_bank_conflict(per_load_coeffs, base_extents, thread_axes)
    if base_conflict <= 1:
        return None

    # Try +1 padding combinations, smallest-pad-first. Single-dim pads
    # (n options) before pair-dim pads (n*(n-1)/2 options). Stop at the
    # first conflict-free configuration.
    candidates: list[tuple[int, ...]] = []
    # 1-dim: prefer innermost-first (likely the right granularity).
    for dim in range(n - 1, -1, -1):
        pad = [0] * n
        pad[dim] = 1
        candidates.append(tuple(pad))
    # 2-dim: every unordered pair.
    for d1 in range(n):
        for d2 in range(d1 + 1, n):
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
        c = max_bank_conflict(per_load_coeffs, padded, thread_axes)
        if c <= 1:
            n_pad_dims = sum(1 for p in pad if p)
            logger.info(
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
        logger.warning(
            "Stage %s: %d-way bank conflict; partial fix → %d-way with pad %s (no perfect fix found)",
            stage.name,
            base_conflict,
            best_c,
            best_pad,
        )
        return dc_replace(stage, pad=best_pad)
    logger.warning("Stage %s: %d-way bank conflict; no padding fix found", stage.name, base_conflict)
    return None
