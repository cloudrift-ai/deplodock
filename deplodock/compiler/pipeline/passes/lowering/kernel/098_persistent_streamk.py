"""Persistent-CTA Stream-K matmul scheduling (atomic variant) — wrapper swap.

Forks a matmul ``GridTile`` into a ``PersistentTile``: instead of one CTA per
output tile (``M_blocks · N_blocks`` CTAs, a fractional last wave that idles the
tail SMs), launch exactly ``num_sms`` CTAs and have each walk a contiguous range
of tile-work units. This kills the wave-quantization tail at compute-bound
matmul sizes (see ``plans/persistent-cta-streamk.md``).

**Why this runs as a late KERNEL pass (098), not at tile-018 as the plan
sketched.** This is the *atomic* variant: the rewrite is a pure launch-geometry
swap of the outer wrapper — every scheduling decision (smem, ring buffers, TMA,
cp.async, pipelining, swizzle, register-tile flattening, Init placement) is
already baked into the ``GridTile.body`` by the tile passes (020–090) AND the
kernel passes (005–095: register-axis split, sibling-cell fuse, Init hoist,
vectorize, interleave). Running before those ``GridTile``-keyed kernel passes
(register-tile flattening, ``020_place_inits``, …) would leave them skipping the
``PersistentTile`` body, so the inner ``RegisterTile`` would reach
``100_materialize`` unflattened and ``partition_tma_groups`` would miss the
staged loads. Running here — after 095_interleave_loads, just before
100_materialize — the body is fully lowered and flat; we carry it verbatim and
teach only the materializer + launch-bounds about ``PersistentTile``. Smem decls
hoisted into the body land inside the per-CTA work loop, which is legal CUDA
(static alloc, reused per tile); ``Sync`` / mbarrier ops fire per-K-iteration, so
re-entering the K-loop per tile is correct.

**Two work-unit granularities, selected by whether split-K is active:**

- **No split-K (M3): full tiles, one owner per tile, no atomics.** Each work unit
  is a whole output tile (full K-loop); the persistent grid covers ``(M_b, N_b)``
  and every tile is written by exactly one CTA, so the output ``Write`` stays a
  plain store.
- **With split-K (M4): atomic-based Stream-K.** When the matmul was lowered with
  ``SPLITK > 1``, its ``GridTile`` already carries a ``K_s`` block axis and the
  per-``K_s`` partial K-loop, and ``Body.coordination`` already marks the output
  Write atomic (``K_s`` ∉ the Write index). Persisting over the full
  ``(K_s, M_b, N_b)`` grid distributes ``SPLITK · M_blocks · N_blocks`` work units
  across ``num_sms`` CTAs — every unit ``atomicAdd``\\s its partial into the
  pre-zeroed output (``zero_outputs``, wired by 010_lower_kernelop from the same
  ``atomic_axes`` signal). This is the atomic variant of the plan: the K split is
  at fixed ``SPLITK`` granularity rather than adaptive mid-tile MAC ranges, so it
  reuses the legacy split-K atomic path verbatim and just swaps the grid for a
  balanced persistent walk that kills the wave-quantization tail.

Swizzle (``swizzle_group_m``) is dropped by the swap for now — persistent CTAs
walk tiles in work-range order, so the L2 swizzle story differs; revisit when
tuning the schedule. Adaptive mid-tile MAC-range boundaries (the atomic-free
combine in the plan's Phase B) are a separate, larger follow-up.
"""

from __future__ import annotations

from math import prod

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.ir.tile.ir import GridTile, PersistentTile, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import STREAMK

PATTERN = [Pattern("root", TileOp)]

# Above this many waves the launch already amortizes its tail across enough
# CTAs that the persistent walk can't recover a meaningful fraction — and the
# Stream-K bookkeeping (work-range loads, atomicAdd contention) would only cost.
# Below it, the fractional last wave idles a real slice of the SMs. The fork is
# offered to the autotuner only inside this regime (an explicit DEPLODOCK_STREAMK
# pin bypasses the gate for A/B benchmarking).
_MAX_WAVES_FOR_STREAMK = 8


def _outer_grid(op: TileOp) -> GridTile | None:
    """The single outermost ``GridTile`` of ``op``, or ``None``."""
    for s in op.body:
        if isinstance(s, GridTile):
            return s
    return None


def rewrite(ctx: Context, match: Match, root) -> list[TileOp] | None:  # noqa: ARG001 — match required by dispatch
    op: TileOp = root.op
    if STREAMK.name in op.knobs:
        raise RuleSkipped("STREAMK already pinned (idempotence)")
    # Compile-time shape gate: Stream-K needs a live SM count to size the grid.
    if ctx.num_sms <= 0:
        raise RuleSkipped("no live SM count — Stream-K is a live-device perf fork")
    grid = _outer_grid(op)
    if grid is None:
        raise RuleSkipped("no outer GridTile (pointwise / already persistent)")
    # Matmul register-tile shape: a 2-D+ block grid carrying the FM cell knob.
    # Pointwise / cooperative-reduce kernels don't qualify (single block axis or
    # no FM), and masking the wave tail there isn't the lever. A split-K K_s
    # axis (if present) makes grid.axes 3-D — that's the atomic path, kept.
    if len(grid.axes) < 2 or "FM" not in op.knobs:
        raise RuleSkipped("not a matmul block grid")

    # Wave-tail shape gate (autotune only): fork Stream-K just where the launch
    # is in the few-wave regime its tail dominates. An explicit env pin is
    # authoritative (A/B benching at any shape) and skips the gate.
    if config.knob_raw(STREAMK.name) is None and all(ax.extent.is_static for ax in grid.axes):
        total_ctas = prod(ax.extent.as_static() for ax in grid.axes)
        if total_ctas > _MAX_WAVES_FOR_STREAMK * ctx.num_sms:
            raise RuleSkipped(f"{total_ctas} CTAs / {ctx.num_sms} SMs — waves saturated, tail negligible")

    candidates = STREAMK.narrow(STREAMK.hints)
    if not candidates:
        raise RuleSkipped("STREAMK pin matches no candidate")

    variants: list[TileOp] = []
    for use_streamk in candidates:
        if not use_streamk:
            # Off: structurally identical, tagged so the cache key + idempotence
            # distinguish it. False first → greedy default keeps the GridTile.
            variants.append(TileOp(body=op.body, name=op.name, knobs={**op.knobs, STREAMK.name: False}))
            continue
        # On: swap the GridTile wrapper for a PersistentTile carrying the same
        # block axes + body. Launch geometry (grid = num_sms) and the two
        # work-range arrays are derived downstream in 010_lower_kernelop.
        persistent = PersistentTile(axes=grid.axes, body=grid.body)
        new_body = tuple(persistent if s is grid else s for s in op.body)
        variants.append(TileOp(body=new_body, name=op.name, knobs={**op.knobs, STREAMK.name: True}))
    return variants
