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

**Current scope: persistent-over-tiles, one owner per tile, no atomics.** Each
work unit is a whole output tile (full K-loop); the persistent grid covers
``(M_b, N_b)`` and every tile is written by exactly one CTA, so the output
``Write`` stays a plain store. This kills the *launch-grid* quantization (no
fractional CTA wave) but is **perf-neutral** on its own: rebalancing which CTA
owns which tile doesn't shorten the critical path (``ceil(units / SMs)``
tile-times either way). The actual wave-tail win needs the K reduction split so
the *fractional* wave gets filled — that's Phase B below.

**Mutually exclusive with split-K.** Stream-K's win comes from splitting the K
reduction across CTAs — the same job ``SPLITK`` does. The plan's Stream-K does it
*adaptively* (only the boundary tiles, at mid-tile MAC offsets sized to fill the
fractional wave), which *replaces* fixed Split-K rather than composing with it.
So this rule self-skips when a split-K ``K_s`` axis is already present; the two
never combine.

**Phase B (in progress — the actual lever).** Adaptive mid-tile MAC-range
splitting: a CTA walks a contiguous range of ``(tile, k_chunk)`` MAC units, runs
a *runtime-bounded* partial K-loop for the boundary tiles it shares, writes full
tiles directly and boundary partials to a scratch workspace, and a tiny combine
kernel (atomic-free, reusing ``017_atomic_free_splitk``'s reduce skeleton) sums
the partials. That is the only form that beats Split-K (each output cell written
once → no atomic tax) and targets the wave tail. Not yet implemented; today the
rule only does the perf-neutral wrapper swap above.

Swizzle (``swizzle_group_m``) is dropped by the swap for now — persistent CTAs
walk tiles in work-range order, so the L2 swizzle story differs; revisit when
tuning the schedule.
"""

from __future__ import annotations

from math import prod

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.ir.tile.ir import GridTile, PersistentTile, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import STREAMK
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import find_split_k_axis_name

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
    # Matmul register-tile shape: a 2-D block grid (M_b, N_b) carrying the FM
    # cell knob. Pointwise / cooperative-reduce kernels don't qualify.
    if len(grid.axes) < 2 or "FM" not in op.knobs:
        raise RuleSkipped("not a matmul block grid")
    # Mutually exclusive with split-K: Stream-K does its own K-split (adaptively,
    # in Phase B). A K_s axis already splits the K reduction across CTAs — the two
    # never combine. (This is why the plan gates "STREAMK ⇒ SPLITK=1".)
    if find_split_k_axis_name(op) is not None:
        raise RuleSkipped("split-K present — Stream-K supplies its own K-split, the two are exclusive")

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
