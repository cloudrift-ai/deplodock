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

from dataclasses import replace
from math import prod

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Body, Cond, Write
from deplodock.compiler.ir.tile.ir import (
    STREAMK_K_HI,
    STREAMK_K_LO,
    AsyncWait,
    GridTile,
    PersistentTile,
    SerialTile,
    TileOp,
)
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

    # When STREAMK is explicitly pinned ON, every "can't apply" below is a user
    # error — fail loudly rather than silently compiling a non-Stream-K kernel
    # (the fail-loud / no-defensive-fallbacks convention). When it's not pinned,
    # these are just autotune-fork eligibility gates → RuleSkipped (the fork
    # simply isn't offered for this shape).
    raw = config.knob_raw(STREAMK.name)
    pinned_on = raw is not None and bool(STREAMK.parse(raw))

    def _cant(reason: str, hint: str = "") -> None:
        if pinned_on:
            raise ValueError(f"DEPLODOCK_STREAMK=1 pinned but cannot be applied here: {reason}.{(' ' + hint) if hint else ''}")
        raise RuleSkipped(reason)

    # Compile-time shape gate: Stream-K needs a live SM count to size the grid.
    if ctx.num_sms <= 0:
        _cant("no live SM count", "Stream-K is a live-device perf fork; run on a GPU.")
    grid = _outer_grid(op)
    if grid is None:
        _cant("no outer GridTile (pointwise / cooperative-reduce kernel)", "Stream-K applies to matmul.")
    # Matmul register-tile shape: a 2-D block grid (M_b, N_b) carrying the FM
    # cell knob. Pointwise / cooperative-reduce kernels don't qualify.
    if len(grid.axes) < 2 or "FM" not in op.knobs:
        _cant("not a matmul block grid")
    # Mutually exclusive with split-K: Stream-K does its own K-split. A K_s axis
    # already splits the K reduction across CTAs — the two never combine.
    if find_split_k_axis_name(op) is not None:
        _cant("split-K is active", "STREAMK and SPLITK both split K and are mutually exclusive — unset SPLITK.")
    # Adaptive Stream-K needs the chunked-K outer loop to re-bound; without one
    # (degenerate single-chunk matmul) there's nothing to split.
    k_blocks = _k_blocks(grid)
    if k_blocks is None or k_blocks < 2:
        _cant("only one K chunk (nothing to split)", "lower BK so K / BK >= 2.")
    # The runtime-bounded K-loop is correct for any staging whose load sits
    # *inside* a single K-loop iteration — SYNC, BUFFERED ring, and cp.async ring
    # (load-then-wait per iteration). *Temporal pipelining* (080_pipeline_stages)
    # and TMA split the loop into a prologue that prefetches chunk 0 + a steady
    # body + an epilogue, with slot/phase constants baked for [0, K_blocks) — so
    # re-bounding to [k_lo, k_hi) breaks the prologue. Those stay gated (their
    # explicit AsyncWait / pipeline-kind loop is the tell).
    if _is_pipelined(grid):
        _cant(
            "the K-loop is temporally pipelined (prologue/steady/epilogue) or uses TMA",
            "Stream-K needs the load inside the K-loop — set TMA=0,PIPELINE_STAGES=0 (cp.async ring is fine; pipelined/TMA pending).",
        )

    # Wave-tail shape gate (autotune only): fork Stream-K just where the launch
    # is in the few-wave regime its tail dominates. An explicit env pin is
    # authoritative (A/B benching at any shape) and skips the gate.
    if not pinned_on and all(ax.extent.is_static for ax in grid.axes):
        total_ctas = prod(ax.extent.as_static() for ax in grid.axes)
        if total_ctas > _MAX_WAVES_FOR_STREAMK * ctx.num_sms:
            raise RuleSkipped(f"{total_ctas} CTAs / {ctx.num_sms} SMs — waves saturated, tail negligible")

    candidates = STREAMK.narrow(STREAMK.hints)
    if not candidates:
        raise RuleSkipped("STREAMK pin matches no candidate")

    out_names = frozenset(op.outputs)
    variants: list[TileOp] = []
    for use_streamk in candidates:
        if not use_streamk:
            # Off: structurally identical, tagged so the cache key + idempotence
            # distinguish it. False first → greedy default keeps the GridTile.
            variants.append(TileOp(body=op.body, name=op.name, knobs={**op.knobs, STREAMK.name: False}))
            continue
        # On: rewrite the matmul body to the adaptive MAC-segment form and wrap it
        # in an adaptive PersistentTile (grid = num_sms; work units = tiles ×
        # K_blocks, derived downstream in 010_lower_kernelop). The body's K-loop
        # runs the per-CTA sub-range [k_lo, k_hi); the output Write branches on
        # whether the CTA owns the whole tile (direct store) or just a slice
        # (atomicAdd into the pre-zeroed output — the combine becomes atomic-free
        # in B4).
        adaptive_body = Body(_adaptivize(tuple(grid.body), out_names, k_blocks))
        persistent = PersistentTile(axes=grid.axes, body=adaptive_body, k_blocks=k_blocks)
        new_body = tuple(persistent if s is grid else s for s in op.body)
        variants.append(TileOp(body=new_body, name=op.name, knobs={**op.knobs, STREAMK.name: True}))
    return variants


def _k_blocks(grid: GridTile) -> int | None:
    """K-loop trip count = extent of the ``serial_outer`` chunked-K loop, or
    ``None`` if the matmul has no such loop."""
    for s in grid.body.iter():
        if isinstance(s, SerialTile) and s.kind == "serial_outer" and s.axis.extent.is_static:
            return s.axis.extent.as_static()
    return None


def _is_pipelined(grid: GridTile) -> bool:
    """True if the K-loop is temporally pipelined — a ``pipeline``-kind serial
    loop, or an explicit ``AsyncWait`` (emitted by 080_pipeline_stages / TMA when
    the loop is split into a chunk-0 prologue + steady + epilogue). That prologue
    bakes slot/phase constants for ``[0, K_blocks)``, incompatible with a
    runtime-bounded ``[k_lo, k_hi)`` loop (pending). SYNC, BUFFERED, and cp.async
    rings keep the load *inside* a single K-loop iteration (no prologue, no
    AsyncWait at this stage) and are fine to re-bound."""
    for s in grid.body.iter():
        if isinstance(s, AsyncWait):
            return True
        if isinstance(s, SerialTile) and s.kind == "pipeline":
            return True
    return False


def _adaptivize(stmts: tuple, out_names: frozenset[str], k_blocks: int) -> list:
    """Rewrite a matmul ThreadTile body into the adaptive Stream-K form:

    - the ``serial_outer`` K-loop gains runtime bounds ``[streamk_k_lo,
      streamk_k_hi)`` (the per-CTA K sub-range, bound by the adaptive
      PersistentTile render);
    - each output ``Write`` is wrapped in ``Cond(is_full)``: a full-tile owner
      (``k_lo == 0 && k_hi == K_blocks``) stores directly, a boundary partial
      ``atomicAdd``\\s into the pre-zeroed output.

    Recurses children-first via ``Stmt.nested`` / ``with_bodies`` so the rewrite
    reaches the K-loop and Writes wherever the staging passes left them.
    """
    is_full = BinaryExpr(
        "&&",
        BinaryExpr("==", Var(STREAMK_K_LO), Literal(0, "int")),
        BinaryExpr("==", Var(STREAMK_K_HI), Literal(k_blocks, "int")),
    )
    out: list = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(_adaptivize(tuple(b), out_names, k_blocks)) for b in bodies))
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            s = replace(s, lo=Var(STREAMK_K_LO), hi=Var(STREAMK_K_HI))
        if isinstance(s, Write) and s.output in out_names:
            partial = replace(s, atomic=True)
            s = Cond(cond=is_full, body=Body((s,)), else_body=Body((partial,)))
        out.append(s)
    return out
