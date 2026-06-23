"""stage pass (enumeration fork) — the first ``Schedule``-move pass.

``plans/tile-ir-block-dag.md`` R1: ``stage(read)`` writes
``Schedule.staged[edge] = SYNC`` for a reused gmem read; it inserts **no block**
and edits **no body** — the smem slab + cooperative producer are synthesized later
by ``assembly/_slab`` from the annotation.

Post-F3-b this is a **pre-assemble** schedule fork over the **stored, fully-tiled
algorithm** (the inversion R1 carried before F3-b — ``stage`` running *after* a
separate monolithic build pass — is gone). The body moves (``010_reduce_tile`` /
``030_register_tile``) already refined ``op.tilegraph`` in place, so ``stage`` reads
its derived ``Block.reads`` + the ranked stageable read-sites directly and forks on
the stage *mask*, writing the chosen ``Edge``s straight into ``Schedule.staged`` —
the source of truth ``assemble`` reads. The mask string is the variant identity the
perf DB / learned prior key on; ordering is decision order (``stage`` reads derived
projections of the fully-tiled algorithm, so it runs once every free-axis tile is
pinned: ``MAP_N_REG``).

R1 scope: scalar (no-MMA) reduce regimes. ``stage_candidates`` returns nothing for
pointwise (no K-tower) / no-reuse kernels, so this pass is a no-op there
(``RuleSkipped`` → ``apply_off_defaults`` stamps ``STAGE=""``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp, Transport
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import prospective_sources
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG, STAGE
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._stage import stage_candidates

PATTERN = [Pattern("root", TileGraphOp)]


def _slab_bytes(graph, ranked) -> dict[str, int]:
    """Smem bytes ``assemble`` would allocate per staged buffer's slab — the budget
    the auto-enumerated fork ranks against. Each buffer's slab is independent of which
    others stage, so size them once from the all-staged ``prospective_sources``
    (the byte count matches ``KernelOp.smem_bytes`` exactly — the slab is
    ``∏(cache_extent · block) · dtype.nbytes``, pre-padding). A buffer whose source
    can't be sized (an exotic non-affine layout) is omitted ⇒ treated as free, so the
    filter never removes a staging it can't price (preserves the legacy offer)."""
    full = replace(graph, schedule=replace(graph.schedule, staged={e: Transport.SYNC for e in ranked}))
    try:
        sources = prospective_sources(full)
    except Exception:  # noqa: BLE001 — best-effort sizing; fall back to no filter
        return {}
    out: dict[str, int] = {}
    for s in sources:
        block = getattr(s.addressing, "block", ()) or ()
        elems = 1
        for i, ax in enumerate(s.cache_axes):
            if not ax.extent.is_static:
                break
            elems *= ax.extent.as_static() * (block[i] if block else 1)
        else:
            out[s.buf] = elems * (s.dtype.nbytes if s.dtype else 4)
    return out


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if MAP_N_REG.name not in op.knobs or STAGE.name in op.knobs:
        raise RuleSkipped("stage runs once the algorithm is fully tiled (idempotence via the STAGE knob)")
    if op.algebra in (AlgebraKind.MONOID, AlgebraKind.TWISTED_MONOID):
        # A cooperative reduce / flash stream stays smem-free: each lane reads its own
        # K-strided slice with no cross-thread reuse (legacy coop / flash never staged).
        raise RuleSkipped("cooperative reduce / flash stays smem-free (no cross-thread read reuse)")
    ranked = stage_candidates(op.tilegraph)
    n = len(ranked)
    if n == 0:
        raise RuleSkipped("no stageable read-sites (pointwise / no reuse / no K-tower)")

    # Env pin (``DEPLODOCK_STAGE=11`` / ``all`` / ``none``) collapses the fork to one
    # mask and is authoritative (no budget filter); otherwise offer every subset,
    # most-staged-first (option-0 = stage the most, best when smem fits — matches the
    # search's prefer-deeper-first heuristic).
    raw = STAGE.raw()
    if raw:
        masks = [STAGE.parse(raw, width=n)]
    else:
        # Budget-aware: drop any subset whose slabs exceed the smem cap so greedy's
        # option-0 deterministically picks the largest IN-BUDGET staging (mask 0 = no
        # staging always fits → never empty). Without this the deterministic compile
        # would pick stage-all and fail downstream with no fallback when a large pinned
        # tile over-stages (the multi-accum / masked linear-projection case).
        budget = ctx.max_dynamic_smem if ctx is not None else None
        buf_bytes = _slab_bytes(op.tilegraph, ranked)

        def _fits(m: int) -> bool:
            if budget is None:
                return True
            total = sum(buf_bytes.get(e.buffer, 0) for i, e in enumerate(ranked) if m & (1 << i))
            return total <= budget

        masks = [m for m in sorted(range(1 << n), key=lambda m: (-bin(m).count("1"), m)) if _fits(m)]

    out: list[TileGraphOp] = []
    for mask in masks:
        staged = {e: Transport.SYNC for i, e in enumerate(ranked) if mask & (1 << i)}
        tg = replace(op.tilegraph, schedule=replace(op.tilegraph.schedule, staged=staged))
        out.append(replace(op, tilegraph=tg, knobs={**op.knobs, STAGE.name: STAGE.pretty(mask, width=n)}))
    return out
