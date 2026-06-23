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
from deplodock.compiler.ir.tile.ir import TileGraphOp, Transport
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG, STAGE
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._stage import stage_candidates

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if MAP_N_REG.name not in op.knobs or STAGE.name in op.knobs:
        raise RuleSkipped("stage runs once the algorithm is fully tiled (idempotence via the STAGE knob)")
    ranked = stage_candidates(op.tilegraph)
    n = len(ranked)
    if n == 0:
        raise RuleSkipped("no stageable read-sites (pointwise / no reuse / no K-tower)")

    # Env pin (``DEPLODOCK_STAGE=11`` / ``all`` / ``none``) collapses the fork to one
    # mask; otherwise offer every subset, most-staged-first (option-0 = stage all,
    # best when smem fits — matches the search's prefer-deeper-first heuristic).
    raw = STAGE.raw()
    masks = [STAGE.parse(raw, width=n)] if raw else sorted(range(1 << n), key=lambda m: (-bin(m).count("1"), m))

    out: list[TileGraphOp] = []
    for mask in masks:
        staged = {e: Transport.SYNC for i, e in enumerate(ranked) if mask & (1 << i)}
        tg = replace(op.tilegraph, schedule=replace(op.tilegraph.schedule, staged=staged))
        out.append(replace(op, tilegraph=tg, knobs={**op.knobs, STAGE.name: STAGE.pretty(mask, width=n)}))
    return out
