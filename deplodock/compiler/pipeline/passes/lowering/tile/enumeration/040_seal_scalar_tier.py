"""Seal-scalar-tier pass (deterministic) — stamp the scalar-tier OFF sentinels.

``plans/tile-ir-block-dag.md`` F3-b: this pass is **knob-only** — it does not touch
the stored algorithm (the body moves run in ``010_reduce_tile`` / ``030_register_tile``).
It seals the reduce regime's warp-tier knobs to their scalar-tier OFF sentinels
(``MMA=0 / WM=0 / WN=0`` plus ``BR=1`` — serial, not the warp-OFF ``BR=0``) so every
reduce variant carries the full knob set the perf DB / learned prior key on.

It runs at the **post-register** level (gate: every free-axis tile pinned), exactly
where the old monolithic ``040_lower`` stamped these sentinels — so the per-level
greedy ranking sees the same partial knob set at every fork, keeping the deployed pick
byte-identical. It is deterministic (one rewrite, no ``Fork``); a pointwise (``MAP``)
nest carries no warp/reduce knobs and is left untouched.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    op: TileGraphOp = root.op
    if MAP_N_REG.name not in op.knobs or not op.target_names or "MMA" in op.knobs:
        raise RuleSkipped("not fully tiled / not a reduce regime / already sealed")
    return replace(op, knobs={**op.knobs, "MMA": "0", "WM": 0, "WN": 0, "BR": 1})
