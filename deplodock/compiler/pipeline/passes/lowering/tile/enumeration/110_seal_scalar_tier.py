"""Seal-scalar-tier pass (deterministic) — stamp the scalar-tier OFF sentinels.

``plans/tile-ir-block-dag.md`` F3-b: this pass is **knob-only** — it does not touch
the stored algorithm (the body moves run in ``060_reduce_tile`` / ``100_register_tile``).
It seals the scalar ``SEMIRING`` reduce's warp-tier knobs to their OFF sentinels
(``MMA=0 / WM=0 / WN=0``) so every reduce variant carries the full knob set the perf DB
/ learned prior key on. The cooperative ``REDUCE@<axis>.coop`` factor already rides the
native reduce value (no separate ``BR`` sentinel).

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
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.SEMIRING or MAP_N_REG.name not in op.knobs or "MMA" in op.knobs:
        raise RuleSkipped("not a fully-tiled scalar SEMIRING reduce / already sealed / warp tier")
    return replace(op, knobs={**op.knobs, "MMA": "0", "WM": 0, "WN": 0})
