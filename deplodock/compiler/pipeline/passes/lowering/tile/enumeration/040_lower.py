"""Lower pass (deterministic) — build the chosen ``TileGraph`` from the pinned knobs.

The terminal of the F3-a enumeration split (``plans/tile-ir-block-dag.md``): once
every tile-knob group is pinned, ``build_dag`` materializes the invariant
algorithm + the reference ``Schedule`` (binding) from ``op.knobs`` once, and the
result rides ``op.tilegraph`` for the separate ``assembly/010_assemble`` pass. No
fork — one deterministic rewrite. A reduce regime stamps the warp-tier OFF
sentinels (scalar tier) so every variant carries the full knob set.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import build_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> TileGraphOp:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is not None or MAP_N_REG.name not in op.knobs:
        raise RuleSkipped("not fully tiled / already built")
    tg = build_dag(op.dag, op.knobs, kernel_name=op.name, target_names=op.target_names, buffers=op.buffers)
    knobs = {**op.knobs, "MMA": "0", "WM": 0, "WN": 0, "BR": 1} if op.target_names else op.knobs
    return replace(op, tilegraph=tg, knobs=knobs)
