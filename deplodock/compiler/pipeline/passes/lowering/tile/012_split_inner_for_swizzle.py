"""Split TMA stage inner cache axis for swizzle width (stub during refactor).

Pre-refactor: walks every ``TmaBufferedStage`` whose inner cache extent
is a multiple of a TMA swizzle width and splits the inner axis so the
swizzle matches exactly.

Post-refactor (Phase C bucket 12 — TMA-specific): same logic, but
operates on per-Source cache_dims (a Source carries its own cache axes
in the new IR) and rebuilds Source instances instead of the old
``trivial_stage_body`` body.

Stubbed (always RuleSkipped) until Phase C bucket 12 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("012_split_inner_for_swizzle: stubbed during stage-wrap-body refactor")
