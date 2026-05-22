"""TMA copy promotion (stage-wrap-body refactor: stubbed).

Pre-refactor: promoted single-source ``BufferedStage`` to
``TmaBufferedStage`` and inserted a trailing ``AsyncWait`` carrying the
consumer-side mbarrier phase.

Post-refactor (Phase C bucket 12 — TMA-specific): the wait is implicit
at the Stage wrap boundary; TmaBufferedStage materialization emits its
own mbarrier-wait. This pass becomes a pure "promote BufferedStage →
TmaBufferedStage when eligible" rewrite over the new wrap-body shape.

Stubbed (always RuleSkipped) until Phase C bucket 12 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("011_tma_copy: stubbed during stage-wrap-body refactor")
