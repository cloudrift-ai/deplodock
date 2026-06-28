"""Schedule a lifted ``TileOp`` onto the thread grid.

Second of the two tile-lowering steps — ``010_recognize`` lifted the kernel to a
``TileOp`` carrying one op-tree ``AlgebraNode`` and an UNMAPPED :class:`Schedule` (its
parallel ``free`` axes, empty ``grid``). Scheduling is purely geometry: at the scalar
(one-thread-per-output-cell) tier every free axis maps onto the thread grid, so this step
just binds the schedule's ``free`` axes onto ``grid`` (``Schedule.on_grid``). The fold (a
``Monoid`` / ``Semiring`` reduce, or a serial reduce ``Loop`` inside a flat ``Map``) stays
in the op tree and is materialized to loop IR in ``lowering/kernel`` — nothing here, nor
downstream, branches on reduction-vs-softmax-vs-contraction.

The cooperative / cross-CTA reduce schedules and the mma / blocked / split-K contraction
schedules arrive later as richer mappings of the same ``free`` axes on the ``Schedule``
(see ``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    if tile.schedule.is_mapped:
        # Already mapped (grid set), or nothing to map (a scalar-output kernel materializes
        # on an empty grid) — leave it for materialize.
        raise RuleSkipped("schedule already mapped")
    return TileOp(op=tile.op, schedule=tile.schedule.on_grid(), name=tile.name)
