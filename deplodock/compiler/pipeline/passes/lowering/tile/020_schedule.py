"""Schedule a lifted ``TileOp`` onto the thread grid.

Second of the two tile-lowering steps — ``010_recognize`` lifted the kernel to a
``TileOp`` carrying one op-tree ``AlgebraNode`` with its parallel axes on the node's
``free`` field and an **empty** ``grid_axes``. Scheduling is purely geometry: at the
scalar (one-thread-per-output-cell) tier every free axis maps onto the thread grid, so
this step just moves the node's ``free`` axes onto ``TileOp.grid_axes`` and clears the
field. The fold (a ``Monoid`` / ``Semiring`` reduce, or a serial reduce ``Loop`` inside a
flat ``Map``) stays in the op tree and is materialized to loop IR in ``lowering/kernel`` —
nothing here, nor downstream, branches on reduction-vs-softmax-vs-contraction.

The cooperative / cross-CTA reduce schedules and the mma / blocked / split-K contraction
schedules arrive later as richer mappings of the same ``free`` axes (see
``plans/tile-ir-rebuild.md``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    if tile.grid_axes:
        raise RuleSkipped("already scheduled")
    node = tile.op
    free = getattr(node, "free", ())
    if not free:
        raise RuleSkipped("no free axes to map onto the grid")
    # Move the node's parallel axes onto the thread grid (the schedule) and clear them off
    # the op tree — materialize sees a free-less node and lowers its per-cell body.
    return TileOp(op=replace(node, free=()), name=tile.name, grid_axes=tuple(free))
