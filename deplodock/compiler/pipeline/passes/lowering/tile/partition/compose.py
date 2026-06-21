"""Dispatch entry for the move composer.

``010_partition_loops`` calls :func:`try_compose` when the composer is enabled;
it returns a ``Fork`` / ``TileOp`` for a regime the composer covers, or ``None``
to fall through to the legacy planner. Phase 1 covers pointwise only.
"""

from __future__ import annotations

from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import lift_pointwise
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_pointwise_tree


def try_compose(loop_op: LoopOp, *, kernel_name: str) -> Fork | TileOp | None:
    """Compose ``loop_op`` if its regime is covered, else ``None``."""
    skel = lift_pointwise(loop_op)
    if skel is not None:
        return build_pointwise_tree(skel, base_knobs=dict(loop_op.knobs), kernel_name=kernel_name)
    return None
