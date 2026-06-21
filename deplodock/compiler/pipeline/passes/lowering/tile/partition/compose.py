"""Dispatch entry for the move composer.

``010_partition_loops`` calls :func:`try_compose` when the composer is enabled;
it returns a ``Fork`` / ``TileOp`` for a regime the composer covers, or ``None``
to fall through to the legacy planner. Phase 1 covers pointwise only.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import lift_matmul, lift_pointwise
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_matmul_tree, build_pointwise_tree


def try_compose(loop_op: LoopOp, ctx: Context, graph: Graph, *, kernel_name: str) -> Fork | TileOp | None:
    """Compose ``loop_op`` if its regime is covered, else ``None``. ``ctx`` /
    ``graph`` feed tensor-core atom eligibility (compute capability + operand
    dtypes)."""
    base_knobs = dict(loop_op.knobs)
    pointwise = lift_pointwise(loop_op)
    if pointwise is not None:
        return build_pointwise_tree(pointwise, base_knobs=base_knobs, kernel_name=kernel_name)
    matmul = lift_matmul(loop_op)
    if matmul is not None:
        return build_matmul_tree(matmul, loop_op=loop_op, context=ctx, graph=graph, base_knobs=base_knobs, kernel_name=kernel_name)
    return None
