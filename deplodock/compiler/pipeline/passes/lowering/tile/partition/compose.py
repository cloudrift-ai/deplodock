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
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import (
    CoopReduceSkeleton,
    MatmulSkeleton,
    PointwiseSkeleton,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import (
    build_coop_reduce_tree,
    build_matmul_tree,
    build_pointwise_tree,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.walk import walk_nest


def try_compose(loop_op: LoopOp, ctx: Context, graph: Graph, *, kernel_name: str) -> Fork | TileOp | None:
    """Compose ``loop_op`` if the nest walk recognizes it, else ``None``. ``ctx``
    / ``graph`` feed tensor-core atom eligibility (compute capability + operand
    dtypes). One walk tags the nest; the skeleton type selects the builder."""
    base_knobs = dict(loop_op.knobs)
    nest = walk_nest(loop_op, warp_size=ctx.warp_size)
    if isinstance(nest, PointwiseSkeleton):
        return build_pointwise_tree(nest, base_knobs=base_knobs, kernel_name=kernel_name)
    if isinstance(nest, MatmulSkeleton):
        return build_matmul_tree(nest, loop_op=loop_op, context=ctx, graph=graph, base_knobs=base_knobs, kernel_name=kernel_name)
    if isinstance(nest, CoopReduceSkeleton):
        return build_coop_reduce_tree(nest, base_knobs=base_knobs, kernel_name=kernel_name)
    return None
