"""Dispatch entry for the move composer.

``010_partition_loops`` calls :func:`try_compose` when the composer is enabled;
it returns a ``Fork`` / ``TileOp`` for a regime the composer covers, or ``None``
to fall through to the legacy planner.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_partition
from deplodock.compiler.pipeline.passes.lowering.tile.partition.walk import walk_nest


def try_compose(loop_op: LoopOp, ctx: Context, graph: Graph, *, kernel_name: str) -> Fork | TileOp | None:
    """Compose ``loop_op`` if the nest walk recognizes it, else ``None``. ``ctx``
    / ``graph`` feed tensor-core atom eligibility (compute capability + operand
    dtypes). One walk tags the nest; one ``build_partition`` over the derived
    iteration-DAG view materializes it."""
    base_knobs = dict(loop_op.knobs)
    nest = walk_nest(loop_op, warp_size=ctx.warp_size)
    # The materialize builders read free axes / inner body off the DAG (a derived
    # view, byte-identical to the skeleton projections walk_nest returns).
    dag = iter_dag(loop_op)
    return build_partition(nest, dag=dag, loop_op=loop_op, context=ctx, graph=graph, base_knobs=base_knobs, kernel_name=kernel_name)
