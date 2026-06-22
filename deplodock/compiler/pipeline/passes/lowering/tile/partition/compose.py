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


def try_compose(loop_op: LoopOp, ctx: Context, graph: Graph, *, kernel_name: str) -> Fork | TileOp | None:
    """Compose ``loop_op`` if the nest is recognized, else ``None``. ``ctx`` /
    ``graph`` feed tensor-core atom eligibility (compute capability + operand
    dtypes). One ``iter_dag`` derived view is the structure the partition
    consumes — ``build_partition`` classifies the regime + factors the axes off
    it (no typed-skeleton layer)."""
    base_knobs = dict(loop_op.knobs)
    dag = iter_dag(loop_op)
    return build_partition(dag=dag, loop_op=loop_op, context=ctx, graph=graph, base_knobs=base_knobs, kernel_name=kernel_name)
