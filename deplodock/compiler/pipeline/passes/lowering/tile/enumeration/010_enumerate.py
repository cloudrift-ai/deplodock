"""Enumeration pass — lower each fused ``LoopOp`` into the move composer's
generative ``Fork`` tree (or a single ``TileGraphOp``).

The tile phase's sole search pass: the **enumeration** half of the block-DAG
model (``plans/tile-ir-block-dag.md``). It derives the iteration DAG
(``_iterdag.iter_dag``), then ``_tree.build_partition`` classifies the regime off
its reduce axes' ``Loop.algebra_kind`` and builds a generative ``Fork`` tree over
the carrier-trait-licensed moves the two-level MCTS branches on (the search
space). Each resolved leaf emits a ``TileGraphOp`` (a chosen ``Schedule``'s
``TileGraph``); the separate ``assembly/010_assemble`` pass materializes it into
the ``TileOp`` tower. A kernel the composer can't lower raises (no fallback).
See ``plans/algebra-licensed-decomposition-moves.md``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._tree import build_partition

PATTERN = [Pattern("root", LoopOp)]


def rewrite(ctx: Context, root: Node, match) -> Graph | None | TileOp | Fork:
    """Enumerate one fused ``LoopOp``'s schedule space — a ``Fork`` tree (the
    two-level MCTS branches on it) or a single ``TileGraphOp`` for a one-variant
    kernel. ``ctx`` / ``match.graph`` feed tensor-core atom eligibility (compute
    capability + operand dtypes). A kernel the composer can't lower raises (no
    fallback). Idempotence is structural: once the ``LoopOp`` is replaced its
    pattern no longer matches."""
    loop_op: LoopOp = root.op
    # The kernel name was stamped onto the LoopOp by ``loop/stamp/010_stamp_loop_names``.
    kernel_name = loop_op.name
    dag = iter_dag(loop_op)
    composed = build_partition(
        dag=dag, loop_op=loop_op, context=ctx, graph=match.graph, base_knobs=dict(loop_op.knobs), kernel_name=kernel_name
    )
    if composed is None:
        raise RuleSkipped(f"move composer cannot lower kernel {kernel_name!r}")
    return composed
