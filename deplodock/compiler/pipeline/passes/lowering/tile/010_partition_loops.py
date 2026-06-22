"""Partition stage — lower each fused ``LoopOp`` to a ``TileOp`` tower.

The sole partitioner is the hierarchical **move composer**
(``partition/`` — ``iter_dag`` → ``classify`` → ``build_partition``): one
``iter_dag`` derived view over the body, a regime tag off its reduce-axes'
``Loop.algebra_kind``, and a generative ``Fork`` tree of carrier-trait-licensed
``legal_decomps`` moves the two-level MCTS branches on move-by-move. There is no
legacy planner and no fallback — a kernel the composer can't lower surfaces as a
hard error (the gap to close). See ``plans/algebra-licensed-decomposition-moves.md``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.compose import try_compose

PATTERN = [Pattern("root", LoopOp)]


def rewrite(ctx: Context, root: Node, match) -> Graph | None | TileOp | Fork:
    """Lower one fused ``LoopOp`` via the move composer — a ``Fork`` tree (the
    two-level MCTS branches on it) or a bare ``TileOp`` for a single-variant
    kernel. ``ctx`` / ``graph`` feed tensor-core atom eligibility. A kernel the
    composer can't lower raises (no legacy fallback). Idempotence is structural:
    once a ``TileOp`` is built, the ``LoopOp`` pattern no longer matches."""
    loop_op: LoopOp = root.op
    # The kernel name was stamped onto the LoopOp by ``loop/stamp/010_stamp_loop_names``.
    kernel_name = loop_op.name
    composed = try_compose(loop_op, ctx, match.graph, kernel_name=kernel_name)
    if composed is None:
        raise RuleSkipped(f"move composer cannot lower kernel {kernel_name!r}")
    return composed
