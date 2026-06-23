"""Enumeration pass — lower each fused ``LoopOp`` into the move composer's
generative ``Fork`` tree (or a single ``TileOp``).

The tile phase's sole pass: the **enumeration** half of the block-DAG model
(``plans/tile-ir-block-dag.md``). ``try_compose`` derives the iteration DAG
(``_iterdag.iter_dag``), classifies the regime off its reduce axes'
``Loop.algebra_kind``, and builds a generative ``Fork`` tree over the
carrier-trait-licensed moves the two-level MCTS branches on (the search space).
Each resolved leaf **assembles** the chosen ``Schedule`` into the ``TileOp``
tower (``../assembly``). A kernel the composer can't lower raises (no fallback).
See ``plans/algebra-licensed-decomposition-moves.md``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._compose import try_compose

PATTERN = [Pattern("root", LoopOp)]


def rewrite(ctx: Context, root: Node, match) -> Graph | None | TileOp | Fork:
    """Enumerate one fused ``LoopOp``'s schedule space — a ``Fork`` tree (the
    two-level MCTS branches on it) or a bare ``TileOp`` for a single-variant
    kernel. ``ctx`` / ``graph`` feed tensor-core atom eligibility. A kernel the
    composer can't lower raises (no fallback). Idempotence is structural: once a
    ``TileOp`` is built, the ``LoopOp`` pattern no longer matches."""
    loop_op: LoopOp = root.op
    # The kernel name was stamped onto the LoopOp by ``loop/stamp/010_stamp_loop_names``.
    kernel_name = loop_op.name
    composed = try_compose(loop_op, ctx, match.graph, kernel_name=kernel_name)
    if composed is None:
        raise RuleSkipped(f"move composer cannot lower kernel {kernel_name!r}")
    return composed
