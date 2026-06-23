"""Thread-tile pass (fork) — the free-axis ``tile_axis`` THREAD move.

Offer the legal ``(thread_n, thread_m)`` tiles (``_moves.thread_offers``) and fork
— one option per tile, each pinning the THREAD-knob group. Runs after the reduce
tile is pinned (for a ``SEMIRING`` regime) so the cell budget is known downstream.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_THREAD, RED_BK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, thread_knobs, thread_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is not None or MAP_N_THREAD.name in op.knobs:
        raise RuleSkipped("thread tile not applicable / already pinned")
    if op.algebra is AlgebraKind.SEMIRING and RED_BK.name not in op.knobs:
        raise RuleSkipped("reduce tile not yet pinned")
    offers = thread_offers(op.dag, Budget())
    if not offers:
        raise RuleSkipped("no legal thread tile")
    return [replace(op, knobs={**op.knobs, **thread_knobs(op.dag, t)}) for t in offers]
