"""Thread-tile pass (fork) — pins the free-axis THREAD knob.

Offer the legal ``(thread_n, thread_m)`` tiles (``_moves.thread_offers``) and fork
— one option per tile, each pinning the THREAD-knob group. Runs after the reduce
tile is pinned (for a ``SEMIRING`` regime) so the cell budget is known downstream.

It pins a knob but applies **no body move**: the free-axis σ-split needs both the
thread *and* register knob to lay out ``A → A_b·(T·R) + A_t·R + A_r`` byte-identically
(register innermost), and the register knob isn't pinned until ``100_register_tile``
— so the single free-axis body move (``_build.free_tile``) is applied there, with this
pass only fixing the thread extent the search ranks on (F3-b).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_THREAD
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, thread_knobs, thread_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if MAP_N_THREAD.name in op.knobs or mma_atom(op.knobs) is not None:
        raise RuleSkipped("thread tile already pinned / warp tier")
    if op.algebra is AlgebraKind.MONOID:
        raise RuleSkipped("cooperative-reduce tier (070_coop_reduce owns the MONOID free-axis tile)")
    if op.algebra is AlgebraKind.SEMIRING and fam.reduce_key(op.dag.k_node.loop.axis.name) not in op.knobs:
        raise RuleSkipped("reduce tile not yet pinned")
    # A SEMIRING (matmul) 2-D output tile wants a balanced, coalesced shape; a MAP
    # (pointwise) nest stays bandwidth-bound / wide-N. (MONOID is skipped above.)
    offers = thread_offers(op.dag, Budget(), balanced=op.algebra is AlgebraKind.SEMIRING)
    if not offers:
        raise RuleSkipped("no legal thread tile")
    return [replace(op, knobs={**op.knobs, **thread_knobs(op.dag, t)}) for t in offers]
