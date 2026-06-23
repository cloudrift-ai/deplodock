"""Reduce-tile pass (fork) — the ``TileSerial`` move on a contraction axis.

For a ``SEMIRING`` seed, offer the carrier-licensed ``(bk, fk, splitk)`` K-tilings
(``_moves.reduce_offers`` → ``legal_decomps``) and fork — each option pins one
reduce-knob group onto the in-flight ``TileGraphOp``. A ``MAP`` nest has no
contraction, so this passes through (``RuleSkipped``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import RED_BK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import reduce_knobs, reduce_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is not None or op.algebra is not AlgebraKind.SEMIRING or RED_BK.name in op.knobs:
        raise RuleSkipped("reduce tile not applicable / already pinned")
    offers = reduce_offers(op.dag)
    if not offers:
        raise RuleSkipped("no legal reduce tiling")
    return [replace(op, knobs={**op.knobs, **reduce_knobs(r)}) for r in offers]
