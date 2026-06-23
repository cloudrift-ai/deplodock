"""Reduce-tile pass (fork) — the ``TileSerial`` move on a contraction axis.

For a ``SEMIRING`` seed, offer the carrier-licensed ``(bk, fk, splitk)`` K-tilings
(``_moves.reduce_offers`` → ``legal_decomps``) and fork — each option **applies the
reduce-decomposition body move** (``_build.reduce_decomp``: re-bracket K into the
``K_o`` / ``K_i`` tower in ``Block.compute``) to the stored algorithm and pins its
reduce-knob group. The first of the F3-b incremental body moves; the free-axis split
follows at ``030_register_tile``. A ``MAP`` nest has no contraction, so this passes
through (``RuleSkipped``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import reduce_decomp
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import RED_BK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import reduce_knobs, reduce_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.SEMIRING or RED_BK.name in op.knobs or mma_atom(op.knobs) is not None:
        raise RuleSkipped("reduce tile not applicable / already pinned / warp tier")
    offers = reduce_offers(op.dag)
    if not offers:
        raise RuleSkipped("no legal reduce tiling")
    out = []
    for r in offers:
        knobs = {**op.knobs, **reduce_knobs(r)}
        tg = reduce_decomp(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
