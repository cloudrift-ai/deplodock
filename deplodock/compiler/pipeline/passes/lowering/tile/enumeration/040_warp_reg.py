"""Warp register-tile pass (fork) — pins the per-warp register cells ``(FM, FN)``.

``plans/tile-ir-block-dag.md`` R4: the third warp-tier fork (after the geometry
is pinned). Offers the legal ``(fm, fn)`` cells under the register-file budget
(``_moves.warp_reg_offers``) and forks — still knob-only; the warp build (the
``atomize`` body move) lands at ``050_warp_build`` once the K chunk is pinned
too. Fires only on a warp variant whose geometry is set.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import TC_REG_M, WARP_M
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import warp_reg_knobs, warp_reg_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    kind = mma_atom(op.knobs)
    if kind is None or WARP_M.name not in op.knobs or TC_REG_M.name in op.knobs:
        raise RuleSkipped("warp reg tile applies once, after the geometry is pinned")
    offers = warp_reg_offers(ATOM_REGISTRY[kind])
    if not offers:
        raise RuleSkipped("no legal warp register tile")
    return [replace(op, knobs={**op.knobs, **warp_reg_knobs(fm, fn)}) for fm, fn in offers]
