"""Warp register-tile pass (fork) — pins the per-warp register cells ``(FM, FN)``.

The third warp-tier fork (after the geometry
is pinned). Offers the legal ``(fm, fn)`` cells under the register-file budget
(``_moves.warp_reg_offers``) and forks — still knob-only; the warp build (the
``atomize`` body move) lands at ``050_warp_build`` once the K chunk is pinned
too. Fires only on a warp variant whose geometry is set.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.tile.ir import ATOM_REGISTRY, TileGraphOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import mma_atom
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import warp_reg_knobs, warp_reg_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    kind = mma_atom(op.knobs)
    if kind is None or op.dag is None:
        raise RuleSkipped("warp reg tile applies to a warp (MMA atom) variant with a dag")
    nkey = fam.split_key(op.dag.inner_n.axis.name)
    if nkey not in op.knobs or fam.split_complete(op.knobs[nkey]):
        raise RuleSkipped("warp reg tile applies once, after the geometry is pinned")
    offers = warp_reg_offers(op.dag, ATOM_REGISTRY[kind])
    if not offers:
        raise RuleSkipped("no legal warp register tile")
    return [replace(op, knobs={**op.knobs, **warp_reg_knobs(op.dag, op.knobs, fm, fn)}) for fm, fn in offers]
