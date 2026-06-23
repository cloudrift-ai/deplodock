"""Warp-geometry pass (fork) — pins the per-CTA warp counts ``(WM, WN)``.

``plans/tile-ir-block-dag.md`` R4: the second warp-tier fork (after
``005_tensorize`` chose the atom). Offers the legal ``(wm, wn)`` warp tilings
(``_moves.warp_offers``) and forks — knob-only, no body move (the warp build
needs the register + K-chunk knobs too, applied at ``009_warp_build``). Fires
only on a warp variant (a concrete ``MMA`` atom pinned); scalar variants skip.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import WARP_M
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import warp_geom_knobs, warp_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    kind = mma_atom(op.knobs)
    if kind is None or WARP_M.name in op.knobs:
        raise RuleSkipped("warp geometry applies once, to a warp (MMA atom) variant")
    offers = warp_offers(ATOM_REGISTRY[kind])
    if not offers:
        raise RuleSkipped("no legal warp geometry")
    return [replace(op, knobs={**op.knobs, **warp_geom_knobs(wm, wn)}) for wm, wn in offers]
