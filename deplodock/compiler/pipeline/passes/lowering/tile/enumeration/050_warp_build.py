"""Warp-build pass (fork) — pins the K chunk ``BK`` + applies the ``atomize`` move.

``plans/tile-ir-block-dag.md`` R4: the last warp-tier enumeration fork. With the
atom + geometry + register cells already pinned, this offers the legal ``BK``
K-chunks (``_moves.warp_bk_offers``, in atom-K units) and, for each, **applies the
warp build body move** (``_build.warp_build``: σ-split each output axis four ways
GRID/WARP/REGISTER/ATOM, re-bracket K at ``atom_k`` granularity, fuse the cell
into an ``Mma``). ``assemble`` reconstructs the AtomTile/WarpTile tower from the
resulting ``Block.domain`` + ``Mma``. Fires only on a warp variant whose register
tile is pinned; scalar variants skip.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import warp_build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import warp_bk_knobs, warp_bk_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    kind = mma_atom(op.knobs)
    if kind is None or op.dag is None:
        raise RuleSkipped("warp build applies to a warp (MMA atom) variant with a dag")
    nkey = fam.split_key(op.dag.inner_n.axis.name)
    if (
        nkey not in op.knobs
        or not fam.split_complete(op.knobs[nkey])
        or fam.reduce_key(op.dag.k_node.loop.axis.name) in op.knobs
    ):
        raise RuleSkipped("warp build applies once, after the register tile is pinned")
    atom = ATOM_REGISTRY[kind]
    offers = warp_bk_offers(op.dag, atom)
    if not offers:
        raise RuleSkipped("no legal warp K chunk")
    out = []
    for bk in offers:
        knobs = {**op.knobs, **warp_bk_knobs(op.dag, atom, bk)}
        tg = warp_build(op.tilegraph, op.dag, knobs, atom=atom)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
