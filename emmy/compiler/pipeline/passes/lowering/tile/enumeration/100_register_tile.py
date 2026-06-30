"""Register-tile pass (fork) — the free-axis ``tile_axis`` body move.

Offer the legal ``(reg_n, reg_m)`` cells under the per-thread budget and fork. The
candidate menu is the only algebra-conditioned ranking: bandwidth-biased for a
``MAP`` nest (``map_reg_offers``), compute/ILP-biased for a reduce regime
(``reduce_reg_offers``, sized by the pinned ``FK``). Runs after the thread tile, so
both free-axis knobs are pinned — this is where the **free-axis σ-split body move**
(``_build.free_tile``: ``A → A_b·(T·R) + A_t·R + A_r``, laid into ``Block.domain`` +
``Schedule.binding``, masked-axis guards applied) lands on the stored algorithm. The
last of the F3-b incremental body moves; the algorithm is fully tiled afterward.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.context import Context
from emmy.compiler.graph import Node
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.tile.ir import TileGraphOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._build import free_tile
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, map_reg_offers, reduce_reg_offers, reg_knobs

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.dag is None:
        raise RuleSkipped("already-built kernel (no dag — e.g. the split-K combine)")
    nkey = fam.split_key(op.dag.inner_n.axis.name)
    if nkey not in op.knobs or fam.split_complete(op.knobs[nkey]):
        raise RuleSkipped("register tile not applicable / thread tile not pinned / already complete")
    if op.algebra is AlgebraKind.MONOID:
        raise RuleSkipped("cooperative-reduce tier (070_coop_reduce owns the MONOID free-axis tile)")
    if op.algebra is AlgebraKind.SEMIRING:
        fold = fam.dec_reduce(op.knobs[fam.reduce_key(op.dag.k_node.loop.axis.name)]).fold
        offers = reduce_reg_offers(op.dag, Budget(), fold)
    else:
        offers = map_reg_offers(op.dag, Budget())
    if not offers:
        raise RuleSkipped("no legal register tile")
    out = []
    for reg in offers:
        knobs = {**op.knobs, **reg_knobs(op.dag, op.knobs, reg)}
        tg = free_tile(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
