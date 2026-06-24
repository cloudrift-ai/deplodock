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

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import free_tile
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG, MAP_N_THREAD, RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, map_reg_offers, reduce_reg_offers, reg_knobs

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if MAP_N_REG.name in op.knobs or MAP_N_THREAD.name not in op.knobs:
        raise RuleSkipped("register tile not applicable / already pinned")
    if op.algebra is AlgebraKind.MONOID:
        raise RuleSkipped("cooperative-reduce tier (070_coop_reduce owns the MONOID free-axis tile)")
    if op.algebra is AlgebraKind.SEMIRING:
        offers = reduce_reg_offers(op.dag, Budget(), op.knobs[RED_FK.name])
    else:
        offers = map_reg_offers(op.dag, Budget())
    if not offers:
        raise RuleSkipped("no legal register tile")
    out = []
    for reg in offers:
        knobs = {**op.knobs, **reg_knobs(op.dag, reg)}
        tg = free_tile(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
