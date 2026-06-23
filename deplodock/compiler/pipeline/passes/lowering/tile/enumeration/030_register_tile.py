"""Register-tile pass (fork) — the free-axis ``tile_axis`` REGISTER move.

Offer the legal ``(reg_n, reg_m)`` cells under the per-thread budget and fork. The
candidate menu is the only algebra-conditioned ranking: bandwidth-biased for a
``MAP`` nest (``map_reg_offers``), compute/ILP-biased for a reduce regime
(``reduce_reg_offers``, sized by the pinned ``FK``). Runs after the thread tile.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAP_N_REG, MAP_N_THREAD, RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, map_reg_offers, reduce_reg_offers, reg_knobs

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.tilegraph is not None or MAP_N_REG.name in op.knobs or MAP_N_THREAD.name not in op.knobs:
        raise RuleSkipped("register tile not applicable / already pinned")
    if op.algebra is AlgebraKind.SEMIRING:
        offers = reduce_reg_offers(op.dag, Budget(), op.knobs[RED_FK.name])
    else:
        offers = map_reg_offers(op.dag, Budget())
    if not offers:
        raise RuleSkipped("no legal register tile")
    return [replace(op, knobs={**op.knobs, **reg_knobs(op.dag, reg)}) for reg in offers]
