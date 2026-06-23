"""Cooperative-reduce pass (fork) — the ``MONOID`` whole-CTA reduce.

``plans/tile-ir-block-dag.md`` R2: a ``MONOID`` nest (a plain reduce — softmax LSE /
rmsnorm stat / mean / max) partitions its contraction axis across ``BR`` cooperative
THREAD lanes (the carrier's commutative-licensed partition placed on a THREAD axis,
not split-K's BLOCK axis), then a warp-shuffle / hierarchical-tree combine folds the
per-thread partials. The combine is **not** in the body — the reduce ``Accum.axes``
carry the ``K_c`` lane through σ and ``kernel/100_materialize_tile`` synthesizes it.

Unlike the scalar ``SEMIRING`` chain (``010``/``020``/``030`` — three sequential
forks), a cooperative reduce is one ``(bk, fk, br)`` decision (legacy
``build_coop_reduce_tree``), so this is a single fork that applies the whole
``coop_build`` body move per leaf. The free-axis THREAD tiles (``BN``/``BM``, default
1 = whole-CTA, or the pinned strided-cooperative value) ride every leaf. The scalar
passes ``020``/``030``/``040`` and ``050_stage`` gate off ``MONOID`` (a cooperative
reduce stays smem-free — each lane reads its own ``K_c``-strided slice with no
cross-thread reuse), so this pass owns the regime end to end.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import coop_build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import COOP_BR, MAP_M_REG, MAP_N_REG, RED_SPLITK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    coop_free_thread_knobs,
    coop_reduce_knobs,
    coop_reduce_offers,
)

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.MONOID or COOP_BR.name in op.knobs:
        raise RuleSkipped("cooperative reduce applies once, to a MONOID seed")
    offers = coop_reduce_offers(op.dag)
    if not offers:
        raise RuleSkipped("no legal cooperative-reduce decomposition")
    free_knobs = coop_free_thread_knobs(op.dag)
    out = []
    for r in offers:
        # The free-axis register tile is forced to 1 (one element per cell-owner);
        # SPLITK=1 (the cooperative partition rides THREAD, no cross-CTA split).
        # MMA / WM / WN / STAGE OFF-fill via apply_off_defaults (warp / staging off).
        knobs = {**op.knobs, **free_knobs, **coop_reduce_knobs(r), MAP_N_REG.name: 1, RED_SPLITK.name: 1}
        if op.dag.outer_m is not None:
            knobs[MAP_M_REG.name] = 1
        tg = coop_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
