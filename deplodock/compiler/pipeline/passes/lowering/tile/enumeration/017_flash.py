"""Flash pass (fork) — the streaming ``TWISTED_MONOID`` online-softmax attention.

``plans/tile-ir-block-dag.md`` R6: a ``TWISTED_MONOID`` nest (a fused flash-attention
kernel — the ``loop/recognize/010_recognize_flash`` rewrite produces one streaming
online-softmax kernel) tiles its free output axes (q-rows / head-dim) like a pointwise
nest, then serial-transforms BOTH contraction axes (the streaming KV reduce + its
nested QK^T reduce) with ``bk=fk=splitk=1`` — each output element streams its own KV;
the coupled ``Monoid`` carrier can't span register cells or split-K. The carrier's
m/l/O rescale is **not** a search dimension and rides through σ untouched
(``kernel/100_materialize_tile`` + ``kernel/_combine`` synthesize it).

Like the cooperative reduce (``015_coop_reduce``), this is one fork that owns the
regime end to end: it enumerates the free-axis THREAD tile (``thread_offers``) with the
register tile forced to ``FM=FN=1``, pins ``BK=FK=SPLITK=1``, and applies the whole
``flash_build`` body move per leaf. The scalar / warp / coop passes gate off
(``005``/``010`` need ``SEMIRING``, ``015`` needs ``MONOID``); ``040_seal`` stamps the
scalar-tier OFF sentinels and ``050_stage`` skips (flash is smem-free — the streaming
carrier reuses no cross-thread slab). Mirrors the legacy ``build_flash_tree``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import flash_build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    MAP_M_REG,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    RED_SPLITK,
)
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import Budget, thread_knobs, thread_offers

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.TWISTED_MONOID or MAP_N_THREAD.name in op.knobs:
        raise RuleSkipped("flash applies once, to a TWISTED_MONOID seed")
    offers = thread_offers(op.dag, Budget())
    if not offers:
        raise RuleSkipped("no legal free-axis thread tile for the flash nest")
    out: list[TileGraphOp] = []
    for t in offers:
        # FM=FN=1 (the streaming carrier can't span register cells); BK=FK=SPLITK=1
        # (no intra/cross-CTA K split — each output element streams its own KV).
        knobs = {**op.knobs, **thread_knobs(op.dag, t), MAP_N_REG.name: 1, RED_BK.name: 1, RED_FK.name: 1, RED_SPLITK.name: 1}
        if op.dag.outer_m is not None:
            knobs[MAP_M_REG.name] = 1
        tg = flash_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
