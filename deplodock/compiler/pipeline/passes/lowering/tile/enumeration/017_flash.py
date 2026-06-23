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

**Cooperative-KV (``BR > 1``).** A pinned ``DEPLODOCK_BR`` partitions the **static**
streaming KV axis across ``BR`` cooperative THREAD lanes (the same
commutative-licensed THREAD partition the ``MONOID`` coop reduce uses, here on the
streaming ``TWISTED_MONOID`` carrier): each lane streams a strided KV slice into its
own ``(m, l, O)`` online-softmax partial, and the per-lane partials merge via the
carrier's ``combine_states`` (``kernel/100_materialize_tile`` emits the warp-shuffle /
smem-tree combine, exactly as for a plain monoid reduce). The free-axis THREAD tile is
budgeted by ``BR`` and the cross-lane layout is constrained (``flash_coop_geometry_ok``:
whole-CTA tree vs strided intra-warp segment). Default ``BR = 1`` keeps the serial-KV
streaming form, so cooperative-KV is opt-in and a symbolic streaming KV stays serial.
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
    COOP_BR,
    MAP_M_REG,
    MAP_N_REG,
    MAP_N_THREAD,
    MAX_THREADS_PER_CTA,
    RED_BK,
    RED_FK,
    RED_SPLITK,
)
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    Budget,
    flash_br_offers,
    flash_coop_geometry_ok,
    thread_knobs,
    thread_offers,
)

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.TWISTED_MONOID or MAP_N_THREAD.name in op.knobs:
        raise RuleSkipped("flash applies once, to a TWISTED_MONOID seed")
    # Cooperative-KV (``BR > 1``): the streaming KV axis splits across BR THREAD lanes,
    # the per-lane online-softmax partials merged via the carrier's cross-thread
    # ``combine_states``. Default ``BR = 1`` (the serial-KV streaming form) — the
    # cooperative split only fires when pinned (``DEPLODOCK_BR``). The free-axis THREAD
    # tile is budgeted by BR (BR lanes share the CTA), and the cross-lane combine
    # geometry constrains the free×BR layout (whole-CTA tree vs strided intra-warp).
    out: list[TileGraphOp] = []
    for br in flash_br_offers(op.dag):
        offers = thread_offers(op.dag, Budget(max_threads=max(1, MAX_THREADS_PER_CTA // br)))
        for t in offers:
            if not flash_coop_geometry_ok(br, t[0] * t[1]):
                continue
            # FM=FN=1 (the streaming carrier can't span register cells); BK=FK=SPLITK=1
            # (no intra/cross-CTA K split — each output element streams its own KV).
            knobs = {
                **op.knobs,
                **thread_knobs(op.dag, t),
                MAP_N_REG.name: 1,
                RED_BK.name: 1,
                RED_FK.name: 1,
                RED_SPLITK.name: 1,
                COOP_BR.name: br,
            }
            if op.dag.outer_m is not None:
                knobs[MAP_M_REG.name] = 1
            tg = flash_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
            out.append(replace(op, tilegraph=tg, knobs=knobs))
    if not out:
        raise RuleSkipped("no legal free-axis thread tile for the flash nest")
    return out
