"""Streaming pass (fork) — the streaming ``TWISTED_MONOID`` regime (e.g. online-softmax).

``plans/tile-ir-block-dag.md`` R6: a ``TWISTED_MONOID`` nest (e.g. a fused flash-attention
kernel — the ``loop/recognize/010_recognize_flash`` rewrite produces one streaming
online-softmax kernel) tiles its free output axes (for attention, q-rows / head-dim)
like a MAP nest, then serial-transforms BOTH contraction axes (the streaming reduce + its
nested inner reduce) with ``bk=fk=splitk=1`` — each output element streams its own
reduction; the coupled ``Monoid`` carrier can't span register cells or split-K. The
carrier's m/l/O rescale is **not** a search dimension and rides through σ untouched
(``kernel/100_materialize_tile`` + ``kernel/_combine`` synthesize it).

Like the cooperative reduce (``015_coop_reduce``), this is one fork that owns the
regime end to end: it enumerates the free-axis THREAD tile (``thread_offers``) with the
register tile forced to ``FM=FN=1``, pins ``BK=FK=SPLITK=1``, and applies the whole
``streaming_build`` body move per leaf. The scalar / warp / coop passes gate off
(``005``/``010`` need ``SEMIRING``, ``015`` needs ``MONOID``); ``040_seal`` stamps the
scalar-tier OFF sentinels and ``050_stage`` skips (a streaming nest is smem-free — the
streaming carrier reuses no cross-thread slab).

**Cooperative stream (``BR > 1``).** A pinned ``DEPLODOCK_BR`` partitions the **static**
streaming axis (for attention, the KV axis) across ``BR`` cooperative THREAD lanes (the
same commutative-licensed THREAD partition the ``MONOID`` coop reduce uses, here on the
streaming ``TWISTED_MONOID`` carrier): each lane streams a strided slice into its own
``(m, l, O)`` online-softmax partial, and the per-lane partials merge via the carrier's
``combine_states`` (``kernel/100_materialize_tile`` emits the warp-shuffle / smem-tree
combine, exactly as for a plain monoid reduce). The free-axis THREAD tile is budgeted by
``BR`` and the cross-lane layout is constrained (``streaming_coop_geometry_ok``: whole-CTA
tree vs strided intra-warp segment). Default ``BR = 1`` keeps the serial-stream form, so
cooperative streaming is opt-in and a symbolic streaming axis stays serial.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import streaming_build
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
    streaming_br_offers,
    streaming_coop_geometry_ok,
    thread_knobs,
    thread_offers,
)

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.TWISTED_MONOID or MAP_N_THREAD.name in op.knobs:
        raise RuleSkipped("streaming applies once, to a TWISTED_MONOID seed")
    # Cooperative stream (``BR > 1``): the streaming axis splits across BR THREAD lanes,
    # the per-lane online-softmax partials merged via the carrier's cross-thread
    # ``combine_states``. Default ``BR = 1`` (the serial-stream form) — the
    # cooperative split only fires when pinned (``DEPLODOCK_BR``). The free-axis THREAD
    # tile is budgeted by BR (BR lanes share the CTA), and the cross-lane combine
    # geometry constrains the free×BR layout (whole-CTA tree vs strided intra-warp).
    out: list[TileGraphOp] = []
    for br in streaming_br_offers(op.dag):
        offers = thread_offers(op.dag, Budget(max_threads=max(1, MAX_THREADS_PER_CTA // br)))
        for t in offers:
            if not streaming_coop_geometry_ok(br, t[0] * t[1]):
                continue
            # FM=FN=1 (the streaming carrier can't span register cells); BK=FK=SPLITK=1
            # (no intra/cross-CTA K split — each output element streams its own reduction).
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
            tg = streaming_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
            out.append(replace(op, tilegraph=tg, knobs=knobs))
    if not out:
        raise RuleSkipped("no legal free-axis thread tile for the streaming nest")
    return out
