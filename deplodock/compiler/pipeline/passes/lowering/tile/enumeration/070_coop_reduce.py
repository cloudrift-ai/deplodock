"""MONOID pass (fork) — the one pass for the cooperative reduce AND the streaming flash.

``plans/tile-ir-block-dag.md`` R2/R6 + ``plans/tensor-core-streaming-flash-mma.md`` Phase 0:
a ``MONOID`` nest lowers through ONE build move (``_build.monoid_build``) regardless of
whether it is a **flat** reduce (softmax LSE / rmsnorm stat / mean / max) or a **nested**
streaming flash (online-softmax over a nested QK^T contraction — a twisted monoid is a
monoid, selected structurally not as a distinct kind). The move applies the
reduce-decomposition tower to **each** contraction axis the DAG exposes (one for a flat
monoid; the outer KV stream + the inner QK^T for a nested one), with the carrier's
commutative-licensed ``K_c`` THREAD lane (``BR`` cooperative lanes per row) on the primary
axis. The cross-thread / cross-lane combine is **not** in the body — the carrier's ``axes``
(``Accum.axes`` / ``Monoid.axes``) carry ``K_c`` through σ and ``kernel/100_materialize_tile``
+ ``kernel/_combine`` synthesize the warp-shuffle / tree / online-softmax rescale.

What differs between the two is only the **offer set** (the algebra-conditioned ranking
heuristic, not a code path): a flat reduce searches ``(bk, fk, br)`` over the whole-CTA /
strided-cooperative free tile (``coop_reduce_offers``); a streaming flash searches the
free-axis THREAD tile (``thread_offers``) with ``BK = FK = 1`` and ``BR`` over the **static**
KV axis (``streaming_br_offers`` — cooperative-KV, opt-in; a symbolic streaming axis stays
serial). Both pin ``REGISTER = 1`` (one element per cell-owner) and ``SPLITK = 1`` (the
partition rides THREAD, not a cross-CTA split). This single fork owns the regime end to end:
the scalar passes ``090``/``100``/``110`` and ``120_stage`` gate off ``MONOID`` (a monoid
reduce stays smem-free — each lane reads its own ``K_c``-strided slice, no cross-thread reuse).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import monoid_build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    COOP_BR,
    MAP_M_REG,
    MAP_N_REG,
    MAX_THREADS_PER_CTA,
    RED_BK,
    RED_FK,
    RED_SPLITK,
)
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    Budget,
    coop_free_thread_knobs,
    coop_reduce_knobs,
    coop_reduce_offers,
    streaming_br_offers,
    streaming_coop_geometry_ok,
    thread_knobs,
    thread_offers,
)

PATTERN = [Pattern("root", TileGraphOp)]


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.MONOID or COOP_BR.name in op.knobs:
        raise RuleSkipped("MONOID build applies once, to a MONOID seed")
    leaves = _streaming_leaves(op) if op.dag.streaming else _coop_leaves(op)
    if not leaves:
        raise RuleSkipped("no legal MONOID decomposition")
    return leaves


def _coop_leaves(op: TileGraphOp) -> list[TileGraphOp]:
    """Flat cooperative reduce: the ``(bk, fk, br)`` decomposition × the whole-CTA /
    strided-cooperative free THREAD tile (default ``BN = BM = 1``, whole-CTA)."""
    offers = coop_reduce_offers(op.dag)
    free_knobs = coop_free_thread_knobs(op.dag)
    out: list[TileGraphOp] = []
    for r in offers:
        # REGISTER forced to 1 (one element per cell-owner); SPLITK=1 (the cooperative
        # partition rides THREAD, no cross-CTA split). MMA / WM / WN OFF-fill downstream.
        knobs = {**op.knobs, **free_knobs, **coop_reduce_knobs(r), MAP_N_REG.name: 1, RED_SPLITK.name: 1}
        if op.dag.outer_m is not None:
            knobs[MAP_M_REG.name] = 1
        tg = monoid_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out


def _streaming_leaves(op: TileGraphOp) -> list[TileGraphOp]:
    """Streaming flash: the free-axis THREAD tile × cooperative ``BR`` over the static KV
    axis, with ``BK = FK = SPLITK = 1`` (each output element streams its own reduction;
    the coupled ``Monoid`` carrier can't span register cells or split-K). ``BR > 1`` lays
    the ``K_c`` lane on the streaming axis (cooperative-KV), constrained by the cross-lane
    combine geometry (whole-CTA tree vs strided intra-warp segment)."""
    out: list[TileGraphOp] = []
    for br in streaming_br_offers(op.dag):
        offers = thread_offers(op.dag, Budget(max_threads=max(1, MAX_THREADS_PER_CTA // br)))
        for t in offers:
            if not streaming_coop_geometry_ok(br, t[0] * t[1]):
                continue
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
            tg = monoid_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
            out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out
