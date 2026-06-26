"""MONOID pass (fork) — the one pass for the cooperative reduce AND the streaming flash.

A ``MONOID`` nest lowers through ONE build move (``_build.monoid_build``) regardless of
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
partition rides THREAD, not a cross-CTA split).

A streaming flash in the **16-bit tensor-core scope** (``_is_warp_flash`` — fp16/bf16,
``D%16==0``, GQA-or-equal-head, static ``S%16==0`` OR symbolic ``seq_len``) instead deploys the
**warp-tier chain**: this pass marks the kv-stream axis ``Schedule.carry`` and hands the logical
seed to assembly, where ``_assemble.realize_flash`` realizes the fragment-tier online-softmax (the
former ``split/005_warp_chain`` route, folded in here). Symbolic is the deployed default (the
~100× win); static is a ``DEPLODOCK_CHAIN`` opt-in. This single fork owns the regime end to end:
the scalar passes ``090``/``100``/``110`` and ``120_stage`` gate off ``MONOID`` (a monoid
reduce stays smem-free — each lane reads its own ``K_c``-strided slice, no cross-thread reuse).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock import config
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering._flash_geom import flash_params
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import chain_build, monoid_build, warp_chain_build
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import MAX_THREADS_PER_CTA
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    Budget,
    coop_free_thread_knobs,
    coop_reduce_knobs,
    coop_reduce_offers,
    free_split_knobs,
    streaming_br_offers,
    streaming_coop_geometry_ok,
    thread_offers,
)


def _streaming_bk(dag) -> int:
    """The KV-tile factor ``BK`` re-bracketing the streaming reduce,
    ``S_k → S_k/BK · BK``. Honored from a
    ``DEPLODOCK_BK`` pin only when it divides EVERY reduce extent the move tiles (the KV
    stream + the nested QK^T), since ``_rebracket_k`` applies it uniformly; else ``1``
    (the serial-stream default). A symbolic axis (no static extent) stays ``1``.

    This is the **serial** re-bracket (each ``K_i`` step still folds one key through the
    ``Monoid``); it makes ``BK`` a realized knob on the streaming axis — closing the Phase-0
    validator-collapse gap (``BK`` was legal on the MONOID tier but forced to 1 here) — and
    is the foundation for the register-score / P@V-cell surfacing the tensor-core tier needs.
    Default (unpinned) is ``1``, so greedy / the scalar tier are unchanged."""
    bk = fam.reduce_fields(dag, dag.k_node.loop.axis.name)[0]
    if not bk or bk <= 1:
        return 1
    for n in dag.reduce:
        ext = n.loop.axis.extent
        if not ext.is_static or ext.as_static() % bk != 0:
            return 1
    return bk


PATTERN = [Pattern("root", TileGraphOp)]


def _is_warp_flash(op: TileGraphOp) -> bool:
    """Whether this streaming flash deploys the **warp-tier tensor-core chain** (vs the
    cooperative / scalar streaming nest): a carried-contraction chain in the 16-bit warp scope
    (the shared ``flash_params`` — fp16/bf16, ``D%16==0`` with ``16 ≤ D ≤ 256``, GQA group
    dividing H, static ``S%16==0`` OR symbolic) AND either a symbolic ``seq_len`` (the deployed
    default — the ~100× win) or a static shape under ``DEPLODOCK_CHAIN``. (Folds the former
    ``split/005_warp_chain`` routing shim into this MONOID fork.)"""
    if op.dag.chain is None:
        return False
    fp = flash_params(op.buffers, op.tilegraph.blocks[0].writes[0].buffer)
    if fp is None:
        return False
    if not fp.symbolic and not fam.pin_inline_chain():
        return False
    if fp.group < 1 or fp.H % fp.group != 0 or fp.D % 16 != 0 or not (16 <= fp.D <= 256):
        return False
    return True if fp.symbolic else (fp.S % 16 == 0 and fp.S >= 16)


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.MONOID or fam.reduce_key(op.dag.k_node.loop.axis.name) in op.knobs:
        raise RuleSkipped("MONOID build applies once, to a MONOID seed")
    if op.dag.streaming and _is_warp_flash(op):
        # Warp-tier tensor-core flash: mark the kv-stream axis ``Schedule.carry`` (the
        # fragment-tier realization) and hand the seed to assembly — no build move, no cooperative
        # leaves (the warp chain replaces them here, matching the old ``split/005_warp_chain``
        # route). ``assembly/010_assemble`` dispatches on ``carry``. A terminal leaf: the later
        # scalar passes gate off MONOID. Under ``DEPLODOCK_FLASHWALK`` the seed is σ-tiled +
        # atomized by ``warp_chain_build`` (it stamps ``carry`` itself) so assembly's generic
        # ``carry_scope_from_graph`` walk realizes it; else the seed is marked for ``realize_flash``.
        if config.flash_walk():
            return [replace(op, tilegraph=warp_chain_build(op))]
        sched = replace(op.tilegraph.schedule, carry=frozenset({op.dag.chain.hinge_name}))
        return [replace(op, tilegraph=replace(op.tilegraph, schedule=sched))]
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
        # ``free_knobs`` already completes SPLIT@<axis> with reg=1 (one element per
        # cell-owner — the MONOID tier has no register fork); the cooperative partition
        # rides THREAD (``coop`` factor), no cross-CTA split (``cta=1``, the REDUCE
        # default). MMA / WM / WN OFF-fill downstream.
        knobs = {**op.knobs, **free_knobs, **coop_reduce_knobs(op.dag, r)}
        tg = monoid_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
        out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out


def _streaming_leaves(op: TileGraphOp) -> list[TileGraphOp]:
    """Streaming flash: the free-axis THREAD tile × cooperative ``BR`` over the static KV
    axis, with ``FK = SPLITK = 1`` (the coupled ``Monoid`` carrier can't span register
    cells or split-K) and ``BK`` the KV-tile re-bracket of the streaming axis (Phase 1 —
    default 1, honored from a ``DEPLODOCK_BK`` pin via :func:`_streaming_bk`). ``BR > 1``
    lays the ``K_c`` lane on the streaming axis (cooperative-KV), constrained by the
    cross-lane combine geometry (whole-CTA tree vs strided intra-warp segment)."""
    bk = _streaming_bk(op.dag)  # Phase 1: KV-tile re-bracket of the streaming axis (pin-honored)
    symbolic = op.dag.k_bound is not None
    out: list[TileGraphOp] = []
    for br in streaming_br_offers(op.dag):
        budget = Budget(max_threads=max(1, MAX_THREADS_PER_CTA // br))
        offers = [t for t in thread_offers(op.dag, budget) if streaming_coop_geometry_ok(br, t[0] * t[1])]
        # A **symbolic** streaming (KV) axis is serial-locked (``BR = BK = 1``). With a
        # carried-contraction chain, ``chain_build`` (the FA-2 shared-score restructuring)
        # makes it efficient — the QK^T score is computed ONCE per KV step and shared across
        # the P@V output ``d`` (register vector ``O[d]``), not recomputed per ``d``. That is
        # the symbolic DEFAULT: ``monoid_build`` would recompute the score per ``d`` and run
        # unboundedly long (Finding 1, qwen3-emb-0.6b layer 0). A static stream keeps the
        # pin-gated opt-in (greedy stays the scalar nest until the search-fork integration).
        use_chain = _chain_applicable(op, br) and (symbolic or fam.pin_inline_chain())
        # Without a chain (a streaming monoid with no inner contraction) a symbolic axis
        # falls back to ``monoid_build``'s serial stream, where the free-axis tile can't move
        # the reduce-bound kernel — so collapse the futile fork to one canonical leaf.
        if symbolic and not use_chain:
            offers = offers[:1]
        for t in offers:
            knobs = {
                **op.knobs,
                **free_split_knobs(op.dag, t, (1, 1)),  # complete SPLIT, register forced to 1
                fam.reduce_key(op.dag.k_node.loop.axis.name): fam.enc_reduce(serial=bk, fold=1, cta=1, coop=br),
            }
            if use_chain:
                # Phase 1c: the FA-2 shared-score restructuring (register O[d] + the score
                # edge placed INLINE + the split carrier).
                knobs[fam.place_key(op.dag.chain.score)] = fam.INLINE
                tg = chain_build(op.tilegraph, op.dag, knobs)
            else:
                tg = monoid_build(op.tilegraph, op.dag, knobs, target_names=op.target_names)
            out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out


def _chain_applicable(op: TileGraphOp, br: int) -> bool:
    """Whether ``chain_build`` covers this nest: a carried-contraction chain and no
    cooperative-KV (``BR == 1`` — the cooperative combine isn't wired through the split
    carrier yet). The **hinge** (KV stream) axis MAY be symbolic — ``chain_build`` keeps it
    a serial runtime-bounded loop (no tiling → no masking, every ``kv < seq_len`` is valid);
    every OTHER contraction (the inner QK^T score reduce) must be static, since the score is
    a register-shared reduce rather than a masked one."""
    chain = op.dag.chain
    if chain is None or br != 1:
        return False
    return all(n.loop.axis.extent.is_static for n in op.dag.reduce if n.loop.axis.name != chain.hinge_name)
