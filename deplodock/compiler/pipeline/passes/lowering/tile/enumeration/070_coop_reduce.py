"""MONOID pass (fork) — the one pass for the cooperative reduce AND the streaming flash.

A ``MONOID`` nest lowers through ONE build move (``_build.build_monoid``) regardless of
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

Both regimes are **one** :class:`MonoidReduction` composition (``op.dag.reduction`` — an associative
carrier folding a primary reduce axis, optionally composed over an inner SEMIRING ``Contraction``),
and ``reduction_build`` is the **one dispatch** that routes it: ``inner is None`` → the cooperative
leaves; a streaming reduction → the streaming leaves; an eligible streaming reduction → the
warp-tier flash. The three regimes are branches of the ONE ``build_monoid``, selected by the carrier
``Combiner`` it is handed (``ScalarCombiner`` / ``MmaTwist``) + the score-placement knob — not three
functions.

A streaming flash deploys the **warp-tier chain** off **general DAG invariants**, never a flash
shape match — the dispatch reads three orthogonal facts: (1) the reduction composes an inner
contraction (``reduction.inner``), (2) it **tensorizes** (``_atom.inner_atomizes`` — the SAME
``cell_atomizes`` atom-fit the standalone SEMIRING warp matmul gates on, applied to
``reduction.inner`` with its own score coords, so the 16-bit-operand / ``D%cell_k`` /
classifiable-cell facts live in one shared predicate; a warp chain is just a ``MonoidReduction``
whose inner contraction tensorizes), AND (3) the realizer can build it and policy deploys it. The
score coords are a DAG invariant on the inner ``Contraction`` (``out_index``); the free-axis
geometry is walked off the composition at emit time (``_iterdag.chain_free_axes``), so neither
routing nor the build moves walk the lowered tile to recover them. The v1 realizer's scope ceilings
(``_warp_chain_buildable`` — no additive mask, ``D≤256``) and the deployment policy
(``_deploy_warp_chain`` — symbolic-default, static under ``DEPLODOCK_CHAIN``) are named, orthogonal
guards: what the realizer can build today + an env/extent policy, not graph facts.
This pass then hands the logical seed to ``_build.build_monoid`` (with ``combiner=MmaTwist``), which
σ-tiles + atomizes the two chained contractions (stamping the kv-stream ``Schedule.carry`` + the
score→A handoff edge);
assembly's generic ``_assemble.carry_scope_from_graph`` walk then realizes the fragment-tier
online-softmax around those cells (the former ``split/005_warp_chain`` route, folded in here).
Symbolic is the deployed default (the ~100× win); static is a ``DEPLODOCK_CHAIN`` opt-in.
This single fork owns the regime end to end:
the scalar passes ``090``/``100``/``110`` and ``120_stage`` gate off ``MONOID`` (a monoid
reduce stays smem-free — each lane reads its own ``K_c``-strided slice, no cross-thread reuse).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.ir.twist import MmaTwist, ScalarCombiner
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import inner_atomizes
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import build_monoid
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


def _dtype_of(op: TileGraphOp):
    """A buffer-name → dtype lookup over the op's buffers (the atom-fit gate reads operand
    dtypes), ``None`` for an unknown name."""
    return lambda n: op.buffers[n].dtype if n in op.buffers else None


def _warp_chain_buildable(op: TileGraphOp) -> bool:
    """The v1 fragment-softmax realizer's scope **ceilings** — kept SEPARATE from the eligibility
    invariant (``reduction.inner`` + ``inner_atomizes``) because they are "what the realizer can
    build today", not "is this a warp chain". The realizer folds no **additive mask** (a 4th rank-4
    input — itself a structural ``Add``, so a later realizer could handle it like the causal
    ``Select`` rather than decline), and bounds the head dim at ``256`` (register / smem pressure).
    Shrinks toward empty as the realizer generalizes."""
    block = op.tilegraph.blocks[0]
    ins = [b for n, b in op.buffers.items() if len(b.shape) == 4 and n != block.writes[0].buffer]
    if len(ins) != 3:  # a 4th rank-4 input is an additive mask
        return False
    d = op.dag.reduction.inner.loop.axis.extent
    return d.is_static and d.as_static() <= 256


def _deploy_warp_chain(op: TileGraphOp, reduction) -> bool:
    """The warp-tier chain's **deployment policy** — orthogonal to eligibility (it gates on env +
    the runtime extent, not the graph). A **symbolic** KV stream deploys by default (the ~100×
    win); a **static** stream stays the scalar nest unless ``DEPLODOCK_CHAIN`` opts in, and then
    only when it is 16-aligned (the warp tile owns a 16-key slab). The hinge (KV stream) extent IS
    the deployment axis."""
    seq = reduction.hinge.axis.extent
    if seq.is_static:
        return fam.pin_inline_chain() and seq.as_static() % 16 == 0 and seq.as_static() >= 16
    return True


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if op.algebra is not AlgebraKind.MONOID or fam.reduce_key(op.dag.k_node.loop.axis.name) in op.knobs:
        raise RuleSkipped("MONOID build applies once, to a MONOID seed")
    leaves = reduction_build(ctx, op)
    if not leaves:
        raise RuleSkipped("no legal MONOID decomposition")
    return leaves


def reduction_build(ctx: Context, op: TileGraphOp) -> list[TileGraphOp]:
    """The **unified MONOID codegen dispatch** — read the one :class:`MonoidReduction` and route it to
    the ONE build move ``_build.build_monoid``, varying only the carrier ``Combiner`` it is handed
    (and the score-placement knob), never the function (``op.dag.reduction``):

    - ``inner`` present + tensorizes + buildable + deployable → the **warp-tier tensor-core flash**
      (``build_monoid(combiner=MmaTwist)`` — a terminal leaf; the later scalar passes gate off MONOID).
    - a **streaming** reduction → the streaming leaves (``build_monoid(combiner=ScalarCombiner)``; the
      scalar FA-2 geometry when the inner chain applies — signalled by the ``INLINE`` score knob — else
      the serial-stream tower).
    - a **flat** reduction (``inner is None``, not streaming) → the cooperative leaves
      (``build_monoid(combiner=ScalarCombiner)``).

    Warp-tier flash routes off **general DAG invariants**, never a flash shape match: a composed
    inner contraction (``reduction.inner``) that independently tensorizes (``inner_atomizes`` — the
    SAME atom-fit the standalone SEMIRING matmul gates on). The realizer scope ceiling
    (``_warp_chain_buildable``) and the deployment policy (``_deploy_warp_chain`` — symbolic-default,
    static under ``DEPLODOCK_CHAIN``) are explicit, orthogonal guards: what the v1 realizer can build
    today + an env/extent policy, not graph facts. The three regimes are the honest 2×2 of (chained
    pair vs single contraction) × (scalar vs warp) — branches of ONE ``build_monoid``, parametrized by
    the combiner, not three functions (they still produce genuinely different kernels — a cooperative
    tree-reduce, a scalar FA-2 stream, a tensor-core flash)."""
    reduction = op.dag.reduction
    if (
        reduction.inner is not None
        and inner_atomizes(reduction.inner, compute_capability=ctx.compute_capability, dtype_of=_dtype_of(op))
        and _warp_chain_buildable(op)
        and _deploy_warp_chain(op, reduction)
    ):
        # ``build_monoid(combiner=MmaTwist)`` σ-tiles the two chained contractions to the warp geometry
        # and fuses them via the generic ``atomize_cell`` (stamping the kv-stream ``Schedule.carry`` +
        # the score→A handoff edge); assembly's generic ``carry_scope_from_graph`` walk realizes the
        # fragment-tier phases (softmax / scale / mask / handoff / epilogue) around those cells.
        return [replace(op, tilegraph=build_monoid(op, op.knobs, combiner=MmaTwist))]
    return _streaming_leaves(op) if reduction.nested else _coop_leaves(op)


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
        tg = build_monoid(op, knobs, combiner=ScalarCombiner)
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
    # Cross-CTA split-KV (Flash-Decoding) of the streaming flash — **pin-gated** (an explicit
    # ``REDUCE`` ``c<cta>`` on the KV stream): each CTA folds a slice of the KV stream into its
    # own partial ``(m, l, O)`` state, ``150_cross_cta_finalize`` writes the 3 state workspaces
    # and the carrier-generic combine merges them (``deferred_combine_tilegraph``). Static KV only
    # (the slice must divide); the serial-stream geometry carries the split-K grid (the FA-2
    # shared-score geometry doesn't thread it yet), so cta>1 forces the serial-stream path (no
    # INLINE score knob).
    _, _, sk_pin, _ = fam.reduce_fields(op.dag, op.dag.k_node.loop.axis.name)
    stream_ext = op.dag.k_node.loop.axis.extent
    cta = sk_pin if (sk_pin and sk_pin > 1 and stream_ext.is_static and stream_ext.as_static() % sk_pin == 0) else 1
    out: list[TileGraphOp] = []
    for br in streaming_br_offers(op.dag):
        budget = Budget(max_threads=max(1, MAX_THREADS_PER_CTA // br))
        offers = [t for t in thread_offers(op.dag, budget) if streaming_coop_geometry_ok(br, t[0] * t[1])]
        # A **symbolic** streaming (KV) axis is serial-locked (``BR = BK = 1``). With a
        # carried-contraction chain, the FA-2 shared-score geometry (``build_monoid`` with the score
        # placed INLINE) makes it efficient — the QK^T score is computed ONCE per KV step and shared
        # across the P@V output ``d`` (register vector ``O[d]``), not recomputed per ``d``. That is
        # the symbolic DEFAULT: the serial-stream geometry would recompute the score per ``d`` and run
        # unboundedly long (Finding 1, qwen3-emb-0.6b layer 0). A static stream keeps the
        # pin-gated opt-in (greedy stays the scalar nest until the search-fork integration).
        # Split-KV (cta>1) rides the serial-stream geometry — force it (the FA-2 chain doesn't thread
        # the split-K grid yet) and stamp the bare ``c<cta>`` finalize-pending.
        use_chain = cta == 1 and _chain_applicable(op, br) and (symbolic or fam.pin_inline_chain())
        # Without a chain (a streaming monoid with no inner contraction) a symbolic axis
        # falls back to the serial-stream geometry, where the free-axis tile can't move
        # the reduce-bound kernel — so collapse the futile fork to one canonical leaf.
        if symbolic and not use_chain:
            offers = offers[:1]
        for t in offers:
            knobs = {
                **op.knobs,
                **free_split_knobs(op.dag, t, (1, 1)),  # complete SPLIT, register forced to 1
                fam.reduce_key(op.dag.k_node.loop.axis.name): fam.enc_reduce(serial=bk, fold=1, cta=cta, coop=br),
            }
            if use_chain:
                # Phase 1c: the FA-2 shared-score restructuring — placing the score INLINE is the only
                # signal ``build_monoid`` needs to pick the register-``O[d]`` shared-score geometry over
                # the serial-stream tower (same combiner, same call; the knob drives the geometry).
                knobs[fam.place_key(op.dag.reduction.score)] = fam.INLINE
            tg = build_monoid(op, knobs, combiner=ScalarCombiner)
            out.append(replace(op, tilegraph=tg, knobs=knobs))
    return out


def _chain_applicable(op: TileGraphOp, br: int) -> bool:
    """Whether the FA-2 shared-score geometry covers this nest: a streaming reduction with a composed
    inner contraction (``reduction.inner``) and no cooperative-KV (``BR == 1`` — the cooperative combine
    isn't wired through the split carrier yet). The **hinge** (KV stream) axis MAY be symbolic — the
    FA-2 geometry keeps it a serial runtime-bounded loop (no tiling → no masking, every
    ``kv < seq_len`` is valid); every OTHER contraction (the inner QK^T score reduce) must be static,
    since the score is a register-shared reduce rather than a masked one."""
    reduction = op.dag.reduction
    if reduction is None or reduction.inner is None or br != 1:
        return False
    return all(n.loop.axis.extent.is_static for n in op.dag.reduce if n.loop.axis.name != reduction.hinge_name)
