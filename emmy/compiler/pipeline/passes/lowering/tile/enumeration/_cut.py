"""The cut-offer policy — the derived tier-monotonicity predicate (R7 edge placement).

The cut-decision module. All cut-offer logic lives here, one auditable place, so the
fork pass (``split/010_split_demoted``) holds **no** decision logic and ``eval``-style
introspection can ask "what cut does ``_cut`` offer for this kernel, and why" directly
— mirroring how ``_stage.py`` owns the staging offer set and ``_atom.py`` owns atom
eligibility.

The offer is a **derived-view query**, not a per-op cone pattern-match: every demoted
matmul is the *same* trigger — inlining a computed operand forces its consumer matmul
below the warp tier, and materializing the operand restores it. So "is this worth
cutting" generalizes to a **monotonicity check on a derived tier lattice**:

> Each compute body has a derived **best achievable tier** — ``tier(dag)`` — a function
> of its carrier + atom eligibility (``MAP`` < ``SCALAR_REDUCE`` < ``COOP_REDUCE`` <
> ``WARP_MMA``, with ``UNBUILDABLE`` below them for a body the move composer can't
> lower at all). **Offer a cut iff materializing the demoted operand strictly raises
> the consumer's maximal tier.**

Mechanically this is the existing ``eligible_atoms`` / ``classify`` machinery read on
the fused (inline) body: a demoted matmul's cone operand defeats the clean
``[Load, Load, mul, Accum]`` cell, so ``classify`` returns ``None`` and ``tier`` is
``UNBUILDABLE`` — exactly when the fused form has no buildable regime. The fission
(``_extract.extract_block``) then materializes the operand into a clean ``Load``,
so the consumer rebuilds at a real tier (``WARP_MMA`` for an f16 gemm, ``SCALAR_REDUCE``
for an f32 one); the tier strictly rises. A demoted cell with a cone operand is **never**
atom-eligible (the cell isn't ``[Load, Load, mul, Accum]``), so ``tier(inline) is
UNBUILDABLE`` holds *iff* ``classify(fused) is None`` — the predicate reproduces the
legacy force condition exactly while expressing it through the lattice.

**Why it is tight — exactly today's cuts, no more.** A pointwise→pointwise body is
``MAP`` inline → no drop → no offer; a clean gemm (plain-load operands) is already at
its max tier inline → no offer. The strict inequality fires only where fusion is
provably lossy (a layout-changing / reduction cone — rotary, softmax-norm — that cannot
preserve the atom layout inline). Profitability stays the search's job: a ``GMEM`` cut
still pays a gmem round-trip, so the offer is a **necessary condition** (a tier gain is
available), not a verdict.

**The keep-vs-cut fork.** A demoted matmul whose fused body is ``UNBUILDABLE`` *inline*
may still be buildable as an **SMEM fused edge** — the producer cone materialized into an
on-chip smem slab the consumer ``ldmatrix``-reads, one kernel, no gmem round-trip (the
cut-beating form). That realization is ``_extract.seed_fused``'s expressibility check, so
``cut_offers`` takes it as the ``smem_fusible`` input: when the fused edge lowers, the cut
is **offered, not forced** (``force=False``) — a real keep(SMEM)-vs-cut(GMEM) fork the
search prices. The cut stays **forced** (``force=True``) only when the fused edge is *not*
expressible (multi-cone rotary / multi-accum gated-MLP that ``assemble_fused`` can't fuse
yet), so the GMEM cut is the lone lowerable option. Splitting the tier verdict (here) from
the fission expressibility (``seed_fused`` / ``extract_block``, paired at the fork pass)
keeps this module free of the body-surgery dependency.

Prefixed ``_`` so the pipeline rule loader skips it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._atom import eligible_atoms
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag


class Tier(enum.IntEnum):
    """The derived best-achievable tier lattice a fused body is pinned to (the
    *meet* of its blocks' tiers). Ordered so a new tier slots in by extending the
    lattice, not by editing call sites. ``UNBUILDABLE`` is below ``MAP``: a body the
    move composer cannot lower at all (``classify`` returns ``None`` — the demoted
    matmul whose cone operand defeats every regime)."""

    UNBUILDABLE = 0
    MAP = 1
    SCALAR_REDUCE = 2
    COOP_REDUCE = 3
    WARP_MMA = 4


_ALGEBRA_TIER = {
    AlgebraKind.MAP: Tier.MAP,
    AlgebraKind.SEMIRING: Tier.SCALAR_REDUCE,
    AlgebraKind.MONOID: Tier.COOP_REDUCE,  # cooperative reduce AND streaming flash (both MONOID)
}


def tier(dag, *, compute_capability: tuple[int, int], dtype_of) -> Tier:
    """The best achievable :class:`Tier` of ``dag`` — the single source of truth for
    "what would fusion cost here." Warp-tier (``WARP_MMA``) when any atom is eligible;
    otherwise the carrier's tier from ``classify``; ``UNBUILDABLE`` when the body has
    no buildable regime (a demoted matmul's cone operand)."""
    if eligible_atoms(dag, compute_capability=compute_capability, dtype_of=dtype_of):
        return Tier.WARP_MMA
    regime = classify(dag)
    if regime is None:
        return Tier.UNBUILDABLE
    return _ALGEBRA_TIER.get(regime.algebra, Tier.MAP)


@dataclass(frozen=True)
class CutOffer:
    """One ranked cuttable edge — a single keep-vs-cut decision the ``CUT`` BINMASK
    indexes (bit ``i`` = cut ranked offer ``i``). A demoted ``LoopOp`` exposes exactly
    **one** offer today: the whole-cone cut is all-or-nothing (the per-edge multi-producer
    fission — cut cone A, keep cone B in one graph — is the deferred follow-up that lands
    with multi-producer ``assemble_fused``), so the mask is width-1. The list shape is the
    additive-widening seam: when fission becomes per-edge, ``cut_offers`` returns one
    ``CutOffer`` per independently-fusible cone and the mask widens with no re-key.
    ``tier_inline`` (the fused body's tier the cut raises — ``UNBUILDABLE`` today) is
    recorded for ``eval`` introspection + the future ``D_*`` edge-pricing features."""

    tier_inline: Tier


@dataclass(frozen=True)
class CutDecision:
    """The typed offer the ``split/010_split_demoted`` fork consumes. ``offers``: the
    ranked cuttable edges (each a tier-monotonic ``GMEM`` cut whose materialization raises
    the consumer's tier); the ``CUT`` mask width is ``len(offers)``, empty when no cut is
    available. ``force``: the cut must be taken — the fused form is ``UNBUILDABLE`` inline
    AND not expressible as an SMEM fused edge, so the GMEM cut is the lone lowerable
    option. When the SMEM fused edge IS expressible (``smem_fusible``) the cut is
    offered-not-forced — a keep(SMEM)-vs-cut(GMEM) fork. ``tier_inline``: the fused body's
    tier, for ``eval`` introspection."""

    offers: tuple[CutOffer, ...]
    force: bool
    tier_inline: Tier

    @property
    def offered(self) -> bool:
        """A tier-monotonic ``GMEM`` cut is available (at least one ranked offer)."""
        return bool(self.offers)


def cut_offers(loop_op: LoopOp, *, compute_capability: tuple[int, int], dtype_of, smem_fusible: bool = False) -> CutDecision:
    """The cut-offer verdict for a (still-un-tiled) fused ``LoopOp`` — read off the
    derived ``tier`` of its inline body. Offer the ``GMEM`` cut iff the fused form is
    ``UNBUILDABLE``: a demoted matmul whose operand cone keeps it below any buildable
    tier, which materializing the operand strictly raises. Returns the ranked
    ``offers`` (one whole-cone offer today — the per-edge fission is deferred) so the
    ``CUT`` mask width derives from it. **Force** the cut only when the fused edge isn't
    also expressible on-chip (``smem_fusible`` False) — otherwise the SMEM fused edge is a
    lowerable keep option, so the cut is offered-not-forced (a real fork). Holds **no**
    fission logic — the fork pass supplies ``smem_fusible`` (the ``seed_fused``
    expressibility check) and pairs this verdict with ``extract_block``."""
    t = tier(iter_dag(loop_op), compute_capability=compute_capability, dtype_of=dtype_of)
    unbuildable = t is Tier.UNBUILDABLE
    offers = (CutOffer(tier_inline=t),) if unbuildable else ()
    return CutDecision(offers=offers, force=unbuildable and not smem_fusible, tier_inline=t)
