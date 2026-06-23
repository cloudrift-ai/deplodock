"""The cut-offer policy — the derived tier-monotonicity predicate (R7 edge placement).

``plans/dag-edge-placement-split-as-enumeration.md`` → "Cut-decision module —
``enumeration/_cut.py``". All cut-offer logic lives here, one auditable place, so the
fork pass (``split/005_split_demoted``) holds **no** decision logic and ``eval``-style
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
(``split/_extract.extract_block``) then materializes the operand into a clean ``Load``,
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

**v1 scope.** Only the *forced* cut is wired (``offered`` ⟺ ``force`` ⟺ ``UNBUILDABLE``
— the fused form can't lower, so there is no keep-fused option). A buildable-fused
demoted matmul whose operand *would* reach ``WARP_MMA`` if materialized is a genuine
keep-vs-split **fork**, but that needs the lowerable fused-prologue regime the R7
backlog defers, so it stays fused for now (the offer predicate would price it, the fork
is future work).

Prefixed ``_`` so the pipeline rule loader skips it.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import eligible_atoms
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag


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
    AlgebraKind.MONOID: Tier.COOP_REDUCE,
    AlgebraKind.TWISTED_MONOID: Tier.COOP_REDUCE,  # streaming flash — a cooperative reduce regime
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
class CutDecision:
    """The typed offer the ``split/005_split_demoted`` fork consumes. ``offered``: a
    tier-monotonic ``GMEM`` cut is available (materializing the demoted operand raises
    the consumer's tier). ``force``: the cut must be taken — the fused form is
    ``UNBUILDABLE``, so there is no lowerable keep-fused option (v1: ``offered`` ⟺
    ``force``; the buildable-fused keep-vs-split fork is R7). ``tier_inline``: the
    fused body's tier, for ``eval`` introspection."""

    offered: bool
    force: bool
    tier_inline: Tier


def cut_offers(loop_op: LoopOp, *, compute_capability: tuple[int, int], dtype_of) -> CutDecision:
    """The cut-offer verdict for a (still-un-tiled) fused ``LoopOp`` — read off the
    derived ``tier`` of its inline body. Offer (and force) the ``GMEM`` cut iff the
    fused form is ``UNBUILDABLE``: a demoted matmul whose operand cone keeps it below
    any buildable tier, which materializing the operand (the ``extract_block`` fission)
    strictly raises. Holds **no** fission logic — the fork pass pairs this verdict with
    ``extract_block``'s expressibility check."""
    t = tier(iter_dag(loop_op), compute_capability=compute_capability, dtype_of=dtype_of)
    unbuildable = t is Tier.UNBUILDABLE
    return CutDecision(offered=unbuildable, force=unbuildable, tier_inline=t)
