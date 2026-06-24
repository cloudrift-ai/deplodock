"""Offer the demoted-matmul cut as a structural fork — the thin shell (R7 edge placement).

The **pre-build structural-fork head** of the tile phase. Runs before
``enumeration/000_build`` on the still-un-tiled ``LoopOp`` (the only dialect where every
piece of a split can re-enter planning with its own tiling), and the demoted matmul never
builds a seed anyway (``classify`` returns ``None`` for a cone-operand cell, so
``000_build`` would ``RuleSkip`` it) — which is why the cut stays a pre-build operation
and ``split/`` survives rather than folding into ``enumeration/`` as an edge-placement
move (``plans/dag-edge-placement-split-as-enumeration.md`` → "Status / step 2.5").

This rule holds **no** decision logic — it is the thin fork the plan calls for, pairing
three relocated pieces:

- :func:`~deplodock.compiler.pipeline.passes.lowering.tile.enumeration._cut.cut_offers`
  (``enumeration/_cut.py``) — the **offer policy**: the derived tier-monotonicity
  predicate. Offer the ``GMEM`` cut iff the fused body is ``UNBUILDABLE`` (a demoted
  matmul whose cone operand keeps it below any buildable tier — which materializing the
  operand strictly raises). **Force** it (single option) only when the on-chip fused edge
  isn't also expressible.
- :func:`~deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract.seed_fused`
  (``enumeration/_extract.py``) — the **keep(SMEM) realization**: the same fission laid out
  as a fused 2-block ``TileGraphOp`` (clean matmul consumer ``blocks[0]`` + producer cone
  ``blocks[1]``, the ``xn`` edge ``SMEM``-placed) the enumeration tiles into ONE kernel —
  the producer rides the consumer's smem slab, no gmem round-trip. ``None`` (a multi-cone /
  multi-accum body it can't fuse yet) is its expressibility check, supplied to
  ``cut_offers`` as ``smem_fusible``.
- :func:`~deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract.extract_block`
  (``enumeration/_extract.py``) — the **cut(GMEM) fission**: lift each computed/K-folded cone
  into an ``xn`` producer kernel and rebuild the consumer reading the materialized
  intermediate, wired into a ``Graph`` fragment the engine splices (a kernel-set change →
  the **outer** two-level tree). ``None`` is its expressibility check.

The familiar instance is the **score-materializing SDPA**: the fused softmax-prologue +
P@V ``k_sdpa_reduce`` un-fuses into a softmax-normalizing ``xn`` producer + a clean
(static **or** symbolic-K) gemm consumer that both lower. When the cone fuses on-chip
(``seed_fused`` expressible — a single pointwise MAP or single-reduce RMSNorm cone) the cut
is offered-not-forced (``cut_offers`` reports ``force=False``); otherwise (multi-cone
rotary, two-pass SDPA softmax, multi-accum gated-MLP) the cut is **forced** — the GMEM cut
is the lone lowerable option.

**Greedy default = keep(SMEM).** When the SMEM fused edge is expressible the rule emits the
real ``[keep, cut]`` fork. Greedy's "a cold compile never changes kernel sets" rule
(``search/policy/greedy._pick_structural`` filters the structural cut when the prior is
cold) deploys the kernel-set-preserving keep(SMEM) — the fused edge, robust at the warp tier
(the cold pick). The *trained* prior prices the cut's GMEM Σ vs keep's SMEM Σ and deploys
the cut when it predicts faster; ``tune``'s MCTS walks both branches. ``DEPLODOCK_SPLIT_CONE
=0`` (or ``DEPLODOCK_CUT=0``) pins the fused edge, ``=1`` the GMEM cut.

Both branches stamp the decision into ``op.knobs`` — keep carries the ``CUT`` mask
``"0"`` (cut nothing), every split-fragment kernel carries ``"1"`` (cut every ranked
edge) — the standard considered-vs-declined idiom: the knob is the rule's idempotence
guard AND the learned prior's training signal, and it keeps each decision state's
``op_cache_key`` distinct from its parent so the search tree never self-parents. The
mask is width-1 today (the whole-cone cut is all-or-nothing); a per-edge mask (cut cone
A, keep cone B) is the additive follow-up gated on multi-producer ``assemble_fused``.

Termination (measured, not assumed). The split branch terminates structurally: its
LoopOps re-match this rule but fail the fission (the gemm has no cone; the producer has
no matmul reduce; the combine has no matmul) — or, if a piece is itself still cuttable, a
further split is a legitimate offer, not a loop. The keep branch terminates structurally
too: the SMEM fused realization is a ``TileGraphOp`` (this rule matches only ``LoopOp``);
the un-fusible keep-fused ``LoopOp`` terminates via the ``CUT`` knob guard.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from deplodock.compiler import target as target_mod
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._cut import cut_offers
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract import extract_block, seed_fused

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

PATTERN = [Pattern("root", LoopOp)]

# BINMASK knob over ``_cut.cut_offers``' ranked cuttable edges (bit ``i`` = cut
# ranked offer ``i``): at a demoted matmul (a fused computed-operand cone keeps
# the warp tier structurally unreachable — e.g. the gated-MLP norm prologue) this
# rule forks between keeping the cone fused (mask ``0``, emitted first = the
# greedy cold pick) and cutting it into producer kernel(s) (mask all-ones).
# Width-1 today (the whole-cone cut is all-or-nothing); the mask is the substrate
# for per-edge fission (cut cone A, keep cone B) once multi-producer fusion lands,
# so widening it later is additive — no perf-DB re-key. Stamped into ``op.knobs``
# on both branches (decision = idempotence guard = prior training signal).
# DELIBERATELY no ``off=``: ``_off_fill_pass`` would stamp an off-default onto
# every knob-bearing TileOp at the pass boundary, erasing the absent-vs-declined
# distinction the prior trains on. ``SPLIT_CONE`` alias keeps ``DEPLODOCK_SPLIT_CONE``
# pinning the width-1 mask (``1`` ↔ old ``True``, ``0`` ↔ old ``False``).
CUT = Knob(
    "CUT",
    KnobType.BINMASK,
    aliases=("SPLIT_CONE",),
    help="BINMASK over ranked cuttable edges (bit i = cut ranked offer i) of a demoted matmul's "
    "computed / K-folded operand cone(s) into producer kernel(s) (xn producer(s) + clean gemm; "
    "multi-accum also extracts one gemm per accum + the pointwise combine) vs keep them fused on-chip. "
    "Width-1 today (the whole-cone cut is all-or-nothing); DEPLODOCK_SPLIT_CONE=1/0 pins it.",
)


def _stamp(split: Graph, mask_str: str) -> Graph:
    """Stamp the cut ``CUT`` mask onto every split kernel's ``op.knobs``."""
    for node in split.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op.knobs = {**node.op.knobs, CUT.name: mask_str}
    return split


def _dtype_of(loop_op: LoopOp, graph: Graph):
    """Resolve a buffer name to its dtype — off the ``LoopOp``'s own I/O first, then
    the enclosing graph. The tier predicate's ``eligible_atoms`` reads operand dtypes
    through this."""

    def f(buf):
        if buf in loop_op.inputs:
            return loop_op.inputs[buf].dtype
        if buf in loop_op.outputs:
            return loop_op.outputs[buf].dtype
        node = graph.nodes.get(buf)
        return node.output.dtype if node is not None else None

    return f


def rewrite(ctx: Context | None, match: Match, root: Node) -> Graph | Op | list:
    if CUT.name in root.op.knobs:
        raise RuleSkipped("split fork already considered for this kernel")
    split = extract_block(root.op, graph=match.graph, node_id=root.id, out_tensor=root.output)
    if split is None:
        raise RuleSkipped("not a cuttable demoted matmul")
    # The offer policy (enumeration/_cut.cut_offers): a demoted cone whose operand keeps
    # the matmul below any buildable tier (the gated-MLP RMSNorm, the SDPA softmax-
    # prologue) offers a cut that restores it. ``decision.width`` is the number of ranked
    # cuttable edges = the ``CUT`` mask width (1 today, the all-or-nothing whole-cone cut;
    # ``or 1`` keeps the fork width-1 even in the unreached cuttable-but-not-offered case).
    cc = ctx.compute_capability if ctx is not None else target_mod.compute_capability()
    fused = seed_fused(root.op, graph=match.graph, node_id=root.id, out_tensor=root.output)
    smem_fusible = fused is not None
    decision = cut_offers(root.op, compute_capability=cc, dtype_of=_dtype_of(root.op, match.graph), smem_fusible=smem_fusible)
    n = decision.width or 1
    keep_mask = CUT.pretty(0, width=n)  # "0" — cut nothing (keep the cone fused)
    cut_mask = CUT.pretty((1 << n) - 1, width=n)  # "1" — cut every ranked edge
    # The "keep" option: the SMEM fused edge (the producer cone on-chip, one kernel) when
    # expressible, else the un-tiled fused LoopOp (a buildable-fused operand index). Only
    # genuine offer sites carry the decision knob — never-offered kernels stay knob-free
    # (the prior's "not considered" NaN state).
    if smem_fusible:
        fused.knobs = {**root.op.knobs, CUT.name: keep_mask}
        keep = fused
    else:
        keep = replace(root.op, knobs={**root.op.knobs, CUT.name: keep_mask})
    # Env pin (``DEPLODOCK_CUT`` / ``DEPLODOCK_SPLIT_CONE`` = ``1`` / ``0`` / ``all`` /
    # ``none``) collapses the fork to one mask, mirroring ``STAGE`` — ``narrow`` rejects
    # BINMASK, so pin via ``raw()`` + ``parse(width=n)``. Any non-zero mask = cut.
    raw = CUT.raw()
    if raw:
        return keep if CUT.parse(raw, width=n) == 0 else _stamp(split, cut_mask)
    if decision.force:
        return _stamp(split, cut_mask)
    # Offered-not-forced (``smem_fusible``): a real keep(SMEM)-vs-cut(GMEM) fork. Greedy's
    # "a cold compile never changes kernel sets" rule (search/policy/greedy._pick_structural
    # filters the structural cut when the prior is cold) deploys the kernel-set-PRESERVING
    # keep(SMEM) — the fused edge, robust at the warp tier (the cold pick). The trained prior
    # prices the cut's GMEM Σ vs keep's SMEM Σ and deploys the cut when it predicts faster;
    # ``tune``'s MCTS walks both. Keep first = the documented no-prior emission-order
    # fallback (the SMEM fused edge — no gmem round-trip). ``DEPLODOCK_SPLIT_CONE=1/0`` pins.
    return [keep, _stamp(split, cut_mask)]
