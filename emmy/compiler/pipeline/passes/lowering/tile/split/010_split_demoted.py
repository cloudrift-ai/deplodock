"""Offer the demoted-matmul cut as a structural fork — the thin shell (R7 edge placement).

The **pre-build structural-fork head** of the tile phase. Runs before
``enumeration/010_build`` on the still-un-tiled ``LoopOp`` (the only dialect where every
piece of a split can re-enter planning with its own tiling), and the demoted matmul never
builds a seed anyway (``classify`` returns ``None`` for a cone-operand cell, so
``010_build`` would ``RuleSkip`` it) — which is why the cut stays a pre-build operation
and ``split/`` survives rather than folding into ``enumeration/`` as an edge-placement
move.

This rule holds **no** decision logic — it is the thin fork the plan calls for, pairing
three relocated pieces:

- :func:`~emmy.compiler.pipeline.passes.lowering.tile.enumeration._cut.cut_offers`
  (``enumeration/_cut.py``) — the **offer policy**: the derived tier-monotonicity
  predicate. Offer the ``GMEM`` cut iff the fused body is ``UNBUILDABLE`` (a demoted
  matmul whose cone operand keeps it below any buildable tier — which materializing the
  operand strictly raises). **Force** it (single option) only when the on-chip fused edge
  isn't also expressible.
- :func:`~emmy.compiler.pipeline.passes.lowering.tile.enumeration._extract.seed_fused`
  (``enumeration/_extract.py``) — the **keep(SMEM) realization**: the same fission laid out
  as a fused 2-block ``TileGraphOp`` (clean matmul consumer ``blocks[0]`` + producer cone
  ``blocks[1]``, the ``xn`` edge ``SMEM``-placed) the enumeration tiles into ONE kernel —
  the producer rides the consumer's smem slab, no gmem round-trip. ``None`` (a multi-cone /
  multi-accum body it can't fuse yet) is its expressibility check, supplied to
  ``cut_offers`` as ``smem_fusible``.
- :func:`~emmy.compiler.pipeline.passes.lowering.tile.enumeration._extract.extract_block`
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
the cut when it predicts faster; ``tune``'s MCTS walks both branches. ``EMMY_SPLIT_CONE
=0`` (or ``EMMY_CUT=0``) pins the fused edge, ``=1`` the GMEM cut.

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

from emmy.compiler import target as target_mod
from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.base import Op
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import Knob, KnobType
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._cut import cut_offers
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._extract import extract_block, seed_fused

if TYPE_CHECKING:
    from emmy.compiler.context import Context

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
# distinction the prior trains on. ``SPLIT_CONE`` alias keeps ``EMMY_SPLIT_CONE``
# pinning the cut. The native decision is the ``PLACE@cone`` placement
# (``inline`` keep / ``cut`` materialize — ``_families``); this ``CUT`` ``Knob`` stays
# registered only so the legacy ``EMMY_CUT`` / ``EMMY_SPLIT_CONE`` env namespace
# + display resolve (ingested via ``_knob_legacy.cut_pin``).
CUT = Knob(
    "CUT",
    KnobType.BINMASK,
    aliases=("SPLIT_CONE",),
    help="(legacy spelling of PLACE@cone) cut a demoted matmul's computed / K-folded operand "
    "cone(s) into producer kernel(s) (xn producer(s) + clean gemm) vs keep them fused on-chip. "
    "EMMY_SPLIT_CONE=1/0 pins it; the native key is PLACE@cone=cut/inline.",
)


def _stamp(split: Graph, place: str) -> Graph:
    """Stamp the cone placement (``cut``) onto every split kernel's ``PLACE@cone``."""
    for node in split.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op.knobs = {**node.op.knobs, fam.cone_key(): place}
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
    if fam.cone_key() in root.op.knobs:
        raise RuleSkipped("split fork already considered for this kernel")
    split = extract_block(root.op, graph=match.graph, node_id=root.id, out_tensor=root.output)
    if split is None:
        raise RuleSkipped("not a cuttable demoted matmul")
    # The offer policy (enumeration/_cut.cut_offers): a demoted cone whose operand keeps
    # the matmul below any buildable tier (the gated-MLP RMSNorm, the SDPA softmax-
    # prologue) offers a cut that restores it.
    cc = ctx.compute_capability if ctx is not None else target_mod.compute_capability()
    fused = seed_fused(root.op, graph=match.graph, node_id=root.id, out_tensor=root.output)
    smem_fusible = fused is not None
    decision = cut_offers(root.op, compute_capability=cc, dtype_of=_dtype_of(root.op, match.graph), smem_fusible=smem_fusible)
    # The "keep" option places the cone ``inline`` (it rides inside the consumer — the SMEM
    # fused edge when expressible, else the un-tiled fused LoopOp); the "cut" option places
    # it ``cut`` (materialized to a gmem intermediate kernel). Only genuine offer sites
    # carry the ``PLACE@cone`` decision — never-offered kernels stay knob-free.
    if smem_fusible:
        fused.knobs = {**root.op.knobs, fam.cone_key(): fam.INLINE}
        keep = fused
    else:
        keep = replace(root.op, knobs={**root.op.knobs, fam.cone_key(): fam.INLINE})
    # Env pin (native ``EMMY_PLACE_CONE=cut`` / legacy ``EMMY_CUT`` /
    # ``EMMY_SPLIT_CONE``) collapses the fork — True = cut, False = keep.
    pin = fam.pin_cut()
    if pin is not None:
        return _stamp(split, fam.CUT) if pin else keep
    if decision.force:
        return _stamp(split, fam.CUT)
    # Offered-not-forced (``smem_fusible``): a real keep(SMEM)-vs-cut(GMEM) fork. Greedy's
    # "a cold compile never changes kernel sets" rule (search/policy/greedy._pick_structural
    # filters the structural cut when the prior is cold) deploys the kernel-set-PRESERVING
    # keep(SMEM) — the fused edge, robust at the warp tier (the cold pick). The trained prior
    # prices the cut's GMEM Σ vs keep's SMEM Σ and deploys the cut when it predicts faster;
    # ``tune``'s MCTS walks both. Keep first = the documented no-prior emission-order
    # fallback (the SMEM fused edge — no gmem round-trip). ``EMMY_SPLIT_CONE=1/0`` pins.
    return [keep, _stamp(split, fam.CUT)]
