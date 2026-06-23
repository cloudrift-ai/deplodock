"""Offer the demoted-matmul cut as a structural fork — the thin shell (R7 edge placement).

The **pre-build structural-fork head** of the tile phase. Runs before
``enumeration/000_build`` on the still-un-tiled ``LoopOp`` (the only dialect where every
piece of a split can re-enter planning with its own tiling), and the demoted matmul never
builds a seed anyway (``classify`` returns ``None`` for a cone-operand cell, so
``000_build`` would ``RuleSkip`` it) — which is why the cut stays a pre-build operation
and ``split/`` survives rather than folding into ``enumeration/`` as an edge-placement
move (``plans/dag-edge-placement-split-as-enumeration.md`` → "Status / step 2.5").

This rule holds **no** decision logic — it is the thin fork the plan calls for, pairing
two relocated pieces:

- :func:`~deplodock.compiler.pipeline.passes.lowering.tile.enumeration._cut.cut_offers`
  (``enumeration/_cut.py``) — the **offer policy**: the derived tier-monotonicity
  predicate. Offer (and, v1, force) the ``GMEM`` cut iff the fused body is
  ``UNBUILDABLE`` (a demoted matmul whose cone operand keeps it below any buildable
  tier — which materializing the operand strictly raises).
- :func:`~deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract.extract_block`
  (``enumeration/_extract.py``) — the **fission**: lift each computed/K-folded cone into an
  ``xn`` producer kernel and rebuild the consumer reading the materialized intermediate,
  wired into a ``Graph`` fragment the engine splices (a kernel-set change → the **outer**
  two-level tree). ``None`` is its expressibility check.

The familiar instance is the **score-materializing SDPA**: the fused softmax-prologue +
P@V ``k_sdpa_reduce`` un-fuses into a softmax-normalizing ``xn`` producer + a clean
(static **or** symbolic-K) gemm consumer that both lower. The cut is **forced** (single
option) whenever it fires: the fused form is ``UNBUILDABLE``, so there is no lowerable
keep-fused branch. The buildable-fused keep-vs-split *fork* needs the lowerable
fused-prologue regime the R7 backlog defers, so it is not offered yet.

Emission order is load-bearing: the keep-fused option (when offered) comes FIRST, the
documented greedy/no-prior fallback. ``DEPLODOCK_SPLIT_CONE=1/0`` pins either branch.

Both branches stamp the decision into ``op.knobs`` — keep-fused carries ``SPLIT_CONE:
False``, every split-fragment kernel carries ``SPLIT_CONE: True`` — the standard
considered-vs-declined idiom: the knob is the rule's idempotence guard AND the learned
prior's training signal, and it keeps each decision state's ``op_cache_key`` distinct
from its parent so the search tree never self-parents.

Termination (measured, not assumed). The split branch terminates structurally: its
LoopOps re-match this rule but fail the fission (the gemm has no cone; the producer has
no matmul reduce; the combine has no matmul) — or, if a piece is itself still cuttable, a
further split is a legitimate offer, not a loop. The keep-fused branch terminates via the
``SPLIT_CONE`` knob guard.
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
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._extract import extract_block

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

PATTERN = [Pattern("root", LoopOp)]

# BOOL knob: at a demoted matmul (a fused computed-operand cone keeps the warp
# tier structurally unreachable — e.g. the gated-MLP norm prologue) this rule
# forks between keeping the fused kernel (False, emitted first = the greedy
# cold pick) and splitting it apart (True). Stamped into ``op.knobs`` on both
# branches (decision = idempotence guard = prior training signal). DELIBERATELY
# no ``off=``: ``_off_fill_pass`` would stamp an off-default onto every
# knob-bearing TileOp at the pass boundary, erasing the absent-vs-declined
# distinction the prior trains on.
SPLIT_CONE = Knob(
    "SPLIT_CONE",
    KnobType.BOOL,
    hints=(False, True),
    help="Split a demoted matmul's computed / K-folded operand cone(s) into producer kernel(s) "
    "(xn producer(s) + clean gemm; multi-accum also extracts one gemm per accum + the pointwise combine)",
)


def _stamp(split: Graph) -> Graph:
    """Stamp ``SPLIT_CONE: True`` onto both split kernels' ``op.knobs``."""
    for node in split.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op.knobs = {**node.op.knobs, SPLIT_CONE.name: True}
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
    if SPLIT_CONE.name in root.op.knobs:
        raise RuleSkipped("split fork already considered for this kernel")
    split = extract_block(root.op, graph=match.graph, node_id=root.id, out_tensor=root.output)
    if split is None:
        raise RuleSkipped("not a cuttable demoted matmul")
    # Only genuine offer sites carry the decision knob — never-offered kernels
    # stay knob-free (the prior's "not considered" NaN state).
    keep_fused = replace(root.op, knobs={**root.op.knobs, SPLIT_CONE.name: False})
    pinned = SPLIT_CONE.narrow((False, True))
    if pinned == (False,):
        return keep_fused
    if pinned == (True,):
        return _stamp(split)
    # The offer policy (enumeration/_cut.cut_offers): force the cut iff the fused
    # body is UNBUILDABLE — a demoted cone whose operand keeps the matmul below any
    # buildable tier (the gated-MLP RMSNorm, the SDPA softmax-prologue), so there is
    # no lowerable keep-fused regime and every cut yields composable pieces. A
    # buildable-fused demoted matmul (a merely computed operand *index* the composer
    # lowers fused) keeps fused — the keep-vs-split fork over that is R7.
    cc = ctx.compute_capability if ctx is not None else target_mod.compute_capability()
    if cut_offers(root.op, compute_capability=cc, dtype_of=_dtype_of(root.op, match.graph)).force:
        return _stamp(split)
    return keep_fused
