"""Offer the demoted-matmul split as a structural fork — see ``_split_demoted``.

Runs before ``010_partition_loops`` so the body is still un-tiled (the only dialect where
both halves of a split can re-enter planning with their own tilings). This rule owns only
the OFFER; the cut itself lives in ``_split_demoted.try_split_demoted`` — one rule, no
per-shape cases: each multiply operand is independently a plain Load (stays put) or a
computed cone (becomes an ``xn`` producer materialized over exactly the axes it reads,
with K second-to-last for an N-reading cone so the consumer keeps the canonical B layout).
Its checks are the cut's well-formedness conditions, not a profitability prediction.
Whether the split pays is the search's question: the tuner compares both branches as outer
terminals (``search/two_level.py``); greedy ``compile`` / ``run`` deploy the split only when
the *trained* prior prices it cheaper (``policy/greedy._pick_structural``) — cold kernel
sets are unchanged.

Emission order is load-bearing: the keep-fused option comes FIRST, the documented
greedy/no-prior fallback. ``DEPLODOCK_SPLIT_CONE=1/0`` pins either branch.

Both branches stamp the decision into ``op.knobs`` — keep-fused carries ``SPLIT_CONE:
False``, the split fragment's two kernels carry ``SPLIT_CONE: True`` — the standard
considered-vs-declined idiom (``020_stage_inputs``'s declined ``STAGE`` row; see
``search/keys.py``): the knob is the rule's idempotence guard AND the learned prior's
training signal (absent = never offered → NaN-filled; ``False`` / ``True`` = the measured
decision), and it keeps each decision state's ``op_cache_key`` distinct from its parent so
the search tree never self-parents. The stamp is deterministic per offer site, so
structurally identical kernels across graphs stamp identically and keep sharing perf rows.

Termination (measured, not assumed). The split branch terminates structurally: its two
LoopOps re-match this rule but fail the cut classification (the gemm has no cone; the
producer has no matmul reduce) — or, if a piece is itself still cuttable, a further split
is a legitimate offer, not a loop. The keep-fused branch terminates via the knob guard:
without it, a multi-demotion-site batch re-offers compoundingly in fork children (the fork
fires on a non-last match and every child re-enumerates the batch; measured: a two-site
graph stops yielding terminals entirely) — the engine's quiescence contract explicitly
relies on rule-side idempotence guards (``Candidate.try_rewrite`` docstring). A node CAN
still be offered once per *sibling branch* of an earlier fork point (each branch copied
the graph before the later node's stamp) — the intended cross-product of independent
decisions, bounded at one offer per node per branch.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork import OptionFork
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._split_demoted import try_split_demoted

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

PATTERN = [Pattern("root", LoopOp)]

# BOOL knob: at a demoted matmul (a fused computed-operand cone keeps the warp
# tier structurally unreachable — e.g. the gated-MLP norm prologue) this rule
# forks between keeping the fused kernel (False, emitted first = the greedy
# cold pick) and splitting it in two (True). Stamped into ``op.knobs`` on both
# branches (decision = idempotence guard = prior training signal — see the
# module docstring). DELIBERATELY no ``off=``: ``_off_fill_pass`` would stamp
# an off-default onto every knob-bearing TileOp at the pass boundary, erasing
# the absent-vs-declined distinction the prior trains on (and churning every
# tile-dialect ``op_cache_key``).
SPLIT_CONE = Knob(
    "SPLIT_CONE",
    KnobType.BOOL,
    hints=(False, True),
    help="Split a demoted matmul's computed operand cone(s) into producer kernel(s) (xn producer(s) + clean gemm)",
)


def _stamp(split: Graph) -> Graph:
    """Stamp ``SPLIT_CONE: True`` onto both split kernels' ``op.knobs``."""
    for node in split.nodes.values():
        if isinstance(node.op, LoopOp):
            node.op.knobs = {**node.op.knobs, SPLIT_CONE.name: True}
    return split


def rewrite(ctx: Context | None, match: Match, root: Node) -> Graph | Op | list:
    if SPLIT_CONE.name in root.op.knobs:
        raise RuleSkipped("split fork already considered for this kernel")
    split = try_split_demoted(root.op, ctx, graph=match.graph, node_id=root.id, out_tensor=root.output)
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
    # The split fork's ranking knobs carry the offer site's full knob base
    # (its ``S_*`` identity) under the decision delta — mirroring the keep
    # side, whose lifted OptionFork copies the Op's knob dict — so the outer
    # search's ``_node_knobs`` at the two siblings is feature-identical to
    # the composed Σ rows ``two_level._decomposition_rows`` trains the prior
    # on. Ranking metadata only; the spliced kernels' own knobs are stamped
    # by ``_stamp`` / the cut builder.
    return [
        keep_fused,
        OptionFork(option=_stamp(split), knobs={**root.op.knobs, SPLIT_CONE.name: True}),
    ]
