"""Offer the demoted-matmul split as a structural fork — see ``_split_demoted``.

Runs before ``010_partition_loops`` so the body is still un-tiled (the only dialect where
both halves of a split can re-enter planning with their own tilings). This rule owns only
the OFFER; the cut itself — slicing the computed A-operand cone (with its prologue deps)
into an ``xn`` producer + the clean gemm — lives in ``_split_demoted.try_split_demoted``,
whose checks are the cut's well-formedness conditions, not a profitability prediction.
Whether the split pays is the search's question: the tuner measures both branches inside
the op's slice; greedy ``compile`` / ``run`` never pick the structural option while an Op
variant exists (``policy/greedy._is_structural``), so cold kernel sets are unchanged.

Emission order is load-bearing: the keep-fused option (a no-op rebind of ``root.op`` to
itself — ``Graph.copy`` shares ops, so apply short-circuits) comes FIRST, the documented
greedy/no-prior fallback. ``DEPLODOCK_SPLIT_CONE=1/0`` pins either branch.

Termination (all measured, not assumed). The split branch terminates structurally: its two
LoopOps re-match this rule but fail the cut classification (the gemm has no cone; the
producer has no matmul reduce) — or, if a piece is itself still cuttable, a further split
is a legitimate offer, not a loop. A SINGLE demotion site also self-terminates without
help: its match is the batch's last, so the keep-fused child's apply advances the cursor
and partition consumes the LoopOp before this rule re-runs. The hazard is MULTIPLE
demotion sites in one batch: the fork fires on a non-last match, every child re-enumerates
the batch against a graph where the keep-fused node is unchanged, and the re-offers
compound combinatorially (measured: a two-site graph stops yielding terminals entirely) —
the engine's quiescence contract explicitly relies on rule-side idempotence guards
(``Candidate.try_rewrite`` docstring). So the rule stamps a ``tile.split_cone.offered``
node hint before returning the fork and skips marked nodes. Hints are advisory metadata,
excluded from structural digests — zero ``op_cache_key`` churn. A node CAN still be
offered once per *sibling branch* of an earlier fork point (each branch copied the graph
before the later node's hint was stamped) — that's the intended cross-product of
independent decisions, bounded at one offer per node per branch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Node
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
# cold pick) and splitting it in two (True). Search-layer identity only — it
# rides ``OptionFork.knobs``, never ``op.knobs``. DELIBERATELY no ``off=``:
# ``_off_fill_pass`` would stamp an off-default onto every knob-bearing TileOp
# at the pass boundary and change every tile-dialect ``op_cache_key`` (busting
# the tune DB).
SPLIT_CONE = Knob(
    "SPLIT_CONE",
    KnobType.BOOL,
    hints=(False, True),
    help="Split a demoted matmul's computed A-operand cone into its own kernel (xn producer + clean gemm)",
)

# Idempotence marker (see "Termination" above): the keep-fused branch leaves the
# LoopOp matchable, so the offered fork is remembered on the node.
_OFFERED_HINT = "tile.split_cone.offered"


def rewrite(ctx: Context | None, match: Match, root: Node) -> Graph | list[OptionFork]:
    if root.hints.get(_OFFERED_HINT):
        raise RuleSkipped("split fork already offered for this node")
    pinned = SPLIT_CONE.narrow((False, True))
    if pinned == (False,):
        raise RuleSkipped("SPLIT_CONE pinned off — keep the fused kernel")
    split = try_split_demoted(root.op, ctx, graph=match.graph, node_id=root.id, out_tensor=root.output)
    if split is None:
        raise RuleSkipped("not a cuttable demoted matmul")
    if pinned == (True,):
        return split
    # Mark before returning: both fork children inherit the parent snapshot, so
    # neither branch re-offers on the next pass rescan.
    root.hints.set(_OFFERED_HINT, True)
    return [
        OptionFork(option=root.op, knobs={SPLIT_CONE.name: False}),
        OptionFork(option=split, knobs={SPLIT_CONE.name: True}),
    ]
