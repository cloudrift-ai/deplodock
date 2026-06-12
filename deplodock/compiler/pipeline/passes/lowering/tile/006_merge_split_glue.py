"""Re-fuse the split's glue kernels into neighbors — fusion as post-split cleanup.

``005_split_demoted`` buys the MMA tier by cutting kernels apart, and the cut leaves glue:
``xn`` operand materializations, per-accum ``mm{i}`` gemms, and the rebuilt pointwise
combine. At deploy size these are launch-latency-floor kernels (the Qwen3-Embedding layer-0
findings measured ~23 of 48 per-launch µs in glue at 1–23% DRAM — see
``plans/qwen3-embedding-layer0-tune-findings.md``, finding 1). This rule re-runs the loop
fusion *mechanism* (``splice_graph`` via ``loop/fusion/_helpers.build_merged_op``) on the
post-split ``Graph[LoopOp]`` so glue fuses into a neighbor's epilogue — while a guard set
keeps every protection the split paid for. Unconditional cleanup, NOT a fork: a single
deterministic ``Graph`` rewrite adds no outer-tree fork points; the tuner prices the merged
kernel set through the same ``SPLIT_CONE`` decision it already owns.

Target merges (all producer→consumer, the only direction the matcher offers):

- gemm ``mm_i`` → pointwise combine: the merged op is a single-accum gemm whose SiLU·up
  epilogue sits in the outer-N body, loading the *other* ``mm_j`` buffer elementwise —
  exactly the shape ``classify_fragment_epilogue`` folds onto the fragment store.
- reduce kernel (e.g. q/k-norm) → pure-pointwise ``xn`` producer (e.g. RoPE): the merged op
  is the reduce kernel with the xn's compute as epilogue.
- gemm → pure-copy layout ``xn``: the merged op writes the contiguized layout directly
  (guard 6 declines it if that would cost the producer its MMA eligibility).

Guard set (each raises ``RuleSkipped``; order = cheapest first):

1. scope key — fire only when a matched op carries ``SPLIT_CONE: True`` (a split product).
   Inert on split-free graphs, in the loop tier, and on the keep-fused branch
   (``SPLIT_CONE: False``).
2. one-level marker — never merge a node carrying the ``GLUE_HINT`` node hint (this rule's
   own product). Guarantees one match batch ≡ quiescence: the full pipeline gives this rule
   exactly ONE LoopOp batch per scan (``010_partition_loops`` converts the dialect in the
   same scan) while the two-level outer head loops to quiescence — second-order merges
   firing only in the latter would split ``op_cache_key``s between the outer search and
   greedy replay. The marker is ``Node.hints`` metadata, NOT an ``op.knobs`` entry: knobs
   ride into the prior's training rows, and this is plumbing, not a decision to learn.
3. K-cell protection — skip when the consumer reads the producer's buffer inside a reduce
   loop: inlining there would re-pollute a matmul cell and re-demote it (the exact state
   the split escaped). The ``xn`` producers' own copy loops are not reduce loops
   (``Loop.is_reduce`` = immediate body holds an ``Accum``), so backward glue merges pass.
4. one gemm per kernel — skip when both bodies carry an ``Accum``: blocks re-merging the
   second ``mm_i`` (undoing the multi-accum extraction) and blocks reduce-bearing cones
   (e.g. the gated-MLP norm ``xn``) from riding a gemm epilogue they can't tile into.
5. blowup + broadcast-materialization — the base fusion economics, unchanged. The base
   rule's multi-load-of-reduce-heavy guard is deliberately NOT applied: the splicer dedups
   a row-stat ``Accum`` to one emission, and the post-split shapes that read a reduce
   producer through two Loads (RoPE reading the normed row) are exactly the merges this
   rule exists for.
6. eligibility preservation — if either constituent is atom-eligible but the merged op is
   not, decline: never trade an MMA kernel for a fused scalar one. Calls the real gate
   (``_atom.is_atom_eligible``), so there is no simulated-gate drift.

The merged op is stamped ``SPLIT_CONE: True`` (without it, ``005_split_demoted``'s
idempotence guard no longer fires and the rule re-offers a split of the merged kernel —
split→merge→re-split never terminates); the composite marker rides the merged NODE's
hints (``GLUE_HINT``). ``source`` is forwarded from a stamped constituent so
``two_level._decomposition_rows`` attributes the merged kernel to its pre-split offer
ancestor. ``S_*`` features and the kernel name are restamped by the
``007``/``008``/``009`` aliases after the body settles. ``005_split_shared_indexmap`` is
not aliased: the only multi-consumer split product (the shared gated-MLP ``xn``) is
compute-bearing, which that rule's pure-indexmap gate excludes anyway.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.loop import Accum, Load, Loop, LoopOp
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.loop.fusion._helpers import build_merged_op, is_pure_indexmap, wrap_merge_fragment
from deplodock.compiler.pipeline.passes.lowering.tile._atom import ATOM_REGISTRY, is_atom_eligible

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

_base = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.010_merge_loop_ops")

PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]

_SPLIT_CONE = "SPLIT_CONE"
# Node-hint key marking a re-fused composite (guard 2's one-level contract).
# DELIBERATELY a node hint, not an ``op.knobs`` entry: every knobs key rides
# into the prior's training rows (``knob.knob_features`` float-coerces even
# unregistered keys), and this is plumbing, not a tuning decision to learn.
GLUE_HINT = "tile.split_glue"


def _has_accum(op: LoopOp) -> bool:
    return any(isinstance(s, Accum) for s in op.body.iter())


def _reads_buf_in_reduce_scope(op: LoopOp, buf: str) -> bool:
    """True when ``op`` Loads ``buf`` anywhere under a reduce loop (the K cell)."""

    def walk(stmts: Body, in_reduce: bool) -> bool:
        for s in stmts:
            if isinstance(s, Loop):
                if walk(s.body, in_reduce or s.is_reduce):
                    return True
            elif isinstance(s, Load) and s.input == buf and in_reduce:
                return True
        return False

    return walk(op.body, False)


def rewrite(ctx: Context | None, match: Match, producer: Node, consumer: Node) -> Graph | None:
    graph = match.graph
    if not isinstance(producer.op, LoopOp) or not isinstance(consumer.op, LoopOp):
        raise RuleSkipped("producer or consumer is no longer a LoopOp")
    if producer.id not in consumer.inputs:
        raise RuleSkipped(f"producer {producer.id!r} is not an input of consumer {consumer.id!r}")

    # Guard 1: scope key — only split products and their direct neighbors.
    if not (producer.op.knobs.get(_SPLIT_CONE) is True or consumer.op.knobs.get(_SPLIT_CONE) is True):
        raise RuleSkipped("neither op is a split product (SPLIT_CONE=True) — loop-tier fusion already settled this pair")

    # Guard 2: one-level marker — never compound re-fusions (one-batch ≡ quiescence).
    if producer.hints.get(GLUE_HINT) or consumer.hints.get(GLUE_HINT):
        raise RuleSkipped("already a re-fused composite — second-order merges are out of contract")

    # Guard 3: K-cell protection — inlining into a reduce scope re-demotes the matmul.
    if _reads_buf_in_reduce_scope(consumer.op, producer.id):
        raise RuleSkipped(f"consumer reads {producer.id!r} inside a reduce loop — merging would re-pollute the K cell")

    # Guard 4: one gemm per kernel — don't undo the multi-accum extraction.
    if _has_accum(producer.op) and _has_accum(consumer.op):
        raise RuleSkipped("both ops carry an Accum — merging would rebuild a multi-reduce kernel the split cut apart")

    merged = build_merged_op(graph, producer, consumer)
    if merged is None:
        raise RuleSkipped(f"splice_graph rejected pattern: {producer.id!r} -> {consumer.id!r}")

    # Guard 5: the base fusion economics (blowup + broadcast materialization).
    pre_work = _base._total_work(producer.op) + _base._total_work(consumer.op)
    pre_reads = _base._total_reads(producer.op) + _base._total_reads(consumer.op)
    if _base._total_work(merged) > _base._BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={_base._total_work(merged)} > {_base._BLOWUP_FACTOR}× pre={pre_work}")
    if _base._total_reads(merged) > _base._BLOWUP_FACTOR * pre_reads:
        raise RuleSkipped(f"read blowup: post={_base._total_reads(merged)} > {_base._BLOWUP_FACTOR}× pre={pre_reads}")
    consumer_wider = _base._output_numel(consumer.op) > _base._output_numel(producer.op)
    if is_pure_indexmap(consumer.op) and not is_pure_indexmap(producer.op) and consumer_wider:
        raise RuleSkipped("broadcast materialization: pure-indexmap consumer is wider than its compute producer")

    # Guard 6: eligibility preservation — never trade an MMA kernel for fused scalar glue.
    if ctx is not None:
        pre_eligible = any(
            is_atom_eligible(atom, op, ctx, graph=graph) for op in (producer.op, consumer.op) for atom in ATOM_REGISTRY.values()
        )
        if pre_eligible and not any(is_atom_eligible(atom, merged, ctx, graph=graph) for atom in ATOM_REGISTRY.values()):
            raise RuleSkipped("merge would cost the constituent gemm its atom (MMA) eligibility — glue stays separate")

    # Stamp: SPLIT_CONE keeps 005's idempotence guard firing on the merged op
    # (termination) — a real decision knob the prior already trains on. The
    # composite marker (guard 2) goes on the fragment NODE's hints, not
    # ``op.knobs``: knobs entries become prior training features
    # (``knob.knob_features``), and the marker is plumbing, not a decision.
    # ``source`` forwards the pre-split offer ancestor for two_level's Σ
    # attribution.
    merged.knobs = {_SPLIT_CONE: True}
    merged.source = next(
        (op.source for op in (consumer.op, producer.op) if op.knobs.get(_SPLIT_CONE) is True and op.source is not None),
        None,
    )

    frag = wrap_merge_fragment(graph, merged, consumer)
    frag.nodes[f"merged_{consumer.id}"].hints.set(GLUE_HINT, True)
    match.output = consumer.id
    match.consumed = {producer.id, consumer.id}
    return frag
