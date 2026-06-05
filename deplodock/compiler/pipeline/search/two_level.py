"""Two-level autotuning: an outer fusion MCTS whose terminal reward comes from
an inner, separable, per-op search.

``deplodock tune`` used to run one SP-MCTS tree over the whole graph. Because
the pipeline applies rules sequentially, op-variant forks (tile / pad / stage
choices for one kernel) and fusion forks (how ops group into kernels) **nest**
and cross-product under one global patience — so the bottleneck op starves the
deep ops (see ``project_mcts_exploration_limit``). The two kinds of decision
have opposite structure:

- **op-variant forks are separable** — every multi-option fork today is an
  in-place ``Op`` rebind that leaves the graph unchanged, so the whole-graph
  time is ``Σ_k t_k`` with each ``t_k`` depending only on op ``k``'s variant;
- **fusion forks are NOT separable** — they change *which ops exist*.

So we split the search in two:

- **Outer** (:func:`run_two_level_tune`) drives only the graph-changing passes
  (:data:`OUTER_PASSES` = ``frontend`` + ``loop``). A terminal is the state
  where the cursor reaches ``partition_loops`` — every op post-fusion and
  structurally final. Each terminal is a candidate fused graph; its reward is
  ``1 / Σ best-per-op time`` from the inner search, backpropagated by the
  reused :class:`TuningSearch`. Today fusion is deterministic (no multi-option
  fusion forks) so the outer tree has one terminal — but this is the clean
  insertion point for fusion search when those forks exist.
- **Inner** (:func:`inner_reward`) tunes each finalized kernel *independently*
  in its own single-node slice (:func:`single_node_graph`) with a plain
  :class:`TuningSearch` over :data:`LOWERING_PASSES` only. Results key
  structurally (:func:`op_cache_key`), so they transfer to the assembled graph
  unchanged AND are shared across outer terminals (a shared op is a DB hit).

An op is skipped when it is already ``terminated`` — its recorded inner-search
effort meets the requested patience (an exhausted op records ``inf``). That
makes re-runs idempotent, higher patience re-deepen only under-tuned ops, and
the whole thing resumable across sessions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pipeline, TuningSearch
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import PerfStats, SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.slice import single_node_graph

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph

logger = logging.getLogger(__name__)

# The graph-changing passes the outer search drives (any fusion forks live
# here). A terminal is reached when the cursor would advance into
# ``lowering/tile`` (``partition_loops``).
OUTER_PASSES = LOOP_PASSES

# Lowering-only passes (post-fusion): ``tile → kernel → cuda``. The inner
# per-op search runs these on a single-node slice so the finalized LoopOp body
# — and thus its ``op_cache_key`` — is never re-touched by ``loop/fusion``,
# which is what keeps inner-tuned ``perf`` / ``lowering`` rows transferable to
# the assembled graph. Sliced as the tail of ``CUDA_PASSES`` so it tracks
# pass-list edits automatically.
LOWERING_PASSES = CUDA_PASSES[len(LOOP_PASSES) :]

# Per-op latency stand-in when the inner search produced no clean ``ok``
# measurement — large enough to sink the outer reward, finite so the Σ stays a
# real number for the separability report.
_FAIL_US = 1e12


@dataclass
class OpResult:
    """One unique kernel's inner-search outcome, for the per-op summary.

    ``multiplicity`` is the number of structurally-identical ``LoopOp`` nodes
    in the fused graph that share this ``op_key`` — 24 for a 24-layer
    RMSNorm, 1 for a singleton. The outer reward's ``total_us`` weights
    ``best_us`` by ``multiplicity`` so the Σ across ``per_op`` equals the
    whole-graph latency (every node position counts, even though dedup
    means we only run the inner search and DB lookup once per key).
    """

    name: str
    op_key: str
    best_us: float | None
    benched: bool  # True when the inner search ran; False when skipped (terminated)
    multiplicity: int = 1


@dataclass
class InnerReward:
    """Result of evaluating one outer terminal: ``Σ best-per-op time``."""

    total_us: float
    ok: bool  # every kernel had a clean ``ok`` measurement
    per_op: list[OpResult] = field(default_factory=list)


@dataclass
class TwoLevelResult:
    """Outcome of :func:`run_two_level_tune`."""

    best_fused: Graph | None  # winning fused graph (finalized LoopOps)
    best_reward: InnerReward | None  # its Σ-per-op breakdown
    n_terminals: int  # outer terminals evaluated (1 today)
    assembled: Graph | None  # greedy DB-best Graph[CudaOp] assembled from the bests


def _point_stats(us: float) -> PerfStats:
    """A degenerate :class:`PerfStats` carrying a single aggregate value
    (``n_samples=0`` marks it as a derived total, not a raw sample set)."""
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)


def _kernel_nodes(graph: Graph) -> list[tuple[str, object]]:
    """Post-fusion kernel nodes — ``(node_id, op)`` for every ``LoopOp``."""
    from deplodock.compiler.ir.loop import LoopOp  # noqa: PLC0415

    return [(nid, n.op) for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp)]


def inner_reward(
    fused_graph: Graph,
    *,
    ctx: Context,
    db: SearchDB,
    backend,
    patience: int,
    ucb_c: float = TuningSearch.DEFAULT_UCB_C,
    progress=None,
) -> InnerReward:
    """Tune every post-fusion kernel of ``fused_graph`` in its own single-node
    slice and return ``Σ best-per-op time`` — the outer terminal reward.

    Each kernel is gated on persisted effort: an op already tuned to
    ``>= patience`` (or exhausted, recorded ``inf``) is skipped and its cached
    best is summed directly. Otherwise its slice is tuned by a plain inner
    :class:`TuningSearch` over :data:`LOWERING_PASSES`, the effort is recorded
    (``inf`` when the inner tree drained, else ``patience``), and the new best
    is summed. Benches scale as ``Σ_k n_k`` (per op), never the product.

    ``progress`` (a duck-typed :class:`~deplodock.commands.tune_progress.TuneProgress`,
    or ``None``) drives the CLI progress bar: one op leaf ticked per *unique*
    kernel (24-layer RMSNorm = 1 tick, not 24), the live tail updated per
    benched variant. Kept duck-typed so this module carries no dependency on
    ``commands/``.

    Leaves are deduped by ``op_cache_key`` before iteration. The inner search
    keys every DB write on the structural key, so iterating per occurrence
    (337 LoopOp nodes on Qwen3-Embedding-0.6B) only differs from per-unique-key
    iteration (~14) by an O(positions) ``db.terminated`` / ``best_per_op_time``
    cache lookup — and gives the user a misleading progress denominator.
    Multiplicity is preserved: ``total_us`` weights each unique kernel's best
    by its node count, so the outer MCTS reward is bit-for-bit identical to
    the per-node-iterated total."""
    from collections import OrderedDict  # noqa: PLC0415

    from deplodock.compiler.pipeline.pipeline import variant_label  # noqa: PLC0415

    ctx_key = ctx.structural_key()
    backend_name = getattr(backend, "name", "cuda")
    total = 0.0
    ok = True
    per_op: list[OpResult] = []
    # Group structurally-identical LoopOps under one ``op_cache_key`` —
    # insertion order = first occurrence (drives the progress tail name).
    # Ops with no cache key are unreachable through the bench path so they
    # don't enter the dedup map at all (matches the previous filter).
    unique: OrderedDict[str, tuple[str, object, int]] = OrderedDict()
    # One score cache for ALL the per-op inner searches: ``Fork.score_key``
    # is value-keyed (plan cache key + knob row), so slices of structurally
    # identical kernels — and re-visits of the same kernel — share every
    # planner-prior score instead of recomputing ~40k ``lazy_score`` rows
    # per slice.
    score_cache: dict = {}
    for nid, op in _kernel_nodes(fused_graph):
        key = op_cache_key(op)
        if key is None:
            continue
        if key in unique:
            rep_nid, rep_op, count = unique[key]
            unique[key] = (rep_nid, rep_op, count + 1)
        else:
            unique[key] = (nid, op, 1)
    if progress is not None:
        progress.start_terminal(len(unique))
    for key, (nid, op, count) in unique.items():
        name = getattr(op, "name", None) or nid
        if progress is not None:
            progress.op_start(name)
        benched = False
        if not db.terminated(ctx_key, key, patience):
            sub = single_node_graph(fused_graph, nid)
            inner = TuningSearch(patience=patience, ucb_c=ucb_c, score_cache=score_cache)
            for cand in Pipeline.build(LOWERING_PASSES).tune(sub, search=inner, ctx=ctx, backend=backend, db=db):
                if progress is not None:
                    st = inner.last_stats
                    best_us = (1.0 / inner.tree.best_reward) if inner.tree.best_reward > 0 else None
                    progress.variant(
                        name,
                        variant_label(cand.graph),
                        median_us=st.median if st is not None else None,
                        status=inner.last_status or "",
                        best_us=best_us,
                    )
            # The inner MCTS's best reward is ``1 / min whole-slice total``
            # (``_bench_terminal`` sums every CudaOp in the slice, so a split-K
            # main + combine both count). Record that total under the LoopOp
            # key so ``best_per_op_time`` reads the true per-op cost — and a
            # skipped (terminated) re-run still finds it.
            best_total = 1.0 / inner.tree.best_reward if inner.tree.best_reward > 0 else None
            if best_total is not None:
                db.record_perf(ctx_key, key, backend=backend_name, status="ok", stats=_point_stats(best_total))
            # ``stop_reason is None`` ⟺ the queue drained (every variant
            # measured) ⟺ the op's tree is exhausted; record ``inf`` so any
            # future patience treats it as terminated. A patience stop records
            # exactly the patience it ran with (incremental deepening).
            exhausted = inner.stop_reason is None
            db.record_effort(ctx_key, key, math.inf if exhausted else float(patience))
            benched = True
        best = db.best_per_op_time(ctx_key, key, backend=backend_name)
        per_op.append(OpResult(name=name, op_key=key, best_us=best, benched=benched, multiplicity=count))
        # Multiplicity-weighted accumulation — a 24-layer RMSNorm contributes
        # 24 × best, matching the per-node total before dedup.
        if best is None:
            ok = False
            total += _FAIL_US * count
        else:
            total += best * count
        if progress is not None:
            progress.op_done(name)
    return InnerReward(total_us=total, ok=ok, per_op=per_op)


def run_two_level_tune(
    graph: Graph,
    *,
    ctx: Context,
    db: SearchDB,
    backend,
    patience: int,
    ucb_c: float = TuningSearch.DEFAULT_UCB_C,
    dump=None,
    progress=None,
) -> TwoLevelResult:
    """Drive the outer fusion search, scoring each terminal by
    :func:`inner_reward`, then greedy-assemble the DB-best kernels and bench
    the whole graph once for the separability check.

    The outer uses :meth:`Pipeline.search` directly (manual ``observe``)
    because its terminal reward comes from the inner tuning, not
    ``_bench_terminal``. Today there are no multi-option fusion forks, so the
    outer yields a single terminal and this reduces to "tune each op once, sum,
    assemble"."""
    from deplodock.compiler import provenance  # noqa: PLC0415
    from deplodock.compiler.pipeline.pipeline import Cursor  # noqa: PLC0415

    provenance.seed(graph)
    outer = TuningSearch(patience=patience, ucb_c=ucb_c)
    # The outer drives only the graph-changing passes — no dump here; the
    # winning config's full stage artifacts (incl. per-kernel ``.torch.json``
    # reproducers) come from the final assembled CUDA_PASSES run below.
    outer_pipeline = Pipeline.build(OUTER_PASSES)
    outer.push(Candidate(ctx=ctx, graph=graph, cursor=Cursor(pipeline=outer_pipeline)).lazy())

    best_fused: Graph | None = None
    best_reward: InnerReward | None = None
    n_terminals = 0
    for fused in outer_pipeline.search(outer, ctx, db=db):
        n_terminals += 1
        reward = inner_reward(fused.graph, ctx=ctx, db=db, backend=backend, patience=patience, ucb_c=ucb_c, progress=progress)
        stats = PerfStats(median=reward.total_us, min=reward.total_us, max=reward.total_us, mean=reward.total_us, variance=0.0, n_samples=0)
        outer.observe(stats, "ok" if reward.ok else "bench_fail")
        positions = sum(r.multiplicity for r in reward.per_op)
        logger.info(
            "[tune] fused terminal #%d: Σ per-op = %.2f us (%d unique kernels, %d positions)",
            n_terminals,
            reward.total_us,
            len(reward.per_op),
            positions,
        )
        if best_reward is None or (reward.ok and reward.total_us < best_reward.total_us):
            best_fused, best_reward = fused.graph, reward

    assembled: Graph | None = None
    if best_fused is not None:
        # Greedy replay over the *original* graph re-derives the same fused
        # LoopOps and lowers each via the DB-best forks the inner search
        # recorded. No backend → ``_bench_terminal`` persists nothing (so the
        # 1.0us stub never clobbers a tuned row). The dump (if any) rides here
        # so it captures the winning config's full stage artifacts. The
        # whole-graph separability bench used to run here too; dropped — its
        # only output was a small advisory gap line that nobody read, and at
        # the tight tune compile budget it could (and did) abort whole-model
        # tunes when a slow-compiling kernel raised. ``--bench`` re-benches
        # the assembled graph at -O3 anyway, which is the deployable number.
        assembled = Pipeline.build(CUDA_PASSES, dump=dump).run(graph, ctx=ctx, db=db)
    return TwoLevelResult(best_fused=best_fused, best_reward=best_reward, n_terminals=n_terminals, assembled=assembled)
