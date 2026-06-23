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

So we split the search in two, drawing the boundary on the fork's *effect*
(the ``Op``-rebind / ``Graph``-splice classification stamped at the engine's
spawn site — see ``plans/structural-forks-in-two-level.md``), not on a fixed
pass index:

- **Outer** (:func:`run_two_level_tune`) drives the graph-changing passes —
  ``frontend`` + ``loop`` plus the pre-partition head of ``lowering/tile``
  (:func:`outer_pipeline`), where the structural fork emitters live (today
  ``005_split_demoted``'s keep-vs-split offer). A terminal is the state where
  the cursor reaches ``partition_loops`` with every structural fork resolved —
  every op post-fusion and structurally final, split producers/consumers
  included as real ``LoopOp`` nodes. Each terminal is a candidate fused graph;
  its reward is ``1 / Σ best-per-op time`` from the inner search,
  backpropagated by the reused :class:`TuningSearch` — so keep-vs-split is an
  outer-terminal comparison, the natural cost model for a kernel-set decision.
  Identical offer sites within a trajectory take the same side — the engine
  replays the decision read off the trajectory's own graph via the
  ``Op.source`` decomposition links and the stamped decision knobs
  (``pipeline._replay_structural_decision``) — keeping the tree linear in
  *unique* kernels. Fusion itself is still
  deterministic (no multi-option fusion forks); this remains the clean
  insertion point for fusion search when those forks exist.
- **Inner** (:func:`_inner_reward_async`) tunes each finalized kernel *independently*
  in its own single-node slice (:func:`single_node_graph`) with a plain
  :class:`TuningSearch` over :data:`LOWERING_PASSES` only. Results key
  structurally (:func:`op_cache_key`), so they transfer to the assembled graph
  unchanged AND are shared across outer terminals (a shared op is a DB hit).

The inner search runs for **every** op on every pass — it is never skipped on
prior effort. Replay is cheap, not gated: each benched terminal hits the
per-variant ``perf`` cache (:class:`pipeline._TerminalBench`), so a variant
already measured is served from the DB with no GPU bench. An identical re-run
(same prior) re-walks the same deterministic trajectory → every terminal is a
cache hit → zero benches and the same total. But the global learned prior keeps
changing (it refits across ops and runs), so the same patience can steer the
MCTS down a *different* trajectory — re-running lets it reach and bench the
genuinely-new variants the improved prior surfaces, replaying the rest for free.
(The old per-op ``op_effort`` "skip already-tuned" gate is gone: it skipped the
whole op, which would suppress exactly that prior-driven re-exploration.)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline import CUDA_PASSES, LOOP_PASSES, Pass, Pipeline, TuningSearch
from deplodock.compiler.pipeline.search.db import PerfStats, SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.policy.greedy import PARTITION_RULE
from deplodock.compiler.pipeline.search.slice import single_node_graph

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph

logger = logging.getLogger(__name__)

# ``PARTITION_RULE`` (imported from the greedy policy, which prices kernels at
# that fork) is the rule that finalizes the kernel set: everything in
# ``lowering/tile`` *before* it is a structural (kernel-set-changing) decision
# the outer search owns; everything from it on is op-variant lowering the
# inner search tunes.

# Lowering-only passes (post-fusion): ``tile → kernel → cuda``. The inner
# per-op search runs these on a single-node slice so the finalized LoopOp body
# — and thus its ``op_cache_key`` — is never re-touched by ``loop/fusion``,
# which is what keeps inner-tuned ``perf`` / ``lowering`` rows transferable to
# the assembled graph. Sliced as the tail of ``CUDA_PASSES`` so it tracks
# pass-list edits automatically. The pre-partition tile rules (005's split
# offer) run here too, but every outer-terminal op already carries their
# decision knob, so the rules' idempotence guards skip — the inner never
# re-opens an outer-owned structural decision.
LOWERING_PASSES = CUDA_PASSES[len(LOOP_PASSES) :]


def outer_pipeline() -> Pipeline:
    """The graph-changing passes the outer search drives: ``frontend`` +
    ``loop`` (any fusion forks) **plus the pre-partition head of
    ``lowering/tile``** — the structural fork emitters that change which
    kernels exist before partition finalizes them (today ``005_split_demoted``;
    the boundary is the *effect*, not the pass index — see
    ``plans/structural-forks-in-two-level.md`` step 2). An outer terminal is a
    fused graph whose kernel set is final: the cursor reached
    :data:`PARTITION_RULE` with every structural fork resolved, so split
    producers/consumers are real ``LoopOp`` nodes picked up by
    :func:`_inner_reward_async` like any kernel — own slice, own patience, own
    progress leaf, deduped by ``op_cache_key`` across layers and terminals.

    Sub-partition splices (``017_atomic_free_splitk``'s combine) stay on the
    inner slice-sum path — their trigger knob (``SPLITK``) doesn't exist until
    partition runs."""
    passes = [Pass.load(name, i) for i, name in enumerate(LOOP_PASSES)]
    tile_rules = [r.name for r in Pass.load("lowering/tile", index=len(passes)).rules]
    head = tile_rules[: tile_rules.index(PARTITION_RULE)]
    # Re-load with ``select`` (rather than slicing the loaded rules) so the
    # partial pass's ``declared_knobs`` covers only the pre-partition rule
    # modules — the pass-boundary OFF-fill must not stamp post-partition
    # lowering knobs onto fused LoopOps (that would churn every op_cache_key
    # away from what the assembled greedy run derives).
    passes.append(Pass.load("lowering/tile", index=len(passes), select=set(head)))
    return Pipeline(passes=passes)


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
    multiplicity: int = 1
    # Inner-search effort, for ML-experiment tracking (search speed): total GPU
    # benches this op did, the bench index at which the best was found
    # (benches-to-best), and why the search stopped (``patience`` / ``max_visits``).
    # Read off the live ``TuningSearch`` at op completion — see ``_inner_reward_async``.
    benches: int = 0
    benches_to_best: int | None = None
    stop_reason: str | None = None


@dataclass
class InnerReward:
    """Result of evaluating one outer terminal: ``Σ best-per-op time``."""

    total_us: float
    ok: bool  # every kernel had a clean ``ok`` measurement
    per_op: list[OpResult] = field(default_factory=list)
    total_benches: int = 0  # Σ benches across ``per_op`` — the terminal's search cost
    # Learned-prior end-of-run sanity block(s) — printed by the command after
    # the progress bar closes.
    prior_summaries: list[str] = field(default_factory=list)


@dataclass
class TwoLevelResult:
    """Outcome of :func:`run_two_level_tune`."""

    best_fused: Graph | None  # winning fused graph (finalized LoopOps)
    best_reward: InnerReward | None  # its Σ-per-op breakdown
    n_terminals: int  # outer terminals evaluated (1 today)
    assembled: Graph | None  # greedy DB-best Graph[CudaOp] assembled from the bests
    prior_summaries: list[str] = field(default_factory=list)  # learned-prior stats


def _point_stats(us: float) -> PerfStats:
    """A degenerate :class:`PerfStats` carrying a single aggregate value
    (``n_samples=0`` marks it as a derived total, not a raw sample set)."""
    return PerfStats(median=us, min=us, max=us, mean=us, variance=0.0, n_samples=0)


def _kernel_nodes(graph: Graph) -> list[tuple[str, object]]:
    """Post-fusion kernel nodes — ``(node_id, op)`` for every ``LoopOp``."""
    from deplodock.compiler.ir.loop import LoopOp  # noqa: PLC0415

    return [(nid, n.op) for nid, n in graph.nodes.items() if isinstance(n.op, LoopOp)]


def _decomposition_rows(graph: Graph, per_op: list[OpResult], ctx: Context) -> list[tuple[dict, float]]:
    """Composed value-of-position training rows for structural decompositions.

    A terminal's kernels attribute back to the op a structural decision was
    taken on via ``Op.source`` (``Candidate.apply`` stamps loop→loop splice
    hops; the keep-side ``Op`` rebind stamps as always). Group the terminal's
    unique kernels by that pre-decision ancestor + the decision-knob delta the
    hop stamped (``SPLIT_CONE: True/False``, never the per-body ``S_*``
    restamps), and label each group with the **Σ of its kernels' tuned bests**
    — the kernel-set cost of taking that side. The row's features
    (``{ctx, pre-decision op knobs, decision delta}``) are exactly what the
    outer MCTS's PUCT queries at the structural fork's siblings, so these rows
    are what make that selection informed instead of uniform. They are
    estimates for *search ordering*; greedy's deploy decision keeps the sharper
    compositional probe (``policy/greedy._pick_structural``)."""
    from deplodock.compiler.pipeline.search.keys import dialect_of  # noqa: PLC0415

    best = {r.op_key: r.best_us for r in per_op}
    unique: dict[str, object] = {}
    for _nid, op in _kernel_nodes(graph):
        key = op_cache_key(op)
        if key is not None and key not in unique:
            unique[key] = op
    groups: dict[tuple, tuple[dict, list[float | None]]] = {}
    for key, op in unique.items():
        site = op.source
        if site is None or dialect_of(site) != "loop":
            continue
        site_key = op_cache_key(site)
        if site_key is None:
            continue
        # The decision delta: knobs this hop introduced. ``S_*`` keys are
        # excluded — fragment kernels restamp their own structural features,
        # which describe the child body, not the decision.
        delta = {k: v for k, v in op.knobs.items() if k not in site.knobs and not k.startswith("S_")}
        if not delta:
            continue  # not a decision hop (e.g. a name-only rebind ancestor)
        gkey = (site_key, tuple(sorted((k, str(v)) for k, v in delta.items())))
        feats, labels = groups.setdefault(gkey, ({**ctx.features(), **site.knobs, **delta}, []))
        labels.append(best.get(key))
    return [(feats, float(sum(labels))) for feats, labels in groups.values() if labels and all(us is not None for us in labels)]


async def _inner_reward_async(fused_graph, *, ctx, db, pool, patience, ucb_c, explore_eps, seed, progress, prior) -> InnerReward:
    """Tune every post-fusion kernel of ``fused_graph`` in its own single-node slice
    and return ``Σ best-per-op time`` — the outer terminal reward.

    One coroutine per unique kernel over a slot queue of ``len(pool)`` device-pinned
    :class:`CudaBackend`s (one in-flight bench per slot), each op's inner search
    running on its slot via :meth:`Pipeline.tune_async`. Single event loop, single
    thread — the shared ``db`` / ``prior`` are touched only between bench ``await``s,
    so they're atomic with no locks. A single-element ``pool`` is the serial case:
    coroutines acquire the lone worker in ``op_idx`` order → strictly sequential.

    ``prior`` (a single shared :class:`~deplodock.compiler.pipeline.search.prior.Prior`,
    or ``None``) drives every inner search's PUCT — ONE **global** model across all
    kernels: each op's search trains it on ``archived + this op's tree``; when the op
    finishes its rows are archived and the prior is checkpointed (keyed by regime), so
    a later compile / tune reloads it.

    Every kernel's slice is tuned by a plain inner :class:`TuningSearch` over
    :data:`LOWERING_PASSES` on every pass — never skipped on prior effort. The cost is
    paid at the bench, not gated at the op: :class:`pipeline._TerminalBench` serves any
    already-measured variant from the ``perf`` cache, so an identical re-run benches
    nothing while a prior-shifted trajectory benches only its genuinely-new variants.
    Benches scale as ``Σ_k n_k`` (per op), never the product.

    ``progress`` (a duck-typed :class:`~deplodock.commands.tune_progress.TuneProgress`,
    or ``None``) drives the CLI progress bar: one op leaf ticked per *unique* kernel,
    the live tail updated per benched variant. Leaves are deduped by ``op_cache_key``
    before iteration; multiplicity is preserved so ``total_us`` weights each unique
    kernel's best by its node count (order-stable, identical to per-node iteration)."""
    from collections import OrderedDict  # noqa: PLC0415

    from deplodock.compiler.pipeline.pipeline import variant_label  # noqa: PLC0415

    ctx_key = ctx.structural_key()
    backend_name = getattr(pool[0], "name", "cuda")
    # Group structurally-identical LoopOps under one ``op_cache_key`` —
    # insertion order = first occurrence (drives the progress tail name).
    # Ops with no cache key are unreachable through the bench path so they
    # don't enter the dedup map at all (matches the previous filter).
    unique: OrderedDict[str, tuple[str, object, int]] = OrderedDict()
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

    # Slot queue: each coroutine pops a device-pinned backend, benches its op's
    # whole inner search on it, returns it. ``len(pool)`` benches run at once.
    slots: asyncio.Queue = asyncio.Queue()
    for b in pool:
        slots.put_nowait(b)
    results: dict[int, OpResult] = {}

    async def tune_op(op_idx: int, key: str, nid: str, op, count: int) -> None:
        name = getattr(op, "name", None) or nid
        backend = await slots.get()
        try:
            if progress is not None:
                progress.op_start(name, slot=op_idx)
            sub = single_node_graph(fused_graph, nid)
            # Base knobs the prior sees on every row: the LoopOp's ``S_*``
            # structural identity (op-aware rows) + the ``H_*`` host/hardware
            # regime (GPU + nvcc opt level), so one global prior spans ops and
            # regimes from the feature vector alone.
            base_knobs = {**ctx.features(), **op.knobs}
            # Per-op RNG seed so each kernel's ε-greedy stream differs yet the run
            # is reproducible AND execution-order-independent (no wall-clock seed):
            # op ``op_idx`` always seeds ``seed + op_idx`` regardless of which slot
            # or completion order it ran in.
            inner = TuningSearch(
                patience=patience,
                ucb_c=ucb_c,
                explore_eps=explore_eps,
                seed=seed + op_idx,
                prior_model=prior,
                base_knobs=base_knobs,
            )
            async for cand in Pipeline.build(LOWERING_PASSES).tune_async(sub, search=inner, ctx=ctx, backend=backend, db=db):
                if progress is not None:
                    st = inner.last_stats
                    best_us = (1.0 / inner.tree.best_reward) if inner.tree.best_reward > 0 else None
                    progress.variant(
                        name,
                        variant_label(cand.graph),
                        median_us=st.median if st is not None else None,
                        status=inner.last_status or "",
                        best_us=best_us,
                        slot=op_idx,
                    )
            # The inner MCTS's best reward is ``1 / min whole-slice total``
            # (``_bench_terminal`` sums every CudaOp in the slice, so a split-K
            # main + combine both count). Record that total under the LoopOp
            # key so ``best_per_op_time`` reads the true per-op cost.
            best_total = 1.0 / inner.tree.best_reward if inner.tree.best_reward > 0 else None
            if best_total is not None:
                # captured=True: the sweep benches under graph capture by default, so
                # this Σ-best bookkeeping row derives from captured measurements (a
                # rare per-variant capture fallback can contaminate the min — accepted;
                # the capture-wins overwrite then lets a re-tune upgrade it).
                db.record_perf(ctx_key, key, backend=backend_name, status="ok", stats=_point_stats(best_total), captured=True)
            if prior is not None:
                # In-flight refit (single-threaded → no lock): stream this op's
                # value-of-position rows (-O1) plus any -O3 winner samples into the
                # global reservoir; refit + checkpoint once enough new rows
                # accumulate (batched — see ``Prior``). Rows arrive in completion
                # order, so the trained ``prior.json`` varies run-to-run; the per-op
                # DB best below does not (distinct ``key`` per op).
                prior.add_rows(inner._collect_rows() + inner.o3_rows)
                if prior.maybe_refit():
                    prior.checkpoint()
            best = db.best_per_op_time(ctx_key, key, backend=backend_name)
            results[op_idx] = OpResult(
                name=name,
                op_key=key,
                best_us=best,
                multiplicity=count,
                benches=inner.tree.root.visits,
                benches_to_best=inner._visits_at_best,
                stop_reason=inner.stop_reason,
            )
            if progress is not None:
                progress.op_done(name, slot=op_idx)
        finally:
            slots.put_nowait(backend)

    try:
        await asyncio.gather(*[tune_op(i, key, nid, op, count) for i, (key, (nid, op, count)) in enumerate(unique.items())])
    finally:
        # SIGKILL + await-reap each slot's async bench worker (the subprocess
        # transports are bound to this event loop; awaiting the reap cleans them
        # between terminals). Backend objects persist — their workers respawn lazily
        # on the next terminal's first ``benchmark_async`` (same loop, since the whole
        # outer drive runs under one ``asyncio.run`` in ``handle_tune``).
        for b in pool:
            aclose = getattr(b, "aclose_async_worker", None)
            if aclose is not None:
                await aclose()

    # Accumulate in ``op_idx`` order so the reward / ``per_op`` order is
    # execution-order-independent (the float sum is order-stable, matching serial).
    total = 0.0
    ok = True
    per_op: list[OpResult] = []
    for op_idx in range(len(unique)):
        r = results[op_idx]
        per_op.append(r)
        if r.best_us is None:
            ok = False
            total += _FAIL_US * r.multiplicity
        else:
            total += r.best_us * r.multiplicity
    return InnerReward(total_us=total, ok=ok, per_op=per_op, total_benches=sum(r.benches for r in per_op), prior_summaries=[])


async def run_two_level_tune(
    graph: Graph,
    *,
    ctx: Context,
    db: SearchDB,
    backend=None,
    backends=None,
    patience: int,
    ucb_c: float = TuningSearch.DEFAULT_UCB_C,
    explore_eps: float = 0.0,
    dump=None,
    progress=None,
    prior_seed: int = 0,
) -> TwoLevelResult:
    """Drive the outer structural search, scoring each terminal by
    :func:`_inner_reward_async`, then greedy-assemble the DB-best kernels and bench
    the whole graph once for the separability check.

    ``backends`` (a list of device-pinned :class:`CudaBackend`s) fans the inner
    per-kernel search out across GPUs; the default single ``backend`` is the
    one-slot serial pool.

    The outer drives a :class:`Run` directly (manual ``observe``)
    because its terminal reward comes from the inner tuning, not
    ``_bench_terminal_async``. The outer pipeline (:func:`outer_pipeline`) runs
    through the pre-partition tile rules, so each structural fork (the
    keep-vs-split offer of ``005_split_demoted``) branches the outer tree —
    one terminal per kernel-set, compared by Σ-per-op cost. A graph with no
    structural offers yields a single terminal and this reduces to "tune each
    op once, sum, assemble". Identical offer sites within one trajectory
    replay the first decision (read off the graph —
    ``pipeline._replay_structural_decision``), and a terminal whose kernels
    are all known is a pure DB read, so extra terminals stay cheap."""
    from deplodock.compiler import provenance  # noqa: PLC0415
    from deplodock.compiler.pipeline.pipeline import Run  # noqa: PLC0415

    provenance.seed(graph)
    # ONE global prior for the whole run — the learned ``CatBoostPrior`` (warm-
    # started from its checkpoint) behind an ``AnalyticPrior`` cold-start
    # fallback, so the first op's inner search is heuristic-guided, not uniform.
    # Training (add_rows / maybe_refit / checkpoint) delegates to the learned half
    # (lazy import keeps catboost off non-tune callers).
    from deplodock.compiler.pipeline.search.prior import load_prior  # noqa: PLC0415

    prior = load_prior(seed=prior_seed)
    # The global prior drives the outer PUCT too: at a structural fork the
    # siblings' ``_node_knobs`` are ``{ctx, pre-decision op knobs, decision
    # knob}`` — the exact feature shape :func:`_decomposition_rows` trains on
    # below — so once composed Σ rows accumulate, the outer descends the
    # predicted-cheaper kernel set first instead of emission order.
    outer = TuningSearch(patience=patience, ucb_c=ucb_c, prior_model=prior, base_knobs=ctx.features())
    # The outer drives only the graph-changing passes (through the
    # pre-partition tile head) — no dump on this Run; the winning config's
    # full stage artifacts (incl. per-kernel ``.torch.json`` reproducers) come
    # from the final assembled CUDA_PASSES run below.
    outer_run = Run(pipeline=outer_pipeline(), ctx=ctx, search=outer, db=db)

    best_fused: Graph | None = None
    best_reward: InnerReward | None = None
    n_terminals = 0
    prior_summaries: list[str] = []
    pool = list(backends) if backends else [backend]
    for token, fused in outer_run.drive(graph):
        n_terminals += 1
        reward = await _inner_reward_async(
            fused.graph,
            ctx=ctx,
            db=db,
            pool=pool,
            patience=patience,
            ucb_c=ucb_c,
            explore_eps=explore_eps,
            seed=prior_seed,
            progress=progress,
            prior=prior,
        )
        stats = PerfStats(median=reward.total_us, min=reward.total_us, max=reward.total_us, mean=reward.total_us, variance=0.0, n_samples=0)
        outer.observe(token, stats, "ok" if reward.ok else "bench_fail")
        # Composed Σ rows per structural decision this terminal realized —
        # the kernel-set cost of each side, attributed via the ``Op.source``
        # decomposition links. Re-emitted every terminal evaluation, so the
        # reservoir keeps refreshing the sum as per-kernel bests fall.
        rows = _decomposition_rows(fused.graph, reward.per_op, ctx)
        if rows:
            prior.add_rows(rows)
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

    # One global end-of-run sanity block (the prior spans every kernel now).
    if prior.fitted or prior.trajectory:
        prior_summaries.append(prior.summary("global"))
    # Force a final fit so even a small tune that never crossed a refit tier ends
    # with a usable model, then persist (dataset accumulates across runs).
    prior.maybe_refit(force=True)
    prior.checkpoint()

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
        assembled = Pipeline.build(CUDA_PASSES).run(graph, ctx=ctx, db=db, dump=dump)
    return TwoLevelResult(
        best_fused=best_fused, best_reward=best_reward, n_terminals=n_terminals, assembled=assembled, prior_summaries=prior_summaries
    )
