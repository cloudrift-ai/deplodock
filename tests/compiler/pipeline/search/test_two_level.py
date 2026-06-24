"""Two-level autotuning: the inner separable per-op reward.

The inner search tunes each post-fusion kernel in its own single-node slice.
These tests pin the separability properties with a fake counting backend (no
GPU): benches scale as ``Σ_k n_k`` not the product, the bests land in the DB,
re-runs are idempotent under the effort gate, a higher patience re-deepens
only under-tuned ops, and a kernel shared by two terminals is tuned once.

Target is forced to sm_80 so lowering is deterministic and GPU-independent —
the fake backend never launches anything, it just hands back per-launch
latencies keyed off each CudaOp's structural key.
"""

from __future__ import annotations

import zlib

import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler.backend.base import BenchmarkResult, LaunchTime
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline, TuningSearch
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.slice import single_node_graph
from deplodock.compiler.pipeline.search.two_level import (
    LOWERING_PASSES,
    OpResult,
    _decomposition_rows,
    outer_pipeline,
)
from tests.compiler.conftest import drain_tune, run_inner_reward, run_two_level

# Moderate patience: each kernel explores several variants then stops on
# stagnation (the fake backend gives a stable but arbitrary per-variant
# signal). Enough to exercise "Σ_k n_k, not the product" without paying for a
# full tree drain — exhaustion (∞ effort) is covered directly in test_db.py.
_PATIENCE = 8


@pytest.fixture(autouse=True)
def _force_target(monkeypatch, tmp_path):
    from deplodock.compiler import target as target_mod

    # Isolate the learned-prior checkpoint: ``run_two_level_tune`` trains and
    # checkpoints the global prior, and these fake-backend rows must never
    # pollute the host's real ``~/.cache/deplodock/prior.json``.
    monkeypatch.setenv("DEPLODOCK_PRIOR_FILE", str(tmp_path / "prior.json"))
    target_mod.set_target((8, 0))
    yield
    target_mod.set_target(None)


class _CountingBackend:
    """Fake backend: counts ``benchmark`` calls and returns a deterministic
    per-CudaOp latency keyed off the op's structural key, so the inner search
    sees real signal and a unique best without touching a GPU."""

    name = "cuda"
    bench_run_timeout_s = 1.0

    def __init__(self) -> None:
        self.calls = 0

    def benchmark(self, graph, num_iters="auto") -> BenchmarkResult:  # noqa: ARG002
        self.calls += 1
        cuda = [n for n in graph.nodes.values() if isinstance(n.op, CudaOp)]
        per: list[LaunchTime] = []
        for i, n in enumerate(cuda):
            # crc32, not hash(): str hashes are salted per process (PYTHONHASHSEED),
            # which made the MCTS path — and the bench counts the separability test
            # bounds — vary run to run.
            us = 1.0 + (zlib.crc32(op_cache_key(n.op).encode()) % 100)
            per.append(LaunchTime(idx=i, kernel_name=getattr(n.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,)))
        return BenchmarkResult(time_ms=sum(p.time_ms for p in per), num_launches=len(per), per_launch=per)

    async def benchmark_async(self, graph, num_iters="auto") -> BenchmarkResult:
        # The two-level driver benches through the async path (``Pipeline.tune_async``);
        # the fake has no real I/O, so delegate to the deterministic sync bench. The
        # signature mirrors ``benchmark`` exactly (no ``nvcc_flags``) so the -O3
        # re-bench rejects the same way → identical bench counts to the sync path.
        return self.benchmark(graph, num_iters=num_iters)


def _matmul(g: Graph, prefix: str, M: int, K: int, N: int) -> str:
    a, b, c = f"{prefix}a", f"{prefix}b", f"{prefix}c"
    g.add_node(InputOp(), [], Tensor(a, (M, K)), node_id=a)
    g.add_node(InputOp(), [], Tensor(b, (K, N)), node_id=b)
    g.add_node(MatmulOp(), [a, b], Tensor(c, (M, N)), node_id=c)
    return c


def _two_distinct_matmuls() -> Graph:
    g = Graph()
    c1 = _matmul(g, "x", 64, 128, 48)
    c2 = _matmul(g, "y", 96, 64, 32)
    g.inputs = ["xa", "xb", "ya", "yb"]
    g.outputs = [c1, c2]
    return g


def _two_identical_matmuls() -> Graph:
    g = Graph()
    c1 = _matmul(g, "x", 64, 128, 48)
    c2 = _matmul(g, "y", 64, 128, 48)
    g.inputs = ["xa", "xb", "ya", "yb"]
    g.outputs = [c1, c2]
    return g


def _fuse(graph: Graph) -> Graph:
    return Pipeline.build(LOOP_PASSES).run(graph, db=SearchDB())


def _loop_ids(fused: Graph) -> list[str]:
    return [nid for nid, n in fused.nodes.items() if isinstance(n.op, LoopOp)]


def _tune_one_slice(fused: Graph, nid: str, patience: int) -> int:
    """Tune a single kernel's slice in isolation; return the bench count."""
    backend = _CountingBackend()
    sub = single_node_graph(fused, nid)
    pipeline = Pipeline.build(LOWERING_PASSES)
    search = TuningSearch(patience=patience)
    drain_tune(pipeline, sub, search=search, ctx=Context.from_target((8, 0)), backend=backend, db=SearchDB())
    return backend.calls


def test_inner_reward_is_separable_not_a_product() -> None:
    """Total benches across two kernels == n1 + n2 (per-op), never n1 * n2."""
    fused = _fuse(_two_distinct_matmuls())
    loops = _loop_ids(fused)
    assert len(loops) == 2

    n1 = _tune_one_slice(fused, loops[0], _PATIENCE)
    n2 = _tune_one_slice(fused, loops[1], _PATIENCE)
    assert n1 > 1 and n2 > 1, "kernels must have multiple variants to make the point"

    backend = _CountingBackend()
    db = SearchDB()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=db, backend=backend, patience=_PATIENCE)

    # Separability: the shared run must not blow up to the cross-product
    # (n1 * n2 — the old whole-graph SP-MCTS bug this test guards against).
    # Per-op sharing through the DB perf cache is allowed — an already-measured
    # variant replays without a bench. The exact share count is sensitive to
    # MCTS exploration order: the
    # ``_CountingBackend`` fakes latency from ``crc32(op_cache_key)``, so any
    # structural-digest perturbation (e.g. a Source-field rename) shifts the
    # path and the count by a few benches. The hard guarantee is the
    # cross-product upper bound; tighter ``n1+n2`` / ``max(n1,n2)`` bounds
    # are sanity checks with slack.
    assert backend.calls < n1 * n2, "separable sum must be below the cross-product"
    # Patience-noise slack: per-op MCTS can stretch its patience window by up
    # to a handful of benches when interleaved with another kernel's search
    # in the shared run.
    slack = max(8, (n1 + n2) // 4)
    assert backend.calls <= n1 + n2 + slack, f"expected ≤ {n1 + n2 + slack} (separable+slack) benches, got {backend.calls}"
    # Every kernel measured; total is the sum of the per-op bests. Two distinct
    # structural keys → two ``per_op`` entries, each at ``multiplicity=1``.
    assert reward.ok
    assert len(reward.per_op) == 2
    assert all(r.best_us is not None for r in reward.per_op)
    assert all(r.multiplicity == 1 for r in reward.per_op)
    assert reward.total_us == pytest.approx(sum(r.best_us for r in reward.per_op))


def test_inner_reward_rerun_is_replay_dominated() -> None:
    """A second pass at the same patience is replay-dominated and never regresses:
    the warm perf cache serves almost every terminal, so the rerun benches far
    fewer variants than the cold run, and the per-op best total only improves (or
    ties), never worsens.

    Two things changed vs the old idempotence invariant. Ranking moved from the
    priority-sorted enumeration to the ``Prior`` (``AnalyticPrior`` cold), so the
    cold search walks a real prior-ranked frontier instead of finding the best at
    option-0; that frontier interacts with the cache's cross-op kernel sharing, so
    a warm rerun wanders into a handful of new frontier variants while replaying
    the rest. Those extra benches can only LOWER the per-op best — ``record_perf``
    keeps the minimum and ``best_per_op_time`` reads it — so ``second.total_us <=
    first.total_us`` always (it does not converge to the *same* total; it converges
    *downward*). The exact bench count is exploration-order-sensitive (same caveat
    as ``test_inner_reward_separability``), so only the two robust invariants are
    pinned."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    cold_backend = _CountingBackend()
    first = run_inner_reward(fused, ctx=ctx, db=db, backend=cold_backend, patience=_PATIENCE)
    rerun_backend = _CountingBackend()
    second = run_inner_reward(fused, ctx=ctx, db=db, backend=rerun_backend, patience=_PATIENCE)

    # The DB's per-op best is monotone non-increasing — more benches never worsen it.
    assert second.total_us <= first.total_us + 1e-6, "rerun must not regress the per-op best total"
    # Warm rerun replays most terminals from the perf cache → fewer benches than cold.
    # (Only "fewer", not a fixed ratio: the exact count is exploration-order-
    # sensitive — see the docstring — so a `cold // 2`-style bound is not robust.)
    assert rerun_backend.calls < cold_backend.calls, "warm rerun must bench fewer variants than the cold run"


def test_inner_reward_deeper_patience_benches_new_variants() -> None:
    """A higher patience re-runs the search (never skipped) and reaches new
    variants the shallow pass never measured — those miss the perf cache and
    bench, while the already-measured ones replay for free."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    run_inner_reward(fused, ctx=ctx, db=db, backend=_CountingBackend(), patience=1)

    deep_backend = _CountingBackend()
    run_inner_reward(fused, ctx=ctx, db=db, backend=deep_backend, patience=_PATIENCE)
    assert deep_backend.calls > 0, "a deeper pass must bench the new variants it reaches"


def test_inner_reward_shares_identical_kernel() -> None:
    """Two identical kernels in one terminal collapse to a single ``per_op``
    entry under one ``op_cache_key`` with ``multiplicity=2``. The inner
    search runs once; the outer total still costs 2× the shared best so the
    outer MCTS reward stays bit-for-bit identical to the per-node-iterated
    formulation."""
    fused = _fuse(_two_identical_matmuls())
    loops = _loop_ids(fused)
    assert len(loops) == 2
    # Same body ⇒ same structural key.
    keys = {op_cache_key(fused.nodes[nid].op) for nid in loops}
    assert len(keys) == 1, "the two matmuls must share one structural key"

    single = _tune_one_slice(fused, loops[0], _PATIENCE)
    backend = _CountingBackend()
    reward = run_inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backend=backend, patience=_PATIENCE)

    assert backend.calls == single, "shared kernel must bench once, not twice"
    assert len(reward.per_op) == 1, "identical kernels collapse to one per_op entry"
    assert reward.per_op[0].multiplicity == 2, "both node positions are counted"
    # Total weights the shared best by its multiplicity — both kernels still cost time.
    assert reward.total_us == pytest.approx(2 * reward.per_op[0].best_us)


def test_inner_reward_parallel_matches_serial() -> None:
    """The core multi-GPU invariant: tuning the unique kernels concurrently across
    a pool of N device-pinned backends yields the SAME per-op bests and summed
    reward as the one-slot serial path. Each op's search is seeded by ``op_idx``
    (execution-order-independent) and the fake backend's latency keys off
    ``op_cache_key`` (slot-independent), so completion order can't change the
    result. ``prior=None`` keeps this off the learned-prior (catboost) path."""
    fused = _fuse(_two_distinct_matmuls())
    ctx = Context.from_target((8, 0))

    serial = run_inner_reward(fused, ctx=ctx, db=SearchDB(), backend=_CountingBackend(), patience=_PATIENCE)
    parallel = run_inner_reward(fused, ctx=ctx, db=SearchDB(), backends=[_CountingBackend(), _CountingBackend()], patience=_PATIENCE)

    assert parallel.total_us == pytest.approx(serial.total_us)
    assert parallel.ok == serial.ok
    s_by_key = {r.op_key: (r.best_us, r.multiplicity) for r in serial.per_op}
    p_by_key = {r.op_key: (r.best_us, r.multiplicity) for r in parallel.per_op}
    assert p_by_key == s_by_key, "per-op bests must be identical regardless of slot count"


def _norm_linear(prefix: str, g: Graph | None = None) -> Graph:
    """RMSNorm → Linear (f16): fusion yields the prologue-demoted matmul whose
    keep-vs-split offer (``tile/010_split_demoted``) is a structural fork."""
    f16 = _dt.get("f16")
    g = g if g is not None else Graph()
    x, nw, wg, xn, o = (f"{prefix}{n}" for n in ("x", "nw", "wg", "xn", "o"))
    g.add_node(InputOp(), [], Tensor(x, (1, 32, 1024), f16), node_id=x)
    g.add_node(InputOp(), [], Tensor(nw, (1024,), f16), node_id=nw)
    g.add_node(InputOp(), [], Tensor(wg, (3072, 1024), f16), node_id=wg)
    g.add_node(RmsNormOp(eps=1e-6), [x, nw], Tensor(xn, (1, 32, 1024), f16), node_id=xn)
    g.add_node(LinearOp(), [xn, wg], Tensor(o, (1, 32, 3072), f16), node_id=o)
    g.inputs += [x, nw, wg]
    g.outputs += [o]
    return g


class _RecordingProgress:
    """Duck-typed ``TuneProgress``: captures per-terminal op denominators and
    the tuned op-leaf names."""

    def __init__(self) -> None:
        self.terminal_sizes: list[int] = []
        self.ops: list[str] = []

    def start_terminal(self, n_ops: int) -> None:
        self.terminal_sizes.append(n_ops)

    def op_start(self, name: str, *, slot: int = 0) -> None:  # noqa: ARG002
        self.ops.append(name)

    def variant(self, *a, **kw) -> None:  # noqa: ANN002
        pass

    def op_done(self, name: str, *, slot: int = 0) -> None:
        pass


class _RecordingPrior:
    """Minimal ``Prior`` stand-in for ``load_prior`` monkeypatching: unfitted
    (uniform PUCT, greedy keeps cold behavior), captures ``add_rows`` traffic."""

    fitted = False

    def __init__(self) -> None:
        self.rows: list[tuple[dict, float]] = []
        self.trajectory: list = []

    def add_rows(self, rows) -> None:
        self.rows.extend(rows)

    def maybe_refit(self, *, force: bool = False) -> bool:
        return False

    def checkpoint(self) -> None:
        pass

    def record_bench(self, knobs, median, status) -> None:
        pass

    def score(self, knobs) -> float:
        return 0.0

    def mean_score(self, knobs) -> float:
        return 0.0

    def mean_scores(self, rows) -> list[float]:
        return [0.0] * len(rows)

    def summary(self, label) -> str:
        return ""


def _is_decomposition_row(knobs: dict) -> bool:
    """Composed Σ rows carry the decision knob but no tile-level knobs — every
    inner per-kernel / branch row carries at least one partition-level knob."""
    return "CUT" in knobs and not any(k in knobs for k in ("BM", "BN", "BR", "MMA", "WM", "FM"))


def test_outer_branches_on_structural_fork(monkeypatch) -> None:
    """The outer drives through the pre-partition tile head: 005's keep-vs-split
    offer branches the OUTER tree, so both kernel sets appear as outer terminals
    and the split producer/consumer are their own tuned op leaves (own progress
    denominator), not sub-explorations inside the fused kernel's slice. Each
    terminal also feeds the prior one composed Σ row per structural decision —
    the kernel-set cost of the side it realized."""
    from deplodock.compiler import target as target_mod
    from deplodock.compiler.pipeline.search import prior as prior_pkg

    for k, v in {"WM": "2", "WN": "2", "FM": "1", "FN": "8", "BK": "2", "BM": "8", "BN": "64", "BR": "1", "SPLITK": "1", "FK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    target_mod.set_target((12, 0))
    rec = _RecordingPrior()
    monkeypatch.setattr(prior_pkg, "load_prior", lambda *a, **kw: rec)
    progress = _RecordingProgress()
    db = SearchDB()
    result = run_two_level(
        _norm_linear("a"),
        ctx=Context.from_target((12, 0)),
        db=db,
        backend=_CountingBackend(),
        patience=4,
        progress=progress,
    )
    assert result.n_terminals == 2, "keep-vs-split must branch the outer tree into two terminals"
    # One terminal carries the fused kernel (1 op leaf), the other the split
    # producer + consumer (2 op leaves) — the denominators the progress bar sees.
    assert sorted(progress.terminal_sizes) == [1, 2]
    assert any(name.endswith("_xn") for name in progress.ops), f"split producer must be its own op leaf, got {progress.ops}"
    # Both sides' composed rows reached the prior: CUT="0" (keep) labeled
    # with the fused best, CUT="1" (cut) with the split kernels' Σ.
    decomp = [(k, us) for k, us in rec.rows if _is_decomposition_row(k)]
    assert {k["CUT"] for k, _ in decomp} == {"0", "1"}
    assert all(us > 0 for _, us in decomp)
    # The decision hop never enters the ``lowering`` table (one best child per
    # parent — a multi-kernel decomposition's parent must not resolve through
    # ONE fragment kernel's median): the pre-decision op has no lowering row.
    site = next(_site(n.op) for n in result.best_fused.nodes.values() if _site(n.op) is not None)
    assert db.lookup_lowering(op_cache_key(site)) is None


def _outer_terminals(graph: Graph) -> list[Graph]:
    search = TuningSearch(patience=10**6)
    drained = drain_tune(outer_pipeline(), graph, search=search, ctx=Context.from_target((12, 0)), db=SearchDB())
    return [cand.graph for cand in drained]


def _kernels(graph: Graph) -> list:
    """Outer-terminal kernel ops. A terminal sits past ``010_build``, so its
    kernels are ``TileGraphOp`` seeds (the keep(SMEM) fused kernel + the cut's
    producer/consumer); a ``LoopOp`` survives only when ``010_build`` skipped it."""
    from deplodock.compiler.ir.tile.ir import TileGraphOp

    return [n.op for n in graph.nodes.values() if isinstance(n.op, (LoopOp, TileGraphOp))]


def _site(op):
    """The pre-decision offer site — the deepest ``S_*``-stamped loop ancestor,
    mirroring ``two_level._decomposition_rows`` (the original demoted matmul,
    before the fission re-stamped each fragment's own shallower ``S_*``)."""
    from deplodock.compiler.pipeline.search.keys import dialect_of, source_chain

    site = None
    for anc in source_chain(op):
        if dialect_of(anc) == "loop" and any(k.startswith("S_") for k in getattr(anc, "knobs", {})):
            site = anc
    return site


def test_split_kernels_attribute_to_pre_decision_op() -> None:
    """Both sides of the structural fork attribute to the offer-site op via
    ``Op.source`` — the decomposition link the composed Σ rows group by. The
    split side is stamped at the splice; the keep side is stamped by the
    engine's UNCONDITIONAL rebind stamp (005 builds the keep option via
    ``dataclasses.replace``, which copies the root's own knob-less ancestor
    into ``source`` — honoring the copy used to point the link past the offer
    site)."""
    terminals = _outer_terminals(_norm_linear("a"))
    split = next(t for t in terminals if len(_kernels(t)) == 2)
    kernels = _kernels(split)
    assert all(k.knobs.get("CUT") == "1" for k in kernels)
    assert all(_site(k) is not None for k in kernels)
    assert len({op_cache_key(_site(k)) for k in kernels}) == 1, "both fragment kernels must attribute to one pre-decision op"
    site = _site(kernels[0])
    assert "CUT" not in site.knobs
    assert any(k.startswith("S_") for k in site.knobs), "the site is the S_*-stamped offer op, not a bare ancestor"

    fused = next(t for t in terminals if len(_kernels(t)) == 1)
    keep = _kernels(fused)[0]
    assert keep.knobs.get("CUT") == "0"
    assert _site(keep) is not None and op_cache_key(_site(keep)) == op_cache_key(site), (
        "the keep side must attribute to the SAME offer-site op as the split side"
    )


def test_outer_descends_prior_preferred_branch_first() -> None:
    """With a prior ranking the split side cheaper, the outer PUCT explores it
    FIRST — including past the fork's resolve: the resolved branch's
    continuation keeps its ``CUT`` delta (``LazyCandidate.resolved_knobs``),
    so it isn't out-scored by the unresolved keep-fused sibling as a knob-less
    generic row (the regression this test pins)."""

    class _SplitCheapPrior(_RecordingPrior):
        def score(self, knobs) -> float:
            if knobs.get("CUT") == "1":
                return 1.0
            return 2.0 if "CUT" in knobs else 3.0

    ctx = Context.from_target((12, 0))
    search = TuningSearch(patience=10**6, prior_model=_SplitCheapPrior(), base_knobs=ctx.features())
    drained = drain_tune(outer_pipeline(), _norm_linear("a"), search=search, ctx=ctx, db=SearchDB(), on=lambda c: True)
    first = drained[0]
    assert len(_kernels(first.graph)) == 2, "the prior-preferred (split) kernel set must reach its terminal first"


def test_decomposition_rows_sum_kernel_set_costs() -> None:
    """One composed row per structural decision: features = the offer site's
    knobs + the decision delta (never the kids' restamped ``S_*``), label =
    the Σ of the side's per-kernel bests."""
    terminals = _outer_terminals(_norm_linear("a"))
    by_size = {len(_kernels(t)): t for t in terminals}
    ctx = Context.from_target((12, 0))

    split = by_size[2]
    per_op = [OpResult(name="k", op_key=op_cache_key(op), best_us=us) for op, us in zip(_kernels(split), (7.0, 13.0), strict=True)]
    rows = _decomposition_rows(split, per_op, ctx)
    assert len(rows) == 1
    feats, label = rows[0]
    assert label == pytest.approx(20.0), "the split side's price is the kernel-set Σ"
    assert feats["CUT"] == "1"
    site = _site(_kernels(split)[0])
    site_s_feats = {k: v for k, v in site.knobs.items() if k.startswith("S_")}
    assert site_s_feats and all(feats[k] == v for k, v in site_s_feats.items()), "the row rides the SITE's S_* identity"

    fused = by_size[1]
    fop = _kernels(fused)[0]
    rows = _decomposition_rows(fused, [OpResult(name="k", op_key=op_cache_key(fop), best_us=42.0)], ctx)
    assert len(rows) == 1
    feats, label = rows[0]
    assert label == pytest.approx(42.0)
    assert feats["CUT"] == "0"


def test_identical_offer_sites_take_the_same_side() -> None:
    """Two structurally identical offer sites take the same side within a
    trajectory — the engine replays the first decision read off the graph via
    the ``Op.source`` links + stamped decision knobs
    (``pipeline._replay_structural_decision``): the outer tree yields 2
    terminals (all-fused, all-split), not the 2^sites cross-product."""
    g = _norm_linear("b", _norm_linear("a"))
    terminals = [
        cand.graph
        for cand in drain_tune(outer_pipeline(), g, search=TuningSearch(patience=10**6), ctx=Context.from_target((12, 0)), db=SearchDB())
    ]
    sizes = sorted(len(_kernels(t)) for t in terminals)
    assert sizes == [2, 4], f"expected all-fused (2 kernels) and all-split (4) terminals only, got {sizes}"


def test_run_two_level_tune_single_terminal_assembles_bests() -> None:
    """With no fusion forks today the outer yields one terminal; the assembled
    graph greedy-replays the per-op bests."""
    result = run_two_level(
        _two_distinct_matmuls(),
        ctx=Context.from_target((8, 0)),
        db=SearchDB(),
        backend=_CountingBackend(),
        patience=_PATIENCE,
    )
    assert result.n_terminals == 1, "no multi-option fusion forks today → exactly one outer terminal"
    assert result.best_reward is not None and result.best_reward.ok
    assert len(result.best_reward.per_op) == 2

    # The winning fusion was greedy-assembled into a Graph[CudaOp] from the DB.
    assert result.assembled is not None
    assert any(isinstance(n.op, CudaOp) for n in result.assembled.nodes.values())
