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
from deplodock.compiler.pipeline.search.two_level import LOWERING_PASSES, inner_reward, outer_pipeline, run_two_level_tune

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
            us = 1.0 + (abs(hash(op_cache_key(n.op))) % 100)
            per.append(LaunchTime(idx=i, kernel_name=getattr(n.op, "kernel_name", "k"), time_ms=us / 1000.0, samples=(us / 1000.0,)))
        return BenchmarkResult(time_ms=sum(p.time_ms for p in per), num_launches=len(per), per_launch=per)


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
    list(pipeline.tune(sub, search=search, ctx=Context.from_target((8, 0)), backend=backend, db=SearchDB()))
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
    reward = inner_reward(fused, ctx=Context.from_target((8, 0)), db=db, backend=backend, patience=_PATIENCE)

    # Separability: the shared run must not blow up to the cross-product
    # (n1 * n2 — the old whole-graph SP-MCTS bug this test guards against).
    # Per-op sharing through the DB perf cache is allowed — an already-measured
    # variant replays without a bench. The exact share count is sensitive to
    # MCTS exploration order: the
    # ``_CountingBackend`` fakes latency from ``hash(op_cache_key)``, so any
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
    as ``test_inner_reward_separability``); the ``_CountingBackend``'s hash-derived
    latencies make it host-dependent, so only the two robust invariants are pinned."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    cold_backend = _CountingBackend()
    first = inner_reward(fused, ctx=ctx, db=db, backend=cold_backend, patience=_PATIENCE)
    rerun_backend = _CountingBackend()
    second = inner_reward(fused, ctx=ctx, db=db, backend=rerun_backend, patience=_PATIENCE)

    # The DB's per-op best is monotone non-increasing — more benches never worsen it.
    assert second.total_us <= first.total_us + 1e-6, "rerun must not regress the per-op best total"
    # Warm rerun replays most terminals from the perf cache → fewer benches than cold.
    # (Only "fewer", not a fixed ratio: the exact count is exploration-order- and
    # host-dependent — see the docstring — so a `cold // 2`-style bound is not robust.)
    assert rerun_backend.calls < cold_backend.calls, "warm rerun must bench fewer variants than the cold run"


def test_inner_reward_deeper_patience_benches_new_variants() -> None:
    """A higher patience re-runs the search (never skipped) and reaches new
    variants the shallow pass never measured — those miss the perf cache and
    bench, while the already-measured ones replay for free."""
    fused = _fuse(_two_distinct_matmuls())
    db = SearchDB()
    ctx = Context.from_target((8, 0))

    inner_reward(fused, ctx=ctx, db=db, backend=_CountingBackend(), patience=1)

    deep_backend = _CountingBackend()
    inner_reward(fused, ctx=ctx, db=db, backend=deep_backend, patience=_PATIENCE)
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
    reward = inner_reward(fused, ctx=Context.from_target((8, 0)), db=SearchDB(), backend=backend, patience=_PATIENCE)

    assert backend.calls == single, "shared kernel must bench once, not twice"
    assert len(reward.per_op) == 1, "identical kernels collapse to one per_op entry"
    assert reward.per_op[0].multiplicity == 2, "both node positions are counted"
    # Total weights the shared best by its multiplicity — both kernels still cost time.
    assert reward.total_us == pytest.approx(2 * reward.per_op[0].best_us)


def _norm_linear(prefix: str, g: Graph | None = None) -> Graph:
    """RMSNorm → Linear (f16): fusion yields the prologue-demoted matmul whose
    keep-vs-split offer (``tile/005_split_demoted``) is a structural fork."""
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

    def op_start(self, name: str) -> None:
        self.ops.append(name)

    def variant(self, *a, **kw) -> None:  # noqa: ANN002
        pass

    def op_done(self, name: str) -> None:
        pass


def test_outer_branches_on_structural_fork(monkeypatch) -> None:
    """The outer drives through the pre-partition tile head: 005's keep-vs-split
    offer branches the OUTER tree, so both kernel sets appear as outer terminals
    and the split producer/consumer are their own tuned op leaves (own progress
    denominator), not sub-explorations inside the fused kernel's slice."""
    from deplodock.compiler import target as target_mod

    for k, v in {"WM": "2", "WN": "2", "FM": "1", "FN": "8", "BK": "2", "BM": "8", "BN": "64", "BR": "1", "SPLITK": "1", "FK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)
    target_mod.set_target((12, 0))
    progress = _RecordingProgress()
    result = run_two_level_tune(
        _norm_linear("a"),
        ctx=Context.from_target((12, 0)),
        db=SearchDB(),
        backend=_CountingBackend(),
        patience=4,
        progress=progress,
    )
    assert result.n_terminals == 2, "keep-vs-split must branch the outer tree into two terminals"
    # One terminal carries the fused kernel (1 op leaf), the other the split
    # producer + consumer (2 op leaves) — the denominators the progress bar sees.
    assert sorted(progress.terminal_sizes) == [1, 2]
    assert any(name.endswith("_xn") for name in progress.ops), f"split producer must be its own op leaf, got {progress.ops}"


def test_structural_memo_collapses_identical_sites() -> None:
    """Two structurally identical offer sites take the same side within a
    trajectory (the per-``(rule, op_cache_key)`` decision memo): the outer tree
    yields 2 terminals (all-fused, all-split), not the 2^sites cross-product."""
    g = _norm_linear("b", _norm_linear("a"))
    terminals = [
        cand.graph
        for cand in outer_pipeline().tune(g, search=TuningSearch(patience=10**6), ctx=Context.from_target((12, 0)), db=SearchDB())
    ]
    sizes = sorted(sum(1 for n in t.nodes.values() if isinstance(n.op, LoopOp)) for t in terminals)
    assert sizes == [2, 4], f"expected all-fused (2 kernels) and all-split (4) terminals only, got {sizes}"


def test_run_two_level_tune_single_terminal_assembles_bests() -> None:
    """With no fusion forks today the outer yields one terminal; the assembled
    graph greedy-replays the per-op bests."""
    result = run_two_level_tune(
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
