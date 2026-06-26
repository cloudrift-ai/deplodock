"""Golden-config evaluation harness (``search/analytic``) + the cold-start
:class:`AnalyticPrior` it ranks with.

CPU-only (no CUDA): ``search/analytic`` reconstructs a matmul shape's enumeration
and ranks it with a ``Prior`` (the :class:`AnalyticPrior` by default — the fixed
linear model over ``knob.knob_features`` that replaced the old
``score_matmul_thread`` / ``_priority_matmul_*`` heuristic). These pin the
load-bearing properties — degenerate tiles score *above* (slower than) golden
ones, picks land in the geometry band, and the warp tier dispatches by dtype —
without re-running the offline weight fit (that lives in
``scripts/golden_knob_heuristics.py``).
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.search.analytic import pick_matmul
from deplodock.compiler.pipeline.search.prior import AnalyticPrior


def _ctx() -> Context:
    # sm_120 (RTX 5090) — the regime the golden set + prior weights target.
    return Context.from_target((12, 0))


def _score(knobs: dict, M: int, N: int, K: int) -> float:
    """AnalyticPrior latency proxy (lower = better) with the shape / regime
    features the prior featurizes merged in — mirrors what the planner stamps."""
    base = {**_ctx().features(), "S_ext_free_prod": float(M * N), "S_ext_reduce_prod": float(K), "S_ext_reduce_max": float(K)}
    return AnalyticPrior().score({**base, **knobs})


def test_golden_row_outscores_degenerate_tile():
    # The recorded golden for 2048² is BN=32, BM=8; a degenerate BN=1, BM=256 tile
    # is far slower. The analytic prior must rank the golden strictly better —
    # i.e. its latency proxy is LOWER (lower = better, matching CatBoostPrior).
    shape = (2048, 2048, 2048)
    golden = {"BN": 32, "BM": 8, "FM": 26, "FN": 4, "FK": 1, "BK": 32, "SPLITK": 1, "BR": 1}
    degenerate = {"BN": 1, "BM": 256, "FM": 1, "FN": 128, "FK": 1, "BK": 32, "SPLITK": 1, "BR": 1}
    assert _score(golden, *shape) < _score(degenerate, *shape)


def test_pick_matmul_lands_in_geometry_band():
    # Across a spread of shapes the argmin pick should respect the prior's
    # dominant bands: coalesced inner axis 16..64, short outer axis 8..16,
    # large K-chunk, light split-K — never a degenerate BN=1 / BM=256 tile.
    ctx = _ctx()
    for M, N, K in [(64, 64, 64), (128, 256, 128), (512, 1024, 1024)]:
        r = pick_matmul(M, N, K, "fp32", ctx)
        # Native SPLIT@<axis>: the matmul free axes are a1 (inner N) and a0 (outer M).
        n_par, _ = fam.dec_split(r[fam.split_key("a1")])
        m_par, _ = fam.dec_split(r[fam.split_key("a0")])
        assert 16 <= n_par <= 64, (M, N, K, r)
        assert 8 <= m_par <= 16, (M, N, K, r)
        decomp = fam.dec_reduce(r[next(k for k in r if k.startswith("REDUCE@"))])
        assert decomp.serial >= 32, (M, N, K, r)  # native REDUCE serial (legacy BK)
        assert decomp.cta <= 2, (M, N, K, r)  # native REDUCE cta (legacy SPLITK)


def test_pick_matmul_warp_dispatch_by_dtype():
    ctx = _ctx()
    r16 = pick_matmul(256, 256, 256, "fp16", ctx)
    assert r16.get(fam.atom_key(fam.MATMUL_CELL)) == "mma_m16n8k16_f16"
    wn = fam.dec_split(r16[fam.split_key("a1")])[0]
    wm = fam.dec_split(r16[fam.split_key("a0")])[0]
    assert wm * wn != 1  # single-warp tiles are pruned
    r_bf = pick_matmul(256, 256, 256, "bf16", ctx)
    assert r_bf.get(fam.atom_key(fam.MATMUL_CELL)) == "mma_m16n8k16_bf16"


def test_dynamic_weight_set_selected_on_symbolic_flag():
    """A symbolic-axis row (``S_ext_n_symbolic_axis > 0``) ranks under the
    dynamic weight set; a static row under the static one. With deliberately
    opposed weight sets the same knobs must score differently across the flag."""
    from deplodock.compiler.pipeline.search.prior.analytic import AnalyticPrior

    p = AnalyticPrior(weights={"D_l2_bm": 1.0}, weights_dynamic={"D_l2_bm": -1.0})
    static = {"BN": 16, "BM": 8, "FM": 4, "FN": 2, "BK": 64, "SPLITK": 1, "BR": 1, "S_ext_free_prod": 4096.0}
    dynamic = {**static, "S_ext_n_symbolic_axis": 1.0}
    assert p.score(static) != p.score(dynamic)
    # Polarity: bigger BM is rewarded statically (+w → lower proxy), penalized dynamically.
    static_big = {**static, "BM": 16}
    dynamic_big = {**dynamic, "BM": 16}
    assert p.score(static_big) < p.score(static)
    assert p.score(dynamic_big) > p.score(dynamic)


def test_atomic_free_split_preference_above_threshold():
    """The atomic-free split-K gate (plans/atomic-free-monoid-combine.md): a wide
    split (SPLITK >= threshold) should prefer NOATOMIC=True (workspace + reduce);
    a narrow split should keep the atomicAdd fast-path. The two variants share
    identical matmul geometry — the NOATOMIC term is the sole differentiator."""
    p = AnalyticPrior(atomic_free_split_threshold=4.0, atomic_free_weight=5.0)
    base = {"BN": 32, "BM": 8, "FM": 26, "FN": 4, "FK": 1, "BK": 32, "BR": 1, "S_ext_free_prod": 4.0e6}
    # Wide split: atomic-free wins (lower latency proxy).
    wide_atomic = {**base, "SPLITK": 8, "NOATOMIC": False}
    wide_free = {**base, "SPLITK": 8, "NOATOMIC": True}
    assert p.score(wide_free) < p.score(wide_atomic)
    # Narrow split: the atomic fast-path wins.
    narrow_atomic = {**base, "SPLITK": 2, "NOATOMIC": False}
    narrow_free = {**base, "SPLITK": 2, "NOATOMIC": True}
    assert p.score(narrow_atomic) < p.score(narrow_free)
    # The term is gated off for NOATOMIC=False: its score matches a weight-0 prior;
    # NOATOMIC=True diverges (the term fires).
    p0 = AnalyticPrior(atomic_free_weight=0.0)
    assert p.score(wide_atomic) == p0.score(wide_atomic)
    assert p.score(wide_free) != p0.score(wide_free)
