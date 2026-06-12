"""Tests for the harmonized data layer — :class:`ShapeKey` / :class:`Sample` /
:class:`Dataset` over golden / DB / prior sources.

The load-bearing acceptance gates are the two round-trips: a DB row and a golden
config must produce the *same* feature vector through ``Sample`` as the inline
code each consumer used before — otherwise the learned prior's ranking degrades
silently.
"""

from __future__ import annotations

from deplodock.compiler.pipeline import knob
from deplodock.compiler.pipeline.search.data import Dataset, Sample, ShapeKey
from deplodock.compiler.pipeline.search.db import PerfSample, PerfStats, SearchDB


def _stats(median: float) -> PerfStats:
    return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)


def _pretty(kernel_name: str) -> str:
    """A realistic ``cuda_op.pretty`` header — the renderer's
    ``extern "C" __global__\\n__launch_bounds__(N) void name(`` form."""
    return f'extern "C" __global__\n__launch_bounds__(256) void {kernel_name}(const float* x) {{ }}\n'


def _seed(db: SearchDB, key: str, pretty: str, knobs: dict, us: float) -> None:
    db.record_cuda_op(key, kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty=pretty)
    db.record_perf("ctx", key, backend="cuda", status="ok", stats=_stats(us), knobs=knobs)


# --- ShapeKey ---------------------------------------------------------------


def test_shapekey_from_matmul_arith() -> None:
    sk = ShapeKey.from_matmul(512, 256, 64, "fp32")
    assert (sk.free_prod, sk.reduce_max, sk.is_warp) == (512 * 256, 64, False)
    assert sk.s_features_arith() == {"S_ext_free_prod": 131072.0, "S_ext_reduce_prod": 64.0, "S_ext_reduce_max": 64.0}
    assert ShapeKey.from_matmul(64, 64, 64, "fp16").is_warp is True


def test_shapekey_from_s_features_joins_with_from_matmul() -> None:
    """The two constructors are the two sides of every golden ↔ measured-data join:
    a stamped op-side histogram and the golden's (M,N,K) must produce equal keys,
    and the fp32/fp16 twins must never merge. The dtype flag comes from the
    ``S_dtype_f32`` load multiset — ``S_n_mma`` is 0.0 on every stamped row (the
    stamp runs before the tile tier emits ``Mma``) and must not enter the key."""
    fp32 = {"S_ext_free_prod": 131072.0, "S_ext_reduce_max": 64.0, "S_dtype_f32": 2.0, "S_n_mma": 0.0}
    fp16 = {"S_ext_free_prod": 131072.0, "S_ext_reduce_max": 64.0, "S_dtype_f16": 2.0, "S_n_mma": 0.0}
    assert ShapeKey.from_s_features(fp32) == ShapeKey.from_matmul(512, 256, 64, "fp32")
    assert ShapeKey.from_s_features(fp16) == ShapeKey.from_matmul(512, 256, 64, "fp16")
    assert ShapeKey.from_s_features(fp32) != ShapeKey.from_s_features(fp16)  # twins split on dtype
    # Missing extents degrade to zeros, not a crash (e.g. an op stamped without dims).
    assert ShapeKey.from_s_features({}) == ShapeKey(free_prod=0, reduce_max=0, is_warp=True)


# --- Sample round-trips (the acceptance gates) ------------------------------


def test_db_row_round_trip_is_lossless() -> None:
    """``Sample.from_perf_sample`` splits the stamped dict by prefix; ``all_knobs``
    re-merges to the exact original, and ``features`` matches ``knob_features`` of
    the raw dict — so the prior sees an identical vector and regret sees identical
    rows."""
    raw = {"BN": 32, "BM": 8, "SPLITK": 1, "S_ext_free_prod": 4194304.0, "S_n_mma": 1.0, "H_cc": 120.0, "H_opt": 3.0}
    s = Sample.from_perf_sample(PerfSample(pretty=_pretty("k_matmul"), knobs=raw, latency_us=42.0))
    assert s.all_knobs() == raw
    assert s.features() == knob.knob_features(raw)
    assert s.name == "k_matmul"
    assert s.knobs == {"BN": 32, "BM": 8, "SPLITK": 1}  # tunable only


def test_kernel_name_skips_device_helper_preludes() -> None:
    """MMA/TMA kernel sources open with ``__device__`` helper preludes; the name
    must come from the ``__global__`` entry point, not the first ``void name(``
    in the source (which would be the helper)."""
    src = (
        "static __device__ __forceinline__ void dpl_ldmatrix_x4(unsigned* r, const void* smem) { }\n"
        "static __device__ __forceinline__ void mbarrier_init(unsigned long long* mbar, int count) { }\n"
        'extern "C" __global__\n__launch_bounds__(128) void k_linear_reduce_735349(const __half* x) { }\n'
    )
    s = Sample.from_perf_sample(PerfSample(pretty=src, knobs={"WM": 16}, latency_us=5.0))
    assert s.name == "k_linear_reduce_735349"
    # No __launch_bounds__ qualifier is also fine.
    s2 = Sample.from_perf_sample(PerfSample(pretty='extern "C" __global__ void k_mean(float* x) { }', knobs={}, latency_us=1.0))
    assert s2.name == "k_mean"


def test_prior_row_round_trip_is_lossless() -> None:
    raw = {"BM": 16, "S_kind": 1.0, "H_cc": 90.0}
    s = Sample.from_prior_row(raw, 3.5)
    assert s.all_knobs() == raw
    assert s.features() == knob.knob_features(raw)
    assert s.source == "prior"


def test_golden_compile_s_feats_matches_inline(monkeypatch) -> None:
    """The high-risk gate: a golden's full-histogram feature vector via
    ``Sample.from_golden(compile_s_feats=True)`` equals the old inline
    compile-and-scrape (eval's ``_emit_golden_features``), and the cheap arithmetic
    extents are a subset that agrees on the shared keys."""
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
    from deplodock.compiler.pipeline.knob import STRUCT_PREFIX
    from deplodock.compiler.pipeline.search.golden import GOLDEN_CONFIGS, MatmulGoldenConfig

    g = next(c for c in GOLDEN_CONFIGS if isinstance(c, MatmulGoldenConfig) and c.name == "square.512")

    graph, _, _ = graph_from_code(g.snippet())
    compiled = Pipeline.build(LOOP_PASSES).run(graph)
    s_feats: dict = {}
    for n in compiled.nodes.values():
        s_feats.update({k: v for k, v in (getattr(n.op, "knobs", {}) or {}).items() if k.startswith(STRUCT_PREFIX)})
    inline = knob.knob_features({**Context.from_target(g.compute_cap).features(), **s_feats, **g.knobs})

    assert Sample.from_golden(g, compile_s_feats=True).features() == inline

    arith = Sample.from_golden(g).s_features()
    full = Sample.from_golden(g, compile_s_feats=True).s_features()
    assert set(arith) <= set(full)
    assert all(arith[k] == full[k] for k in arith)


# --- Dataset adapters + grouping --------------------------------------------


def test_from_golden_filters() -> None:
    assert len(Dataset.from_golden()) == len(Dataset.from_golden(kernel="")) > 0
    assert all("square" in s.name for s in Dataset.from_golden(kernel="square"))
    assert all(s.dtype == "fp16" for s in Dataset.from_golden(dtype="fp16"))
    named = Dataset.from_golden(name="square.512").samples
    assert named and all(s.name == "square.512" for s in named)


def test_from_db_grouping(tmp_path) -> None:
    path = tmp_path / "t.db"
    db = SearchDB(path)
    # two shapes of one kernel name (different S_ext_free_prod) + a second kernel
    _seed(db, "a1", _pretty("k_matmul"), {"BM": 8, "S_ext_free_prod": 1024.0, "S_n_mma": 1.0}, 10.0)
    _seed(db, "a2", _pretty("k_matmul"), {"BM": 16, "S_ext_free_prod": 1024.0, "S_n_mma": 1.0}, 8.0)
    _seed(db, "b1", _pretty("k_rms"), {"FK": 4, "S_ext_free_prod": 64.0, "S_reduce_add": 1.0}, 3.0)
    ds = Dataset.from_db(path)

    by_name = ds.group_by_kernel_name()
    assert set(by_name) == {"k_matmul", "k_rms"}
    assert len(by_name["k_matmul"]) == 2
    assert ds.group_by_kernel_name(min_variants=2) == {"k_matmul": by_name["k_matmul"]}
    assert set(ds.group_by_kernel_name(kernel="rms")) == {"k_rms"}

    # group_by_op keys on the full S_* signature → the two matmul rows share a group
    by_op = ds.group_by_op()
    assert len(by_op) == 2
    assert max(len(v) for v in by_op.values()) == 2


def test_from_prior_group_by_op_matches_op_sig() -> None:
    class _P:
        _dataset = [({"S_kind": 1.0, "BM": bm}, 100.0 / bm) for bm in (2, 4, 8)]

    groups = Dataset.from_prior(_P()).group_by_op()
    assert len(groups) == 1
    (sig,) = groups
    assert sig == (("S_kind", 1.0),)  # only S_* enters the signature
    assert len(groups[sig]) == 3
