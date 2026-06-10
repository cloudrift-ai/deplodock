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


def _seed(db: SearchDB, key: str, pretty: str, knobs: dict, us: float) -> None:
    db.record_cuda_op(key, kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty=pretty)
    db.record_perf("ctx", key, backend="cuda", status="ok", stats=_stats(us), knobs=knobs)


# --- ShapeKey ---------------------------------------------------------------


def test_shapekey_from_matmul_arith() -> None:
    sk = ShapeKey.from_matmul(512, 256, 64, "fp32")
    assert (sk.free_prod, sk.reduce_max, sk.is_warp) == (512 * 256, 64, False)
    assert sk.s_features_arith() == {"S_ext_free_prod": 131072.0, "S_ext_reduce_prod": 64.0, "S_ext_reduce_max": 64.0}
    assert ShapeKey.from_matmul(64, 64, 64, "fp16").is_warp is True


# --- Sample round-trips (the acceptance gates) ------------------------------


def test_db_row_round_trip_is_lossless() -> None:
    """``Sample.from_perf_sample`` splits the stamped dict by prefix; ``all_knobs``
    re-merges to the exact original, and ``features`` matches ``knob_features`` of
    the raw dict — so the prior sees an identical vector and regret sees identical
    rows."""
    raw = {"BN": 32, "BM": 8, "SPLITK": 1, "S_ext_free_prod": 4194304.0, "S_n_mma": 1.0, "H_cc": 120.0, "H_opt": 3.0}
    s = Sample.from_perf_sample(PerfSample(pretty="void k_matmul(float*)", knobs=raw, latency_us=42.0))
    assert s.all_knobs() == raw
    assert s.features() == knob.knob_features(raw)
    assert s.name == "k_matmul"
    assert s.knobs == {"BN": 32, "BM": 8, "SPLITK": 1}  # tunable only


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
    _seed(db, "a1", "void k_matmul(float*)", {"BM": 8, "S_ext_free_prod": 1024.0, "S_n_mma": 1.0}, 10.0)
    _seed(db, "a2", "void k_matmul(float*)", {"BM": 16, "S_ext_free_prod": 1024.0, "S_n_mma": 1.0}, 8.0)
    _seed(db, "b1", "void k_rms(float*)", {"FK": 4, "S_ext_free_prod": 64.0, "S_reduce_add": 1.0}, 3.0)
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
