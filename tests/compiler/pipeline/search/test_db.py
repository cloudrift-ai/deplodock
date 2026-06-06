"""Tests for :class:`SearchDB` — the on-disk inventory + perf store.

Schema covers four per-dialect op tables, a best-known ``lowering``
table (idempotent for Tile→Kernel / Kernel→Cuda; best-of upsert for
Loop→Tile), and a backend-partitioned generic ``perf`` table.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.db import PerfStats, SearchDB


def _stats(median: float) -> PerfStats:
    return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)


# ---------------------------------------------------------------------------
# perf — upsert / lookup
# ---------------------------------------------------------------------------


def test_perf_miss_returns_none() -> None:
    db = SearchDB()
    assert db.lookup_perf("ctx", "k", backend="cuda") is None


def test_record_perf_then_lookup() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(12.0), knobs={})
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row is not None
    assert row.stats.median == 12.0
    assert row.status == "ok"
    assert row.backend == "cuda"


def test_record_perf_keeps_best() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(2.5))
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(1.0))
    assert db.lookup_perf("ctx", "k", backend="cuda").stats.median == 1.0
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(5.0))
    assert db.lookup_perf("ctx", "k", backend="cuda").stats.median == 1.0


def test_bench_fail_never_overrides_ok() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(10.0))
    db.record_perf("ctx", "k", backend="cuda", status="bench_fail", stats=_stats(1.0))
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row.status == "ok" and row.stats.median == 10.0


def test_ok_overrides_prior_bench_fail() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="bench_fail", stats=_stats(2_000_000.0))
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(10.0))
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row.status == "ok" and row.stats.median == 10.0


def test_different_backends_dont_clobber() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(10.0))
    db.record_perf("ctx", "k", backend="loop", status="ok", stats=_stats(123.0))
    assert db.lookup_perf("ctx", "k", backend="cuda").stats.median == 10.0
    assert db.lookup_perf("ctx", "k", backend="loop").stats.median == 123.0


def test_min_latency_filters_by_backend() -> None:
    db = SearchDB()
    db.record_perf("ctx", "a", backend="cuda", status="ok", stats=_stats(10.0))
    db.record_perf("ctx", "b", backend="loop", status="ok", stats=_stats(1.0))
    assert db.min_latency_for_context("ctx", backend="cuda") == 10.0
    assert db.min_latency_for_context("ctx", backend="loop") == 1.0
    assert db.min_latency_for_context("ctx") == 1.0  # aggregates across backends


# ---------------------------------------------------------------------------
# op inventory
# ---------------------------------------------------------------------------


def test_record_op_inventory_is_idempotent() -> None:
    db = SearchDB()
    db.record_tile_op("t", '{"dialect":"tile"}', "tile pretty")
    db.record_tile_op("t", '{"dialect":"tile","mutated":true}', "won't replace")
    row = db._conn.execute("SELECT body_json, pretty FROM tile_op WHERE key = ?", ("t",)).fetchone()
    assert row == ('{"dialect":"tile"}', "tile pretty")


def test_record_cuda_op_persists_launch_params() -> None:
    db = SearchDB()
    db.record_cuda_op(
        "c",
        kernel_source="__global__ void k() {}",
        arg_order=["x", "y"],
        grid=[1, 1, 1],
        block=[32, 1, 1],
        smem_bytes=4096,
        pretty="__global__ void k() {}",
    )
    row = db._conn.execute(
        "SELECT kernel_source, arg_order, grid, block, smem_bytes, pretty FROM cuda_op WHERE key = ?",
        ("c",),
    ).fetchone()
    assert row[0] == "__global__ void k() {}"
    assert row[1] == '["x", "y"]'
    assert row[2] == "[1, 1, 1]"
    assert row[3] == "[32, 1, 1]"
    assert row[4] == 4096


# ---------------------------------------------------------------------------
# lowering — uniform best-of across every dialect
# ---------------------------------------------------------------------------


def test_lowering_best_of_applies_across_dialects() -> None:
    """Every hop replays via the same best-median upsert — including
    Tile→Kernel and Kernel→Cuda, which used to be first-write-wins.
    The chain-replay design needs uniform semantics so intra-Tile
    autotune hops (blockify, split_register_axes) get the same treatment as
    Loop→Tile."""
    db = SearchDB()
    db.record_lowering("tile-A", "tile", "kernel-A", "kernel", measured_median_us=10.0)
    db.record_lowering("tile-A", "tile", "kernel-B", "kernel", measured_median_us=5.0)
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        ("tile-A",),
    ).fetchone()
    assert row[0] == "kernel-B" and row[1] == 5.0


def test_lowering_loop_to_tile_keeps_best() -> None:
    db = SearchDB()
    # First TileOp variant for this LoopOp: 100 us.
    db.record_lowering("loop-A", "loop", "tile-X", "tile", measured_median_us=100.0)
    # A faster variant: replaces.
    db.record_lowering("loop-A", "loop", "tile-Y", "tile", measured_median_us=40.0)
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        ("loop-A",),
    ).fetchone()
    assert row[0] == "tile-Y"
    assert row[1] == 40.0
    # A slower variant: ignored.
    db.record_lowering("loop-A", "loop", "tile-Z", "tile", measured_median_us=80.0)
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        ("loop-A",),
    ).fetchone()
    assert row[0] == "tile-Y"


def test_lowering_loop_to_tile_ignores_none_measurement() -> None:
    """``record_lowering`` with ``measured_median_us=None`` (e.g. a
    bench_fail terminal) must not overwrite a known-good row."""
    db = SearchDB()
    db.record_lowering("loop-A", "loop", "tile-X", "tile", measured_median_us=50.0)
    db.record_lowering("loop-A", "loop", "tile-Y", "tile", measured_median_us=None)
    row = db._conn.execute(
        "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
        ("loop-A",),
    ).fetchone()
    assert row[0] == "tile-X" and row[1] == 50.0


# ---------------------------------------------------------------------------
# best_per_op_time — walk lowering chain to the cuda terminal, read perf
# ---------------------------------------------------------------------------


def test_best_per_op_time_walks_chain_to_cuda() -> None:
    """A LoopOp key resolves to the median of the CudaOp it lowers to by
    following the best-known ``lowering`` child links."""
    db = SearchDB()
    db.record_lowering("loop", "loop", "tile", "tile", measured_median_us=9.0)
    db.record_lowering("tile", "tile", "kernel", "kernel", measured_median_us=9.0)
    db.record_lowering("kernel", "kernel", "cuda", "cuda", measured_median_us=9.0)
    db.record_perf("ctx", "cuda", backend="cuda", status="ok", stats=_stats(9.0))
    assert db.best_per_op_time("ctx", "loop", backend="cuda") == 9.0


def test_best_per_op_time_direct_cuda_key() -> None:
    db = SearchDB()
    db.record_perf("ctx", "cuda", backend="cuda", status="ok", stats=_stats(3.5))
    assert db.best_per_op_time("ctx", "cuda", backend="cuda") == 3.5


def test_best_per_op_time_missing_returns_none() -> None:
    db = SearchDB()
    # Chain present but the terminal CudaOp has no perf row.
    db.record_lowering("loop", "loop", "cuda", "cuda", measured_median_us=None)
    assert db.best_per_op_time("ctx", "loop", backend="cuda") is None
    # No chain at all.
    assert db.best_per_op_time("ctx", "nope", backend="cuda") is None


def test_best_per_op_time_ignores_bench_fail() -> None:
    db = SearchDB()
    db.record_lowering("loop", "loop", "cuda", "cuda", measured_median_us=None)
    db.record_perf("ctx", "cuda", backend="cuda", status="bench_fail", stats=_stats(1e6))
    assert db.best_per_op_time("ctx", "loop", backend="cuda") is None


def test_best_per_op_time_prefers_direct_loop_row() -> None:
    """The two-level inner search records the best whole-slice total (which
    counts e.g. a split-K combine) under the LoopOp key itself — that direct
    row wins over the single-CudaOp chain walk."""
    db = SearchDB()
    db.record_lowering("loop", "loop", "cuda", "cuda", measured_median_us=6.0)
    db.record_perf("ctx", "cuda", backend="cuda", status="ok", stats=_stats(6.0))
    # Direct whole-slice total (main 6.0 + combine 1.0) recorded on the loop key.
    db.record_perf("ctx", "loop", backend="cuda", status="ok", stats=_stats(7.0))
    assert db.best_per_op_time("ctx", "loop", backend="cuda") == 7.0
