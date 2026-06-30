"""Tests for :class:`SearchDB` — the on-disk inventory + perf store.

Schema covers four per-dialect op tables, a best-known ``lowering``
table (idempotent for Tile→Kernel / Kernel→Cuda; best-of upsert for
Loop→Tile), and a backend-partitioned generic ``perf`` table.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from emmy.compiler.pipeline.search.db import NodeRow, PerfStats, SearchDB


def _stats(median: float) -> PerfStats:
    return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)


def _seed_cuda(db: SearchDB, key: str, pretty: str) -> None:
    db.record_cuda_op(key, kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty=pretty)


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


def test_captured_overrides_uncaptured_even_if_slower() -> None:
    # Captured (pure-GPU) and uncaptured (wall) numbers aren't comparable;
    # the captured measurement is the better truth and replaces the wall one.
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(3.0), captured=False)
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(5.0), captured=True)
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row.captured is True and row.stats.median == 5.0


def test_uncaptured_never_overrides_captured() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(5.0), captured=True)
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(1.0), captured=False)
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row.captured is True and row.stats.median == 5.0


def test_captured_rows_keep_best_among_themselves() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(5.0), captured=True)
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(3.0), captured=True)
    assert db.lookup_perf("ctx", "k", backend="cuda").stats.median == 3.0
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(9.0), captured=True)
    assert db.lookup_perf("ctx", "k", backend="cuda").stats.median == 3.0


def test_bench_fail_never_overrides_captured_ok() -> None:
    db = SearchDB()
    db.record_perf("ctx", "k", backend="cuda", status="ok", stats=_stats(5.0), captured=True)
    db.record_perf("ctx", "k", backend="cuda", status="bench_fail", stats=_stats(1.0), captured=True)
    row = db.lookup_perf("ctx", "k", backend="cuda")
    assert row.status == "ok" and row.captured is True


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
# read-only open + perf ⋈ cuda_op samples
# ---------------------------------------------------------------------------


def test_iter_perf_samples_joins_cuda_op() -> None:
    db = SearchDB()
    _seed_cuda(db, "k1", "void k_matmul(float*)")
    _seed_cuda(db, "k2", "void k_rms(float*)")
    db.record_perf("ctx", "k1", backend="cuda", status="ok", stats=_stats(10.0), knobs={"BM": 8, "S_n_mma": 1.0})
    db.record_perf("ctx", "k2", backend="cuda", status="ok", stats=_stats(5.0), knobs={"BN": 1})
    db.record_perf("ctx", "k1", backend="cuda", status="bench_fail", stats=_stats(1.0))  # ok stays; fail not yielded

    samples = sorted(db.iter_perf_samples(), key=lambda s: s.latency_us)
    assert [(s.pretty, s.latency_us) for s in samples] == [("void k_rms(float*)", 5.0), ("void k_matmul(float*)", 10.0)]
    assert samples[1].knobs == {"BM": 8, "S_n_mma": 1.0}


def test_iter_perf_samples_backend_and_min_latency() -> None:
    db = SearchDB()
    _seed_cuda(db, "k1", "void k(float*)")
    _seed_cuda(db, "k2", "void k(float*)")
    db.record_perf("ctx", "k1", backend="cuda", status="ok", stats=_stats(10.0))
    db.record_perf("ctx", "k2", backend="loop", status="ok", stats=_stats(2.0))
    assert {s.latency_us for s in db.iter_perf_samples(backend="cuda")} == {10.0}
    assert {s.latency_us for s in db.iter_perf_samples(backend=None)} == {10.0, 2.0}
    assert {s.latency_us for s in db.iter_perf_samples(backend=None, min_latency_us=5.0)} == {10.0}


def test_record_perf_error_text_round_trips() -> None:
    """A ``bench_fail`` row's failure text is whitespace-collapsed, truncated, and
    readable back through the ``perf ⋈ cuda_op`` sample view; ``ok`` rows carry
    ``None``."""
    db = SearchDB()
    _seed_cuda(db, "k1", "void k_matmul(float*)")
    _seed_cuda(db, "k2", "void k_rms(float*)")
    db.record_perf("ctx", "k1", backend="cuda", status="bench_fail", stats=_stats(1.0), error="HungKernelError:\n  watchdog " + "x" * 400)
    db.record_perf("ctx", "k2", backend="cuda", status="ok", stats=_stats(5.0))

    (fail,) = db.iter_perf_samples(status="bench_fail")
    assert fail.error.startswith("HungKernelError: watchdog x")  # newline collapsed
    assert len(fail.error) <= 300
    (ok,) = db.iter_perf_samples()
    assert ok.error is None


def test_perf_error_column_added_to_pre_migration_db(tmp_path) -> None:
    """Opening an old DB (no ``error`` column) as a writer adds the column in
    place; a read-only open of an unmigrated DB degrades the sample ``error`` to
    ``None`` instead of failing the select."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE cuda_op (key TEXT PRIMARY KEY, kernel_source TEXT NOT NULL, arg_order TEXT NOT NULL,
            grid TEXT NOT NULL, block TEXT NOT NULL, smem_bytes INTEGER NOT NULL, pretty TEXT NOT NULL);
        CREATE TABLE perf (context_key TEXT NOT NULL, op_key TEXT NOT NULL, backend TEXT NOT NULL,
            status TEXT NOT NULL, latency_us_median REAL NOT NULL, latency_us_min REAL NOT NULL,
            latency_us_max REAL NOT NULL, latency_us_mean REAL NOT NULL, latency_us_variance REAL NOT NULL,
            n_samples INTEGER NOT NULL, measured_at TEXT NOT NULL, knobs TEXT NOT NULL DEFAULT '{}',
            captured INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (context_key, op_key, backend));
        INSERT INTO cuda_op VALUES ('k1', '', '[]', '[1,1,1]', '[1,1,1]', 0, 'void k(float*)');
        INSERT INTO perf VALUES ('ctx', 'k1', 'cuda', 'bench_fail', 1.0, 1, 1, 1, 0, 1, 'now', '{}', 0);
        """
    )
    con.commit()
    con.close()

    ro = SearchDB.open_readonly(path)  # unmigrated: error degrades to None
    (s,) = ro.iter_perf_samples(status="bench_fail")
    assert s.error is None
    ro.close()

    db = SearchDB(path)  # writer migrates in place
    _seed_cuda(db, "k2", "void k(float*)")
    db.record_perf("ctx", "k2", backend="cuda", status="bench_fail", stats=_stats(1.0), error="boom")
    assert db._has_perf_error_column()
    assert {s.error for s in db.iter_perf_samples(status="bench_fail")} == {None, "boom"}


def test_open_readonly_reads_without_mutating(tmp_path) -> None:
    path = tmp_path / "tune.db"
    db = SearchDB(path)
    _seed_cuda(db, "k1", "void k_matmul(float*)")
    db.record_perf("ctx", "k1", backend="cuda", status="ok", stats=_stats(7.0), knobs={"BM": 8})
    db.close()

    ro = SearchDB.open_readonly(path)
    assert [s.latency_us for s in ro.iter_perf_samples()] == [7.0]
    with pytest.raises(sqlite3.OperationalError):  # ?mode=ro rejects writes
        ro.record_perf("ctx", "k2", backend="cuda", status="ok", stats=_stats(1.0))
    ro.close()


def test_open_readonly_missing_file_raises(tmp_path) -> None:
    with pytest.raises(sqlite3.OperationalError):
        SearchDB.open_readonly(tmp_path / "nope.db")


# ---------------------------------------------------------------------------
# node — search-tree node store
# ---------------------------------------------------------------------------


def _node_row(node_key: str, value_us: float, *, parent_key=None, op_sig="sig", features=None, depth=1, gpu="") -> NodeRow:
    return NodeRow(
        node_key=node_key,
        parent_key=parent_key,
        context_key="ctx",
        op_sig=op_sig,
        features=features or {},
        value_us=value_us,
        depth=depth,
        gpu=gpu,
    )


def test_record_nodes_then_read() -> None:
    db = SearchDB()
    db.record_nodes([_node_row("n1", 5.0, features={"BM": 8, "S_n_mma": 1.0}, depth=2)])
    row = db._conn.execute(
        "SELECT parent_key, context_key, op_sig, features, value_us, depth, n_updates FROM node WHERE node_key = ?",
        ("n1",),
    ).fetchone()
    assert row[0] is None
    assert (row[1], row[2]) == ("ctx", "sig")
    assert json.loads(row[3]) == {"BM": 8, "S_n_mma": 1.0}  # full feature dict round-trips
    assert (row[4], row[5], row[6]) == (5.0, 2, 1)


def test_record_nodes_keeps_min() -> None:
    db = SearchDB()
    db.record_nodes([_node_row("n1", 5.0)])
    db.record_nodes([_node_row("n1", 2.0)])  # improves → kept
    db.record_nodes([_node_row("n1", 9.0)])  # worse → value untouched, n_updates still bumps
    val, n = db._conn.execute("SELECT value_us, n_updates FROM node WHERE node_key = ?", ("n1",)).fetchone()
    assert val == 2.0
    assert n == 3


def test_record_nodes_parent_link() -> None:
    db = SearchDB()
    db.record_nodes([_node_row("parent", 2.0, depth=1), _node_row("child", 4.0, parent_key="parent", depth=2)])
    (parent_value,) = db._conn.execute(
        "SELECT p.value_us FROM node c JOIN node p ON c.parent_key = p.node_key WHERE c.node_key = ?",
        ("child",),
    ).fetchone()
    assert parent_value == 2.0


def test_node_table_autocreated_on_pre_node_db(tmp_path) -> None:
    """A DB written before the ``node`` table existed gains it on the next writer
    open — ``CREATE TABLE IF NOT EXISTS`` auto-creates it, no ALTER needed."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE perf (context_key TEXT NOT NULL, op_key TEXT NOT NULL, backend TEXT NOT NULL,
            status TEXT NOT NULL, latency_us_median REAL NOT NULL, latency_us_min REAL NOT NULL,
            latency_us_max REAL NOT NULL, latency_us_mean REAL NOT NULL, latency_us_variance REAL NOT NULL,
            n_samples INTEGER NOT NULL, measured_at TEXT NOT NULL, knobs TEXT NOT NULL DEFAULT '{}',
            captured INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (context_key, op_key, backend));
        """
    )
    con.commit()
    con.close()

    db = SearchDB(path)  # writer open creates the absent ``node`` table
    db.record_nodes([_node_row("n1", 3.0)])
    (val,) = db._conn.execute("SELECT value_us FROM node WHERE node_key = 'n1'").fetchone()
    assert val == 3.0


def test_node_survives_version_bump_that_drops_lowering(tmp_path) -> None:
    """``node`` is content-keyed like ``perf`` (not topology-keyed like
    ``lowering``), so a schema-version mismatch — which wipes ``lowering`` — leaves
    ``node`` rows intact."""
    path = tmp_path / "t.db"
    db = SearchDB(path)
    db.record_lowering("p", "loop", "c", "tile", knobs={"BN": 64}, measured_median_us=1.0)
    db.record_nodes([_node_row("n1", 3.0)])
    db._conn.execute("PRAGMA user_version = 0")  # simulate an older on-disk schema
    db.close()

    reopened = SearchDB(path)  # version mismatch → drops+recreates ``lowering``; ``node`` survives
    assert reopened.lookup_lowering("p") is None
    (val,) = reopened._conn.execute("SELECT value_us FROM node WHERE node_key = 'n1'").fetchone()
    assert val == 3.0


def test_iter_nodes_roundtrips_features_and_parent(tmp_path) -> None:
    """``iter_nodes`` yields a NodeRow per stored node with the JSON feature dict
    parsed back and the parent pointer intact."""
    db = SearchDB(tmp_path / "t.db")
    db.record_nodes(
        [
            _node_row("p", 2.0, op_sig="mm", features={"S_n_mma": 1.0}, depth=1),
            _node_row("c", 4.0, parent_key="p", op_sig="mm", features={"S_n_mma": 1.0, "BM": 8}, depth=2),
        ]
    )
    by_key = {n.node_key: n for n in db.iter_nodes()}
    assert set(by_key) == {"p", "c"}
    assert by_key["c"].parent_key == "p" and by_key["p"].parent_key is None
    assert by_key["c"].features == {"S_n_mma": 1.0, "BM": 8}  # JSON round-trips
    assert (by_key["c"].value_us, by_key["c"].depth, by_key["c"].op_sig) == (4.0, 2, "mm")
    # op_sig scoping filters
    assert [n.node_key for n in db.iter_nodes(op_sig="other")] == []


def test_iter_nodes_missing_table_degrades(tmp_path) -> None:
    """A read-only open of a DB that predates the ``node`` table yields nothing
    instead of raising ``no such table`` (mirrors the perf error-column degrade)."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript("CREATE TABLE perf (context_key TEXT);")  # a node-less DB
    con.commit()
    con.close()
    ro = SearchDB.open_readonly(path)
    assert not ro._has_node_table()
    assert list(ro.iter_nodes()) == []
    ro.close()


def test_record_nodes_gpu_column_roundtrips(tmp_path) -> None:
    db = SearchDB(tmp_path / "t.db")
    db.record_nodes([_node_row("n1", 5.0, features={"BM": 8}, gpu="NVIDIA H200 141GB")])
    (n,) = list(db.iter_nodes())
    assert n.gpu == "NVIDIA H200 141GB"


_PRE_GPU_NODE_SCHEMA = (
    "CREATE TABLE node (node_key TEXT PRIMARY KEY, parent_key TEXT, context_key TEXT NOT NULL, "
    "op_sig TEXT NOT NULL, features TEXT NOT NULL DEFAULT '{}', value_us REAL NOT NULL, "
    "depth INTEGER NOT NULL, n_updates INTEGER NOT NULL DEFAULT 1, updated_at TEXT NOT NULL); "
    "INSERT INTO node VALUES ('old', NULL, 'ctx', 'mm', '{}', 3.0, 1, 1, 'now');"
)


def test_node_gpu_column_added_to_pre_gpu_db(tmp_path) -> None:
    """A ``node`` table from the first node-store version (no ``gpu`` column) gains it
    on the next writer open — the ALTER runs *before* the schema loop so the
    ``node_gpu`` index builds; old rows default to ''."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.executescript(_PRE_GPU_NODE_SCHEMA)
    con.commit()
    con.close()
    db = SearchDB(path)  # writer migrates: ALTER adds gpu, then node_gpu index builds
    assert db._has_node_gpu_column()
    assert list(db.iter_nodes())[0].gpu == ""  # pre-existing row defaults
    db.record_nodes([_node_row("new", 2.0, gpu="NVIDIA H100 80GB")])
    assert {n.node_key: n.gpu for n in db.iter_nodes()}["new"] == "NVIDIA H100 80GB"


def test_iter_nodes_pre_gpu_column_readonly_degrades(tmp_path) -> None:
    """A read-only open of a pre-``gpu``-column DB degrades ``gpu`` to '' rather than
    raising (the additive ALTER is writer-side only)."""
    path = tmp_path / "ro.db"
    con = sqlite3.connect(path)
    con.executescript(_PRE_GPU_NODE_SCHEMA)
    con.commit()
    con.close()
    ro = SearchDB.open_readonly(path)
    assert not ro._has_node_gpu_column()
    assert list(ro.iter_nodes())[0].gpu == ""
    ro.close()


def test_merge_nodes_keeps_min_and_coexists_across_cards(tmp_path) -> None:
    """``merge_nodes`` upserts another DB's ``node`` rows keyed on ``node_key`` with
    keep-min semantics, in both directions: a shared key takes the faster source row but
    a *slower* source row never clobbers a faster dest row. Rows under distinct
    ``node_key``s — which different cards always produce, since ``_node_key`` folds the
    ``gpu`` (proven in ``test_online_prior.test_node_key_folds_gpu``) — merge in
    alongside untouched. The cross-hardware accumulation behind ``scripts/merge_node_db.py``."""
    dst = SearchDB(tmp_path / "dst.db")
    dst.record_nodes(
        [
            _node_row("shared", 5.0, gpu="NVIDIA H100 80GB"),  # same key as src, slower here
            _node_row("faster_here", 1.0, gpu="NVIDIA H100 80GB"),  # same key as src, FASTER here
            _node_row("dst_only", 7.0, gpu="NVIDIA H100 80GB"),  # untouched by the merge
        ]
    )
    src = SearchDB(tmp_path / "src.db")
    src.record_nodes(
        [
            _node_row("shared", 2.0, gpu="NVIDIA H100 80GB"),  # improves the shared node
            _node_row("faster_here", 9.0, gpu="NVIDIA H100 80GB"),  # slower — must NOT overwrite dest
            _node_row("h200_only", 3.0, gpu="NVIDIA H200 141GB"),  # a different card's row
        ]
    )
    src.close()  # flush the on-disk file before the read-only open inside merge

    merged = dst.merge_nodes(tmp_path / "src.db")
    assert merged == 3  # source rows processed
    by_key = {n.node_key: n for n in dst.iter_nodes()}
    assert set(by_key) == {"shared", "faster_here", "dst_only", "h200_only"}  # both cards coexist
    assert by_key["shared"].value_us == 2.0  # keep-min took the faster source row
    assert by_key["faster_here"].value_us == 1.0  # slower source row did NOT overwrite the faster dest
    assert by_key["dst_only"].value_us == 7.0  # other rows untouched
    assert by_key["h200_only"].gpu == "NVIDIA H200 141GB"  # cross-card row carried its gpu


def test_merge_nodes_into_pre_node_dest_autocreates(tmp_path) -> None:
    """Merging into a freshly-opened DB (no node rows yet) just inserts the source
    rows — the dest's ``node`` table is created by ``SearchDB.__init__``."""
    src = SearchDB(tmp_path / "src.db")
    src.record_nodes([_node_row("n1", 4.0, gpu="NVIDIA H200 141GB")])
    src.close()
    dst = SearchDB(tmp_path / "dst.db")
    assert dst.merge_nodes(tmp_path / "src.db") == 1
    (n,) = list(dst.iter_nodes())
    assert (n.node_key, n.value_us, n.gpu) == ("n1", 4.0, "NVIDIA H200 141GB")


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
