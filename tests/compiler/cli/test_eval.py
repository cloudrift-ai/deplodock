"""Tests for ``deplodock eval knobs`` / ``eval variants`` — the tune-DB analysis CLIs.

Each test builds a synthetic tune-DB inline (just the two tables the
commands read: ``cuda_op`` and ``perf``), so the suite stays hermetic
and does not depend on a real autotune cache or GPU. The ``variants``
CLI tests pin ``--prior`` to a nonexistent file so the pick comes from
the cold ``AnalyticPrior`` regardless of any prior checkpoint on the host.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def _make_tune_db(path: Path, variants: list[tuple[str, str, dict, float]]) -> None:
    """Write a minimal tune DB to ``path``.

    ``variants`` is a list of ``(op_key, kernel_name, knobs, latency_us)``
    rows; one ``cuda_op`` + one ``perf`` row is written per entry. Other
    real-DB columns (kernel_source, arg_order, grid, block, smem_bytes)
    are filled with dummy values — ``knobs`` only reads ``cuda_op.pretty``
    and ``perf.knobs``/``perf.latency_us_median``.
    """
    con = sqlite3.connect(str(path))
    con.executescript(
        """
        CREATE TABLE cuda_op (
            key           TEXT PRIMARY KEY,
            kernel_source TEXT NOT NULL,
            arg_order     TEXT NOT NULL,
            grid          TEXT NOT NULL,
            block         TEXT NOT NULL,
            smem_bytes    INTEGER NOT NULL,
            pretty        TEXT NOT NULL
        );
        CREATE TABLE perf (
            context_key          TEXT NOT NULL,
            op_key               TEXT NOT NULL,
            backend              TEXT NOT NULL,
            status               TEXT NOT NULL,
            latency_us_median    REAL NOT NULL,
            latency_us_min       REAL NOT NULL,
            latency_us_max       REAL NOT NULL,
            latency_us_mean      REAL NOT NULL,
            latency_us_variance  REAL NOT NULL,
            n_samples            INTEGER NOT NULL,
            measured_at          TEXT NOT NULL,
            knobs                TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (context_key, op_key, backend)
        );
        """
    )
    for op_key, kernel_name, knobs, us in variants:
        pretty = f'extern "C" __global__\n__launch_bounds__(256) void {kernel_name}(const float* x) {{ }}\n'
        con.execute(
            "INSERT INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) "
            "VALUES (?, '', '[]', '[1,1,1]', '[1,1,1]', 0, ?)",
            (op_key, pretty),
        )
        con.execute(
            "INSERT INTO perf (context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs) "
            "VALUES ('ctx', ?, 'cuda', 'ok', ?, 0, 0, 0, 0, 1, '2026-05-24', ?)",
            (op_key, us, json.dumps(knobs)),
        )
    con.commit()
    con.close()


def test_knobs_missing_db(run_cli, tmp_path):
    """A non-existent DB path is not an error: the registry schema still prints and
    the regret analysis is skipped cleanly (exit 0, no traceback)."""
    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(tmp_path / "does_not_exist.db"))
    combined = (stdout + stderr).lower()
    assert rc == 0, f"stderr: {stderr}"
    assert "no tune db" in combined and "skipping" in combined, f"expected graceful skip:\nstdout={stdout!r}"
    assert "traceback" not in combined


def test_knobs_empty_db(run_cli, tmp_path):
    """Empty DB → command exits 0 and reports zero kernels."""
    db = tmp_path / "empty.db"
    _make_tune_db(db, variants=[])
    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "0 (of 0 total)" in stdout


def test_knobs_ranks_higher_impact_knob_first(run_cli, tmp_path):
    """Two knobs with very different regret: ``BIG`` swings 100x, ``SMALL``
    swings 1.1x. ``BIG`` must appear first in the ranked table."""
    db = tmp_path / "ranked.db"
    rows = []
    # 8 variants of one kernel: 4 BIG values × 2 SMALL values.
    # Latency = 1.0 × big_factor[BIG] × small_factor[SMALL].
    big_factor = {16: 1.0, 64: 5.0, 128: 50.0, 256: 100.0}
    small_factor = {1: 1.0, 2: 1.1}
    i = 0
    for big, bf in big_factor.items():
        for small, sf in small_factor.items():
            rows.append((f"k{i}", "k_test_kernel", {"BIG": big, "SMALL": small}, bf * sf))
            i += 1
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    # Both knobs appear, BIG first.
    assert "BIG" in stdout and "SMALL" in stdout
    big_pos = stdout.index("\nBIG ")
    small_pos = stdout.index("\nSMALL ")
    assert big_pos < small_pos, f"BIG should rank above SMALL:\n{stdout}"
    # Regret math: BIG = 100/1 = 100; SMALL = 1.1.
    assert "100.00x" in stdout
    assert "1.10x" in stdout


def test_knobs_min_variants_filter(run_cli, tmp_path):
    """Kernels below ``--min-variants`` are excluded from the table."""
    db = tmp_path / "filter.db"
    # k_small: 3 variants (filtered out at --min-variants 5)
    # k_big: 6 variants (kept)
    rows = []
    for i in range(3):
        rows.append((f"s{i}", "k_small", {"A": i}, 10.0 + i))
    for i in range(6):
        rows.append((f"b{i}", "k_big", {"A": i}, 1.0 + i))
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db), "--min-variants", "5")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 (of 2 total)" in stdout


def test_knobs_kernel_filter(run_cli, tmp_path):
    """``--kernel`` substring restricts the kernel set."""
    db = tmp_path / "kfilter.db"
    rows = []
    for i in range(8):
        rows.append((f"m{i}", "k_matmul", {"BM": 16 if i < 4 else 64}, 10.0 + i))
        rows.append((f"r{i}", "k_rmsnorm", {"BM": 16 if i < 4 else 64}, 1.0 + i))
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db), "--kernel", "matmul")
    assert rc == 0, f"stderr: {stderr}"
    assert "1 (of 1 total)" in stdout


def test_knobs_interaction_matrix_present(run_cli, tmp_path):
    """Output includes the K1\\K2 interaction matrix."""
    db = tmp_path / "interact.db"
    rows = []
    # 8 variants with two knobs that vary; the matrix needs ≥2 values per knob.
    i = 0
    for a in (1, 2):
        for b in (10, 20, 30, 40):
            rows.append((f"x{i}", "k_test", {"A": a, "B": b}, float(a * b)))
            i += 1
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "knob interaction" in stdout
    assert "K1\\K2" in stdout


def test_knobs_tolerates_list_valued_knob(run_cli, tmp_path):
    """Some knob dicts carry a list value (e.g. ``OVERHANG=['a0']`` for a masked
    overhang tile). The regret + interaction analysis groups variants by knob
    value, so a list value must not crash the dict/set keying with
    ``TypeError: unhashable type: 'list'`` — it's coerced to a tuple."""
    db = tmp_path / "overhang.db"
    rows = []
    # 8 variants: BN varies (a normal knob) alongside a list-valued OVERHANG so
    # both code paths (regret grouping + interaction matrix) see the list.
    for i in range(8):
        overhang = ["a0"] if i % 2 else []
        rows.append((f"o{i}", "k_matmul", {"BN": 16 if i < 4 else 64, "OVERHANG": overhang}, 10.0 + i))
    _make_tune_db(db, rows)

    rc, stdout, stderr = run_cli("eval", "knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "traceback" not in (stdout + stderr).lower()
    assert "knob interaction" in stdout  # full report rendered, no crash


def _add_perf_row(path: Path, op_key: str, kernel_name: str, knobs: dict, us: float, *, status: str) -> None:
    """Append one ``cuda_op`` + ``perf`` row with an explicit status (the shared
    ``_make_tune_db`` writes only ``ok`` rows)."""
    con = sqlite3.connect(str(path))
    pretty = f'extern "C" __global__\n__launch_bounds__(256) void {kernel_name}(const float* x) {{ }}\n'
    con.execute(
        "INSERT INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) "
        "VALUES (?, '', '[]', '[1,1,1]', '[1,1,1]', 0, ?)",
        (op_key, pretty),
    )
    con.execute(
        "INSERT INTO perf (context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
        "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs) "
        "VALUES ('ctx', ?, 'cuda', ?, ?, 0, 0, 0, 0, 1, '2026-06-10', ?)",
        (op_key, status, us, json.dumps(knobs)),
    )
    con.commit()
    con.close()


# --- eval variants ----------------------------------------------------------


def test_variants_missing_db(run_cli, tmp_path):
    """A non-existent DB path exits cleanly with a pointer, no traceback."""
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(tmp_path / "does_not_exist.db"), "--prior", str(tmp_path / "p.json"))
    combined = (stdout + stderr).lower()
    assert rc == 0, f"stderr: {stderr}"
    assert "no tune db" in combined
    assert "traceback" not in combined


def test_variants_golden_dataset_rejected(run_cli, tmp_path):
    """``--dataset golden`` is a degenerate source — fail fast with a message."""
    rc, stdout, stderr = run_cli("eval", "variants", "--dataset", "golden")
    assert rc == 2
    assert "no per-variant measurements" in (stdout + stderr)


def test_variants_leaderboard_sorted_with_ranked_pick(run_cli, tmp_path):
    """The leaderboard sorts by latency (fastest first), marks exactly one pick
    row, and prints its rank summary. No prior checkpoint → cold AnalyticPrior,
    no reservoir → no ``-O3 us`` column."""
    db = tmp_path / "v.db"
    _make_tune_db(
        db,
        [
            ("a", "k_matmul", {"BM": 8, "BN": 16}, 30.0),
            ("b", "k_matmul", {"BM": 16, "BN": 16}, 10.0),
            ("c", "k_matmul", {"BM": 64, "BN": 16}, 20.0),
        ],
    )
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--prior", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "k_matmul — 3 measured configs" in stdout
    assert stdout.index("10.0") < stdout.index("20.0") < stdout.index("30.0")  # fastest first
    assert stdout.count("◄") == 1
    assert "pick: rank" in stdout and "/3," in stdout
    assert "-O3 us" not in stdout  # empty reservoir → column omitted


def test_variants_top_truncation_keeps_pick_row(run_cli, tmp_path):
    """``--top N`` truncates to the N fastest but always shows the pick row, and
    reports how many rows were hidden."""
    db = tmp_path / "t.db"
    _make_tune_db(db, [(f"v{i}", "k_matmul", {"BM": 2**i, "BN": 16}, 10.0 + i) for i in range(6)])
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--top", "2", "--prior", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "--top 0 shows all" in stdout  # truncation note printed
    assert stdout.count("◄") == 1  # pick row survives the cut


def test_variants_counts_bench_fail_rows(run_cli, tmp_path):
    """``bench_fail`` rows are counted in the kernel header, not listed as variants."""
    db = tmp_path / "f.db"
    _make_tune_db(db, [("a", "k_matmul", {"BM": 8}, 10.0), ("b", "k_matmul", {"BM": 16}, 20.0)])
    _add_perf_row(db, "x", "k_matmul", {"BM": 64, "TMA": True}, 2_000_000.0, status="bench_fail")
    rc, stdout, stderr = run_cli("eval", "variants", "--db", str(db), "--prior", str(tmp_path / "missing.json"))
    assert rc == 0, f"stderr: {stderr}"
    assert "k_matmul — 2 measured configs, 1 bench_fail" in stdout


def test_failures_clusters_by_kernel_and_error(run_cli, tmp_path):
    """``eval failures`` clusters bench_fail rows by (kernel, error) and reports
    the knob values shared by every failing row in a cluster (the 'all rows have
    TMA=1' forensics signal). Built with the real ``SearchDB`` write path so the
    error column round-trips end to end."""
    from deplodock.compiler.pipeline.search.db import PerfStats, SearchDB

    def stats(median):
        return PerfStats(median=median, min=median, max=median, mean=median, variance=0.0, n_samples=1)

    db_path = tmp_path / "f.db"
    db = SearchDB(db_path)
    for i, bm in enumerate((8, 16)):
        db.record_cuda_op(
            f"f{i}", kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty="void k_mm(float*)"
        )
        db.record_perf(
            "ctx",
            f"f{i}",
            backend="cuda",
            status="bench_fail",
            stats=stats(1.0),
            knobs={"BM": bm, "TMA": True},
            error="cuTensorMapEncodeTiled failed",
        )
    db.record_cuda_op("ok1", kernel_source="", arg_order=[], grid=[1, 1, 1], block=[1, 1, 1], smem_bytes=0, pretty="void k_mm(float*)")
    db.record_perf("ctx", "ok1", backend="cuda", status="ok", stats=stats(10.0), knobs={"BM": 8, "TMA": False})
    db.close()

    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db_path))
    assert rc == 0, f"stderr: {stderr}"
    assert "2 bench_fail rows (beside 1 ok)" in stdout
    assert "k_mm — 2 row(s)" in stdout
    assert "cuTensorMapEncodeTiled failed" in stdout
    assert "TMA=True" in stdout and "BM=" not in stdout  # shared knob only — BM differs across the rows


def test_failures_old_db_without_error_column(run_cli, tmp_path):
    """bench_fail rows from a pre-error-column DB cluster under a placeholder
    instead of failing the read."""
    db = tmp_path / "old.db"
    _make_tune_db(db, [("a", "k_mm", {"BM": 8}, 10.0)])
    _add_perf_row(db, "x", "k_mm", {"BM": 64, "TMA": True}, 2_000_000.0, status="bench_fail")
    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "(no error recorded)" in stdout


def test_failures_none_recorded(run_cli, tmp_path):
    db = tmp_path / "clean.db"
    _make_tune_db(db, [("a", "k_mm", {"BM": 8}, 10.0)])
    rc, stdout, stderr = run_cli("eval", "failures", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "No bench_fail rows" in stdout


def test_o3_reservoir_index_joins_db_rows_by_sig_and_knobs():
    """``_o3_reservoir_index`` keeps only ``H_opt=3`` reservoir rows and keys them
    so a DB sample with the same ``S_*`` signature + tunable knobs joins (the -O3
    re-bench never writes a ``perf`` row, so the reservoir is the only -O3 source)."""
    from deplodock.commands.eval import _o3_reservoir_index, _variant_key
    from deplodock.compiler.pipeline.search.data import Sample
    from deplodock.compiler.pipeline.search.db import PerfSample

    stamped = {"BM": 8, "BN": 16, "S_ext_free_prod": 1024.0}

    class _P:
        _dataset = [
            ({**stamped, "H_opt": 3.0}, 9.0),
            ({**stamped, "H_opt": 1.0}, 12.0),  # -O1 sweep row — excluded
            ({"BM": 64, "BN": 16, "S_ext_free_prod": 1024.0, "H_opt": 3.0}, 7.0),
        ]

    o3 = _o3_reservoir_index(_P())
    assert len(o3) == 2
    db_sample = Sample.from_perf_sample(PerfSample(pretty="void k_m(float*)", knobs=stamped, latency_us=12.5))
    assert o3[_variant_key(db_sample)] == 9.0


def test_emit_variant_table_o3_column_and_deterministic_pick(caplog):
    """With a fake prior the pick is deterministic; the ``-O3 us`` column shows the
    reservoir latency for the matching config and ``—`` elsewhere."""
    import logging

    from deplodock.commands.eval import _emit_variant_table, _variant_key
    from deplodock.compiler.pipeline.search.data import Sample
    from deplodock.compiler.pipeline.search.db import PerfSample

    samples = [
        Sample.from_perf_sample(PerfSample(pretty="void k_m(float*)", knobs={"BM": 8, "BN": 16}, latency_us=30.0)),
        Sample.from_perf_sample(PerfSample(pretty="void k_m(float*)", knobs={"BM": 16, "BN": 16}, latency_us=10.0)),
    ]

    class _P:  # predicts the slow BM=8 config fastest → pick is rank 2
        def mean_score(self, knobs):
            return knobs["BM"]

    o3 = {_variant_key(samples[0]): 9.5}
    with caplog.at_level(logging.INFO, logger="deplodock.commands.eval"):
        _emit_variant_table("k_m", samples, _P(), n_fail=0, o3=o3, top=0)
    out = "\n".join(caplog.messages)
    assert "-O3 us" in out
    assert "9.5" in out and "—" in out
    assert "pick: rank 2/2, 3.00x of best" in out
    assert "<-- misses best" in out


def test_knob_columns_names_in_header_values_in_cells():
    """``knob_columns`` puts the knob name in the column header (canonical knob_sort_key
    order) and value-only cells (no ``NAME=`` prefix), blank where a row lacks a knob;
    ``render_table`` aligns the columns to the widest of header + cells."""
    from deplodock.commands.table import knob_columns, render_table

    cols, cells = knob_columns(
        [
            {"BN": ("16", False), "BK": ("32", False)},
            {"BN": ("32", False), "MMA": ("x", False)},
        ]
    )
    assert [c.name for c in cols] == ["BN", "BK", "MMA"]  # canonical order BN, BK, MMA
    lines = render_table(cols, cells)
    assert lines[0].split() == ["BN", "BK", "MMA"]  # header row carries the names
    assert lines[1].split() == ["16", "32"]  # values only, no "BN=" prefix; trailing MMA blank stripped
    assert "BN=" not in lines[1]
    assert lines[2].split() == ["32", "x"]  # BK column blank between BN and MMA


def test_render_table_ansi_aware_width():
    """A coloured cell is padded by its *visible* length, so embedded ANSI codes never
    throw off column alignment (right- and left-aligned columns both line up)."""
    import re  # noqa: PLC0415

    from deplodock.commands.table import Col, render_table

    lines = render_table([Col("a", "r"), Col("b")], [["1", "x"], [("22", "\033[31m"), "y"]])
    plain = [re.sub(r"\033\[[0-9]+m", "", line) for line in lines]
    assert plain[1] == " 1  x"  # "1" right-aligned in a width-2 column
    assert plain[2] == "22  y"  # coloured "22" fills the column; "y" still aligns under "x"
