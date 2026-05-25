"""Tests for ``deplodock knobs`` — knob-impact analysis CLI.

Each test builds a synthetic tune-DB inline (just the two tables the
command reads: ``cuda_op`` and ``perf``), so the suite stays hermetic
and does not depend on a real autotune cache or GPU.
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
    """Non-existent DB path produces a clean ``not found`` error (logged
    via the CLI's standard error path, not an unhandled exception)."""
    rc, stdout, stderr = run_cli("knobs", "--db", str(tmp_path / "does_not_exist.db"))
    combined = (stdout + stderr).lower()
    assert "not found" in combined, f"expected 'not found' in output:\nstdout={stdout!r}\nstderr={stderr!r}"
    assert "traceback" not in combined


def test_knobs_empty_db(run_cli, tmp_path):
    """Empty DB → command exits 0 and reports zero kernels."""
    db = tmp_path / "empty.db"
    _make_tune_db(db, variants=[])
    rc, stdout, stderr = run_cli("knobs", "--db", str(db))
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

    rc, stdout, stderr = run_cli("knobs", "--db", str(db))
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

    rc, stdout, stderr = run_cli("knobs", "--db", str(db), "--min-variants", "5")
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

    rc, stdout, stderr = run_cli("knobs", "--db", str(db), "--kernel", "matmul")
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

    rc, stdout, stderr = run_cli("knobs", "--db", str(db))
    assert rc == 0, f"stderr: {stderr}"
    assert "knob interaction" in stdout
    assert "K1\\K2" in stdout
