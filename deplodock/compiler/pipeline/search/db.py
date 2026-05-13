"""SQLite-backed inventory + measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op`` — one row per
  op encountered along a lowering chain. Keyed by ``op_cache_key``.
  Each row stores the JSON form (for programmatic inspection) and the
  pretty-printed form (for human inspection).
- ``lowering`` — best-known child for each parent op. For Tile→Kernel
  and Kernel→Cuda the rewrite is deterministic so the first write wins.
  For Loop→Tile autotune explores multiple TileOp variants; the row
  tracks the one whose downstream CudaOp has the lowest measured
  ``best_median_us``, and ``record_lowering`` replaces it whenever a
  faster variant gets measured.
- ``perf`` — backend-agnostic measurement store. ``op_key`` is whichever
  terminal op the backend measured (today: a CudaOp; tomorrow whatever
  other backends lower to). ``backend`` partitions the table so the
  loop interpreter and the CUDA backend can coexist in the same DB.

Concurrency: opened in WAL mode so parallel benches can read while one
writes. The connection is kept open for the DB's lifetime; callers can
share one ``SearchDB`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class PerfStats:
    """Summary statistics over per-iter kernel latencies (microseconds)."""

    median: float
    min: float
    max: float
    mean: float
    variance: float
    n_samples: int


@dataclass(frozen=True)
class PerfRow:
    """One ``perf`` row."""

    context_key: str
    op_key: str
    backend: str
    status: str
    stats: PerfStats
    measured_at: str
    knobs: dict


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/deplodock/autotune.db``).
    """

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS loop_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS tile_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kernel_op (
            key       TEXT PRIMARY KEY,
            body_json TEXT NOT NULL,
            pretty    TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS cuda_op (
            key           TEXT PRIMARY KEY,
            kernel_source TEXT NOT NULL,
            arg_order     TEXT NOT NULL,
            grid          TEXT NOT NULL,
            block         TEXT NOT NULL,
            smem_bytes    INTEGER NOT NULL,
            pretty        TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS lowering (
            parent_key      TEXT PRIMARY KEY,
            parent_dialect  TEXT NOT NULL,
            child_key       TEXT NOT NULL,
            child_dialect   TEXT NOT NULL,
            best_median_us  REAL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS perf (
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
        )
        """,
    ]

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        for stmt in self._SCHEMA:
            self._conn.execute(stmt)

    # ------------------------------------------------------------------
    # Op-inventory writes (idempotent INSERT OR IGNORE)
    # ------------------------------------------------------------------

    def record_loop_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO loop_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_tile_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO tile_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_kernel_op(self, key: str, body_json: str, pretty: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO kernel_op (key, body_json, pretty) VALUES (?, ?, ?)",
            (key, body_json, pretty),
        )

    def record_cuda_op(
        self,
        key: str,
        *,
        kernel_source: str,
        arg_order: list[str],
        grid: list[int],
        block: list[int],
        smem_bytes: int,
        pretty: str,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO cuda_op (key, kernel_source, arg_order, grid, block, smem_bytes, pretty) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                kernel_source,
                json.dumps(list(arg_order)),
                json.dumps(list(grid)),
                json.dumps(list(block)),
                int(smem_bytes),
                pretty,
            ),
        )

    # ------------------------------------------------------------------
    # Lowering edges
    # ------------------------------------------------------------------

    def record_lowering(
        self,
        parent_key: str,
        parent_dialect: str,
        child_key: str,
        child_dialect: str,
        *,
        measured_median_us: float | None,
    ) -> None:
        """Upsert one ``parent_key`` → ``child_key`` lowering edge.

        Tile→Kernel and Kernel→Cuda are deterministic (the rewrite has
        no variants), so the first write wins and ``best_median_us``
        stays NULL for those rows. Loop→Tile is the autotuned step —
        passing the measured median lets this routine swap the row to
        the faster variant whenever one shows up.
        """
        existing = self._conn.execute(
            "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, best_median_us) VALUES (?, ?, ?, ?, ?)",
                (parent_key, parent_dialect, child_key, child_dialect, measured_median_us),
            )
            return
        # Deterministic dialects: don't touch a row that's already correct.
        # If the existing row points at a different child we leave it alone —
        # within a process the rewrite is single-valued; cross-process
        # divergence would point at a structural-key collision, which is
        # the digest's problem to solve, not ours.
        if parent_dialect in ("tile", "kernel"):
            return
        # Loop→Tile: replace the row iff the new measurement is strictly
        # better than the stored best (or the stored best is NULL).
        cur_best = existing[1]
        if measured_median_us is None:
            return
        if cur_best is None or measured_median_us < cur_best:
            self._conn.execute(
                "UPDATE lowering SET child_key = ?, child_dialect = ?, best_median_us = ? WHERE parent_key = ?",
                (child_key, child_dialect, measured_median_us, parent_key),
            )

    # ------------------------------------------------------------------
    # Perf — write
    # ------------------------------------------------------------------

    def record_perf(
        self,
        context_key: str,
        op_key: str,
        *,
        backend: str,
        status: str,
        stats: PerfStats,
        knobs: dict | None = None,
    ) -> None:
        """Upsert one ``perf`` row. Preserves the old keep-best-``ok``
        policy: a ``bench_fail`` never overwrites a prior ``ok`` row,
        and among ``ok`` rows the lowest median wins."""
        existing = self.lookup_perf(context_key, op_key, backend=backend)
        keep_existing_ok = existing is not None and existing.status == "ok" and status != "ok"
        keep_best = existing is not None and existing.status == "ok" and status == "ok" and stats.median >= existing.stats.median
        if keep_best or keep_existing_ok:
            return
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        self._conn.execute(
            "INSERT OR REPLACE INTO perf "
            "(context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            " latency_us_mean, latency_us_variance, n_samples, measured_at, knobs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                context_key,
                op_key,
                backend,
                status,
                stats.median,
                stats.min,
                stats.max,
                stats.mean,
                stats.variance,
                stats.n_samples,
                datetime.now(UTC).isoformat(),
                knobs_json,
            ),
        )

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def lookup_perf(self, context_key: str, op_key: str, *, backend: str) -> PerfRow | None:
        row = self._conn.execute(
            "SELECT context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs "
            "FROM perf WHERE context_key = ? AND op_key = ? AND backend = ?",
            (context_key, op_key, backend),
        ).fetchone()
        return _row_to_perf(row) if row else None

    def min_latency_for_context(self, context_key: str, *, backend: str | None = None) -> float | None:
        if backend is None:
            row = self._conn.execute(
                "SELECT MIN(latency_us_median) FROM perf WHERE context_key = ? AND status = 'ok'",
                (context_key,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT MIN(latency_us_median) FROM perf WHERE context_key = ? AND backend = ? AND status = 'ok'",
                (context_key, backend),
            ).fetchone()
        return row[0] if row and row[0] is not None else None

    def iter_perf(self, context_key: str, *, backend: str | None = None) -> Iterator[PerfRow]:
        if backend is None:
            cur = self._conn.execute(
                "SELECT context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
                "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs "
                "FROM perf WHERE context_key = ?",
                (context_key,),
            )
        else:
            cur = self._conn.execute(
                "SELECT context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
                "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs "
                "FROM perf WHERE context_key = ? AND backend = ?",
                (context_key, backend),
            )
        for row in cur:
            yield _row_to_perf(row)

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


def _row_to_perf(row: tuple) -> PerfRow:
    stats = PerfStats(
        median=row[4],
        min=row[5],
        max=row[6],
        mean=row[7],
        variance=row[8],
        n_samples=row[9],
    )
    knobs = json.loads(row[11]) if row[11] else {}
    return PerfRow(
        context_key=row[0],
        op_key=row[1],
        backend=row[2],
        status=row[3],
        stats=stats,
        measured_at=row[10],
        knobs=knobs,
    )
