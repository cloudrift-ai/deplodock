"""SQLite-backed inventory + measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op`` — one row per
  op encountered along a lowering chain. Keyed by ``op_cache_key``.
  Each row stores the JSON form (for programmatic inspection) and the
  pretty-printed form (for human inspection).
- ``lowering`` — best-known child for each parent op, one row per
  rewrite hop along the lowering chain (Loop→Tile, every intra-Tile
  autotune step, Tile→Kernel, Kernel→Cuda). Each row carries the knob
  delta the rule stamped at that hop, so :class:`GreedySearch` can
  replay the full chain by matching forks against the recorded delta
  at every fork point. ``record_lowering`` upserts uniformly across
  dialects: a strictly better measured median replaces the row; a
  None measurement (bench_fail terminal) never overwrites a
  known-good row. Deterministic rewrites (single option) trivially
  win their own slot via the same path.
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


@dataclass(frozen=True)
class LoweringRow:
    """One ``lowering`` row — best-known child for a parent op.

    ``knobs`` is the delta added at this rewrite step (e.g.
    ``005_blockify_launch`` adds ``{"BN": 64, "BM": 64}``). Greedy
    replay picks the fork whose newly-stamped knobs agree with this
    delta — no need to compare structural keys per fork."""

    parent_key: str
    parent_dialect: str
    child_key: str
    child_dialect: str
    knobs: dict
    best_median_us: float | None


class SearchDB:
    """Persistent inventory of compiled ops + their measured perf.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs pass an explicit path like
    ``~/.cache/deplodock/autotune.db``).
    """

    # Bumped whenever the fork-tree topology shifts in ways that change
    # ``parent_key`` / ``child_key`` for the same physical decision —
    # stale ``lowering`` rows from older versions won't match the new
    # keys and would silently slow the next tune sweep. On version
    # mismatch we drop the ``lowering`` table only; ``perf`` /
    # ``loop_op`` / ``tile_op`` etc. survive (source-hash keyed,
    # parent-tree-independent).
    #
    # Version log:
    #   1: M9.4 — planner-hoisted FM / FN / BN / BM forks. Parent-tree
    #       topology shifted vs. the legacy downstream forks.
    _SCHEMA_VERSION = 1

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
            knobs           TEXT NOT NULL DEFAULT '{}',
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
        # Drop the ``lowering`` table when an older schema is detected;
        # everything else (op inventory, perf rows) is keyed off content
        # hashes and remains valid across fork-tree changes.
        cur_version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if cur_version != self._SCHEMA_VERSION:
            self._conn.execute("DROP TABLE IF EXISTS lowering")
            self._conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
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
        knobs: dict | None = None,
        measured_median_us: float | None,
    ) -> None:
        """Upsert one ``parent_key`` → ``child_key`` lowering edge.

        ``knobs`` is the delta this rewrite step stamps onto the child
        (e.g. partition_loops adds ``{"BN": 64, "BM": 64, ...}``;
        launch_geometry adds nothing). Greedy replay picks forks by
        knob-subset match against this delta, so the row is enough to
        reconstruct the chain without re-querying ``perf``.

        Best-of upsert across every dialect — autotune fork rules live
        at Tile→Tile (blockify, split_register_axes) and used to be excluded
        here; recording every hop is how the chain stays replayable.
        Rows where the rewrite is genuinely deterministic (a single
        option) still trivially win their own slot, just via the same
        upsert path.
        """
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        existing = self._conn.execute(
            "SELECT child_key, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if existing is None:
            self._conn.execute(
                "INSERT INTO lowering (parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (parent_key, parent_dialect, child_key, child_dialect, knobs_json, measured_median_us),
            )
            return
        # Replace iff the new measurement is strictly better than the
        # stored best (or the stored best is NULL). A None measurement
        # never overwrites a known-good row.
        cur_best = existing[1]
        if measured_median_us is None:
            return
        if cur_best is None or measured_median_us < cur_best:
            self._conn.execute(
                "UPDATE lowering SET child_key = ?, child_dialect = ?, knobs = ?, best_median_us = ? WHERE parent_key = ?",
                (child_key, child_dialect, knobs_json, measured_median_us, parent_key),
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

    def lookup_lowering(self, parent_key: str) -> LoweringRow | None:
        """Return the best-known child for ``parent_key``, or ``None``
        when no row exists. Used by :class:`GreedySearch` to pick the
        DB-preferred fork at each fork point."""
        row = self._conn.execute(
            "SELECT parent_key, parent_dialect, child_key, child_dialect, knobs, best_median_us FROM lowering WHERE parent_key = ?",
            (parent_key,),
        ).fetchone()
        if row is None:
            return None
        return LoweringRow(
            parent_key=row[0],
            parent_dialect=row[1],
            child_key=row[2],
            child_dialect=row[3],
            knobs=json.loads(row[4]) if row[4] else {},
            best_median_us=row[5],
        )

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
