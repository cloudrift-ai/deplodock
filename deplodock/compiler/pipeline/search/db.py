"""SQLite-backed inventory + measurement store for the search package.

Pure persistence layer — no MCTS state, no propagation walks. Tables:

- ``loop_op`` / ``tile_op`` / ``kernel_op`` / ``cuda_op`` — one row per
  op encountered along a lowering chain. Keyed by ``op_cache_key``.
  Each row stores the JSON form (for programmatic inspection) and the
  pretty-printed form (for human inspection).
- ``lowering`` — best-known child for each parent op, one row per
  rewrite hop along the lowering chain (Loop→Tile, every intra-Tile
  autotune step, Tile→Kernel, Kernel→Cuda). Each row carries the knob
  delta the rule stamped at that hop plus a best-median upsert — the
  chain :meth:`SearchDB.best_per_op_time` walks to resolve a pre-final
  op's measured cost (greedy fork picks come from the ``Prior``, never
  DB replay). ``record_lowering`` upserts uniformly across
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


def _jsonable_geometry(geometry) -> list:
    """Render a launch grid/block to a JSON-safe nested list for the inventory
    tables. Int / str pass through; nested specs recurse; composite ``Expr``
    factors (ceil-div block extents for hint-driven masked tiles) render to
    their pretty string — inventory rows are for human inspection, not
    re-execution."""

    def conv(x):
        if isinstance(x, (int, str)):
            return x
        if isinstance(x, (tuple, list)):
            return [conv(e) for e in x]
        return x.pretty()  # Expr

    return [conv(spec) for spec in geometry]


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
    """One ``perf`` row.

    ``captured``: the measurement ran under CUDA graph capture (pure GPU time);
    False = wall semantics including per-launch dispatch (all pre-capture rows).
    Both kinds stay usable (replay, prior training); on write, a captured
    measurement supersedes an uncaptured one for the same key — see
    :meth:`SearchDB.record_perf`."""

    context_key: str
    op_key: str
    backend: str
    status: str
    stats: PerfStats
    measured_at: str
    knobs: dict
    captured: bool = False


@dataclass(frozen=True)
class PerfSample:
    """One measured terminal kernel — a ``perf`` row joined to its ``cuda_op``.

    The minimal row the dataset layer reads: the kernel's pretty source (for the C
    identifier), the recorded knobs (which already carry the ``S_*`` / ``H_*``
    features the variant was stamped with), and the median latency. Backs
    :meth:`SearchDB.iter_perf_samples`."""

    pretty: str
    knobs: dict
    latency_us: float
    error: str | None = None  # bench_fail failure text (None on ok rows / pre-error-column DBs)


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


# The ``perf`` SELECT column list — order must match ``_row_to_perf``.
_PERF_COLS = (
    "context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
    "latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured"
)


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
    #   2: explicit-knob OFF sentinels — every variant now stamps every planner
    #       knob (tier-foreign ones get an OFF value: WM/WN/MMA on scalar,
    #       BM/BN/BR/FK on warp), so ``op_cache_key`` (which folds the knob dict)
    #       shifts for every TileOp/KernelOp. Stale ``lowering`` rows won't match.
    _SCHEMA_VERSION = 2

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
            captured             INTEGER NOT NULL DEFAULT 0,
            error                TEXT,
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
        # Additive migration: pre-existing DBs lack the ``error`` column
        # (``CREATE TABLE IF NOT EXISTS`` never alters). Writer-side only —
        # read-only consumers tolerate its absence instead.
        if not self._has_perf_error_column():
            self._conn.execute("ALTER TABLE perf ADD COLUMN error TEXT")

    def _has_perf_error_column(self) -> bool:
        return any(r[1] == "error" for r in self._conn.execute("PRAGMA table_info(perf)"))

    @classmethod
    def open_readonly(cls, path: Path | str) -> SearchDB:
        """Open an existing DB **read-only** — no schema creation, no version
        check, no ``DROP TABLE lowering``, no WAL pragma — so a read-side consumer
        (``eval``, the dataset layer) never contends with a concurrent ``tune``
        writer or mutates the file. The read methods (``iter_perf`` /
        ``iter_perf_samples`` / ``lookup_*``) work; any write raises (the
        connection is ``?mode=ro``). Raises ``sqlite3.OperationalError`` if the
        file is absent."""
        self = cls.__new__(cls)
        self._conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True, check_same_thread=False)
        return self

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
                json.dumps(_jsonable_geometry(grid)),
                json.dumps(_jsonable_geometry(block)),
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
        captured: bool = False,
        error: str | None = None,
    ) -> None:
        """Upsert one ``perf`` row. Keep-best-``ok`` policy: a ``bench_fail``
        never overwrites a prior ``ok`` row, and among same-semantics ``ok``
        rows the lowest median wins. ``captured`` (CUDA-graph-captured, pure GPU
        time) adds a precedence axis: a captured measurement supersedes an
        uncaptured (wall-semantics) one regardless of median — the numbers
        aren't comparable, and captured is the better truth — while an
        uncaptured measurement never overwrites a captured one. ``error`` is the
        failure text for a ``bench_fail`` row (whitespace-collapsed, truncated)
        so failure forensics (``eval failures``) need no tune-log grepping."""
        existing = self.lookup_perf(context_key, op_key, backend=backend)
        if existing is not None and existing.status == "ok":
            if status != "ok":
                return  # a failure never replaces a good measurement
            if existing.captured and not captured:
                return  # wall semantics never overwrites a captured row
            if not (captured and not existing.captured) and stats.median >= existing.stats.median:
                return  # same semantics: keep the best median
        knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
        if error is not None:
            error = " ".join(str(error).split())[:300] or None
        self._conn.execute(
            "INSERT OR REPLACE INTO perf "
            "(context_key, op_key, backend, status, latency_us_median, latency_us_min, latency_us_max, "
            " latency_us_mean, latency_us_variance, n_samples, measured_at, knobs, captured, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                int(captured),
                error,
            ),
        )

    # ------------------------------------------------------------------
    # Perf — read
    # ------------------------------------------------------------------

    def lookup_lowering(self, parent_key: str) -> LoweringRow | None:
        """Return the best-known child for ``parent_key``, or ``None``
        when no row exists. Used by :meth:`best_per_op_time`'s chain
        walk to resolve a pre-final op's measured cost."""
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
            f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ? AND op_key = ? AND backend = ?",  # noqa: S608
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
                f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ?",  # noqa: S608
                (context_key,),
            )
        else:
            cur = self._conn.execute(
                f"SELECT {_PERF_COLS} FROM perf WHERE context_key = ? AND backend = ?",  # noqa: S608
                (context_key, backend),
            )
        for row in cur:
            yield _row_to_perf(row)

    def iter_perf_samples(self, *, backend: str | None = "cuda", status: str = "ok", min_latency_us: float = 0.0) -> Iterator[PerfSample]:
        """Yield one :class:`PerfSample` per measured terminal kernel — ``perf``
        joined to ``cuda_op`` on ``op_key = key``. The single place the two tables
        are joined; backs ``Dataset.from_db``. ``backend=None`` spans every backend.
        Filters to ``status`` (default ``ok``) and ``latency_us_median >
        min_latency_us`` so callers don't re-filter stale / failed rows. The
        ``error`` select degrades to ``NULL`` on a pre-error-column DB opened
        read-only (the additive migration runs writer-side only)."""
        error_col = "perf.error" if self._has_perf_error_column() else "NULL"
        sql = (
            f"SELECT cuda_op.pretty, perf.knobs, perf.latency_us_median, {error_col} "  # noqa: S608 — column name from a fixed two-way choice
            "FROM perf JOIN cuda_op ON perf.op_key = cuda_op.key "
            "WHERE perf.status = ? AND perf.latency_us_median > ?"
        )
        params: list = [status, min_latency_us]
        if backend is not None:
            sql += " AND perf.backend = ?"
            params.append(backend)
        for pretty, knobs_json, us, error in self._conn.execute(sql, params):
            try:
                knobs = json.loads(knobs_json) if knobs_json else {}
            except (TypeError, json.JSONDecodeError):
                continue
            yield PerfSample(pretty=pretty, knobs=knobs, latency_us=us, error=error)

    # ------------------------------------------------------------------
    # Per-op best time (summed into the outer terminal reward)
    # ------------------------------------------------------------------

    def best_per_op_time(self, context_key: str, op_key: str, *, backend: str = "cuda") -> float | None:
        """Best measured median (us) for the kernel that ``op_key`` lowers
        to in ``context_key``, or ``None`` when it has no clean ``ok``
        measurement.

        ``op_key`` is typically a finalized ``LoopOp`` key (the unit the
        outer search hands to the inner per-op tuner). Two ways it carries a
        time:

        1. **Direct row** — the two-level inner search records the best
           *whole-slice* total (``Σ`` over the slice's CudaOps, so split-K
           main + combine are both counted) under the LoopOp key itself.
           Preferred when present.
        2. **Chain walk** — otherwise follow the ``lowering`` best-known child
           links down to the ``cuda`` dialect and read that terminal's
           context-keyed median. A ``CudaOp`` key resolves here directly (no
           lowering row as parent).
        """
        direct = self.lookup_perf(context_key, op_key, backend=backend)
        if direct is not None and direct.status == "ok":
            return direct.stats.median
        cur: str | None = op_key
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            row = self.lookup_lowering(cur)
            if row is None:
                break
            cur = row.child_key
            if row.child_dialect == "cuda":
                break
        if cur is None or cur == op_key:
            return None
        perf = self.lookup_perf(context_key, cur, backend=backend)
        return perf.stats.median if perf is not None and perf.status == "ok" else None

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
        captured=bool(row[12]),
    )
