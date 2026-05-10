"""SQLite-backed tuning cache for the autotune loop.

One table keyed by ``(context_key, op_key)``. ``context_key`` is
:meth:`Context.structural_key` — segregates entries by hardware target
and codegen-affecting knobs. ``op_key`` is the structural digest of a
fully-lowered op (today: ``CudaOp`` — :func:`op_cache_key` digests its
rendered source + launch params).

Stub measurement: until real GPU benchmarking lands, terminal
:class:`CudaOp` nodes are recorded with ``latency_us=1.0``. The search
policy uses cache hits to prioritize candidates whose remaining ops
still need measurement (see :class:`MeasurementPrioritySearch`).

Concurrency: opened in WAL mode so parallel benches can read while one
writes. The connection is kept open for the cache's lifetime; callers
can share one ``TuningCache`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from deplodock.compiler.structural import digest


@dataclass(frozen=True)
class Entry:
    """One row of the tuning cache."""

    context_key: str
    op_key: str
    status: str
    latency_us: float
    measured_at: str


class TuningCache:
    """Persistent key-value store for measured op performance.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs that want cross-run persistence pass an
    explicit path like ``~/.cache/deplodock/autotune.db``).
    """

    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS entries (
            context_key TEXT NOT NULL,
            op_key TEXT NOT NULL,
            status TEXT NOT NULL,
            latency_us REAL NOT NULL,
            measured_at TEXT NOT NULL,
            PRIMARY KEY (context_key, op_key)
        )
    """

    def __init__(self, path: Path | str | None = None) -> None:
        if path is None:
            self._conn = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._SCHEMA)

    def lookup(self, context_key: str, op_key: str) -> Entry | None:
        row = self._conn.execute(
            "SELECT context_key, op_key, status, latency_us, measured_at FROM entries WHERE context_key=? AND op_key=?",
            (context_key, op_key),
        ).fetchone()
        return Entry(*row) if row else None

    def has(self, context_key: str, op_key: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM entries WHERE context_key=? AND op_key=? LIMIT 1",
            (context_key, op_key),
        ).fetchone()
        return row is not None

    def record(self, context_key: str, op_key: str, *, latency_us: float, status: str = "ok") -> None:
        """Insert or replace one entry. ``measured_at`` is stamped now."""
        self._conn.execute(
            "INSERT OR REPLACE INTO entries (context_key, op_key, status, latency_us, measured_at) VALUES (?, ?, ?, ?, ?)",
            (context_key, op_key, status, latency_us, datetime.now(UTC).isoformat()),
        )

    def close(self) -> None:
        self._conn.close()


def op_cache_key(op: object) -> str | None:
    """Cache key for a fully-lowered op, or ``None`` if the op isn't
    terminal yet. Today only :class:`CudaOp` is terminal; we digest its
    rendered source and launch parameters (the bits that determine
    runtime behavior)."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        return digest("CudaOp", op.kernel_source, op.arg_order, op.grid, op.block, op.smem_bytes)
    return None


def _is_kernel_bearing(op: object) -> bool:
    """True for any op that represents one kernel of work in the pipeline
    (lowering states from ``LoopOp`` through ``CudaOp``). Used to count
    work remaining for the priority search."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    return isinstance(op, (LoopOp, TileOp, KernelOp, CudaOp))


def count_unmeasured_ops(graph, cache: TuningCache, context_key: str) -> int:
    """Count kernel-bearing nodes that don't yet have a cache entry.

    A node is "measured" iff its op is fully lowered (``CudaOp``) and its
    :func:`op_cache_key` is present in the cache. Pre-terminal ops
    (``LoopOp`` / ``TileOp`` / ``KernelOp``) always count as unmeasured.
    Used as the priority key by :class:`MeasurementPrioritySearch` —
    candidates with fewer remaining unmeasured ops pop first.
    """
    n = 0
    for node in graph.nodes.values():
        if not _is_kernel_bearing(node.op):
            continue
        key = op_cache_key(node.op)
        if key is None or not cache.has(context_key, key):
            n += 1
    return n


def record_terminal(graph, cache: TuningCache, context_key: str, *, latency_us: float = 1.0) -> None:
    """Record every ``CudaOp`` node in ``graph`` to the cache. Stub
    measurement (default ``latency_us=1.0``) until real GPU benching
    lands — at that point this becomes ``measure_terminal`` and gets
    the real number."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    for node in graph.nodes.values():
        if not isinstance(node.op, CudaOp):
            continue
        key = op_cache_key(node.op)
        if key is None:
            continue
        cache.record(context_key, key, latency_us=latency_us, status="ok")
