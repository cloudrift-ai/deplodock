"""SQLite-backed tuning cache for the autotune loop.

One table keyed by ``(context_key, op_key)``. ``context_key`` is
:meth:`Context.structural_key` — segregates entries by hardware target
and codegen-affecting knobs. ``op_key`` is the structural digest of a
fully-lowered op (today: ``CudaOp`` — :func:`op_cache_key` digests its
rendered source + launch params).

Measurement: :func:`record_terminal` either stubs latency to ``1.0`` (no
backend) or runs ``backend.benchmark`` once per terminal graph and
attributes the per-launch GPU-event time to every ancestor along the
``Op.source`` chain (``CudaOp → KernelOp → TileOp → LoopOp``). The
search policy uses cache hits to prioritize candidates whose remaining
ops still need measurement (see :class:`MeasurementPrioritySearch`).

Concurrency: opened in WAL mode so parallel benches can read while one
writes. The connection is kept open for the cache's lifetime; callers
can share one ``TuningCache`` instance across threads (sqlite3 handles
locking).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from deplodock.compiler.structural import digest

logger = logging.getLogger(__name__)


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
    """Cache key for any kernel-bearing op, or ``None`` if the op isn't
    cacheable.

    Each level of the lowering chain has its own well-defined identity:

    - ``CudaOp`` — digest of rendered kernel source + launch params (the
      bits that determine runtime behavior).
    - ``KernelOp`` / ``TileOp`` / ``LoopOp`` — digest of the dialect tag
      plus :meth:`Body.structural_key` (already canonicalizes SSA, axis,
      commutative-arg, and external-buffer names).

    Same kernel reached via different rewrite paths produces the same
    key — ``Op.source`` is *not* part of the digest, so a fused LoopOp
    and the TileOp lowered from it hash differently (their structures
    differ), but two LoopOps that are structurally identical share a
    key regardless of which graph they live in.
    """
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    if isinstance(op, CudaOp):
        return digest("CudaOp", op.kernel_source, op.arg_order, op.grid, op.block, op.smem_bytes)
    if isinstance(op, (LoopOp, TileOp)):
        return digest(type(op).__name__, op.body.structural_key())
    if isinstance(op, KernelOp):
        # KernelOp bodies contain hardware-primitive stmts (Smem, Sync, ...)
        # that ``Body.structural_key``'s normalize path doesn't yet support.
        # Fall back to ``repr``-based digest — deterministic, but doesn't
        # canonicalize SSA / axis names so structurally-equivalent kernels
        # may hash distinct. Register Kernel-IR stmts for ``rewrite`` to
        # promote this to a real structural digest.
        return digest("KernelOp", repr(op.body))
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


def _source_chain(op):
    """Yield ``op`` and every predecessor along ``Op.source``."""
    cur = op
    while cur is not None:
        yield cur
        cur = cur.source


def record_terminal(
    graph,
    cache: TuningCache,
    context_key: str,
    *,
    backend=None,
    warmup: int = 5,
    num_iters: int = 20,
) -> None:
    """Measure every ``CudaOp`` in ``graph`` and attribute the result to
    every ancestor along its ``.source`` chain.

    When ``backend`` is ``None`` (default — keeps tests and CPU-only
    environments working): records ``latency_us=1.0`` for every kernel.
    This is the "stub" path used by the search policy as a placeholder
    until real measurement is opted in.

    When ``backend`` is provided (typically a ``CudaBackend``): calls
    ``backend.benchmark(graph, warmup=..., num_iters=...)`` once,
    consuming the per-launch GPU-event timings. The i-th
    :class:`LaunchTime` corresponds to the i-th ``CudaOp`` in
    ``graph.topological_order()`` (the same order the backend uses
    internally — see ``backend/cuda/program.py::_launches``).

    For each measured kernel the result is attributed to every ancestor
    along ``CudaOp.source`` (``KernelOp → TileOp → LoopOp``) — the same
    latency at every abstraction level, since they describe the same
    kernel. A future run that sees an equivalent ``LoopOp`` body can
    hit the cache without re-lowering.

    If ``backend.benchmark`` raises (NVRTC compile error, OOM, etc.) the
    whole graph is recorded with ``status="bench_fail"`` and the
    exception is logged — the cache pins the failure so the search
    doesn't re-discover the same dead end.
    """
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    cuda_nodes = [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]
    if not cuda_nodes:
        return

    if backend is None:
        for node in cuda_nodes:
            _attribute(cache, context_key, node.op, latency_us=1.0, status="ok")
        return

    try:
        result = backend.benchmark(graph, warmup=warmup, num_iters=num_iters)
    except Exception as exc:  # noqa: BLE001 — autotune cache must record any failure mode
        logger.warning("cache: backend.benchmark failed (%s) — pinning bench_fail for %d kernel(s)", exc, len(cuda_nodes))
        for node in cuda_nodes:
            _attribute(cache, context_key, node.op, latency_us=0.0, status="bench_fail")
        return

    per_launch = result.per_launch or []
    if len(per_launch) != len(cuda_nodes):
        logger.warning(
            "cache: per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
            len(per_launch),
            len(cuda_nodes),
        )
        avg_us = (result.time_ms * 1000.0) / max(len(cuda_nodes), 1)
        for node in cuda_nodes:
            _attribute(cache, context_key, node.op, latency_us=avg_us, status="ok")
        return

    for node, lt in zip(cuda_nodes, per_launch, strict=True):
        _attribute(cache, context_key, node.op, latency_us=lt.time_ms * 1000.0, status="ok")


def _attribute(cache: TuningCache, context_key: str, cuda_op, *, latency_us: float, status: str) -> None:
    """Record ``latency_us`` against every ancestor in ``cuda_op.source``."""
    for ancestor in _source_chain(cuda_op):
        key = op_cache_key(ancestor)
        if key is None:
            continue
        cache.record(context_key, key, latency_us=latency_us, status=status)
    logger.debug("cache: %s for kernel %s @ %.2f us", status, cuda_op.kernel_name, latency_us)
