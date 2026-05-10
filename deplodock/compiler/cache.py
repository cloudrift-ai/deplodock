"""SQLite-backed tuning cache for the autotune loop — tree-shaped.

Three tables:

- ``cuda_perf`` — one row per measured terminal kernel (``cuda_key``
  produced by :func:`op_cache_key` on a ``CudaOp``).
- ``expansions`` — parent → child edges of the autotune search tree.
  PRIMARY KEY ``(context_key, parent_key, child_key)`` enforces that
  each edge exists at most once; ``INSERT OR IGNORE`` makes re-firing
  a rule on a previously-seen op a no-op.
- ``nodes`` — one row per op state encountered in the search.
  ``expected_terminals`` / ``seen_terminals`` are maintained online via
  upward propagation: each *new* expansion of a parent that had no
  children adds ``n_new - 1`` to ``expected_terminals`` on every
  ancestor (the parent's placeholder "1" is consumed by the first
  child); subsequent expansions of the same parent add ``n_new``. Each
  newly measured terminal adds ``+1`` to ``seen_terminals`` on every
  ancestor. ``seen / expected`` is then the fraction explored at any
  node in O(1).

A node is fully explored when ``seen_terminals == expected_terminals``.
The value can move *down* mid-run when expansion grows the denominator
faster than the numerator — that's the correct semantics ("we just
discovered there's more to explore").

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
class CudaPerf:
    """One ``cuda_perf`` row."""

    context_key: str
    cuda_key: str
    status: str
    latency_us: float
    measured_at: str


@dataclass(frozen=True)
class NodeRow:
    """One ``nodes`` row — autotune-tree state for an op."""

    context_key: str
    node_key: str
    parent_key: str | None
    expected_terminals: int
    seen_terminals: int


class TuningCache:
    """Persistent tree-shaped cache for autotune measurements.

    Pass ``path=None`` for an in-memory database (default — keeps tests
    hermetic; tuning runs that want cross-run persistence pass an
    explicit path like ``~/.cache/deplodock/autotune.db``).
    """

    _SCHEMA = [
        """
        CREATE TABLE IF NOT EXISTS cuda_perf (
            context_key  TEXT NOT NULL,
            cuda_key     TEXT NOT NULL,
            status       TEXT NOT NULL,
            latency_us   REAL NOT NULL,
            measured_at  TEXT NOT NULL,
            PRIMARY KEY (context_key, cuda_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS expansions (
            context_key  TEXT NOT NULL,
            parent_key   TEXT NOT NULL,
            child_key    TEXT NOT NULL,
            PRIMARY KEY (context_key, parent_key, child_key)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS nodes (
            context_key        TEXT NOT NULL,
            node_key           TEXT NOT NULL,
            parent_key         TEXT,
            expected_terminals INTEGER NOT NULL DEFAULT 1,
            seen_terminals     INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (context_key, node_key)
        )
        """,
        # Speeds child-count probe before expand().
        "CREATE INDEX IF NOT EXISTS idx_expansions_parent ON expansions(context_key, parent_key)",
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
    # Node / expansion queries
    # ------------------------------------------------------------------

    def node(self, context_key: str, node_key: str) -> NodeRow | None:
        row = self._conn.execute(
            "SELECT context_key, node_key, parent_key, expected_terminals, seen_terminals "
            "FROM nodes WHERE context_key = ? AND node_key = ?",
            (context_key, node_key),
        ).fetchone()
        return NodeRow(*row) if row else None

    def children(self, context_key: str, parent_key: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT child_key FROM expansions WHERE context_key = ? AND parent_key = ?",
            (context_key, parent_key),
        ).fetchall()
        return [r[0] for r in rows]

    def cuda_perf(self, context_key: str, cuda_key: str) -> CudaPerf | None:
        row = self._conn.execute(
            "SELECT context_key, cuda_key, status, latency_us, measured_at FROM cuda_perf WHERE context_key = ? AND cuda_key = ?",
            (context_key, cuda_key),
        ).fetchone()
        return CudaPerf(*row) if row else None

    def is_fully_explored(self, context_key: str, node_key: str) -> bool:
        """O(1) — true iff every known leaf below ``node_key`` has been
        measured. Reads the maintained counters; no tree walk."""
        n = self.node(context_key, node_key)
        return n is not None and n.seen_terminals >= n.expected_terminals

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def ensure_root(self, context_key: str, node_key: str) -> None:
        """Insert a root node (``parent_key = NULL``) if not present."""
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes (context_key, node_key, parent_key) VALUES (?, ?, NULL)",
            (context_key, node_key),
        )

    def expand(self, context_key: str, parent_key: str, child_keys: list[str]) -> None:
        """Record ``parent_key → child_keys`` edges and maintain
        ``expected_terminals`` on every ancestor.

        Idempotent: re-firing a rule with the same children is a no-op
        (the PRIMARY KEY on ``expansions`` filters duplicates via
        ``INSERT OR IGNORE``). When the rule's option set has grown,
        only the genuinely-new edges propagate.
        """
        if not child_keys:
            return
        # Make sure the parent node exists. If the caller didn't pre-insert
        # it (e.g., it's the root), default to a placeholder with no parent.
        self.ensure_root(context_key, parent_key)

        pre = self._conn.execute(
            "SELECT COUNT(*) FROM expansions WHERE context_key = ? AND parent_key = ?",
            (context_key, parent_key),
        ).fetchone()[0]

        # Insert edges; count how many were genuinely new.
        n_new = 0
        for child_key in child_keys:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO expansions (context_key, parent_key, child_key) VALUES (?, ?, ?)",
                (context_key, parent_key, child_key),
            )
            n_new += cur.rowcount

        if n_new == 0:
            return

        # Insert child node rows (placeholders) for the new edges.
        for child_key in child_keys:
            self._conn.execute(
                "INSERT OR IGNORE INTO nodes (context_key, node_key, parent_key) VALUES (?, ?, ?)",
                (context_key, child_key, parent_key),
            )

        # First-ever expansion of this parent consumes its placeholder "1",
        # so the delta is one less than n_new. Later expansions are pure
        # additions (the parent was already accounting for its children).
        delta = n_new - 1 if pre == 0 else n_new
        if delta != 0:
            self._propagate_expected(context_key, parent_key, delta)

    def record_cuda_perf(
        self,
        context_key: str,
        cuda_key: str,
        *,
        latency_us: float,
        status: str = "ok",
    ) -> bool:
        """Record a terminal measurement and propagate ``+1`` to
        ``seen_terminals`` on every ancestor. The cache keeps the *best*
        ``ok`` latency for a given cuda_key; status transitions (e.g.
        ``bench_fail → ok``) always overwrite. Returns ``True`` if the
        row counts as a newly-measured terminal (i.e. ``seen`` was 0 at
        the corresponding node row), so the caller knows propagation
        ran."""
        existing = self.cuda_perf(context_key, cuda_key)
        keep_best = existing is not None and existing.status == "ok" and status == "ok" and latency_us >= existing.latency_us
        if not keep_best:
            self._conn.execute(
                "INSERT OR REPLACE INTO cuda_perf (context_key, cuda_key, status, latency_us, measured_at) VALUES (?, ?, ?, ?, ?)",
                (context_key, cuda_key, status, latency_us, datetime.now(UTC).isoformat()),
            )

        # Make sure the node row exists for this cuda_key (the engine
        # should have already inserted it via expand(), but rooted
        # single-CudaOp graphs may have skipped the chain).
        self._conn.execute(
            "INSERT OR IGNORE INTO nodes (context_key, node_key, parent_key) VALUES (?, ?, NULL)",
            (context_key, cuda_key),
        )

        # First-time-seen propagation: bump ``seen_terminals = 1`` only if
        # the node's count was zero. Re-recording the same kernel doesn't
        # double-count.
        node = self.node(context_key, cuda_key)
        if node is None or node.seen_terminals >= 1:
            return False
        self._conn.execute(
            "UPDATE nodes SET seen_terminals = 1 WHERE context_key = ? AND node_key = ?",
            (context_key, cuda_key),
        )
        if node.parent_key is not None:
            self._propagate_seen(context_key, node.parent_key, +1)
        return True

    # ------------------------------------------------------------------
    # Internal propagation walks
    # ------------------------------------------------------------------

    def _propagate_expected(self, context_key: str, node_key: str, delta: int) -> None:
        """Add ``delta`` to ``expected_terminals`` of ``node_key`` and
        every ancestor along ``parent_key``."""
        cur_key: str | None = node_key
        while cur_key is not None:
            self._conn.execute(
                "UPDATE nodes SET expected_terminals = expected_terminals + ? WHERE context_key = ? AND node_key = ?",
                (delta, context_key, cur_key),
            )
            row = self._conn.execute(
                "SELECT parent_key FROM nodes WHERE context_key = ? AND node_key = ?",
                (context_key, cur_key),
            ).fetchone()
            cur_key = row[0] if row else None

    def _propagate_seen(self, context_key: str, node_key: str, delta: int) -> None:
        """Add ``delta`` to ``seen_terminals`` of ``node_key`` and every
        ancestor along ``parent_key``."""
        cur_key: str | None = node_key
        while cur_key is not None:
            self._conn.execute(
                "UPDATE nodes SET seen_terminals = seen_terminals + ? WHERE context_key = ? AND node_key = ?",
                (delta, context_key, cur_key),
            )
            row = self._conn.execute(
                "SELECT parent_key FROM nodes WHERE context_key = ? AND node_key = ?",
                (context_key, cur_key),
            ).fetchone()
            cur_key = row[0] if row else None

    # ------------------------------------------------------------------
    # House-keeping
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Op-key derivation
# ---------------------------------------------------------------------------


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
    """Count kernel-bearing nodes that don't yet have a CudaOp
    measurement in the cache."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    n = 0
    for node in graph.nodes.values():
        if not _is_kernel_bearing(node.op):
            continue
        if isinstance(node.op, CudaOp):
            key = op_cache_key(node.op)
            if key is None or cache.cuda_perf(context_key, key) is None:
                n += 1
        else:
            # Pre-terminal ops always count as unmeasured — the search
            # hasn't finished lowering them yet.
            n += 1
    return n


# ---------------------------------------------------------------------------
# Terminal recording
# ---------------------------------------------------------------------------


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
) -> None:
    """Measure every ``CudaOp`` in ``graph`` and record one row per
    kernel in ``cuda_perf``.

    The engine seeds the autotune tree by calling :meth:`TuningCache.expand`
    at every rule application — by the time this function runs, every
    ancestor along ``CudaOp.source`` already has a ``nodes`` row with
    its ``parent_key`` set. We only insert ``cuda_perf`` rows and bump
    ``seen_terminals`` upward; ``expected_terminals`` was maintained at
    expand time.

    When ``backend`` is ``None`` (stub): records ``latency_us=1.0``.
    With a backend: one ``backend.benchmark(graph, num_iters="auto")``
    call; the i-th :class:`LaunchTime` corresponds to the i-th
    ``CudaOp`` in ``graph.topological_order()``.

    Bench failure pins ``status="bench_fail"`` so the search doesn't
    re-explore the same dead end."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    cuda_nodes = [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]
    if not cuda_nodes:
        return

    if backend is None:
        for node in cuda_nodes:
            _record_one(cache, context_key, node.op, latency_us=1.0, status="ok")
        return

    try:
        result = backend.benchmark(graph, num_iters="auto")
    except Exception as exc:  # noqa: BLE001 — autotune cache must record any failure mode
        logger.warning("cache: backend.benchmark failed (%s) — pinning bench_fail for %d kernel(s)", exc, len(cuda_nodes))
        for node in cuda_nodes:
            _record_one(cache, context_key, node.op, latency_us=0.0, status="bench_fail")
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
            _record_one(cache, context_key, node.op, latency_us=avg_us, status="ok")
        return

    for node, lt in zip(cuda_nodes, per_launch, strict=True):
        _record_one(cache, context_key, node.op, latency_us=lt.time_ms * 1000.0, status="ok")


def _record_one(cache: TuningCache, context_key: str, cuda_op, *, latency_us: float, status: str) -> None:
    """Insert one ``cuda_perf`` row keyed on the CudaOp's structural
    digest; the cache walks ``parent_key`` chains internally to update
    ``seen_terminals``."""
    cuda_key = op_cache_key(cuda_op)
    if cuda_key is None:
        return
    cache.record_cuda_perf(context_key, cuda_key, latency_us=latency_us, status=status)
    logger.debug("cache: %s for kernel %s @ %.2f us", status, getattr(cuda_op, "kernel_name", "?"), latency_us)
