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

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from deplodock.compiler.structural import digest

logger = logging.getLogger(__name__)


class TuneAborted(RuntimeError):
    """Raised by :func:`record_terminal` when a bench failure leaves the
    CUDA stream in a state where subsequent benches would block in
    ``_allocate`` (waiting for the still-running timed-out kernel to
    drain). Callers catch this to stop the autotune sweep with whatever
    measurements have been recorded so far."""


@dataclass(frozen=True)
class CudaPerf:
    """One ``cuda_perf`` row."""

    context_key: str
    cuda_key: str
    status: str
    latency_us: float
    measured_at: str
    knobs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class NodeRow:
    """One ``nodes`` row — autotune-tree state for an op.

    ``seen_terminals`` counts every measured terminal under this node
    (ok + fail; used by coverage queries). ``failed_terminals`` tracks
    fails only (diagnostic). ``visits`` is the MCTS denominator — it
    increments both on each ``cache.expand`` (expansion = exploration)
    and on each terminal measurement (measurement = visit). ``total_reward``
    accumulates the MCTS-style reward (``1/latency_us`` for ok, ``0``
    for fail). UCB exploitation uses ``total_reward / visits``."""

    context_key: str
    node_key: str
    parent_key: str | None
    expected_terminals: int
    seen_terminals: int
    failed_terminals: int = 0
    visits: int = 0
    total_reward: float = 0.0


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
            knobs        TEXT NOT NULL DEFAULT '{}',
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
            failed_terminals   INTEGER NOT NULL DEFAULT 0,
            visits             INTEGER NOT NULL DEFAULT 0,
            total_reward       REAL NOT NULL DEFAULT 0.0,
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
            "SELECT context_key, node_key, parent_key, expected_terminals, seen_terminals, failed_terminals, visits, total_reward "
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
            "SELECT context_key, cuda_key, status, latency_us, measured_at, knobs "
            "FROM cuda_perf WHERE context_key = ? AND cuda_key = ?",
            (context_key, cuda_key),
        ).fetchone()
        if row is None:
            return None
        knobs = json.loads(row[5]) if row[5] else {}
        return CudaPerf(row[0], row[1], row[2], row[3], row[4], knobs)

    def is_fully_explored(self, context_key: str, node_key: str) -> bool:
        """O(1) — true iff every known leaf below ``node_key`` has been
        measured. Reads the maintained counters; no tree walk."""
        n = self.node(context_key, node_key)
        return n is not None and n.seen_terminals >= n.expected_terminals

    def root_coverage(self, context_key: str) -> tuple[int, int]:
        """``(seen_terminals, expected_terminals)`` summed over every
        root node (``parent_key IS NULL``) for this context. Used by
        the CLI stopping policy to know how much of the autotune tree
        has been explored so far."""
        row = self._conn.execute(
            "SELECT COALESCE(SUM(seen_terminals), 0), COALESCE(SUM(expected_terminals), 0) "
            "FROM nodes WHERE context_key = ? AND parent_key IS NULL",
            (context_key,),
        ).fetchone()
        return (int(row[0]), int(row[1])) if row else (0, 0)

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

        # Treat the expansion event itself as one "visit" of the parent
        # for MCTS-style UCB selection (separate ``visits`` column;
        # doesn't disturb ``seen_terminals`` which counts measured
        # leaves). Without this, every sibling at a fork has
        # ``visits = 0`` and indistinguishable ``-inf`` UCB priorities,
        # so the selector can't differentiate. With it, a node that's
        # been popped and rule-fired accumulates visits along the way;
        # un-expanded siblings stay at ``visits = 0`` and pop first.
        self._propagate_mcts_visit(context_key, parent_key, visits_delta=1, reward_delta=0.0)

    def record_cuda_perf(
        self,
        context_key: str,
        cuda_key: str,
        *,
        latency_us: float,
        status: str = "ok",
        knobs: dict | None = None,
    ) -> bool:
        """Record a terminal measurement and propagate ``+1`` to
        ``seen_terminals`` on every ancestor. The cache keeps the *best*
        ``ok`` latency for a given cuda_key; status transitions (e.g.
        ``bench_fail → ok``) always overwrite. Returns ``True`` if the
        row counts as a newly-measured terminal (i.e. ``seen`` was 0 at
        the corresponding node row), so the caller knows propagation
        ran."""
        existing = self.cuda_perf(context_key, cuda_key)
        # Never let a bench_fail overwrite a prior ``ok`` row. Distinct
        # autotune variants can render to the same ``cuda_key`` (e.g. a
        # BM > input-dim variant clamps down to the same kernel as the
        # exact-fit one); without this guard the bench_fail's wall-budget
        # row would clobber a known-good measurement.
        keep_existing_ok = existing is not None and existing.status == "ok" and status != "ok"
        keep_best = existing is not None and existing.status == "ok" and status == "ok" and latency_us >= existing.latency_us
        if not keep_best and not keep_existing_ok:
            knobs_json = json.dumps(knobs or {}, sort_keys=True, default=str)
            self._conn.execute(
                "INSERT OR REPLACE INTO cuda_perf (context_key, cuda_key, status, latency_us, measured_at, knobs) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (context_key, cuda_key, status, latency_us, datetime.now(UTC).isoformat(), knobs_json),
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
        # MCTS reward: ``1 / latency_us`` for ok, ``0`` for failure.
        # Failures still count as a visit so the subtree's mean reward
        # drops, which lowers its UCB exploitation term.
        reward = (1.0 / latency_us) if status == "ok" and latency_us > 0 else 0.0
        failed_delta = 0 if status == "ok" else 1
        self._conn.execute(
            "UPDATE nodes SET seen_terminals = 1, failed_terminals = ?, visits = visits + 1, total_reward = ? "
            "WHERE context_key = ? AND node_key = ?",
            (failed_delta, reward, context_key, cuda_key),
        )
        if node.parent_key is not None:
            self._propagate_visit(context_key, node.parent_key, seen_delta=1, failed_delta=failed_delta, reward_delta=reward)
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

    def _propagate_visit(
        self,
        context_key: str,
        node_key: str,
        *,
        seen_delta: int,
        failed_delta: int,
        reward_delta: float,
    ) -> None:
        """Terminal-measurement backprop: bump ``seen_terminals`` /
        ``failed_terminals`` / ``visits`` / ``total_reward`` on
        ``node_key`` and every ancestor along ``parent_key``."""
        cur_key: str | None = node_key
        while cur_key is not None:
            self._conn.execute(
                "UPDATE nodes SET seen_terminals = seen_terminals + ?, "
                "failed_terminals = failed_terminals + ?, "
                "visits = visits + ?, "
                "total_reward = total_reward + ? "
                "WHERE context_key = ? AND node_key = ?",
                (seen_delta, failed_delta, seen_delta, reward_delta, context_key, cur_key),
            )
            row = self._conn.execute(
                "SELECT parent_key FROM nodes WHERE context_key = ? AND node_key = ?",
                (context_key, cur_key),
            ).fetchone()
            cur_key = row[0] if row else None

    def _propagate_mcts_visit(self, context_key: str, node_key: str, *, visits_delta: int, reward_delta: float) -> None:
        """Expansion-event backprop: bump ``visits`` (and optionally
        ``total_reward``) only — used by ``expand`` to treat each rule
        firing as a soft visit for UCB selection, without disturbing
        ``seen_terminals`` (which still means "measured leaves")."""
        cur_key: str | None = node_key
        while cur_key is not None:
            self._conn.execute(
                "UPDATE nodes SET visits = visits + ?, total_reward = total_reward + ? "
                "WHERE context_key = ? AND node_key = ?",
                (visits_delta, reward_delta, context_key, cur_key),
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

    logger.info("[tune] benching %d kernel(s) in graph", len(cuda_nodes))
    try:
        # ``num_iters="auto"`` adapts the iter count to per-call latency
        # (target 100 ms of GPU time, capped at ``_AUTO_MAX_ITERS``) so
        # tiny kernels get many samples for variance reduction and heavy
        # ones don't oversample. GPU time is what we actually want to
        # bound, not wall time — Python/cupy framing overhead is fixed
        # per iter and would otherwise force us into low sample counts
        # on small kernels.
        result = backend.benchmark(graph, num_iters="auto")
    except Exception as exc:  # noqa: BLE001 — autotune cache must record any failure mode
        # Treat the failure's "latency" as the wall-time budget that was
        # exhausted. Makes the cuda_perf row honest (these kernels DID
        # consume that much wall time, even if uselessly) and gives MCTS
        # UCB a small but non-zero reward (``1 / bench_wall_timeout``)
        # instead of zero. Still way below an ``ok`` kernel's reward,
        # so failures stay deprioritized.
        fail_latency_us = float(backend.bench_run_timeout_s) * 1_000_000.0
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d kernel(s)",
            exc,
            fail_latency_us,
            len(cuda_nodes),
        )
        for node in cuda_nodes:
            _record_one(cache, context_key, node.op, latency_us=fail_latency_us, status="bench_fail")
        # The kernel that timed out is still queued on the CUDA stream
        # and will keep executing for an unbounded time. Subsequent
        # ``backend.benchmark`` calls in this process hit cupy's
        # ``_allocate`` which serializes on the stream — that's why an
        # autotune sweep tends to "hang on the variant *after* a
        # bench_fail" instead of on the failing one itself. Re-raise so
        # the engine can abort the sweep before that happens.
        raise TuneAborted(f"autotune aborted after bench_fail; the dirty CUDA stream would hang the next variant") from exc

    per_launch = result.per_launch or []
    if len(per_launch) != len(cuda_nodes):
        logger.warning(
            "[tune] per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
            len(per_launch),
            len(cuda_nodes),
        )
        avg_us = (result.time_ms * 1000.0) / max(len(cuda_nodes), 1)
        for node in cuda_nodes:
            _record_one(cache, context_key, node.op, latency_us=avg_us, status="ok")
        return

    for node, lt in zip(cuda_nodes, per_launch, strict=True):
        _record_one(cache, context_key, node.op, latency_us=lt.time_ms * 1000.0, status="ok")

    # Between successful variants: drain any pending GPU work and let
    # cupy release its memory-pool blocks back to the driver. Drain
    # is microseconds when the stream is clean (the bench loop's own
    # ``_wait_for_event`` already synced every launch), so we don't
    # pay anything in the healthy path. The mempool free prevents
    # cross-variant fragmentation — each variant's compiled buffers
    # come from a fresh allocation rather than a stale pool slab.
    try:
        import cupy as _cp  # noqa: PLC0415

        _cp.cuda.runtime.deviceSynchronize()
        _cp.get_default_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


def _record_one(cache: TuningCache, context_key: str, cuda_op, *, latency_us: float, status: str) -> None:
    """Insert one ``cuda_perf`` row keyed on the CudaOp's structural
    digest; the cache walks ``parent_key`` chains internally to update
    ``seen_terminals``. Knobs accumulated along the rewrite chain are
    persisted alongside the latency so a partial sweep is still
    self-describing without the in-memory candidate."""
    cuda_key = op_cache_key(cuda_op)
    if cuda_key is None:
        return
    knobs = getattr(cuda_op, "knobs", None) or {}
    cache.record_cuda_perf(context_key, cuda_key, latency_us=latency_us, status=status, knobs=knobs)
    # INFO so ``deplodock compile --tune -v`` shows per-kernel progress;
    # at default WARNING level we stay quiet.
    logger.info("[tune]   %s @ %.2f us  (%s)", getattr(cuda_op, "kernel_name", "?"), latency_us, status)
