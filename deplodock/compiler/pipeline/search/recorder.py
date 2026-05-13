"""Top-level autotune recorder: bench → DB writes → tree updates.

Replaces the old monolithic ``record_terminal`` in ``cache.py``. Two
passes:

1. **Bench** — one ``backend.benchmark(graph, num_iters="auto")`` call
   produces one :class:`LaunchTime` per ``CudaOp`` in the graph. Per-iter
   samples come along on each launch (see ``LaunchTime.samples``) so
   downstream stats are computable here without re-running.
2. **Persist** — for each measured kernel walk ``Op.source`` once,
   recording the loop/tile/kernel/cuda inventory rows, then the
   lowering edges between dialect transitions, then the ``perf`` row.
   Finally bump the MCTS tree (one terminal measurement per kernel).

This is the only place that knows about all four moving parts (graph,
DB, tree, backend); the engine and policy stay oblivious to the split.
"""

from __future__ import annotations

import json
import logging
import statistics
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline.search.db import PerfStats, SearchDB
from deplodock.compiler.pipeline.search.keys import (
    _is_kernel_bearing,
    dialect_of,
    op_cache_key,
    source_chain,
)

if TYPE_CHECKING:
    # SearchTree lives next to TuningSearch in policy/mcts.py. We can't
    # import it eagerly here because policy/base.py imports recorder
    # (for ``count_unmeasured_ops``) and policy/mcts.py imports
    # policy/base.py — a runtime import would close the cycle. The
    # annotation is the only use, so TYPE_CHECKING is enough.
    from deplodock.compiler.pipeline.search.policy.mcts import SearchTree

logger = logging.getLogger(__name__)


class TuneAborted(RuntimeError):
    """Raised when a bench failure leaves the autotune sweep
    unrecoverable. Callers catch this to stop with whatever measurements
    have been recorded so far."""


def record_terminal(
    graph,
    db: SearchDB,
    tree: SearchTree | None,
    context_key: str,
    *,
    backend,
) -> None:
    """Bench every ``CudaOp`` in ``graph`` and persist one ``perf`` row
    + the op-inventory rows + the lowering edges along each kernel's
    source chain.

    When ``backend`` is ``None`` (stub): records ``latency_us=1.0`` for
    every CudaOp without running anything. Useful for tests that
    exercise the bookkeeping without spinning up CUDA.

    Bench failure pins ``status="bench_fail"`` on every kernel in the
    graph using the backend's wall-budget as the "latency", so the
    search doesn't re-explore the same dead end."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    cuda_nodes = [graph.nodes[nid] for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, CudaOp)]
    if not cuda_nodes:
        return

    backend_name = getattr(backend, "name", "stub")

    if backend is None:
        for node in cuda_nodes:
            stats = _point_stats(1.0)
            _persist(db, tree, context_key, node.op, stats=stats, status="ok", backend_name=backend_name)
        return

    logger.info("[tune] benching %d kernel(s) in graph", len(cuda_nodes))
    try:
        result = backend.benchmark(graph, num_iters="auto")
    except Exception as exc:  # noqa: BLE001 — autotune cache must record any failure mode
        # Pin every kernel as bench_fail using the backend's wall-budget
        # as the "latency". Makes the perf row honest (these kernels did
        # consume that much wall time, even if uselessly) and gives MCTS
        # UCB a small but non-zero reward (1 / bench_wall_timeout)
        # instead of zero.
        fail_latency_us = float(backend.bench_run_timeout_s) * 1_000_000.0
        logger.warning(
            "[tune] backend.benchmark failed (%s) — pinning bench_fail @ %.1f us for %d kernel(s)",
            exc,
            fail_latency_us,
            len(cuda_nodes),
        )
        stats = _point_stats(fail_latency_us)
        for node in cuda_nodes:
            _persist(db, tree, context_key, node.op, stats=stats, status="bench_fail", backend_name=backend_name)
        return

    per_launch = result.per_launch or []
    if len(per_launch) != len(cuda_nodes):
        logger.warning(
            "[tune] per_launch count (%d) != CudaOp node count (%d); falling back to graph time_ms / N",
            len(per_launch),
            len(cuda_nodes),
        )
        avg_us = (result.time_ms * 1000.0) / max(len(cuda_nodes), 1)
        stats = _point_stats(avg_us)
        for node in cuda_nodes:
            _persist(db, tree, context_key, node.op, stats=stats, status="ok", backend_name=backend_name)
        return

    for node, lt in zip(cuda_nodes, per_launch, strict=True):
        stats = _stats_from_launch(lt)
        _persist(db, tree, context_key, node.op, stats=stats, status="ok", backend_name=backend_name)

    # Between successful variants: drain pending GPU work and let cupy
    # release its memory-pool blocks back to the driver. Drain is
    # microseconds when the stream is clean (the bench loop's own
    # ``_wait_for_event`` already synced every launch), so we don't pay
    # anything in the healthy path. The mempool free prevents
    # cross-variant fragmentation — each variant's compiled buffers
    # come from a fresh allocation rather than a stale pool slab.
    try:
        import cupy as _cp  # noqa: PLC0415

        _cp.cuda.runtime.deviceSynchronize()
        _cp.get_default_memory_pool().free_all_blocks()
    except Exception:  # noqa: BLE001 — best-effort cleanup
        pass


# ---------------------------------------------------------------------------
# Persistence (used by both the success and failure paths)
# ---------------------------------------------------------------------------


def _persist(
    db: SearchDB,
    tree: SearchTree | None,
    context_key: str,
    cuda_op,
    *,
    stats: PerfStats,
    status: str,
    backend_name: str,
) -> None:
    """Walk ``cuda_op``'s source chain; record op-inventory + lowering
    + perf rows; bump the tree."""
    cuda_key = op_cache_key(cuda_op)
    if cuda_key is None:
        return

    # Record every op in the chain into its dialect's inventory table
    # AND record every adjacent rewrite hop in the lowering table —
    # including intra-dialect autotune hops (blockify_launch,
    # register_tile, ...) so greedy replay can reconstruct the full
    # decision chain by walking lowering rows hop-by-hop.
    chain = [op for op in source_chain(cuda_op) if _is_kernel_bearing(op)]
    for op in chain:
        _record_op_inventory(db, op)
    # chain[0] = cuda_op, chain[1] = its source, etc.
    # Lowering edges run *from* the older op (deeper in source chain) *to*
    # the newer op, so iterate in reverse.
    for parent_op, child_op in zip(chain[1:], chain[:-1], strict=False):
        p_dialect = dialect_of(parent_op)
        c_dialect = dialect_of(child_op)
        if p_dialect is None or c_dialect is None:
            continue
        p_key = op_cache_key(parent_op)
        c_key = op_cache_key(child_op)
        if p_key is None or c_key is None:
            continue
        # Knob delta = entries new or changed at this hop. Engine merges
        # ``{**old.knobs, **new.knobs}`` on every rebind so ``child``
        # carries the cumulative set; subtract ``parent`` to recover
        # just this step's contribution.
        p_knobs = getattr(parent_op, "knobs", None) or {}
        c_knobs = getattr(child_op, "knobs", None) or {}
        knobs_delta = {k: v for k, v in c_knobs.items() if p_knobs.get(k) != v}
        db.record_lowering(
            p_key,
            p_dialect,
            c_key,
            c_dialect,
            knobs=knobs_delta,
            measured_median_us=stats.median if status == "ok" else None,
        )

    knobs = getattr(cuda_op, "knobs", None) or {}
    db.record_perf(
        context_key,
        cuda_key,
        backend=backend_name,
        status=status,
        stats=stats,
        knobs=knobs,
    )
    if tree is not None:
        reward = (1.0 / stats.median) if status == "ok" and stats.median > 0 else 0.0
        tree.record_terminal(context_key, cuda_key, reward=reward, status=status)

    logger.info("[tune]   %s @ %.2f us  (%s)", getattr(cuda_op, "kernel_name", "?"), stats.median, status)


def _record_op_inventory(db: SearchDB, op) -> None:
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415
    from deplodock.compiler.ir.kernel.ir import KernelOp  # noqa: PLC0415
    from deplodock.compiler.ir.loop.ir import LoopOp  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import TileOp  # noqa: PLC0415

    key = op_cache_key(op)
    if key is None:
        return
    if isinstance(op, CudaOp):
        db.record_cuda_op(
            key,
            kernel_source=op.kernel_source,
            arg_order=list(op.arg_order),
            grid=list(op.grid),
            block=list(op.block),
            smem_bytes=op.smem_bytes,
            pretty=op.kernel_source,
        )
        return
    if isinstance(op, KernelOp):
        db.record_kernel_op(key, _body_json(op, "kernel"), op.pretty_body())
        return
    if isinstance(op, TileOp):
        db.record_tile_op(key, _body_json(op, "tile"), op.pretty_body())
        return
    if isinstance(op, LoopOp):
        db.record_loop_op(key, _body_json(op, "loop"), op.pretty_body())
        return


def _body_json(op, dialect: str) -> str:
    """Inspection-only JSON form for non-CUDA ops. Not round-trippable —
    just a stable handle for tools that want to grep the DB."""
    return json.dumps(
        {
            "dialect": dialect,
            "name": getattr(op, "name", None) or getattr(op, "kernel_name", None) or "?",
            "body_repr": repr(op.body),
        },
        default=str,
    )


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _stats_from_launch(lt) -> PerfStats:
    """Build a :class:`PerfStats` in microseconds from a
    :class:`LaunchTime`. Falls back to point stats when samples weren't
    recorded (older backends or single-iter benches)."""
    if lt.samples and len(lt.samples) >= 1:
        us = [s * 1000.0 for s in lt.samples]
        median = statistics.median(us)
        return PerfStats(
            median=median,
            min=min(us),
            max=max(us),
            mean=statistics.fmean(us),
            variance=statistics.pvariance(us) if len(us) > 1 else 0.0,
            n_samples=len(us),
        )
    return _point_stats(lt.time_ms * 1000.0)


def _point_stats(latency_us: float) -> PerfStats:
    """Stats bundle for a single-point measurement (e.g. the stub
    backend or a bench_fail row). All five fields collapse to the same
    value; ``n_samples=0`` marks it as "no real distribution"."""
    return PerfStats(
        median=latency_us,
        min=latency_us,
        max=latency_us,
        mean=latency_us,
        variance=0.0,
        n_samples=0,
    )


# ---------------------------------------------------------------------------
# Coverage helper used by the autotune driver
# ---------------------------------------------------------------------------


def count_unmeasured_ops(graph, db: SearchDB, context_key: str, *, backend_name: str = "cuda") -> int:
    """Count kernel-bearing nodes that don't yet have a ``perf`` row in
    ``db`` for ``backend_name``."""
    from deplodock.compiler.ir.cuda.ir import CudaOp  # noqa: PLC0415

    n = 0
    for node in graph.nodes.values():
        if not _is_kernel_bearing(node.op):
            continue
        if isinstance(node.op, CudaOp):
            key = op_cache_key(node.op)
            if key is None or db.lookup_perf(context_key, key, backend=backend_name) is None:
                n += 1
        else:
            # Pre-terminal ops always count as unmeasured — the search
            # hasn't finished lowering them yet.
            n += 1
    return n
