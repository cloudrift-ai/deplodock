"""Smem bank-conflict analysis over Tile IR.

Two layers:

* :func:`lane_bank_distribution` — pure oracle used by the lowering
  passes (009/014). Decodes ``threadIdx.x = warp_id*32 + lane`` per the
  same axis flattening as ``Tile.render`` in ``ir/stmt/blocks.py``,
  evaluates a Load's cache-relative ``Expr`` index per lane against
  row-major strides over ``Stage.alloc_extents`` (which already folds
  in ``Stage.pad``), and tallies per-bank distinct addresses
  (``addr % 32``). Free vars not in ``thread_axes`` and not in
  ``extra_env`` are zero-bound (bank distribution is invariant to
  warp-uniform offsets). For ``BufferedStage`` callers must trim the
  leading slot dim — it's CTA-uniform at one moment and doesn't shift
  the bank pattern.

* :func:`simulate_graph` — runtime trace used by the visualizer.
  Compiles the graph through the CUDA pipeline with the
  ``002_instrument_smem_loads`` pass on (``DEPLODOCK_BANK_TRACE=1``),
  runs once, decodes the ``_debug_buf`` records via the per-load
  sidecar attached to the graph, and packs everything (including
  per-cell access provenance over the inner-loop sweep) into
  :class:`BankConflictResult`. Cross-checked against the oracle in
  ``tests/compiler/diagnostics/test_bank_conflicts.py``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt import Cond, Load, Loop, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp

logger = logging.getLogger(__name__)

WARP_SIZE = 32
BANKS = 32

# Wired up by the kernel-stage instrumentation pass
# (``002_instrument_smem_loads``) when ``DEPLODOCK_BANK_TRACE=1``.
_DEBUG_BUF_NAME = "_debug_buf"
_GATE_ENV = "DEPLODOCK_BANK_TRACE"
_SIDECAR_ATTR = "_bank_trace_sidecar"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageBinding:
    """One ``(Stage, body-Load reading it)`` pair plus its surrounding context."""

    stage: Stage
    load: Load
    tile: Tile
    enclosing_loop_axes: tuple[Axis, ...]  # outermost-first
    tile_op_name: str = ""


@dataclass
class BankConflictResult:
    """Per-lane bank id + summary statistics for one StageBinding.

    Hardware semantics: when multiple lanes target the *same address* in
    a bank, the load is broadcast and only one cycle is spent. The
    actual ``l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld``
    counter increments by ``(distinct_addrs_at_bank - 1)`` per warp-LDS,
    summed across banks. ``max_way`` here is that worst-bank value
    (i.e. ``max(distinct_addrs_at_bank)``) — same-address broadcasts
    don't count.

    ``raw_max_way`` is ``max(lanes_at_bank)`` — the upper bound assuming
    no broadcasting. Useful for cross-checking the broadcast model.
    """

    stage_name: str
    buf: str
    stage_class: str
    rows: int
    cols: int
    pad: tuple[int, ...]
    smem_bytes: int
    load_name: str  # SSA name of the body Load (e.g. ``in3``)
    tile_op_name: str
    index_repr: tuple[str, ...]  # the cache-relative index of the simulated Load
    lane_banks: list[int]  # length WARP_SIZE
    lane_addrs: list[int]  # length WARP_SIZE — pre-mod linear smem address
    counts: list[int]  # length BANKS — lanes per bank
    distinct_addrs: list[int]  # length BANKS — distinct addresses per bank
    max_way: int  # worst broadcast-corrected = max(distinct_addrs)
    raw_max_way: int  # max(counts) — upper bound w/o broadcast
    conflict_events: int  # sum(distinct_addrs[b] - 1 for b with hits) — per-LDS.32 model
    lds128_events: int = 0  # sum(max(0, distinct_addrs[b] - vec_width)); vec_width=1 if standalone, up to 4 if vectorized
    vec_group_size: int = 1  # number of Loads that fuse into one LDS.(N×32) (1 = scalar LDS.32)
    avg_way: float = 0.0
    # Populated by ``simulate_graph`` only (visualizer-facing).
    full_sweep_touched: dict[tuple[int, int], list[tuple[int, int]]] = field(default_factory=dict)
    full_sweep_conflict_cells: set[tuple[int, int]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# IR walk
# ---------------------------------------------------------------------------


def find_all_bindings(graph: Graph, stage_filter: set[str] | None = None) -> list[StageBinding]:
    """Yield one ``StageBinding`` per (Stage, body Load) pair across every
    ``TileOp`` in ``graph``. Each Stage declared inside the (single) Tile
    body is matched against every body Load whose ``input`` is the staged
    name; enclosing ``Loop`` / ``StridedLoop`` axes are propagated so
    callers can pin the inner-iter axis at simulation time.
    """
    out: list[StageBinding] = []
    for node in graph.nodes.values():
        if not isinstance(node.op, TileOp):
            continue
        tile_op = node.op
        for top in tile_op.body:
            if not isinstance(top, Tile):
                continue
            stages: dict[str, Stage] = {}
            for s in _walk(top.body):
                if isinstance(s, Stage) and (stage_filter is None or s.name in stage_filter):
                    stages.setdefault(s.name, s)
            for load, axes in _walk_loads(top.body, ()):
                if load.input in stages:
                    out.append(
                        StageBinding(
                            stage=stages[load.input],
                            load=load,
                            tile=top,
                            enclosing_loop_axes=axes,
                            tile_op_name=tile_op.name or "",
                        )
                    )
    return out


def _walk(body) -> Iterable:
    for s in body:
        yield s
        for attr in ("body", "else_body"):
            inner = getattr(s, attr, None)
            if isinstance(inner, tuple) and inner and hasattr(inner[0], "deps"):
                yield from _walk(inner)


def _walk_loads(body, axes: tuple[Axis, ...]):
    for s in body:
        if isinstance(s, Load):
            yield s, axes
        if isinstance(s, (Loop, StridedLoop)):
            yield from _walk_loads(s.body, (*axes, s.axis))
        elif isinstance(s, Tile):
            yield from _walk_loads(s.body, axes)
        elif isinstance(s, Cond):
            yield from _walk_loads(s.body, axes)
            yield from _walk_loads(s.else_body, axes)


# ---------------------------------------------------------------------------
# Thread-axis decode (mirrors ir/stmt/blocks.py::_render_thread_axis_decode)
# ---------------------------------------------------------------------------


def _thread_axis_env(thread_axes: tuple[Axis, ...], tid: int) -> dict[str, int]:
    """Reproduce the runtime decode of ``threadIdx.x`` into per-axis ints.

    Identical to ``_render_thread_axis_decode`` in ``ir/stmt/blocks.py``:
    outermost axis = ``tid // inner_prod``, inner axes get
    ``(tid // stride) % extent`` with ``stride`` doubling rightward,
    and a single-axis tile collapses to ``tid``.
    """
    if not thread_axes:
        return {}
    extents = [int(ax.extent) for ax in thread_axes]
    if len(extents) == 1:
        return {thread_axes[0].name: tid}
    inner_prod = 1
    for e in extents[1:]:
        inner_prod *= e
    env: dict[str, int] = {thread_axes[0].name: tid // inner_prod}
    stride = 1
    for ax, e in zip(reversed(thread_axes[1:]), reversed(extents[1:]), strict=True):
        env[ax.name] = tid % e if stride == 1 else (tid // stride) % e
        stride *= e
    return env


# ---------------------------------------------------------------------------
# Shared lane→bank kernel
# ---------------------------------------------------------------------------


@dataclass
class BankDistribution:
    """Per-lane bank allocation for one warp evaluating one Load index.

    Pure layout output — no Stage / Load / Tile identifiers. The lowering
    rules in ``compiler/pipeline/passes/lowering/tile`` consume the raw
    fields directly to score candidate smem layouts.
    """

    lane_addrs: list[int]  # length WARP_SIZE — pre-mod linear smem address
    lane_banks: list[int]  # length WARP_SIZE
    counts: list[int]  # length BANKS — lanes per bank
    distinct_addrs: list[int]  # length BANKS — distinct addresses per bank
    max_way: int  # max(distinct_addrs) — broadcast-corrected
    raw_max_way: int  # max(counts) — upper bound w/o broadcast
    conflict_events: int  # sum(d - 1 for d in distinct_addrs if d > 0)


def _row_major_strides(extents: tuple[int, ...]) -> list[int]:
    strides = [1] * len(extents)
    for i in range(len(extents) - 2, -1, -1):
        strides[i] = strides[i + 1] * extents[i + 1]
    return strides


def lane_bank_distribution(
    cache_index: tuple[Expr, ...],
    cache_extents: tuple[int, ...],
    thread_axes: tuple[Axis, ...],
    *,
    extra_env: dict[str, int] | None = None,
    warp_id: int = 0,
) -> BankDistribution | None:
    """Decode ``threadIdx.x = warp_id*WARP_SIZE + lane`` into per-axis
    ints, evaluate ``cache_index`` per lane against row-major strides
    over ``cache_extents``, and tally per-bank distinct-address counts.

    ``cache_index`` must already be trimmed to ``len(cache_extents)``
    cache dimensions (e.g. drop a ``BufferedStage``'s leading slot
    index — uniform across the warp at one moment, doesn't shift the
    bank pattern). Free vars in ``cache_index`` not in ``thread_axes``
    and not in ``extra_env`` are zero-bound; bank distribution is
    invariant to additive warp-uniform offsets. Pass ``extra_env`` to
    pin a specific loop iter or block coord (e.g. ``{k_loop: 5}``).

    Returns ``None`` if ``len(cache_index) != len(cache_extents)`` or
    if ``Expr.eval`` raises (KeyError / TypeError) for any lane.
    """
    if len(cache_index) != len(cache_extents) or not cache_extents:
        return None

    strides = _row_major_strides(cache_extents)

    thread_names = {ax.name for ax in thread_axes}
    free: set[str] = set()
    for e in cache_index:
        free |= e.free_vars()
    base_env: dict[str, int] = {n: 0 for n in free - thread_names}
    if extra_env:
        base_env.update(extra_env)

    lane_addrs: list[int] = []
    lane_banks: list[int] = []
    for lane in range(WARP_SIZE):
        env: dict[str, object] = dict(base_env)
        env.update(_thread_axis_env(thread_axes, warp_id * WARP_SIZE + lane))
        try:
            coords = [int(idx.eval(env)) for idx in cache_index]
        except (KeyError, TypeError):
            return None
        addr = sum(c * s for c, s in zip(coords, strides, strict=True))
        lane_addrs.append(addr)
        lane_banks.append(addr % BANKS)

    counts = [0] * BANKS
    addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
    for b, a in zip(lane_banks, lane_addrs, strict=True):
        counts[b] += 1
        addrs_per_bank[b].add(a)
    distinct = [len(s) for s in addrs_per_bank]
    return BankDistribution(
        lane_addrs=lane_addrs,
        lane_banks=lane_banks,
        counts=counts,
        distinct_addrs=distinct,
        max_way=max(distinct) if distinct else 0,
        raw_max_way=max(counts) if counts else 0,
        conflict_events=sum(d - 1 for d in distinct if d > 0),
    )


# ---------------------------------------------------------------------------
# LDS.128 vectorization model
# ---------------------------------------------------------------------------
#
# nvcc fuses N consecutive scalar fp32 loads from smem into a single
# ``ld.shared.v4`` (LDS.128) when they read 4 contiguous fp32. The warp
# drains in 4 cycles regardless — so the first ``vec_width`` distinct
# addresses per bank are "absorbed" into the natural drain (no extra
# replay cycle). Mirrors the cost model in ``compiler/autotune.py``
# (``effective_b_conflict_cost``, empirically validated on
# ``k_add_5_reduce``: 4-way @ F_N=4 ≈ 1-way @ F_N=1 wall-clock).
#
# Per-bank events under LDS.(N×32):
#     events_at_bank = max(0, distinct_addrs_at_bank - vec_width)
# with ``vec_width = min(N, 4)``.

LDS128_VEC_WIDTH = 4


def annotate_lds128(results: list[BankConflictResult]) -> None:
    """In-place: detect vectorizable runs and rewrite ``lds128_events``.

    Two Loads belong to the same LDS.128 chain iff for every lane,
    ``B.lane_addrs[lane] - A.lane_addrs[lane] == 1`` — i.e. each lane
    reads the next contiguous fp32. Greedy chain on unique address
    signatures (so duplicate Loads from prologue/steady-state pipeline
    phases don't break detection); cap each chain at ``LDS128_VEC_WIDTH``
    (4). Standalone Loads keep ``vec_group_size = 1`` and
    ``lds128_events == conflict_events``.
    """
    by_kernel_stage: dict[tuple[str, str], list[BankConflictResult]] = {}
    for r in results:
        by_kernel_stage.setdefault((r.tile_op_name, r.stage_name), []).append(r)

    for group in by_kernel_stage.values():
        # Bucket by address signature so duplicate Loads (same lane_addrs)
        # are treated as one node in the chain graph.
        sig_buckets: dict[tuple[int, ...], list[BankConflictResult]] = {}
        for r in group:
            sig_buckets.setdefault(tuple(r.lane_addrs), []).append(r)
        # Order signatures by min addr — vectorizable signatures end up
        # adjacent. For each signature, all duplicates share its
        # vec_group_size.
        sigs = sorted(sig_buckets.keys(), key=min)
        i = 0
        while i < len(sigs):
            chain_sigs = [sigs[i]]
            j = i + 1
            while j < len(sigs) and len(chain_sigs) < LDS128_VEC_WIDTH:
                prev, cur = chain_sigs[-1], sigs[j]
                if all(b - a == 1 for a, b in zip(prev, cur, strict=True)):
                    chain_sigs.append(cur)
                    j += 1
                else:
                    break
            if len(chain_sigs) > 1:
                vec = len(chain_sigs)
                for sig in chain_sigs:
                    for r in sig_buckets[sig]:
                        r.vec_group_size = vec
                        r.lds128_events = sum(max(0, d - vec) for d in r.distinct_addrs)
            i = j


# ---------------------------------------------------------------------------
# GPU-trace simulator (visualizer + cross-validation)
# ---------------------------------------------------------------------------


def simulate_graph(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
    load_filter: set[str] | None = None,
) -> list[BankConflictResult]:
    """Compile + run + decode bank-conflict trace for every (Stage, Load) pair.

    ``k_iter`` and ``warp_id`` select which inner-loop iteration / warp
    to take the per-lane snapshot from. The full-sweep cell maps include
    every recorded iteration regardless.
    """
    bindings = find_all_bindings(graph, stage_filter)
    if not bindings:
        return []

    outputs = _compile_and_run(graph)

    sidecar = getattr(graph, _SIDECAR_ATTR, None) or {}
    if not sidecar:
        # Either no smem loads were instrumented or the gate was off.
        return []

    debug = outputs.get(_DEBUG_BUF_NAME)
    if debug is None:
        raise RuntimeError(f"_debug_buf not in run outputs (got {list(outputs)})")
    debug_int = np.rint(debug).astype(np.int64)

    out: list[BankConflictResult] = []
    seen: set[tuple] = set()
    for binding in bindings:
        key = (binding.tile_op_name, binding.stage.name, binding.load.name)
        if key in seen:
            continue
        seen.add(key)
        result = _build_result(binding, sidecar, debug_int, k_iter=k_iter, warp_id=warp_id)
        if result is not None:
            out.append(result)

    annotate_lds128(out)

    if load_filter is not None:
        out = [r for r in out if r.load_name in load_filter]
    return out


def _compile_and_run(graph: Graph) -> dict[str, np.ndarray]:
    """Compile through the CUDA pipeline with the instrumentation gate on,
    run once with default inputs, return the output dict."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    backend = CudaBackend()
    prior = os.environ.get(_GATE_ENV)
    os.environ[_GATE_ENV] = "1"
    try:
        compiled = backend.compile(graph)
        result = backend.run(compiled, input_data=None)
    finally:
        if prior is None:
            os.environ.pop(_GATE_ENV, None)
        else:
            os.environ[_GATE_ENV] = prior
    return result.outputs


def _build_result(binding: StageBinding, sidecar: dict, debug_int: np.ndarray, *, k_iter: int, warp_id: int) -> BankConflictResult | None:
    stage, load = binding.stage, binding.load
    if not stage.axes:
        return None

    # Find the matching load record in the sidecar by (smem_name, ssa_name).
    record = None
    for kernel_data in sidecar.values():
        for rec in kernel_data["loads"].values():
            if rec.smem_name == stage.name and rec.ssa_name == load.name:
                record = rec
                num_warps = kernel_data["num_warps"]
                break
        if record is not None:
            break
    if record is None:
        return None

    iter_total = record.iter_total or 1
    region = debug_int[record.offset : record.offset + iter_total * num_warps * WARP_SIZE]
    if region.size != iter_total * num_warps * WARP_SIZE:
        return None
    region = region.reshape(iter_total, num_warps, WARP_SIZE)

    if warp_id >= num_warps or k_iter >= iter_total:
        return None

    # ``addr + 1`` is the recorded value; subtract 1 (and clamp -1 to 0 for
    # any unwritten slots — slots a thread never reached).
    snapshot = region[k_iter, warp_id]
    valid = snapshot > 0
    lane_addrs_arr = np.where(valid, snapshot - 1, 0).astype(np.int64)
    lane_addrs = [int(x) for x in lane_addrs_arr]
    lane_banks = [a % BANKS for a in lane_addrs]

    counts = [0] * BANKS
    addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
    for lane in range(WARP_SIZE):
        if not valid[lane]:
            continue
        b = lane_banks[lane]
        counts[b] += 1
        addrs_per_bank[b].add(lane_addrs[lane])
    distinct = [len(s) for s in addrs_per_bank]
    max_way = max(distinct) if distinct else 0
    raw_max_way = max(counts) if counts else 0
    conflict_events = sum(d - 1 for d in distinct if d > 0)
    nz = [c for c in counts if c > 0]
    avg = sum(nz) / len(nz) if nz else 0.0

    # Full-sweep cell maps — fold all k_iter records.
    full_sweep_touched, full_sweep_conflict = _full_sweep_from_trace(region, stage=stage, warp_id=warp_id)

    return BankConflictResult(
        stage_name=stage.name,
        buf=stage.buf,
        stage_class=type(stage).__name__,
        rows=int(stage.axes[0].extent),
        cols=int(stage.axes[1].extent) if len(stage.axes) > 1 else 1,
        pad=tuple(int(p) for p in stage.pad),
        smem_bytes=stage.smem_bytes,
        load_name=load.name,
        tile_op_name=binding.tile_op_name,
        index_repr=tuple(e.pretty() for e in load.index[-len(stage.axes) :]),
        lane_banks=lane_banks,
        lane_addrs=lane_addrs,
        counts=counts,
        distinct_addrs=distinct,
        max_way=max_way,
        raw_max_way=raw_max_way,
        conflict_events=conflict_events,
        lds128_events=conflict_events,  # annotate_lds128 may rewrite later
        vec_group_size=1,
        avg_way=avg,
        full_sweep_touched=full_sweep_touched,
        full_sweep_conflict_cells=full_sweep_conflict,
    )


def _full_sweep_from_trace(
    region: np.ndarray, stage: Stage, warp_id: int
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], set[tuple[int, int]]]:
    """Build per-cell access maps from the recorded trace.

    ``region.shape == (iter_total, num_warps, 32)``; values are
    ``smem_addr_in_elements + 1`` (0 = unwritten). Cell ``(r, c)`` is
    derived from ``addr // row_stride`` / ``addr % row_stride`` over
    ``stage.alloc_extents``.
    """
    alloc = list(stage.alloc_extents)
    if not alloc:
        return {}, set()
    strides = [1] * len(alloc)
    for i in range(len(alloc) - 2, -1, -1):
        strides[i] = strides[i + 1] * alloc[i + 1]
    row_stride = strides[0] if strides else 1

    iter_total, num_warps, _ = region.shape
    if warp_id >= num_warps:
        return {}, set()

    touched: dict[tuple[int, int], list[tuple[int, int]]] = {}
    conflicts: set[tuple[int, int]] = set()
    for k in range(iter_total):
        lds_addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
        cell_for_bank: list[tuple[int, int, int] | None] = [None] * WARP_SIZE
        for lane in range(WARP_SIZE):
            v = int(region[k, warp_id, lane])
            if v <= 0:
                continue
            addr = v - 1
            r, c = divmod(addr, row_stride) if row_stride else (0, addr)
            bank = addr % BANKS
            touched.setdefault((r, c), []).append((k, lane))
            lds_addrs_per_bank[bank].add(addr)
            cell_for_bank[lane] = (r, c, bank)
        for entry in cell_for_bank:
            if entry is None:
                continue
            r, c, bank = entry
            if len(lds_addrs_per_bank[bank]) > 1:
                conflicts.add((r, c))
    return touched, conflicts
