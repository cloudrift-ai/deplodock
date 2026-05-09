"""Dynamic bank-conflict simulator — instrument the emitted kernel and
decode the actual smem addresses each lane reads at runtime.

Public entry point:

  :func:`simulate_graph_dynamic` — same signature shape as
  :func:`deplodock.compiler.diagnostics.bank_conflicts.simulate_graph`,
  same return type (``list[BankConflictResult]``). The visualizer can
  swap one for the other transparently.

Pipeline:

1. Discover ``(Stage, Load)`` bindings in the input Tile-IR Graph using
   the existing static :func:`find_all_bindings` — it gives us names,
   alloc_extents, pad, enclosing axes (the per-cell metadata that's not
   visible from a Kernel-IR-only view).
2. Compile the graph through the full CUDA pipeline with
   ``DEPLODOCK_BANK_TRACE=1``; the kernel-stage instrumentation pass
   (``002_instrument_smem_loads``) wraps every smem ``Load`` with a
   deterministic-slot record AND adds the ``_debug_buf`` graph output
   the records target.
3. Run the program once with single-CTA-shaped inputs (default-allocated
   pseudo-random values are fine — the recorded *addresses* don't depend
   on data).
4. Decode the debug buffer using the sidecar attached to the graph by
   the pass (``graph._bank_trace_sidecar``) — per-load offset / extents
   / num_warps. For each load_id we get a per-``(k_iter, warp, lane)``
   integer address.
5. For each ``(Stage, Load)`` binding match the load_id by ``smem_name +
   load_ssa`` and assemble a ``BankConflictResult`` — banks are
   ``(addr % 32)`` per lane; full-sweep cell maps fold over k_iter; the
   shared :func:`annotate_lds128` post-pass runs unchanged.

This module imports ``BankConflictResult`` and the LDS.128 helper from
the static sibling module — only the simulation engine differs, the
output shape is shared.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from deplodock.compiler.diagnostics.bank_conflicts import (
    BANKS,
    WARP_SIZE,
    BankConflictResult,
    annotate_lds128,
    find_all_bindings,
)
from deplodock.compiler.graph import Graph

logger = logging.getLogger(__name__)

_DEBUG_BUF_NAME = "_debug_buf"
_GATE_ENV = "DEPLODOCK_BANK_TRACE"
_SIDECAR_ATTR = "_bank_trace_sidecar"


def simulate_graph_dynamic(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
    load_filter: set[str] | None = None,
) -> list[BankConflictResult]:
    """Dynamic counterpart to :func:`simulate_graph`.

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


# ---------------------------------------------------------------------------
# Compile + run
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def _build_result(binding, sidecar: dict, debug_int: np.ndarray, *, k_iter: int, warp_id: int) -> BankConflictResult | None:
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
    full_sweep_touched, full_sweep_subst, full_sweep_conflict = _full_sweep_from_trace(
        region, valid_mask=region > 0, stage=stage, warp_id=warp_id
    )

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
        enclosing_axes=tuple(ax.name for ax in binding.enclosing_loop_axes),
        full_sweep_touched=full_sweep_touched,
        full_sweep_subst_idx=full_sweep_subst,
        full_sweep_conflict_cells=full_sweep_conflict,
    )


def _full_sweep_from_trace(
    region: np.ndarray, valid_mask: np.ndarray, stage, warp_id: int
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], tuple[str, ...]],
    set[tuple[int, int]],
]:
    """Build per-cell access maps from the recorded trace.

    ``region.shape == (iter_total, num_warps, 32)``; values are
    ``smem_addr_in_elements + 1`` (0 = unwritten). Cell ``(r, c)`` is
    derived from ``addr // row_stride`` / ``addr % row_stride`` over
    ``stage.alloc_extents``.
    """
    alloc = list(stage.alloc_extents)
    if not alloc:
        return {}, {}, set()
    strides = [1] * len(alloc)
    for i in range(len(alloc) - 2, -1, -1):
        strides[i] = strides[i + 1] * alloc[i + 1]
    row_stride = strides[0] if strides else 1

    iter_total, num_warps, _ = region.shape
    if warp_id >= num_warps:
        return {}, {}, set()

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

    # Substituted index strings — unavailable from trace alone (we don't
    # have the symbolic Expr per cell). Visualizer tooltips degrade to
    # empty; the static sibling can fill these post-hoc if needed.
    return touched, {}, conflicts
