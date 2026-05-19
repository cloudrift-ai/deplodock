"""Smem bank-conflict analysis.

Two layers:

* :func:`lane_bank_distribution` — pure oracle used by the lowering
  passes (009/014). Decodes ``threadIdx.x = warp_id*32 + lane`` per the
  same axis flattening as ``Tile.render`` in ``ir/stmt/blocks.py``,
  evaluates a Load's cache-relative ``Expr`` index per lane against
  row-major strides over the smem layout, and tallies per-bank distinct
  addresses (``addr % 32``). Free vars not in ``thread_axes`` and not in
  ``extra_env`` are zero-bound (bank distribution is invariant to
  warp-uniform offsets).

* :func:`simulate_graph` — Kernel-IR-level analyzer used by the
  visualizer. Lowers the graph through ``KERNEL_PASSES`` (idempotent —
  re-applies any unmet Tile lowering, then ``lowering/kernel``), then
  walks each ``KernelOp`` body for smem ``Load``s, evaluates per-lane
  addresses via :func:`lane_bank_distribution` against ``Smem.extents``
  (pad already folded), runs :func:`annotate_lds128` over sibling
  chains, and computes per-cell access provenance over the inner-loop
  sweep. Pure static — no GPU run.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from deplodock.compiler.backend.cuda.dtype import nbytes_of as _nbytes_of
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt import Cond, Load, Loop, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp

WARP_SIZE = 32
BANKS = 32


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
# ``k_add_5_reduce``: 4-way @ FN=4 ≈ 1-way @ FN=1 wall-clock).
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
# Kernel-IR static analyzer (visualizer + cross-validation)
# ---------------------------------------------------------------------------


def simulate_graph(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
    load_filter: set[str] | None = None,
) -> list[BankConflictResult]:
    """Lower ``graph`` through ``KERNEL_PASSES`` and statically analyze every
    smem ``Load`` in each ``KernelOp``.

    ``k_iter`` pins the deepest enclosing loop axis at the per-lane snapshot;
    the full-sweep cell maps fold every iteration regardless. ``warp_id``
    selects the warp within the CTA.
    """
    from deplodock.compiler.ir.kernel.ir import KernelOp, Smem
    from deplodock.compiler.pipeline import KERNEL_PASSES, Pipeline

    g = Pipeline.build(KERNEL_PASSES).run(graph.copy())

    out: list[BankConflictResult] = []
    for node in g.nodes.values():
        if isinstance(node.op, KernelOp):
            out.extend(_analyze_kernel(node.op, Smem, stage_filter, k_iter, warp_id))

    annotate_lds128(out)
    if load_filter is not None:
        out = [r for r in out if r.load_name in load_filter]
    return out


def _analyze_kernel(kernel_op, Smem, stage_filter, k_iter: int, warp_id: int) -> list[BankConflictResult]:
    body = kernel_op.body
    smem_decls = {s.name: s for s in body.iter_of_type(Smem) if stage_filter is None or s.name in stage_filter}
    if not smem_decls:
        return []
    tiles = body.iter_of_type(Tile)
    if not tiles:
        return []
    tile = tiles[0]  # KernelOp body has a single Tile holding thread/block axes

    # Walk for smem Loads with their enclosing Loop / StridedLoop axes.
    bindings: list[tuple[object, Load, tuple[Axis, ...]]] = []

    def walk(b, encl: tuple[Axis, ...]):
        for s in b:
            if isinstance(s, Load) and s.input in smem_decls:
                bindings.append((smem_decls[s.input], s, encl))
            if isinstance(s, (Loop, StridedLoop)):
                walk(s.body, (*encl, s.axis))
            elif isinstance(s, Tile):
                walk(s.body, encl)
            elif isinstance(s, Cond):
                walk(s.body, encl)
                walk(s.else_body, encl)

    walk(body, ())

    out: list[BankConflictResult] = []
    seen: set[tuple] = set()
    for smem, load, encl in bindings:
        key = (kernel_op.name, smem.name, load.name)
        if key in seen:
            continue
        seen.add(key)
        r = _build_kernel_result(kernel_op, smem, load, encl, tile, k_iter, warp_id)
        if r is not None:
            out.append(r)
    return out


def _build_kernel_result(
    kernel_op, smem, load: Load, encl: tuple[Axis, ...], tile: Tile, k_iter: int, warp_id: int
) -> BankConflictResult | None:
    if not smem.extents or len(load.index) < len(smem.extents):
        return None
    cache_idx = tuple(load.index[-len(smem.extents) :])

    extra_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
    if encl:
        for ax in encl[:-1]:
            extra_env.setdefault(ax.name, 0)
        extra_env[encl[-1].name] = k_iter

    dist = lane_bank_distribution(cache_idx, smem.extents, tile.thread_axes, extra_env=extra_env, warp_id=warp_id)
    if dist is None:
        return None

    nz = [c for c in dist.counts if c > 0]
    avg = sum(nz) / len(nz) if nz else 0.0

    full_sweep_touched, full_sweep_conflict = _full_sweep_static(cache_idx, smem.extents, tile, encl, warp_id)

    smem_bytes = _nbytes_of(smem.dtype)
    for e in smem.extents:
        smem_bytes *= int(e)

    return BankConflictResult(
        stage_name=smem.name,
        buf=smem.name,
        stage_class="Smem",
        rows=int(smem.extents[0]),
        cols=int(smem.extents[1]) if len(smem.extents) > 1 else 1,
        pad=(),  # already folded into Smem.extents at materialize_tile
        smem_bytes=smem_bytes,
        load_name=load.name,
        tile_op_name=kernel_op.name or "",
        index_repr=tuple(e.pretty() for e in cache_idx),
        lane_banks=dist.lane_banks,
        lane_addrs=dist.lane_addrs,
        counts=dist.counts,
        distinct_addrs=dist.distinct_addrs,
        max_way=dist.max_way,
        raw_max_way=dist.raw_max_way,
        conflict_events=dist.conflict_events,
        lds128_events=dist.conflict_events,  # annotate_lds128 may rewrite later
        vec_group_size=1,
        avg_way=avg,
        full_sweep_touched=full_sweep_touched,
        full_sweep_conflict_cells=full_sweep_conflict,
    )


def _full_sweep_static(
    cache_idx: tuple[Expr, ...], extents: tuple[int, ...], tile: Tile, encl: tuple[Axis, ...], warp_id: int
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], set[tuple[int, int]]]:
    """Sweep the deepest enclosing loop axis; record every (row, col) cell
    touched by any (lane, k_iter) pair, plus the cells participating in a
    real bank conflict at some k_iter (their LDS had > 1 distinct address
    on the cell's bank).
    """
    if not extents:
        return {}, set()
    strides = _row_major_strides(extents)
    row_stride = strides[0] if strides else 1

    if not encl:
        loop_extent = 1
        loop_axis_name = None
    else:
        loop_extent = int(encl[-1].extent)
        loop_axis_name = encl[-1].name

    base_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
    for ax in encl[:-1]:
        base_env.setdefault(ax.name, 0)

    touched: dict[tuple[int, int], list[tuple[int, int]]] = {}
    conflict_cells: set[tuple[int, int]] = set()

    for k in range(loop_extent):
        env_k = dict(base_env)
        if loop_axis_name is not None:
            env_k[loop_axis_name] = k
        lds_addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
        lds_cells: list[tuple[int, int, int]] = []
        for lane in range(WARP_SIZE):
            env: dict[str, object] = dict(env_k)
            env.update(_thread_axis_env(tile.thread_axes, warp_id * WARP_SIZE + lane))
            try:
                coords = [int(idx.eval(env)) for idx in cache_idx]
            except (KeyError, TypeError):
                return {}, set()
            addr = sum(c * s for c, s in zip(coords, strides, strict=True))
            r, c = divmod(addr, row_stride) if row_stride else (0, addr)
            bank = addr % BANKS
            touched.setdefault((r, c), []).append((k, lane))
            lds_addrs_per_bank[bank].add(addr)
            lds_cells.append((r, c, bank))
        for r, c, bank in lds_cells:
            if len(lds_addrs_per_bank[bank]) > 1:
                conflict_cells.add((r, c))
    return touched, conflict_cells
