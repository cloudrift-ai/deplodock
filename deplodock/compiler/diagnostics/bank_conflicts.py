"""Static smem bank-conflict simulator over Tile IR.

For a given (TileOp, Stage) pair, evaluate every body ``Load`` of the
staged buffer for each warp lane at one inner-loop iteration. The
per-lane smem address is computed by:

1. Decoding ``threadIdx.x = warp_id*32 + lane`` into thread-axis values
   using the same flattening as ``Tile.render`` in ``ir/stmt/blocks.py``
   (``_render_thread_axis_decode``: outermost = ``tid // inner_prod``,
   inner = ``(tid // stride) % extent``, rightmost = ``tid % extent``).
2. Binding the deepest enclosing ``Loop`` / ``StridedLoop`` axis to the
   user-supplied ``k_iter`` value; outer block axes and outer loop axes
   are bound to 0 (we visualize one CTA at one fixed iteration).
3. Substituting that env into the Load's index ``Expr``s via
   ``Expr.eval`` — i.e. running the compiler's own evaluator, not a
   reimplementation.
4. Linearizing the cache coordinates against ``Stage.alloc_extents``
   (which already folds in ``Stage.pad`` as added row stride) and
   taking ``addr % 32`` to get the bank id (banks are 32 fp32-wide).

For ``BufferedStage`` subtypes the leading slot dim of the Load index
is dropped — it's CTA-uniform at one cycle and doesn't shift the bank
pattern.

This is a static model: it predicts what a *coalesced* warp access
would do under the IR's declared smem layout. It does not model
sub-warp scheduling, register-file pressure, or LDS.32 vs LDS.128
issue width — confirm against ncu (``smsp__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ld``)
on a real GPU run when the answer matters.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

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
    enclosing_axes: tuple[str, ...] = ()
    # Per-(row, col) → list of (k_iter, lane_id) pairs that read this cell
    # over the FULL inner-loop sweep (k_iter 0..loop_extent-1). Populated
    # by ``annotate_full_sweep`` so visualizers can show every cell's
    # access provenance, not just the one at k_iter=0.
    full_sweep_touched: dict[tuple[int, int], list[tuple[int, int]]] = field(default_factory=dict)
    # Per-(row, col) → tuple of substituted index strings (one per cache
    # axis) using one example (k_iter, lane) that reaches the cell.
    # E.g. ``[(a2 * 8), a6]`` evaluated at lane=0 / k_iter=5 becomes
    # ``("(0 * 8)", "5")``. Lets visualizers replace symbolic vars with
    # concrete values in the per-cell tooltip.
    full_sweep_subst_idx: dict[tuple[int, int], tuple[str, ...]] = field(default_factory=dict)
    # Cells that participate in a *real* conflict at some k_iter — i.e.
    # the LDS that touches the cell has > 1 distinct addresses on the
    # cell's bank for that k_iter. Distinct from "potential conflict"
    # which is only a layout property (rows aliasing on a bank). Empty
    # iff this Load never has bank conflicts across the full sweep.
    full_sweep_conflict_cells: set[tuple[int, int]] = field(default_factory=set)


# ---------------------------------------------------------------------------
# IR walk
# ---------------------------------------------------------------------------


def find_stage_bindings(tile_op: TileOp, stage_filter: set[str] | None = None) -> list[StageBinding]:
    """Walk ``tile_op`` and yield one StageBinding per (Stage, Load) pair.

    Each ``Stage`` declared inside the (single) ``Tile`` body is matched
    against every body ``Load`` whose ``input`` is the staged name. The
    enclosing ``Loop`` / ``StridedLoop`` axes are propagated so callers
    can pin the inner-iter axis at simulation time.
    """
    out: list[StageBinding] = []
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


def find_all_bindings(graph: Graph, stage_filter: set[str] | None = None) -> list[StageBinding]:
    """Run ``find_stage_bindings`` over every TileOp in ``graph``."""
    out: list[StageBinding] = []
    for node in graph.nodes.values():
        if isinstance(node.op, TileOp):
            out.extend(find_stage_bindings(node.op, stage_filter))
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


def thread_axis_env(thread_axes: tuple[Axis, ...], tid: int) -> dict[str, int]:
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

    Pure layout output — no Stage / Load / Tile identifiers. ``simulate``
    composes this with names and provenance into a ``BankConflictResult``;
    the lowering rules in ``compiler/pipeline/passes/lowering/tile``
    consume the raw fields directly to score candidate smem layouts.
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
    invariant to additive warp-uniform offsets, so this matches the
    "drop ``warp_const``" simplification of the affine analyzer that
    used to live in ``passes/lowering/tile/_helpers.py``. Pass
    ``extra_env`` to pin a specific loop iter or block coord (e.g.
    ``{k_loop: 5}``).

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
        env.update(thread_axis_env(thread_axes, warp_id * WARP_SIZE + lane))
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
# Per-binding simulation
# ---------------------------------------------------------------------------


def simulate(binding: StageBinding, k_iter: int = 0, warp_id: int = 0) -> BankConflictResult | None:
    """Compute per-lane bank ids for one ``StageBinding``.

    Returns ``None`` if the cache rank can't be reconciled (e.g. the
    Load index is shorter than ``Stage.axes``) or if any index Expr
    references a Var the env can't bind.
    """
    stage, load, tile = binding.stage, binding.load, binding.tile

    # Buffered stages prefix the Load index with slot/phase dims that
    # are CTA-uniform at one moment — they don't change the bank
    # pattern. Take the trailing ``len(stage.axes)`` entries.
    if not stage.axes:
        return None
    if len(load.index) < len(stage.axes):
        return None
    cache_idx: tuple[Expr, ...] = tuple(load.index[-len(stage.axes) :])

    extra_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
    enc = list(binding.enclosing_loop_axes)
    if enc:
        for ax in enc[:-1]:
            extra_env.setdefault(ax.name, 0)
        extra_env[enc[-1].name] = k_iter

    dist = lane_bank_distribution(
        cache_idx,
        stage.alloc_extents,
        tile.thread_axes,
        extra_env=extra_env,
        warp_id=warp_id,
    )
    if dist is None:
        return None
    nz = [c for c in dist.counts if c > 0]
    avg = sum(nz) / len(nz) if nz else 0.0

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
        index_repr=tuple(e.pretty() for e in cache_idx),
        lane_banks=dist.lane_banks,
        lane_addrs=dist.lane_addrs,
        counts=dist.counts,
        distinct_addrs=dist.distinct_addrs,
        max_way=dist.max_way,
        raw_max_way=dist.raw_max_way,
        conflict_events=dist.conflict_events,
        # Initial scalar (LDS.32) value; ``annotate_lds128`` upgrades
        # vectorizable runs in-place with the absorption-aware count.
        lds128_events=dist.conflict_events,
        vec_group_size=1,
        avg_way=avg,
        enclosing_axes=tuple(ax.name for ax in binding.enclosing_loop_axes),
    )


def simulate_graph(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
    load_filter: set[str] | None = None,
) -> list[BankConflictResult]:
    """Convenience: bindings + simulate, dropping rank-mismatched probes.

    ``stage_filter`` keeps only Stages with these names; ``load_filter``
    keeps only body Loads with these SSA names. Both filters AND.

    After simulation, ``annotate_lds128`` runs in place to upgrade
    vectorizable Load runs (consecutive +0..+N-1 inner-element offsets)
    with their LDS.128-absorbed event counts.
    """
    # Run simulation + chain detection on the un-load-filtered set: an
    # LDS.128 chain spans sibling Loads of the same stage, so a single-
    # Load drill-down still needs to see its siblings to know its real
    # ``vec_group_size``. Apply ``load_filter`` only after annotation.
    out: list[BankConflictResult] = []
    seen: set[tuple] = set()
    for binding in find_all_bindings(graph, stage_filter):
        key = (binding.tile_op_name, binding.stage.name, binding.load.name)
        if key in seen:
            continue
        seen.add(key)
        r = simulate(binding, k_iter=k_iter, warp_id=warp_id)
        if r is not None:
            out.append(r)
    annotate_lds128(out)
    # Compute the per-cell access map for each result by re-evaluating
    # the same Load's index across the full inner-loop sweep.
    bindings_by_key = {(b.tile_op_name, b.stage.name, b.load.name): b for b in find_all_bindings(graph, stage_filter)}
    for r in out:
        binding = bindings_by_key.get((r.tile_op_name, r.stage_name, r.load_name))
        if binding is not None:
            touched, subst, conflicts = _full_sweep_touched(binding, warp_id=warp_id)
            r.full_sweep_touched = touched
            r.full_sweep_subst_idx = subst
            r.full_sweep_conflict_cells = conflicts
    if load_filter is not None:
        out = [r for r in out if r.load_name in load_filter]
    return out


def _full_sweep_touched(
    binding: StageBinding, warp_id: int = 0
) -> tuple[
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], tuple[str, ...]],
    set[tuple[int, int]],
]:
    """Sweep the deepest enclosing loop axis; record every (row, col)
    cell touched by any (lane, k_iter) pair, plus a substituted index
    string (one per cache axis) using one example (k_iter, lane) that
    reaches the cell, plus the set of cells that participate in a real
    bank conflict at some k_iter (their LDS had > 1 distinct address on
    the cell's bank).

    Returns ``({(row, col) -> [(k_iter, lane), ...]}, {(row, col) ->
    ("(0 * 8)", "5")}, conflict_cells)``. Outer enclosing loops and
    block axes pinned to 0; thread axes decoded per lane. Cell
    coordinates are reconstructed from the linear smem address using
    ``alloc_extents`` (folds in pad), matching ``simulate``.
    """
    from deplodock.compiler.ir.expr import Literal

    stage, load, tile = binding.stage, binding.load, binding.tile
    if not stage.axes or len(load.index) < len(stage.axes):
        return {}, {}, set()
    cache_idx = tuple(load.index[-len(stage.axes) :])

    alloc = list(stage.alloc_extents)
    strides = [1] * len(alloc)
    for i in range(len(alloc) - 2, -1, -1):
        strides[i] = strides[i + 1] * alloc[i + 1]
    row_stride = strides[0] if strides else 1

    enc = list(binding.enclosing_loop_axes)
    if not enc:
        loop_extent = 1
        loop_axis_name = None
    else:
        loop_extent = int(enc[-1].extent)
        loop_axis_name = enc[-1].name
    base_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
    for ax in enc[:-1]:
        base_env.setdefault(ax.name, 0)

    out: dict[tuple[int, int], list[tuple[int, int]]] = {}
    subst: dict[tuple[int, int], tuple[str, ...]] = {}
    conflict_cells: set[tuple[int, int]] = set()
    for k in range(loop_extent):
        env_k = dict(base_env)
        if loop_axis_name is not None:
            env_k[loop_axis_name] = k
        # Per-LDS bank-distinct-addrs accounting (one LDS = one Load at
        # one k_iter) so we can flag cells that participate in real
        # conflicts (their bank had > 1 distinct address in this LDS).
        lds_addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
        lds_cells: list[tuple[int, int, int]] = []  # (r, c, bank) per lane
        for lane in range(WARP_SIZE):
            env = dict(env_k)
            env.update(thread_axis_env(tile.thread_axes, warp_id * WARP_SIZE + lane))
            try:
                coords = [int(idx.eval(env)) for idx in cache_idx]
            except (KeyError, TypeError):
                return {}, {}, set()
            addr = sum(c * s for c, s in zip(coords, strides, strict=True))
            r, c = divmod(addr, row_stride) if row_stride else (0, addr)
            bank = addr % BANKS
            out.setdefault((r, c), []).append((k, lane))
            lds_addrs_per_bank[bank].add(addr)
            lds_cells.append((r, c, bank))
            # First (k, lane) wins as the example for substitution.
            if (r, c) not in subst:
                lit_env = {name: Literal(v) for name, v in env.items()}
                subst[(r, c)] = tuple(idx.substitute(lit_env).pretty() for idx in cache_idx)
        # Mark cells that participate in a conflict at this k_iter.
        for r, c, bank in lds_cells:
            if len(lds_addrs_per_bank[bank]) > 1:
                conflict_cells.add((r, c))
    return out, subst, conflict_cells


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
