"""Smem bank-conflict analysis, against the kernel-IR slab layouts.

Two layers:

* :func:`lane_bank_distribution` — pure lane→bank oracle. Decodes an intra-CTA thread id
  ``warp_id*32 + lane`` per the same axis flattening as ``Tile.render``, evaluates a Load's
  slab-relative ``Expr`` index per lane against row-major strides over the slab extents, and
  tallies per-bank distinct addresses (4-byte banks — a sub-word element dtype packs
  ``4 // elem_bytes`` elements per bank word). Free vars not in ``thread_axes`` and not in
  ``extra_env`` are zero-bound (bank distribution is invariant to warp-uniform offsets).

* :func:`find_all_bindings` / :func:`simulate_graph` — the Kernel-IR walk + analyzer the
  visualizer (``scripts/visualize_bank_conflicts.py``) consumes. ``find_all_bindings`` lowers
  nothing: it walks each ``KernelOp`` body for the ``Smem`` slab decls (the staged operand slabs
  ``_a_smem`` / ``_b_smem``, the shared-row / stat rows) and every scalar ``Load`` reading one,
  paired with the ``Tile``'s intra-CTA thread axes and the enclosing loop axes.
  ``simulate_graph`` lowers the graph through ``KERNEL_PASSES`` first (idempotent), evaluates
  each binding via :func:`lane_bank_distribution`, runs :func:`annotate_lds128` over sibling
  chains, and computes per-cell access provenance over the inner-loop sweep. Pure static — no
  GPU run. ``LdmatrixLoad`` slab reads are NOT modeled (the ldmatrix 8×8 lane protocol has its
  own fixed address pattern); the oracle covers the scalar drains and fills.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.stmt import Load

WARP_SIZE = 32
BANKS = 32


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StageBinding:
    """One ``(smem slab, body-Load reading it)`` pair plus its thread/loop context — the
    kernel-IR unit bank-conflict analysis runs per. ``slab`` is the ``Smem`` decl (extents +
    element dtype, pad already folded into the extents); ``thread_axes`` are the ``Tile``'s
    trailing intra-CTA axes (the ones an ``_gid % block_threads`` decode varies per lane)."""

    slab: object  # the Smem decl (kernel IR)
    load: Load
    thread_axes: tuple[Axis, ...]
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


def _thread_axes_of(tile) -> tuple[Axis, ...]:
    """The ``Tile``'s intra-CTA thread axes — the minimal trailing run of grid axes an
    ``_gid % blockDim`` decode varies per thread. A cooperative tile groups exactly
    ``block_threads`` consecutive cells per CTA (take trailing axes up to that product); the
    scalar tier (``block_threads is None``) has no fixed grouping — take trailing axes until a
    warp is covered (bank patterns repeat past 32 lanes)."""
    axes = list(tile.axes)
    want = tile.block_threads if tile.block_threads else WARP_SIZE
    out: list[Axis] = []
    covered = 1
    while axes and covered < want:
        ax = axes.pop()
        if not ax.extent.is_static:
            break
        out.insert(0, ax)
        covered *= ax.extent.as_static()
    return tuple(out)


def find_all_bindings(graph: Graph, stage_filter: set[str] | None = None) -> list[StageBinding]:
    """Every ``(smem slab, scalar body-Load reading it)`` pair across the graph's ``KernelOp``\\ s
    — the staged operand drains (``_a_smem`` / ``_b_smem``), the shared-row / stat-row reads —
    each with the ``Tile``'s intra-CTA thread axes and its enclosing loop axes. ``stage_filter``
    keeps only the named slabs. Graphs holding no ``KernelOp`` yield nothing (run
    ``KERNEL_PASSES`` first — :func:`simulate_graph` does)."""
    from emmy.compiler.ir.kernel import KernelOp  # noqa: PLC0415
    from emmy.compiler.ir.kernel.ir import Smem, Tile  # noqa: PLC0415
    from emmy.compiler.ir.stmt import Loop, StridedLoop  # noqa: PLC0415

    out: list[StageBinding] = []
    for node in graph.nodes.values():
        if not isinstance(node.op, KernelOp):
            continue
        slabs: dict[str, Smem] = {}
        tiles = [s for s in node.op.body if isinstance(s, Tile)]
        thread_axes = _thread_axes_of(tiles[0]) if tiles else ()

        def walk(stmts, loops: tuple[Axis, ...], *, taxes=thread_axes, slabs=slabs, kname=node.op.name):
            for s in stmts:
                if isinstance(s, Smem):
                    slabs[s.name] = s
                elif isinstance(s, Load) and s.is_scalar and s.input in slabs:
                    if stage_filter is None or s.input in stage_filter:
                        out.append(
                            StageBinding(slab=slabs[s.input], load=s, thread_axes=taxes, enclosing_loop_axes=loops, tile_op_name=kname)
                        )
                inner = (*loops, s.axis) if isinstance(s, (Loop, StridedLoop)) else loops
                for b in s.nested():
                    walk(list(b), inner)

        walk(list(node.op.body), ())
    return out


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
    extents = [ax.extent.as_static() for ax in thread_axes]
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
    elem_bytes: int = 4,
) -> BankDistribution | None:
    """Decode an intra-CTA thread id ``warp_id*WARP_SIZE + lane`` into
    per-axis ints, evaluate ``cache_index`` per lane against row-major
    strides over ``cache_extents``, and tally per-bank distinct-address
    counts. Banks are 4-byte words: ``elem_bytes`` scales element
    addresses into bank words (an ``__half`` slab packs two elements per
    word), and the distinct-address model counts distinct WORDS.

    ``cache_index`` must already be trimmed to ``len(cache_extents)``
    slab dimensions (a ring slot's row offset is warp-uniform at one
    moment, so it doesn't shift the bank pattern). Free vars in
    ``cache_index`` not in ``thread_axes`` and not in ``extra_env`` are
    zero-bound; bank distribution is invariant to additive warp-uniform
    offsets. Pass ``extra_env`` to pin a specific loop iter or block
    coord (e.g. ``{k_loop: 5}``).

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
        word = (addr * elem_bytes) // 4  # element address → 4-byte bank word
        lane_addrs.append(word)
        lane_banks.append(word % BANKS)

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


_ELEM_BYTES = {"float": 4, "__half": 2, "__nv_bfloat16": 2, "double": 8, "int": 4}


def _binding_result(b: StageBinding, k_iter: int, warp_id: int) -> BankConflictResult | None:
    """Evaluate one binding into a :class:`BankConflictResult` (or ``None`` — un-evaluable)."""
    extents = tuple(int(e) for e in b.slab.extents)
    elem_bytes = _ELEM_BYTES.get(b.slab.dtype, 4)
    extra_env: dict[str, int] = {}
    for ax in b.enclosing_loop_axes:
        hi = ax.extent.as_static() - 1 if ax.extent.is_static else 0
        extra_env[ax.name] = min(k_iter, max(hi, 0))
    dist = lane_bank_distribution(tuple(b.load.index), extents, b.thread_axes, extra_env=extra_env, warp_id=warp_id, elem_bytes=elem_bytes)
    if dist is None:
        return None
    rows, cols = (extents[0], extents[1]) if len(extents) == 2 else (1, extents[0])
    return BankConflictResult(
        stage_name=b.slab.name,
        buf=b.slab.name,
        stage_class=b.slab.dtype,
        rows=rows,
        cols=cols,
        pad=(),
        smem_bytes=prod(extents) * elem_bytes,
        load_name=b.load.names[0],
        tile_op_name=b.tile_op_name,
        index_repr=tuple(e.pretty() for e in b.load.index),
        lane_banks=dist.lane_banks,
        lane_addrs=dist.lane_addrs,
        counts=dist.counts,
        distinct_addrs=dist.distinct_addrs,
        max_way=dist.max_way,
        raw_max_way=dist.raw_max_way,
        conflict_events=dist.conflict_events,
        lds128_events=dist.conflict_events,
        avg_way=(sum(d for d in dist.distinct_addrs if d) / max(1, sum(1 for d in dist.distinct_addrs if d))),
    )


def _full_sweep(b: StageBinding, r: BankConflictResult, warp_id: int) -> None:
    """Populate the visualizer's per-cell provenance: sweep the innermost enclosing loop
    (static extents only), touch each lane's 2-D slab cell, and mark the cells of any bank with
    more than one distinct word that iteration. 2-D slabs only (the ring slot offset rides the
    row expr, so it is swept naturally)."""
    extents = tuple(int(e) for e in b.slab.extents)
    if len(extents) != 2:
        return
    elem_bytes = _ELEM_BYTES.get(b.slab.dtype, 4)
    inner = b.enclosing_loop_axes[-1] if b.enclosing_loop_axes else None
    iters = inner.extent.as_static() if inner is not None and inner.extent.is_static else 1
    outer_env = {ax.name: 0 for ax in b.enclosing_loop_axes[:-1]}
    for it in range(min(iters, 512)):  # cap the sweep — the pattern repeats
        env = dict(outer_env)
        if inner is not None:
            env[inner.name] = it
        dist = lane_bank_distribution(tuple(b.load.index), extents, b.thread_axes, extra_env=env, warp_id=warp_id, elem_bytes=elem_bytes)
        if dist is None:
            return
        conflict_banks = {bank for bank, d in enumerate(dist.distinct_addrs) if d > 1}
        for lane, word in enumerate(dist.lane_addrs):
            addr = (word * 4) // elem_bytes  # bank word → element address
            cell = (addr // extents[1], addr % extents[1])
            r.full_sweep_touched.setdefault(cell, []).append((it, lane))
            if word % BANKS in conflict_banks:
                r.full_sweep_conflict_cells.add(cell)


def simulate_graph(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
    load_filter: set[str] | None = None,
) -> list[BankConflictResult]:
    """Kernel-IR static bank-conflict analysis (the visualizer's oracle): lower ``graph`` through
    ``KERNEL_PASSES`` (idempotent — re-applies any unmet tile lowering), find every smem-slab
    scalar Load (:func:`find_all_bindings`), evaluate each via :func:`lane_bank_distribution` at
    loop iter ``k_iter`` / ``warp_id``, annotate LDS.128 vector chains, and populate the per-cell
    sweep provenance. Pure static — no GPU run."""
    from emmy.compiler.pipeline import KERNEL_PASSES, Pipeline  # noqa: PLC0415

    lowered = Pipeline.build(KERNEL_PASSES).run(graph)
    results: list[BankConflictResult] = []
    for b in find_all_bindings(lowered, stage_filter):
        if load_filter is not None and b.load.names[0] not in load_filter:
            continue
        r = _binding_result(b, k_iter, warp_id)
        if r is None:
            continue
        _full_sweep(b, r, warp_id)
        results.append(r)
    annotate_lds128(results)
    return results
