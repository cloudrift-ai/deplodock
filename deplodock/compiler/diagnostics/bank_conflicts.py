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
from dataclasses import dataclass

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
    index_repr: tuple[str, ...]  # the cache-relative index of the simulated Load
    lane_banks: list[int]  # length WARP_SIZE
    lane_addrs: list[int]  # length WARP_SIZE — pre-mod linear smem address
    counts: list[int]  # length BANKS — lanes per bank
    distinct_addrs: list[int]  # length BANKS — distinct addresses per bank
    max_way: int  # worst broadcast-corrected = max(distinct_addrs)
    raw_max_way: int  # max(counts) — upper bound w/o broadcast
    conflict_events: int  # sum(distinct_addrs[b] - 1 for b with hits)
    avg_way: float
    enclosing_axes: tuple[str, ...] = ()


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

    # Strides over alloc_extents (cache extents + per-axis pad).
    alloc = list(stage.alloc_extents)
    strides = [1] * len(alloc)
    for i in range(len(alloc) - 2, -1, -1):
        strides[i] = strides[i + 1] * alloc[i + 1]

    # Base env: zero out block axes and outer enclosing loops; pin the
    # innermost loop axis to ``k_iter``.
    base_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
    enc = list(binding.enclosing_loop_axes)
    if enc:
        for ax in enc[:-1]:
            base_env.setdefault(ax.name, 0)
        base_env[enc[-1].name] = k_iter

    lane_banks: list[int] = []
    lane_addrs: list[int] = []
    for lane in range(WARP_SIZE):
        env = dict(base_env)
        env.update(thread_axis_env(tile.thread_axes, warp_id * WARP_SIZE + lane))
        try:
            coords = [int(idx.eval(env)) for idx in cache_idx]
        except (KeyError, TypeError):
            return None
        addr = sum(c * s for c, s in zip(coords, strides, strict=True))
        lane_addrs.append(addr)
        lane_banks.append(addr % BANKS)

    counts = [0] * BANKS
    addrs_per_bank: list[set[int]] = [set() for _ in range(BANKS)]
    for lane, (b, a) in enumerate(zip(lane_banks, lane_addrs, strict=True)):
        del lane  # unused
        counts[b] += 1
        addrs_per_bank[b].add(a)
    distinct_addrs = [len(s) for s in addrs_per_bank]
    raw_max_way = max(counts)
    max_way = max(distinct_addrs) if distinct_addrs else 0
    conflict_events = sum(d - 1 for d in distinct_addrs if d > 0)
    nz = [c for c in counts if c > 0]
    avg = sum(nz) / len(nz) if nz else 0.0

    return BankConflictResult(
        stage_name=stage.name,
        buf=stage.buf,
        stage_class=type(stage).__name__,
        rows=int(stage.axes[0].extent),
        cols=int(stage.axes[1].extent) if len(stage.axes) > 1 else 1,
        pad=tuple(int(p) for p in stage.pad),
        smem_bytes=stage.smem_bytes,
        index_repr=tuple(e.pretty() for e in cache_idx),
        lane_banks=lane_banks,
        lane_addrs=lane_addrs,
        counts=counts,
        distinct_addrs=distinct_addrs,
        max_way=max_way,
        raw_max_way=raw_max_way,
        conflict_events=conflict_events,
        avg_way=avg,
        enclosing_axes=tuple(ax.name for ax in binding.enclosing_loop_axes),
    )


def simulate_graph(
    graph: Graph,
    stage_filter: set[str] | None = None,
    k_iter: int = 0,
    warp_id: int = 0,
) -> list[BankConflictResult]:
    """Convenience: bindings + simulate, dropping rank-mismatched probes."""
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
    return out
