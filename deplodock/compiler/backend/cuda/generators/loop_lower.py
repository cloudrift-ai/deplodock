"""Lower FusedRegionOp + TileAnalysis to LoopIR.

``lower_generic()`` is the single entry point: it reads a ``Schedule`` and
emits LoopIR through five phases (grid → accumulators → reductions →
epilogue → write).  No pattern-matching — all structural decisions live
in the Schedule.

Legacy per-pattern functions are kept temporarily for TMA and
contraction+softmax fallback paths.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, _needed_by
from deplodock.compiler.backend.loop_ir import (
    Accumulate,
    Alloc,
    Barrier,
    Guard,
    Let,
    Load,
    LoopBuiltin,
    LoopExpr,
    LoopFuncCall,
    LoopLiteral,
    LoopNest,
    LoopProgram,
    LoopTernary,
    LoopVar,
    OpCall,
    ParallelAxis,
    RegAccess,
    SetVar,
    Store,
    WarpReduce,
    WarpShuffleXor,
)
from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReshapeOp, TransposeOp

if TYPE_CHECKING:
    from deplodock.compiler.backend.cuda.schedule import Schedule


# ---------------------------------------------------------------------------
# Generic lowering — single entry point driven by Schedule
# ---------------------------------------------------------------------------


def lower_generic(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    schedule: Schedule,
) -> LoopProgram:
    """Lower a FusedRegionOp to LoopIR using a Schedule.

    Five phases, each parameterized by the Schedule:
    1. Grid setup
    2. Accumulator allocation
    3. Reduction loops
    4. Epilogue ops
    5. Write outputs
    """
    params = _build_params(region)
    params.extend(schedule.dim_params)

    body: list = []

    # Phase 1: Grid setup
    body.extend(_emit_grid(schedule, analysis))

    # Phase 2: Accumulators
    body.extend(_emit_accumulators(schedule, analysis))

    # Phase 3: Reduction loops (0 for pointwise, 1+ for reduce/contraction)
    body.extend(_emit_reductions(schedule, analysis, region, shapes))

    # Phase 4: Epilogue
    body.extend(_emit_epilogue(schedule, analysis, region, shapes))

    # Phase 5: Write outputs
    body.extend(_emit_write(schedule, analysis, region))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=schedule.grid.block_size,
        tile_m=schedule.tile_m,
        tile_n=schedule.tile_n,
    )


# ---------------------------------------------------------------------------
# Phase 1: Grid setup
# ---------------------------------------------------------------------------


def _emit_grid(schedule: Schedule, analysis: TileAnalysis) -> list:
    grid = schedule.grid
    if grid.type == "1d":
        axis_name = "i" if schedule.accum.shape is None else "row"
        return [ParallelAxis(axis_name, "blockIdx.x", grid.bound)]
    if grid.type == "2d_swizzle":
        return _cta_swizzle_grid(
            schedule.thread_m or 8,
            schedule.thread_n or 4,
            schedule.tile_m or 64,
            schedule.tile_n or 128,
            schedule.is_batched,
        )
    # 2d_standard (smem) — handled by smem-specific path
    return []


# ---------------------------------------------------------------------------
# Phase 2: Accumulators
# ---------------------------------------------------------------------------


def _emit_accumulators(schedule: Schedule, analysis: TileAnalysis) -> list:
    ops: list = []
    accum = schedule.accum

    if accum.shape is None:
        # Pointwise: no accumulators
        return ops

    if accum.shape == ():
        # Scalar accumulators (one per reduce op)
        for node_id, op, _ in analysis.op_phases.reduces:
            acc_name = f"acc_{_safe(node_id)}"
            ops.append(Alloc(acc_name, accum.dtype, None, "reg", _reduce_init_expr(op.fn)))
        return ops

    # 2D register tile (contraction)
    ops.append(Alloc("c", accum.dtype, accum.shape, "reg", LoopLiteral(0.0)))

    # Batch pointer aliases for contraction
    if schedule.is_batched and analysis.contraction_a:
        A = LoopVar(_safe(analysis.contraction_a))  # noqa: N806
        B = LoopVar(_safe(analysis.contraction_b))  # noqa: N806
        batch, M, K, N = LoopVar("batch"), LoopVar("M"), LoopVar("K"), LoopVar("N")  # noqa: N806
        ops.append(Let("Ab", A + batch * (M * K), dtype="const float*"))
        ops.append(Let("Bb", B + batch * (K * N), dtype="const float*"))

    return ops


# ---------------------------------------------------------------------------
# Phase 3: Reduction loops
# ---------------------------------------------------------------------------


def _emit_reductions(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    accum = schedule.accum

    if accum.shape is None:
        # Pointwise: no reduction, just inline all ops inside a guard
        return _emit_pointwise_body(analysis, region, shapes, schedule)

    if accum.shape == ():
        # Scalar reduction (row_reduce / reduce_broadcast / multi-reduce)
        return _emit_scalar_reductions(schedule, analysis, region, shapes)

    # 2D register tile contraction: K-loop with FMA
    return _emit_contraction_k_loop(schedule, analysis, region, shapes)


def _emit_pointwise_body(
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
    schedule: Schedule,
) -> list:
    """Pointwise: load inputs, emit all ops, store outputs inside a guard."""
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    guard_body: list = []
    var_map: dict[str, LoopExpr] = {}
    for inp in region.input_names:
        idx = _input_loop_expr(inp, analysis, "i", out_size)
        load_name = f"v_{_safe(inp)}"
        guard_body.append(Load(load_name, _safe(inp), idx, "global"))
        var_map[inp] = LoopVar(load_name)

    _emit_loop_ops(region.region_ops, var_map, "v_", guard_body)

    for out_id in region.output_names:
        val = var_map.get(out_id, LoopLiteral(0.0))
        guard_body.append(Store(_safe(out_id), LoopVar("i"), val, "global"))

    return [Guard(LoopVar("i").lt(LoopVar("n")), guard_body)]


def _emit_scalar_reductions(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit scalar reduction loops (single or multi-reduce)."""
    phases = analysis.op_phases
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))
    has_multi_reduce = len(phases.reduces) > 1

    guarded: list = []

    # Accumulator allocs were already emitted in phase 2; build var tracking.
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        acc_name = f"acc_{_safe(node_id)}"
        reduce_vars[node_id] = (acc_name, op.fn)

    var_map: dict[str, LoopExpr] = {}

    if has_multi_reduce:
        # Multi-reduce: one tile loop per reduce pass (e.g., softmax max → sum)
        for ri, (node_id, _op, input_ids) in enumerate(phases.reduces):
            acc_name, fn = reduce_vars[node_id]
            pass_body: list = []

            pass_var_map: dict[str, LoopExpr] = {}
            for inp in region.input_names:
                idx = _input_loop_expr(inp, analysis, "j", out_size)
                load_name = f"r{ri}ld_{_safe(inp)}"
                pass_body.append(Load(load_name, _safe(inp), idx, "global"))
                pass_var_map[inp] = LoopVar(load_name)

            for prev_nid, (prev_acc, _prev_fn) in reduce_vars.items():
                pass_var_map[prev_nid] = LoopVar(prev_acc)

            all_ops_this_pass = list(phases.inter_reduce[ri - 1]) if ri > 0 else []
            needed = _needed_by(all_ops_this_pass + [(node_id, _op, input_ids)])
            for pid, pop, pinp in phases.prologue:
                if isinstance(pop, ElementwiseOp) and pid in needed:
                    a = pass_var_map.get(pinp[0], LoopLiteral(0.0)) if pinp else LoopLiteral(0.0)
                    b = pass_var_map.get(pinp[1], LoopLiteral(0.0)) if len(pinp) > 1 else LoopLiteral(0.0)
                    var_name = f"r{ri}p_{_safe(pid)}"
                    pass_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                    pass_var_map[pid] = LoopVar(var_name)

            if ri > 0 and phases.inter_reduce:
                _emit_loop_ops(phases.inter_reduce[ri - 1], pass_var_map, f"r{ri}_", pass_body)

            val = pass_var_map.get(input_ids[0], LoopLiteral(0.0))
            pass_body.append(Accumulate(acc_name, fn, val))

            guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), pass_body))
            guarded.append(WarpReduce(acc_name, fn))
            var_map[node_id] = LoopVar(acc_name)
    else:
        # Single reduce: one tile loop
        loop_body: list = []
        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"ld_{_safe(inp)}"
            loop_body.append(Load(load_name, _safe(inp), idx, "global"))
            var_map[inp] = LoopVar(load_name)

        _emit_loop_ops(phases.prologue, var_map, "p_", loop_body)

        for node_id, _op, input_ids in phases.reduces:
            acc_name, fn = reduce_vars[node_id]
            val = var_map.get(input_ids[0], LoopLiteral(0.0))
            loop_body.append(Accumulate(acc_name, fn, val))

        guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), loop_body))

        for node_id, (acc_name, fn) in reduce_vars.items():
            guarded.append(WarpReduce(acc_name, fn))
            var_map[node_id] = LoopVar(acc_name)

    return [Guard(LoopVar("row").lt(LoopVar("rows")), guarded)]


def _emit_contraction_k_loop(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit the K-loop for contraction (naive global-load strategy)."""
    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    a_name = _safe(analysis.contraction_a)
    b_name = _safe(analysis.contraction_b)
    a_src = "Ab" if schedule.is_batched else a_name
    b_src = "Bb" if schedule.is_batched else b_name

    bn, tc, bm, tr = LoopVar("bn"), LoopVar("tc"), LoopVar("bm"), LoopVar("tr")
    k, K, N, M = LoopVar("k"), LoopVar("K"), LoopVar("N"), LoopVar("M")  # noqa: N806

    k_body: list = []
    for c in range(thread_n):
        col = bn + tc + c
        k_body.append(Load(f"b{c}", b_src, k * N + col, "global", guard=col.lt(N)))

    for r in range(thread_m):
        row = bm + tr + r
        k_body.append(Load(f"a{r}", a_src, row * K + k, "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accumulate(f"c{r}{c}", "sum", LoopVar(f"a{r}") * LoopVar(f"b{c}")))

    return [LoopNest("k", LoopLiteral(0, "int"), K, None, k_body)]


# ---------------------------------------------------------------------------
# Phase 4: Epilogue
# ---------------------------------------------------------------------------


def _emit_epilogue(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    accum = schedule.accum
    phases = analysis.op_phases

    if accum.shape is None:
        # Pointwise: epilogue was inlined in the pointwise body
        return []

    if accum.shape != ():
        # Contraction register tile epilogue
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        tile_n = schedule.tile_n or 128
        # Multi-reduce on register tile (contraction+softmax)
        if len(phases.reduces) > 1 and analysis.cols <= tile_n:
            return _contraction_softmax_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n)
        return _contraction_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n)

    # Scalar reduction epilogue
    return _emit_scalar_epilogue(schedule, analysis, region, shapes)


def _emit_scalar_epilogue(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit epilogue for scalar reductions (inside the row guard)."""
    phases = analysis.op_phases
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    if not phases.epilogue and not schedule.epilogue_per_element:
        return []

    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        reduce_vars[node_id] = (f"acc_{_safe(node_id)}", op.fn)

    var_map: dict[str, LoopExpr] = {}
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = LoopVar(acc_name)

    has_multi_reduce = len(phases.reduces) > 1

    if has_multi_reduce or schedule.epilogue_per_element:
        # Per-element epilogue loop
        epi_body: list = []
        epi_var_map: dict[str, LoopExpr] = dict(var_map)

        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"epld_{_safe(inp)}"
            epi_body.append(Load(load_name, _safe(inp), idx, "global"))
            epi_var_map[inp] = LoopVar(load_name)

        # Re-compute prologue + inter_reduce ops
        all_inter_ops = [op for group in phases.inter_reduce for op in group] if has_multi_reduce else []
        recompute_ops = phases.prologue + all_inter_ops if has_multi_reduce else phases.prologue
        needed = _needed_by(phases.epilogue)
        for pid, pop, pinp in recompute_ops:
            if isinstance(pop, ElementwiseOp) and (pid in needed or has_multi_reduce):
                a = epi_var_map.get(pinp[0], LoopLiteral(0.0)) if pinp else LoopLiteral(0.0)
                b = epi_var_map.get(pinp[1], LoopLiteral(0.0)) if len(pinp) > 1 else LoopLiteral(0.0)
                var_name = f"ep_{_safe(pid)}"
                epi_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                epi_var_map[pid] = LoopVar(var_name)

        _emit_loop_ops(phases.epilogue, epi_var_map, "e_", epi_body)

        for out_id in region.output_names:
            val = epi_var_map.get(out_id, LoopLiteral(0.0))
            idx = LoopVar("row") * LoopVar("cols") + LoopVar("j")
            epi_body.append(Store(_safe(out_id), idx, val, "global"))

        return [LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), epi_body)]

    # Scalar epilogue (no per-element loop needed)
    ops: list = []
    _emit_loop_ops(phases.epilogue, var_map, "e_", ops)
    return ops


# ---------------------------------------------------------------------------
# Phase 5: Write outputs
# ---------------------------------------------------------------------------


def _emit_write(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
) -> list:
    accum = schedule.accum

    if accum.shape is None:
        # Pointwise: writes were already emitted in the guard body
        return []

    if accum.shape != ():
        # Contraction register tile write
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        out_id = region.output_names[0]
        return _contraction_write_ops(out_id, thread_m, thread_n, schedule.is_batched, schedule.k_splits)

    # Scalar reduction write
    phases = analysis.op_phases
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        reduce_vars[node_id] = (f"acc_{_safe(node_id)}", op.fn)

    var_map: dict[str, LoopExpr] = {}
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = LoopVar(acc_name)

    if schedule.epilogue_per_element or len(phases.reduces) > 1:
        # Per-element writes were already emitted in the epilogue loop
        if phases.epilogue or schedule.epilogue_per_element:
            return []
        # No epilogue — write last reduce result (thread 0 only)
        ops: list = []
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            ops.append(
                Guard(
                    LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )
        return ops

    # Single reduce without per-element epilogue
    if phases.epilogue:
        # Scalar epilogue was emitted in phase 4; thread 0 writes
        ops = []
        # Re-derive var_map with epilogue results
        _emit_loop_ops(phases.epilogue, var_map, "e_", [])  # just update var_map
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            ops.append(
                Guard(
                    LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )
        return ops

    # No epilogue — write last reduce result (thread 0)
    ops = []
    for out_id in region.output_names:
        val = var_map.get(out_id, LoopLiteral(0.0))
        ops.append(
            Guard(
                LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                [Store(_safe(out_id), LoopVar("row"), val, "global")],
            )
        )
    return ops


# ---------------------------------------------------------------------------
# Legacy dispatch (kept for backward compat during transition)
# ---------------------------------------------------------------------------


def lower_to_loop_ir(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> LoopProgram:
    """Lower a FusedRegionOp to LoopIR.

    Routes through ``lower_generic()`` for all patterns except TMA and
    contraction+softmax which use legacy escape hatches.
    """
    from deplodock.compiler.backend.cuda.schedule import build_schedule

    # Legacy: TMA requires inline asm
    if strategy == "tma_db" and analysis.pattern == "contraction":
        return _lower_contraction_tma(region, name, shapes, analysis, hints or {})

    # Legacy: smem strategy still uses _lower_contraction_smem
    if strategy == "smem" and analysis.pattern == "contraction":
        return _lower_contraction_smem(region, name, shapes, analysis, hints or {})

    schedule = build_schedule(analysis, strategy, hints or {})
    return lower_generic(region, name, shapes, analysis, schedule)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _reduce_init_expr(fn: str) -> LoopLiteral:
    """Initial value for a reduction accumulator."""
    if fn == "sum":
        return LoopLiteral(0.0)
    if fn == "max":
        return LoopLiteral(-1e30)
    return LoopLiteral(0.0)


def _input_loop_expr(inp: str, analysis: TileAnalysis, idx_var: str, out_size: int = 0) -> LoopExpr:
    """Build the index expression for reading an input tensor."""
    acc = analysis.input_access[inp]
    row, cols, j = LoopVar("row"), LoopVar("cols"), LoopVar(idx_var)

    if analysis.pattern == "pointwise":
        if acc.is_scalar:
            return LoopLiteral(0, "int")
        if acc.size < out_size:
            return LoopVar("i") % acc.size
        return LoopVar("i")

    if acc.is_2d:
        return row * cols + j
    if acc.is_per_row:
        return row
    if acc.is_row_vector:
        return j
    return LoopLiteral(0, "int")


def _emit_loop_ops(
    ops: list,
    var_map: dict[str, LoopExpr],
    prefix: str,
    body: list,
) -> None:
    """Walk ops and emit Let(name, OpCall(...)) nodes, updating var_map."""
    for node_id, op, input_ids in ops:
        if isinstance(op, (ReshapeOp, TransposeOp)):
            if input_ids and input_ids[0] in var_map:
                var_map[node_id] = var_map[input_ids[0]]
            continue

        if isinstance(op, ElementwiseOp):
            a = var_map.get(input_ids[0], LoopLiteral(0.0)) if input_ids else LoopLiteral(0.0)
            b = var_map.get(input_ids[1], LoopLiteral(0.0)) if len(input_ids) > 1 else LoopLiteral(0.0)
            var_name = f"{prefix}{_safe(node_id)}"
            args = [a, b] if op.info.arity == 2 and len(input_ids) > 1 else [a]
            body.append(Let(var_name, OpCall(op.fn, args)))
            var_map[node_id] = LoopVar(var_name)


def _build_params(region: FusedRegionOp) -> list[tuple[str, str]]:
    """Build kernel params, deduplicating buffers that are both input and output.

    When a buffer appears in both input_names and output_names (in-place op),
    it is emitted once as ``float*`` (read-write), not duplicated as both
    ``const float*`` and ``float*``.
    """
    params: list[tuple[str, str]] = []
    output_set = set(region.output_names)
    for inp in region.input_names:
        if inp in output_set:
            continue  # will be added as read-write output
        params.append(("const float* __restrict__", _safe(inp)))
    for out in region.output_names:
        params.append(("float* __restrict__", _safe(out)))
    return params


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def _lower_pointwise(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
) -> LoopProgram:
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    # Params
    params = _build_params(region)
    params.append(("int", "n"))

    # Body
    body: list = []
    body.append(ParallelAxis("i", "blockIdx.x", "n"))

    # Guard: if (i >= n) return
    guard_body: list = []

    # Build var_map with input loads
    var_map: dict[str, LoopExpr] = {}
    for inp in region.input_names:
        idx = _input_loop_expr(inp, analysis, "i", out_size)
        load_name = f"v_{_safe(inp)}"
        guard_body.append(Load(load_name, _safe(inp), idx, "global"))
        var_map[inp] = LoopVar(load_name)

    # Emit all ops
    _emit_loop_ops(region.region_ops, var_map, "v_", guard_body)

    # Store outputs
    for out_id in region.output_names:
        val = var_map.get(out_id, LoopLiteral(0.0))
        guard_body.append(Store(_safe(out_id), LoopVar("i"), val, "global"))

    body.append(Guard(LoopVar("i").lt(LoopVar("n")), guard_body))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(256, 1, 1),
    )


# ---------------------------------------------------------------------------
# Single reduce (row_reduce without multi-reduce)
# ---------------------------------------------------------------------------


def _lower_single_reduce(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
) -> LoopProgram:
    phases = analysis.op_phases
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    # Params
    params = _build_params(region)
    params.append(("int", "rows"))
    params.append(("int", "cols"))

    body: list = []

    # Parallel over rows
    body.append(ParallelAxis("row", "blockIdx.x", "rows"))

    # Guard: if (row >= rows) return
    guarded: list = []

    # Accumulator declarations
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        acc_name = f"acc_{_safe(node_id)}"
        guarded.append(Alloc(acc_name, "float", None, "reg", _reduce_init_expr(op.fn)))
        reduce_vars[node_id] = (acc_name, op.fn)

    # Build var_map for inputs
    var_map: dict[str, LoopExpr] = {}
    for inp in region.input_names:
        var_map[inp] = LoopVar(inp)  # placeholder, actual load inside loop

    # Tile loop over columns
    loop_body: list = []

    # Load inputs inside loop
    input_loads: dict[str, str] = {}
    for inp in region.input_names:
        idx = _input_loop_expr(inp, analysis, "j", out_size)
        load_name = f"ld_{_safe(inp)}"
        loop_body.append(Load(load_name, _safe(inp), idx, "global"))
        var_map[inp] = LoopVar(load_name)
        input_loads[inp] = load_name

    # Prologue ops
    _emit_loop_ops(phases.prologue, var_map, "p_", loop_body)

    # Accumulation
    for node_id, _op, input_ids in phases.reduces:
        acc_name, fn = reduce_vars[node_id]
        val = var_map.get(input_ids[0], LoopLiteral(0.0))
        loop_body.append(Accumulate(acc_name, fn, val))

    guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), loop_body))

    # Warp reduce
    for node_id, (acc_name, fn) in reduce_vars.items():
        guarded.append(WarpReduce(acc_name, fn))
        var_map[node_id] = LoopVar(acc_name)

    # Epilogue
    epilogue_ops = phases.epilogue
    if epilogue_ops:
        if analysis.epilogue_needs_per_element:
            # Second per-element pass
            epi_body: list = []
            epi_var_map: dict[str, LoopExpr] = dict(var_map)
            for inp in region.input_names:
                idx = _input_loop_expr(inp, analysis, "j", out_size)
                load_name = f"eld_{_safe(inp)}"
                epi_body.append(Load(load_name, _safe(inp), idx, "global"))
                epi_var_map[inp] = LoopVar(load_name)

            # Re-compute prologue ops needed by epilogue
            needed = _needed_by(epilogue_ops)
            for node_id, op, input_ids in phases.prologue:
                if isinstance(op, ElementwiseOp) and node_id in needed:
                    a = epi_var_map.get(input_ids[0], LoopLiteral(0.0)) if input_ids else LoopLiteral(0.0)
                    b = epi_var_map.get(input_ids[1], LoopLiteral(0.0)) if len(input_ids) > 1 else LoopLiteral(0.0)
                    var_name = f"p_{_safe(node_id)}"
                    epi_body.append(Let(var_name, OpCall(op.fn, [a] if len(input_ids) == 1 else [a, b])))
                    epi_var_map[node_id] = LoopVar(var_name)

            _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

            for out_id in region.output_names:
                val = epi_var_map.get(out_id, LoopLiteral(0.0))
                idx = LoopVar("row") * LoopVar("cols") + LoopVar("j")
                epi_body.append(Store(_safe(out_id), idx, val, "global"))

            guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), epi_body))
        else:
            # Scalar epilogue (thread 0 only)
            _emit_loop_ops(epilogue_ops, var_map, "e_", guarded)
            for out_id in region.output_names:
                val = var_map.get(out_id, LoopLiteral(0.0))
                guarded.append(
                    Guard(
                        LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                        [Store(_safe(out_id), LoopVar("row"), val, "global")],
                    )
                )
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            guarded.append(
                Guard(
                    LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )

    body.append(Guard(LoopVar("row").lt(LoopVar("rows")), guarded))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(256, 1, 1),
    )


# ---------------------------------------------------------------------------
# Multi-reduce (softmax, etc.)
# ---------------------------------------------------------------------------


def _lower_multi_reduce(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
) -> LoopProgram:
    phases = analysis.op_phases
    out_shape = shapes.get(region.output_names[0], (1,))
    out_size = math.prod(d for d in out_shape if isinstance(d, int))

    # Params
    params = _build_params(region)
    params.append(("int", "rows"))
    params.append(("int", "cols"))

    body: list = []
    body.append(ParallelAxis("row", "blockIdx.x", "rows"))

    guarded: list = []

    # Accumulator declarations
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        acc_name = f"acc_{_safe(node_id)}"
        guarded.append(Alloc(acc_name, "float", None, "reg", _reduce_init_expr(op.fn)))
        reduce_vars[node_id] = (acc_name, op.fn)

    var_map: dict[str, LoopExpr] = {}

    # One tile loop per reduce pass
    for ri, (node_id, _op, input_ids) in enumerate(phases.reduces):
        acc_name, fn = reduce_vars[node_id]
        pass_body: list = []

        # Re-map inputs for this pass
        pass_var_map: dict[str, LoopExpr] = {}
        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"r{ri}ld_{_safe(inp)}"
            pass_body.append(Load(load_name, _safe(inp), idx, "global"))
            pass_var_map[inp] = LoopVar(load_name)

        # Previous reduce results are available as accumulators
        for prev_nid, (prev_acc, _prev_fn) in reduce_vars.items():
            pass_var_map[prev_nid] = LoopVar(prev_acc)

        # Re-compute prologue ops needed by this reduce or inter_reduce ops
        all_ops_this_pass = list(phases.inter_reduce[ri - 1]) if ri > 0 else []
        needed = _needed_by(all_ops_this_pass + [(node_id, _op, input_ids)])
        for pid, pop, pinp in phases.prologue:
            if isinstance(pop, ElementwiseOp) and pid in needed:
                a = pass_var_map.get(pinp[0], LoopLiteral(0.0)) if pinp else LoopLiteral(0.0)
                b = pass_var_map.get(pinp[1], LoopLiteral(0.0)) if len(pinp) > 1 else LoopLiteral(0.0)
                var_name = f"r{ri}p_{_safe(pid)}"
                pass_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                pass_var_map[pid] = LoopVar(var_name)

        # Apply inter_reduce ops (e.g. sub, exp between max and sum)
        if ri > 0 and phases.inter_reduce:
            _emit_loop_ops(phases.inter_reduce[ri - 1], pass_var_map, f"r{ri}_", pass_body)

        # Accumulate
        val = pass_var_map.get(input_ids[0], LoopLiteral(0.0))
        pass_body.append(Accumulate(acc_name, fn, val))

        guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), pass_body))

        # Warp shuffle after each pass
        guarded.append(WarpReduce(acc_name, fn))
        var_map[node_id] = LoopVar(acc_name)

    # Final epilogue pass
    epilogue_ops = phases.epilogue
    if epilogue_ops or analysis.epilogue_needs_per_element:
        epi_body: list = []
        epi_var_map: dict[str, LoopExpr] = dict(var_map)

        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"epld_{_safe(inp)}"
            epi_body.append(Load(load_name, _safe(inp), idx, "global"))
            epi_var_map[inp] = LoopVar(load_name)

        # Re-compute ALL prologue + inter_reduce ops
        all_inter_ops = [op for group in phases.inter_reduce for op in group]
        for pid, pop, pinp in phases.prologue + all_inter_ops:
            if isinstance(pop, ElementwiseOp):
                a = epi_var_map.get(pinp[0], LoopLiteral(0.0)) if pinp else LoopLiteral(0.0)
                b = epi_var_map.get(pinp[1], LoopLiteral(0.0)) if len(pinp) > 1 else LoopLiteral(0.0)
                var_name = f"ep_{_safe(pid)}"
                epi_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                epi_var_map[pid] = LoopVar(var_name)

        _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

        for out_id in region.output_names:
            val = epi_var_map.get(out_id, LoopLiteral(0.0))
            idx = LoopVar("row") * LoopVar("cols") + LoopVar("j")
            epi_body.append(Store(_safe(out_id), idx, val, "global"))

        guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), epi_body))
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            guarded.append(
                Guard(
                    LoopBuiltin("threadIdx.x").eq(LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )

    body.append(Guard(LoopVar("row").lt(LoopVar("rows")), guarded))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(256, 1, 1),
    )


# ---------------------------------------------------------------------------
# Contraction helpers
# ---------------------------------------------------------------------------


def _contraction_dims(analysis: TileAnalysis, hints: dict) -> dict:
    """Compute tile dimensions and strategy params for a contraction."""
    tx, ty = 32, 8
    thread_m = int(hints.get("thread_m", 8))
    thread_n = 4
    tile_m = ty * thread_m
    tile_n = tx * thread_n  # 128
    return {
        "tx": tx,
        "ty": ty,
        "thread_m": thread_m,
        "thread_n": thread_n,
        "tile_m": tile_m,
        "tile_n": tile_n,
    }


def _cta_swizzle_grid(
    thread_m: int,
    thread_n: int,
    tile_m: int,
    tile_n: int,
    is_batched: bool,
) -> list:
    """Generate LoopIR ops for CTA-swizzle grid setup.

    Maps a linearized 1D grid (blockIdx.x) to 2D tile coordinates (bm, bn)
    using an 8-way swizzle for improved memory access patterns.
    Produces named variables: tr, tc, bm, bn (and batch if batched).
    """
    ops: list = []
    I = "int"  # noqa: E741
    N, M = LoopVar("N"), LoopVar("M")  # noqa: N806
    bidy, bidx = LoopBuiltin("blockIdx.y"), LoopBuiltin("blockIdx.x")
    tidy, tidx = LoopBuiltin("threadIdx.y"), LoopBuiltin("threadIdx.x")
    gdimx = LoopBuiltin("gridDim.x")
    SWIZ = 8

    if is_batched:
        ops.append(Let("batch", LoopBuiltin("blockIdx.z"), dtype=I))

    ops.append(Let("tr", tidy * thread_m, dtype=I))
    ops.append(Let("tc", tidx * thread_n, dtype=I))
    ops.append(Let("ntx", (N + (tile_n - 1)) / tile_n, dtype=I))
    ops.append(Let("nty", (M + (tile_m - 1)) / tile_m, dtype=I))

    ntx, nty = LoopVar("ntx"), LoopVar("nty")
    pid, grp, rem = LoopVar("pid"), LoopVar("grp"), LoopVar("rem")
    by_s, bx_s = LoopVar("by_s"), LoopVar("bx_s")

    ops.append(Let("pid", bidx + bidy * gdimx, dtype=I))
    ops.append(Let("grp", pid / (ntx * SWIZ), dtype=I))
    ops.append(Let("rem", pid % (ntx * SWIZ), dtype=I))
    ops.append(Let("by_s", grp * SWIZ + rem % SWIZ, dtype=I))
    ops.append(Let("bx_s", rem / SWIZ, dtype=I))

    ops.append(Guard(by_s.ge(nty).or_(bx_s.ge(ntx)), []))

    ops.append(Let("bm", by_s * tile_m, dtype=I))
    ops.append(Let("bn", bx_s * tile_n, dtype=I))

    return ops


def _contraction_epilogue_ops(
    phases,
    shapes: dict[str, tuple],
    input_names: list[str],
    thread_m: int,
    thread_n: int,
) -> list:
    """Generate LoopIR ops for contraction epilogue (bias, activation, etc.).

    Operates on the register array ``c[r][c]`` in-place.  Delegates to
    ``_apply_ops_on_register_tile`` which handles any elementwise op.
    """
    if not phases.epilogue:
        return []

    prev_id = phases.reduces[0][0]
    input_set = set(input_names)
    return _apply_ops_on_register_tile(phases.epilogue, prev_id, shapes, input_set, thread_m, thread_n, "ep")


def _apply_ops_on_register_tile(
    op_list: list,
    prev_id: str,
    shapes: dict[str, tuple],
    input_set: set[str],
    thread_m: int,
    thread_n: int,
    prefix: str,
    extra_vars: dict[str, str] | None = None,
) -> list:
    """Apply a list of elementwise ops to the register tile c[r][c].

    Walks the op chain generically — any elementwise op is supported.
    Binary ops where the "other" input is an external buffer get a Load;
    binary ops where the other is a per-row variable (e.g. rmax, rsum)
    use ``extra_vars`` mapping.  Unary ops apply directly.
    """
    ops: list = []
    extra = extra_vars or {}

    for nid, ir_op, inputs in op_list:
        if not isinstance(ir_op, ElementwiseOp):
            continue
        fn = ir_op.fn

        # Find the "other" input (not the accumulator chain) for binary ops.
        other = None
        if ir_op.info.arity == 2 and len(inputs) == 2:
            for inp in inputs:
                if inp != prev_id:
                    other = inp
                    break
        if other is not None and other not in input_set and other not in extra:
            other = None

        for r in range(thread_m):
            for c in range(thread_n):
                acc = RegAccess("c", [r, c])
                dst = f"c{r}{c}"

                if other is not None and other in extra:
                    # Per-row variable (rmax, rsum)
                    other_val = LoopVar(extra[other].format(r=r))
                    if inputs[0] == prev_id:
                        ops.append(SetVar(dst, OpCall(fn, [acc, other_val])))
                    else:
                        ops.append(SetVar(dst, OpCall(fn, [other_val, acc])))
                elif other is not None:
                    # External buffer input
                    safe = _safe(other)
                    other_shape = shapes.get(other, ())
                    other_size = math.prod(d for d in other_shape if isinstance(d, int))
                    if other_size <= 1:
                        idx = LoopLiteral(0, "int")
                    elif len(other_shape) <= 1:
                        idx = LoopVar("bn") + LoopVar("tc") + c
                    else:
                        row_e = LoopVar("bm") + LoopVar("tr") + r
                        col_e = LoopVar("bn") + LoopVar("tc") + c
                        idx = row_e * LoopVar("N") + col_e
                    ld_name = f"_{prefix}_{safe}_{r}_{c}"
                    ops.append(Load(ld_name, safe, idx, "global"))
                    if inputs[0] == prev_id:
                        ops.append(SetVar(dst, OpCall(fn, [acc, LoopVar(ld_name)])))
                    else:
                        ops.append(SetVar(dst, OpCall(fn, [LoopVar(ld_name), acc])))
                else:
                    # Unary op on accumulator
                    ops.append(SetVar(dst, OpCall(fn, [acc])))

        prev_id = nid

    return ops


def _contraction_softmax_epilogue_ops(
    phases,
    shapes: dict[str, tuple],
    input_names: list[str],
    thread_m: int,
    thread_n: int,
) -> list:
    """Generate LoopIR ops for fused contraction+softmax epilogue.

    After the K-loop, accumulators c[r][c] hold matmul scores. This emits:
    1. Apply inter_reduce[0] ops on register tile
    2. Row max via WarpShuffleXor across threadIdx.x
    3. Apply inter_reduce[1] ops + row sum via WarpShuffleXor
    4. Apply epilogue ops on register tile

    Only valid when N <= tile_n (all columns in one CTA).
    """
    ops: list = []
    input_set = set(input_names)

    # Phase 0: apply inter_reduce[0] ops (e.g. scale) on register tile.
    if phases.inter_reduce:
        prev_id = phases.reduces[0][0]
        ops.extend(_apply_ops_on_register_tile(phases.inter_reduce[0], prev_id, shapes, input_set, thread_m, thread_n, "sc"))

    # Phase 1: row max via WarpShuffleXor.
    for r in range(thread_m):
        rmax = f"rmax{r}"
        ops.append(Alloc(rmax, "float", None, "reg", LoopLiteral(-1e30)))
        for c in range(thread_n):
            col_c = LoopVar("bn") + LoopVar("tc") + c
            ops.append(
                Guard(
                    col_c.lt(LoopVar("N")),
                    [SetVar(rmax, LoopFuncCall("fmaxf", [LoopVar(rmax), RegAccess("c", [r, c])]))],
                )
            )
        ops.append(WarpShuffleXor(rmax, "max"))

    # Phase 2: apply inter_reduce[1] ops + accumulate row sum.
    if len(phases.inter_reduce) > 1:
        # Build extra_vars: the max reduce result is available as rmax{r}
        max_id = phases.reduces[1][0]  # the max reduce node_id
        # The inter_reduce[1] ops reference the max reduce output (prev reduce)
        # and the scaled scores (inter_reduce[0] output). The "prev_id" for
        # inter_reduce[1] is the last node of inter_reduce[0] or reduces[0].
        ir1_prev = phases.inter_reduce[0][-1][0] if phases.inter_reduce[0] else phases.reduces[0][0]

        for r in range(thread_m):
            rsum = f"rsum{r}"
            rmax = f"rmax{r}"
            ops.append(Alloc(rsum, "float", None, "reg", LoopLiteral(0.0)))
            for c in range(thread_n):
                col_c = LoopVar("bn") + LoopVar("tc") + c
                guarded: list = []

                # Apply inter_reduce[1] ops generically, with rmax as extra var
                tile_ops = _apply_ops_on_register_tile(
                    phases.inter_reduce[1],
                    ir1_prev,
                    shapes,
                    input_set,
                    1,  # single row
                    1,  # single col
                    f"ir1_{r}_{c}",
                    extra_vars={max_id: rmax},
                )
                # Fixup: the helper emits c00 for (0,0), but we need c{r}{c}
                for tile_op in tile_ops:
                    if isinstance(tile_op, SetVar) and tile_op.name == "c00":
                        tile_op = SetVar(f"c{r}{c}", _fixup_reg_refs_expr(tile_op.expr, 0, 0, r, c))
                    guarded.append(tile_op)

                guarded.append(Accumulate(rsum, "sum", RegAccess("c", [r, c])))
                ops.append(Guard(col_c.lt(LoopVar("N")), guarded))
            ops.append(WarpShuffleXor(rsum, "sum"))

    # Phase 3: apply epilogue ops generically.
    # The epilogue references the sum reduce output as per-row rsum{r}.
    if phases.epilogue:
        sum_id = phases.reduces[2][0] if len(phases.reduces) > 2 else phases.reduces[-1][0]
        epi_prev = phases.inter_reduce[-1][-1][0] if phases.inter_reduce and phases.inter_reduce[-1] else phases.reduces[-1][0]

        for r in range(thread_m):
            rsum = f"rsum{r}"
            for c in range(thread_n):
                col_c = LoopVar("bn") + LoopVar("tc") + c
                epi_ops = _apply_ops_on_register_tile(
                    phases.epilogue,
                    epi_prev,
                    shapes,
                    input_set,
                    1,
                    1,
                    f"ep_{r}_{c}",
                    extra_vars={sum_id: rsum},
                )
                fixed: list = []
                for tile_op in epi_ops:
                    if isinstance(tile_op, SetVar) and tile_op.name == "c00":
                        tile_op = SetVar(f"c{r}{c}", _fixup_reg_refs_expr(tile_op.expr, 0, 0, r, c))
                    fixed.append(tile_op)
                ops.append(Guard(col_c.lt(LoopVar("N")), fixed))

    return ops


def _fixup_reg_refs_expr(expr: LoopExpr, from_r: int, from_c: int, to_r: int, to_c: int) -> LoopExpr:
    """Replace RegAccess("c", [from_r, from_c]) with RegAccess("c", [to_r, to_c]) in an expression tree."""
    if isinstance(expr, RegAccess) and expr.name == "c" and expr.indices == [from_r, from_c]:
        return RegAccess("c", [to_r, to_c])
    if isinstance(expr, OpCall):
        return OpCall(expr.op, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    if isinstance(expr, LoopFuncCall):
        return LoopFuncCall(expr.name, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    return expr


def _contraction_write_ops(
    output_name: str,
    thread_m: int,
    thread_n: int,
    is_batched: bool,
    k_splits: int,
) -> list:
    """Generate LoopIR ops for the contraction write phase.

    Replaces the W() macro with proper Guard + Store ops.
    """
    ops: list = []
    safe_out = _safe(output_name)
    bm, tr, bn, tc = LoopVar("bm"), LoopVar("tr"), LoopVar("bn"), LoopVar("tc")
    M, N = LoopVar("M"), LoopVar("N")  # noqa: N806

    for r in range(thread_m):
        row = bm + tr + r
        row_guard = row.lt(M)
        row_body: list = []

        for c_idx in range(thread_n):
            col = bn + tc + c_idx
            col_guard = col.lt(N)

            if is_batched:
                idx = LoopVar("batch") * (M * N) + row * N + col
            else:
                idx = row * N + col

            row_body.append(Store(safe_out, idx, RegAccess("c", [r, c_idx]), "global", guard=col_guard, atomic=k_splits > 1))

        ops.append(Guard(row_guard, row_body))

    return ops


# ---------------------------------------------------------------------------
# Contraction lowering
# ---------------------------------------------------------------------------


def _lower_contraction_naive(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (naive strategy) to LoopIR.

    Grid setup uses LoopIR, K-loop uses LoopIR.  Accumulator declaration,
    epilogue, and write phase use proper LoopIR with register arrays.

    """
    dims = _contraction_dims(analysis, hints)
    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    tx, ty = dims["tx"], dims["ty"]
    is_batched = analysis.batch_size > 1
    k_splits = int(hints.get("k_splits", 1))

    A_buf = _safe(analysis.contraction_a)  # noqa: N806
    B_buf = _safe(analysis.contraction_b)  # noqa: N806
    A, B = LoopVar(A_buf), LoopVar(B_buf)  # noqa: N806
    out_id = region.output_names[0]

    # Params
    params = _build_params(region)
    params.extend([("int", "M"), ("int", "N"), ("int", "K")])
    if is_batched:
        params.append(("int", "batch_count"))

    body: list = []

    # Grid setup (CTA-swizzle)
    body.extend(_cta_swizzle_grid(thread_m, thread_n, tile_m, tile_n, is_batched))

    # Batch pointer aliases
    M, N, K = LoopVar("M"), LoopVar("N"), LoopVar("K")  # noqa: N806
    if is_batched:
        batch = LoopVar("batch")
        body.append(Let("Ab", A + batch * (M * K), dtype="const float*"))
        body.append(Let("Bb", B + batch * (K * N), dtype="const float*"))
        a_src, b_src = "Ab", "Bb"
    else:
        a_src, b_src = A_buf, B_buf

    # Accumulator register array
    body.append(Alloc("c", "float", (thread_m, thread_n), "reg", LoopLiteral(0.0)))

    # K-loop with bounds-checked global loads + FMA
    k_body: list = []
    # Load B columns (4 per thread)
    bn, tc, bm, tr = LoopVar("bn"), LoopVar("tc"), LoopVar("bm"), LoopVar("tr")
    k, K, N, M = LoopVar("k"), LoopVar("K"), LoopVar("N"), LoopVar("M")  # noqa: N806
    for c in range(thread_n):
        col = bn + tc + c
        k_body.append(Load(f"b{c}", b_src, k * N + col, "global", guard=col.lt(N)))
    # Load A rows + FMA (unrolled over thread_m)
    for r in range(thread_m):
        row = bm + tr + r
        k_body.append(Load(f"a{r}", a_src, row * K + k, "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accumulate(f"c{r}{c}", "sum", LoopVar(f"a{r}") * LoopVar(f"b{c}")))

    body.append(LoopNest("k", LoopLiteral(0, "int"), LoopVar("K"), None, k_body))

    # Epilogue — proper LoopIR with register arrays
    phases = analysis.op_phases
    if len(phases.reduces) > 1 and analysis.cols <= tile_n:
        body.extend(_contraction_softmax_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n))
    else:
        body.extend(_contraction_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n))

    # Write — proper LoopIR with register arrays
    body.extend(_contraction_write_ops(out_id, thread_m, thread_n, is_batched, k_splits))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(tx, ty, 1),
        tile_m=tile_m,
        tile_n=tile_n,
    )


def _lower_contraction_tma(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (TMA double-buffer) to LoopIR.

    CTA-swizzle grid, accumulator alloc, epilogue, and write use proper
    LoopIR.  The TMA smem setup + mbarrier pipeline + double-buffered
    K-loop uses TMAKLoop (CUDA extension op for inline PTX asm).
    """
    dims = _contraction_dims(analysis, hints)
    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    tx, ty = dims["tx"], dims["ty"]
    is_batched = analysis.batch_size > 1
    k_splits = int(hints.get("k_splits", 1))
    bk = int(hints.get("block_k", 32))
    phases = analysis.op_phases

    A_buf = _safe(analysis.contraction_a)  # noqa: N806
    B_buf = _safe(analysis.contraction_b)  # noqa: N806
    out_id = region.output_names[0]
    a_size = tile_m * bk
    b_size = bk * tile_n
    stage = a_size + b_size

    # TMA params: A/B come via descriptors, only C + epilogue inputs as regular params.
    # M/N/K come via #define from backend.py.
    tma_exclude = {A_buf, B_buf, "M", "N", "K"}
    if is_batched:
        tma_exclude.add("batch_count")
    params = [(dt, nm) for dt, nm in _build_params(region) if nm not in tma_exclude]
    if k_splits > 1:
        params.append(("int", "k_splits"))

    body: list = []

    # Grid setup (CTA-swizzle) — same as naive
    body.extend(_cta_swizzle_grid(thread_m, thread_n, tile_m, tile_n, is_batched))

    # TMA double-buffered K-loop (CUDA extension op)
    from deplodock.compiler.backend.cuda.tma_ops import TMAKLoop

    tma_a_ref = f"&{A_buf}_tma[batch]" if is_batched else f"&{A_buf}_tma"
    tma_b_ref = f"&{B_buf}_tma[batch]" if is_batched else f"&{B_buf}_tma"

    body.append(
        TMAKLoop(
            a_tma_ref=tma_a_ref,
            b_tma_ref=tma_b_ref,
            tile_m=tile_m,
            tile_n=tile_n,
            block_k=bk,
            a_size=a_size,
            stage=stage,
            thread_m=thread_m,
            thread_n=thread_n,
            tx=tx,
            k_splits=k_splits,
            is_batched=is_batched,
        )
    )

    # Epilogue — proper LoopIR (softmax or standard)
    if len(phases.reduces) > 1 and analysis.cols <= tile_n:
        body.extend(_contraction_softmax_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n))
    else:
        body.extend(_contraction_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n))

    # Write — proper LoopIR with register arrays
    body.extend(_contraction_write_ops(out_id, thread_m, thread_n, is_batched, k_splits))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(tx, ty, 1),
        tile_m=tile_m,
        tile_n=tile_n,
        tma_params=[f"{A_buf}_tma", f"{B_buf}_tma"],
        batched=is_batched,
        includes=["cuda.h"],
    )


def _lower_contraction_smem(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (smem strategy) to LoopIR.

    Block: (32, 4) = 128 threads. Shared memory for A tile (bank-conflict
    padded), scalar global loads for B (nvcc auto-vectorizes to float4).
    Fully expressed in LoopIR — no RawLoopOp.
    """
    A = LoopVar(_safe(analysis.contraction_a))  # noqa: N806
    B = LoopVar(_safe(analysis.contraction_b))  # noqa: N806
    out_id = region.output_names[0]
    C = LoopVar(_safe(out_id))  # noqa: N806
    is_batched = analysis.batch_size > 1

    thread_m = int(hints.get("thread_m", 4))
    thread_n = 4
    bk = int(hints.get("block_k", 32))
    k_splits = int(hints.get("k_splits", 1))
    tx, ty = 32, 4
    tile_m = ty * thread_m
    tile_n = tx * thread_n  # 128
    smem_stride = bk + 1  # bank conflict padding
    I = "int"  # noqa: E741

    params = _build_params(region)
    params.extend([("int", "M"), ("int", "N"), ("int", "K")])
    if is_batched:
        params.append(("int", "batch_count"))
    elif k_splits > 1:
        params.append(("int", "k_splits"))

    body: list = []

    # Grid setup
    bidy, bidx = LoopBuiltin("blockIdx.y"), LoopBuiltin("blockIdx.x")
    tidy, tidx = LoopBuiltin("threadIdx.y"), LoopBuiltin("threadIdx.x")
    K = LoopVar("K")  # noqa: N806

    if is_batched:
        body.append(Let("batch", LoopBuiltin("blockIdx.z"), dtype=I))
    body.append(Let("row_base", (bidy * ty + tidy) * thread_m, dtype=I))
    body.append(Let("col_base", (bidx * tx + tidx) * thread_n, dtype=I))
    body.append(Let("sr", tidy * thread_m, dtype=I))

    # Shared memory for A tile (bank-conflict padded stride)
    body.append(Alloc("As", "float", (tile_m * smem_stride,), "smem"))

    # K-range for k_splits
    if k_splits > 1:
        bidz = LoopBuiltin("blockIdx.z")
        k_per, k_start = LoopVar("k_per"), LoopVar("k_start")
        body.append(Let("k_per", K / bk / LoopVar("k_splits") * bk, dtype=I))
        body.append(Let("k_start", bidz * k_per, dtype=I))
        body.append(Let("k_end", LoopTernary(bidz.eq(LoopLiteral(k_splits - 1, I)), K, k_start + k_per), dtype=I))
    else:
        body.append(Let("k_start", LoopLiteral(0, I), dtype=I))
        body.append(Let("k_end", K, dtype=I))

    # Accumulator register array
    body.append(Alloc("c", "float", (thread_m, thread_n), "reg", LoopLiteral(0.0)))

    # Batch pointer aliases
    M, N = LoopVar("M"), LoopVar("N")  # noqa: N806
    if is_batched:
        batch = LoopVar("batch")
        body.append(Let("Ab", A + batch * (M * K), dtype="const float*"))
        body.append(Let("Bb", B + batch * (K * N), dtype="const float*"))
    a_src = LoopVar("Ab") if is_batched else A
    b_src = LoopVar("Bb") if is_batched else B

    # K-tile loop
    tk_body: list = []

    # Load A tile into shared memory (guarded, per-thread_m rows)
    row_base, col_base = LoopVar("row_base"), LoopVar("col_base")
    sr, tk, kk = LoopVar("sr"), LoopVar("tk"), LoopVar("kk")
    tidx = LoopBuiltin("threadIdx.x")

    for r in range(thread_m):
        row_r = row_base + r
        k_col = tk + tidx
        tk_body.append(Load(f"As_ld_{r}", a_src, row_r * K + k_col, "global", guard=row_r.lt(M).and_(k_col.lt(K))))
        tk_body.append(Store("As", (sr + r) * smem_stride + tidx, LoopVar(f"As_ld_{r}"), "smem"))

    tk_body.append(Barrier())

    # Inner kk loop: read A from smem, read B from global (scalar), FMA
    kk_body: list = []
    for c in range(thread_n):
        col_c = col_base + c
        kk_body.append(Load(f"b{c}", b_src, (tk + kk) * N + col_c, "global", guard=col_c.lt(N)))

    for r in range(thread_m):
        kk_body.append(Load(f"a{r}", "As", (sr + r) * smem_stride + kk, "smem"))
        for c in range(thread_n):
            kk_body.append(Accumulate(f"c{r}{c}", "sum", LoopVar(f"a{r}") * LoopVar(f"b{c}")))

    tk_body.append(LoopNest("kk", LoopLiteral(0, I), LoopLiteral(bk, I), None, kk_body))
    tk_body.append(Barrier())

    body.append(LoopNest("tk", LoopVar("k_start"), LoopVar("k_end"), LoopLiteral(bk, I), tk_body))

    # Write
    c_local = LoopVar("Cb") if is_batched else C
    if is_batched:
        body.append(Let("Cb", C + LoopVar("batch") * (M * N), dtype="float*"))
    for r in range(thread_m):
        row = row_base + r
        row_body: list = []
        for c in range(thread_n):
            col = col_base + c
            idx = row * N + col
            row_body.append(Store(c_local, idx, RegAccess("c", [r, c]), "global", guard=col.lt(N), atomic=k_splits > 1))
        body.append(Guard(row.lt(M), row_body))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(tx, ty, 1),
        tile_m=tile_m,
        tile_n=tile_n,
        grid_2d=True,
    )
