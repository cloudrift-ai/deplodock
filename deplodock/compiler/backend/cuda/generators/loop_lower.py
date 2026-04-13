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
from deplodock.compiler.backend.ir.loop_ir import (
    Accumulate,
    Alloc,
    Barrier,
    Builtin,
    FuncCall,
    Guard,
    Let,
    Literal,
    Load,
    LoopExpr,
    LoopNest,
    LoopProgram,
    OpCall,
    ParallelAxis,
    RegAccess,
    SetVar,
    Store,
    Ternary,
    Var,
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
    ops.append(Alloc("c", accum.dtype, accum.shape, "reg", Literal(0.0)))

    # Batch pointer aliases for contraction
    if schedule.is_batched and analysis.contraction_a:
        A = Var(_safe(analysis.contraction_a))  # noqa: N806
        B = Var(_safe(analysis.contraction_b))  # noqa: N806
        batch, M, K, N = Var("batch"), Var("M"), Var("K"), Var("N")  # noqa: N806
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
        var_map[inp] = Var(load_name)

    _emit_loop_ops(region.region_ops, var_map, "v_", guard_body)

    for out_id in region.output_names:
        val = var_map.get(out_id, Literal(0.0))
        guard_body.append(Store(_safe(out_id), Var("i"), val, "global"))

    return [Guard(Var("i").lt(Var("n")), guard_body)]


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
                pass_var_map[inp] = Var(load_name)

            for prev_nid, (prev_acc, _prev_fn) in reduce_vars.items():
                pass_var_map[prev_nid] = Var(prev_acc)

            all_ops_this_pass = list(phases.inter_reduce[ri - 1]) if ri > 0 else []
            needed = _needed_by(all_ops_this_pass + [(node_id, _op, input_ids)])
            for pid, pop, pinp in phases.prologue:
                if isinstance(pop, ElementwiseOp) and pid in needed:
                    a = pass_var_map.get(pinp[0], Literal(0.0)) if pinp else Literal(0.0)
                    b = pass_var_map.get(pinp[1], Literal(0.0)) if len(pinp) > 1 else Literal(0.0)
                    var_name = f"r{ri}p_{_safe(pid)}"
                    pass_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                    pass_var_map[pid] = Var(var_name)

            if ri > 0 and phases.inter_reduce:
                _emit_loop_ops(phases.inter_reduce[ri - 1], pass_var_map, f"r{ri}_", pass_body)

            val = pass_var_map.get(input_ids[0], Literal(0.0))
            pass_body.append(Accumulate(acc_name, fn, val))

            guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), pass_body))
            guarded.append(WarpReduce(acc_name, fn))
            var_map[node_id] = Var(acc_name)
    else:
        # Single reduce: one tile loop
        loop_body: list = []
        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"ld_{_safe(inp)}"
            loop_body.append(Load(load_name, _safe(inp), idx, "global"))
            var_map[inp] = Var(load_name)

        _emit_loop_ops(phases.prologue, var_map, "p_", loop_body)

        for node_id, _op, input_ids in phases.reduces:
            acc_name, fn = reduce_vars[node_id]
            val = var_map.get(input_ids[0], Literal(0.0))
            loop_body.append(Accumulate(acc_name, fn, val))

        guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), loop_body))

        for node_id, (acc_name, fn) in reduce_vars.items():
            guarded.append(WarpReduce(acc_name, fn))
            var_map[node_id] = Var(acc_name)

    return [Guard(Var("row").lt(Var("rows")), guarded)]


def _emit_contraction_k_loop(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit the K-loop for contraction (naive global-load strategy)."""
    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    A = Var(_safe(analysis.contraction_a))  # noqa: N806
    B = Var(_safe(analysis.contraction_b))  # noqa: N806
    a_src = Var("Ab") if schedule.is_batched else A
    b_src = Var("Bb") if schedule.is_batched else B

    bn, tc, bm, tr = Var("bn"), Var("tc"), Var("bm"), Var("tr")
    k, K, N, M = Var("k"), Var("K"), Var("N"), Var("M")  # noqa: N806

    k_body: list = []
    for c in range(thread_n):
        col = bn + tc + c
        k_body.append(Load(f"b{c}", b_src, k * N + col, "global", guard=col.lt(N)))

    for r in range(thread_m):
        row = bm + tr + r
        k_body.append(Load(f"a{r}", a_src, row * K + k, "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accumulate(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))

    return [LoopNest("k", Literal(0, "int"), K, None, k_body)]


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
        var_map[node_id] = Var(acc_name)

    has_multi_reduce = len(phases.reduces) > 1

    if has_multi_reduce or schedule.epilogue_per_element:
        # Per-element epilogue loop
        epi_body: list = []
        epi_var_map: dict[str, LoopExpr] = dict(var_map)

        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"epld_{_safe(inp)}"
            epi_body.append(Load(load_name, _safe(inp), idx, "global"))
            epi_var_map[inp] = Var(load_name)

        # Re-compute prologue + inter_reduce ops
        all_inter_ops = [op for group in phases.inter_reduce for op in group] if has_multi_reduce else []
        recompute_ops = phases.prologue + all_inter_ops if has_multi_reduce else phases.prologue
        needed = _needed_by(phases.epilogue)
        for pid, pop, pinp in recompute_ops:
            if isinstance(pop, ElementwiseOp) and (pid in needed or has_multi_reduce):
                a = epi_var_map.get(pinp[0], Literal(0.0)) if pinp else Literal(0.0)
                b = epi_var_map.get(pinp[1], Literal(0.0)) if len(pinp) > 1 else Literal(0.0)
                var_name = f"ep_{_safe(pid)}"
                epi_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                epi_var_map[pid] = Var(var_name)

        _emit_loop_ops(phases.epilogue, epi_var_map, "e_", epi_body)

        for out_id in region.output_names:
            val = epi_var_map.get(out_id, Literal(0.0))
            idx = Var("row") * Var("cols") + Var("j")
            epi_body.append(Store(_safe(out_id), idx, val, "global"))

        return [LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), epi_body)]

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
        return _contraction_write_ops(Var(_safe(out_id)), thread_m, thread_n, schedule.is_batched, schedule.k_splits)

    # Scalar reduction write
    phases = analysis.op_phases
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        reduce_vars[node_id] = (f"acc_{_safe(node_id)}", op.fn)

    var_map: dict[str, LoopExpr] = {}
    for node_id, (acc_name, _fn) in reduce_vars.items():
        var_map[node_id] = Var(acc_name)

    if schedule.epilogue_per_element or len(phases.reduces) > 1:
        # Per-element writes were already emitted in the epilogue loop
        if phases.epilogue or schedule.epilogue_per_element:
            return []
        # No epilogue — write last reduce result (thread 0 only)
        ops: list = []
        for out_id in region.output_names:
            val = var_map.get(out_id, Literal(0.0))
            ops.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), Var("row"), val, "global")],
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
            val = var_map.get(out_id, Literal(0.0))
            ops.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), Var("row"), val, "global")],
                )
            )
        return ops

    # No epilogue — write last reduce result (thread 0)
    ops = []
    for out_id in region.output_names:
        val = var_map.get(out_id, Literal(0.0))
        ops.append(
            Guard(
                Builtin("threadIdx.x").eq(Literal(0, "int")),
                [Store(_safe(out_id), Var("row"), val, "global")],
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

    Contractions route to ``_lower_contraction()`` with the strategy.
    All other patterns go through ``build_schedule()`` + ``lower_generic()``.
    """
    if analysis.pattern == "contraction":
        strat = "tma" if strategy == "tma_db" else strategy
        return _lower_contraction(region, name, shapes, analysis, hints or {}, strategy=strat)

    from deplodock.compiler.backend.cuda.schedule import build_schedule

    schedule = build_schedule(analysis, strategy, hints or {})
    return lower_generic(region, name, shapes, analysis, schedule)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _reduce_init_expr(fn: str) -> Literal:
    """Initial value for a reduction accumulator."""
    if fn == "sum":
        return Literal(0.0)
    if fn == "max":
        return Literal(-1e30)
    return Literal(0.0)


def _input_loop_expr(inp: str, analysis: TileAnalysis, idx_var: str, out_size: int = 0) -> LoopExpr:
    """Build the index expression for reading an input tensor."""
    acc = analysis.input_access[inp]
    row, cols, j = Var("row"), Var("cols"), Var(idx_var)

    if analysis.pattern == "pointwise":
        if acc.is_scalar:
            return Literal(0, "int")
        if acc.size < out_size:
            return Var("i") % acc.size
        return Var("i")

    if acc.is_2d:
        return row * cols + j
    if acc.is_per_row:
        return row
    if acc.is_row_vector:
        return j
    return Literal(0, "int")


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
            a = var_map.get(input_ids[0], Literal(0.0)) if input_ids else Literal(0.0)
            b = var_map.get(input_ids[1], Literal(0.0)) if len(input_ids) > 1 else Literal(0.0)
            var_name = f"{prefix}{_safe(node_id)}"
            args = [a, b] if op.info.arity == 2 and len(input_ids) > 1 else [a]
            body.append(Let(var_name, OpCall(op.fn, args)))
            var_map[node_id] = Var(var_name)


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
        var_map[inp] = Var(load_name)

    # Emit all ops
    _emit_loop_ops(region.region_ops, var_map, "v_", guard_body)

    # Store outputs
    for out_id in region.output_names:
        val = var_map.get(out_id, Literal(0.0))
        guard_body.append(Store(_safe(out_id), Var("i"), val, "global"))

    body.append(Guard(Var("i").lt(Var("n")), guard_body))

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
        var_map[inp] = Var(inp)  # placeholder, actual load inside loop

    # Tile loop over columns
    loop_body: list = []

    # Load inputs inside loop
    input_loads: dict[str, str] = {}
    for inp in region.input_names:
        idx = _input_loop_expr(inp, analysis, "j", out_size)
        load_name = f"ld_{_safe(inp)}"
        loop_body.append(Load(load_name, _safe(inp), idx, "global"))
        var_map[inp] = Var(load_name)
        input_loads[inp] = load_name

    # Prologue ops
    _emit_loop_ops(phases.prologue, var_map, "p_", loop_body)

    # Accumulation
    for node_id, _op, input_ids in phases.reduces:
        acc_name, fn = reduce_vars[node_id]
        val = var_map.get(input_ids[0], Literal(0.0))
        loop_body.append(Accumulate(acc_name, fn, val))

    guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), loop_body))

    # Warp reduce
    for node_id, (acc_name, fn) in reduce_vars.items():
        guarded.append(WarpReduce(acc_name, fn))
        var_map[node_id] = Var(acc_name)

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
                epi_var_map[inp] = Var(load_name)

            # Re-compute prologue ops needed by epilogue
            needed = _needed_by(epilogue_ops)
            for node_id, op, input_ids in phases.prologue:
                if isinstance(op, ElementwiseOp) and node_id in needed:
                    a = epi_var_map.get(input_ids[0], Literal(0.0)) if input_ids else Literal(0.0)
                    b = epi_var_map.get(input_ids[1], Literal(0.0)) if len(input_ids) > 1 else Literal(0.0)
                    var_name = f"p_{_safe(node_id)}"
                    epi_body.append(Let(var_name, OpCall(op.fn, [a] if len(input_ids) == 1 else [a, b])))
                    epi_var_map[node_id] = Var(var_name)

            _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

            for out_id in region.output_names:
                val = epi_var_map.get(out_id, Literal(0.0))
                idx = Var("row") * Var("cols") + Var("j")
                epi_body.append(Store(_safe(out_id), idx, val, "global"))

            guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), epi_body))
        else:
            # Scalar epilogue (thread 0 only)
            _emit_loop_ops(epilogue_ops, var_map, "e_", guarded)
            for out_id in region.output_names:
                val = var_map.get(out_id, Literal(0.0))
                guarded.append(
                    Guard(
                        Builtin("threadIdx.x").eq(Literal(0, "int")),
                        [Store(_safe(out_id), Var("row"), val, "global")],
                    )
                )
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, Literal(0.0))
            guarded.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), Var("row"), val, "global")],
                )
            )

    body.append(Guard(Var("row").lt(Var("rows")), guarded))

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
            pass_var_map[inp] = Var(load_name)

        # Previous reduce results are available as accumulators
        for prev_nid, (prev_acc, _prev_fn) in reduce_vars.items():
            pass_var_map[prev_nid] = Var(prev_acc)

        # Re-compute prologue ops needed by this reduce or inter_reduce ops
        all_ops_this_pass = list(phases.inter_reduce[ri - 1]) if ri > 0 else []
        needed = _needed_by(all_ops_this_pass + [(node_id, _op, input_ids)])
        for pid, pop, pinp in phases.prologue:
            if isinstance(pop, ElementwiseOp) and pid in needed:
                a = pass_var_map.get(pinp[0], Literal(0.0)) if pinp else Literal(0.0)
                b = pass_var_map.get(pinp[1], Literal(0.0)) if len(pinp) > 1 else Literal(0.0)
                var_name = f"r{ri}p_{_safe(pid)}"
                pass_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                pass_var_map[pid] = Var(var_name)

        # Apply inter_reduce ops (e.g. sub, exp between max and sum)
        if ri > 0 and phases.inter_reduce:
            _emit_loop_ops(phases.inter_reduce[ri - 1], pass_var_map, f"r{ri}_", pass_body)

        # Accumulate
        val = pass_var_map.get(input_ids[0], Literal(0.0))
        pass_body.append(Accumulate(acc_name, fn, val))

        guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), pass_body))

        # Warp shuffle after each pass
        guarded.append(WarpReduce(acc_name, fn))
        var_map[node_id] = Var(acc_name)

    # Final epilogue pass
    epilogue_ops = phases.epilogue
    if epilogue_ops or analysis.epilogue_needs_per_element:
        epi_body: list = []
        epi_var_map: dict[str, LoopExpr] = dict(var_map)

        for inp in region.input_names:
            idx = _input_loop_expr(inp, analysis, "j", out_size)
            load_name = f"epld_{_safe(inp)}"
            epi_body.append(Load(load_name, _safe(inp), idx, "global"))
            epi_var_map[inp] = Var(load_name)

        # Re-compute ALL prologue + inter_reduce ops
        all_inter_ops = [op for group in phases.inter_reduce for op in group]
        for pid, pop, pinp in phases.prologue + all_inter_ops:
            if isinstance(pop, ElementwiseOp):
                a = epi_var_map.get(pinp[0], Literal(0.0)) if pinp else Literal(0.0)
                b = epi_var_map.get(pinp[1], Literal(0.0)) if len(pinp) > 1 else Literal(0.0)
                var_name = f"ep_{_safe(pid)}"
                epi_body.append(Let(var_name, OpCall(pop.fn, [a] if len(pinp) == 1 else [a, b])))
                epi_var_map[pid] = Var(var_name)

        _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

        for out_id in region.output_names:
            val = epi_var_map.get(out_id, Literal(0.0))
            idx = Var("row") * Var("cols") + Var("j")
            epi_body.append(Store(_safe(out_id), idx, val, "global"))

        guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), epi_body))
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, Literal(0.0))
            guarded.append(
                Guard(
                    Builtin("threadIdx.x").eq(Literal(0, "int")),
                    [Store(_safe(out_id), Var("row"), val, "global")],
                )
            )

    body.append(Guard(Var("row").lt(Var("rows")), guarded))

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
    N, M = Var("N"), Var("M")  # noqa: N806
    bidy, bidx = Builtin("blockIdx.y"), Builtin("blockIdx.x")
    tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
    gdimx = Builtin("gridDim.x")
    SWIZ = 8

    if is_batched:
        ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))

    ops.append(Let("tr", tidy * thread_m, dtype=I))
    ops.append(Let("tc", tidx * thread_n, dtype=I))
    ops.append(Let("ntx", (N + (tile_n - 1)) / tile_n, dtype=I))
    ops.append(Let("nty", (M + (tile_m - 1)) / tile_m, dtype=I))

    ntx, nty = Var("ntx"), Var("nty")
    pid, grp, rem = Var("pid"), Var("grp"), Var("rem")
    by_s, bx_s = Var("by_s"), Var("bx_s")

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
    """Generate LoopIR ops for contraction epilogue on the register tile.

    Handles both simple epilogues (bias, activation) and multi-reduce
    epilogues (softmax: inter_reduce ops + WarpShuffleXor between reduces).
    Walks phases generically — no pattern-specific checks.

    Structure:
    1. For each inter_reduce group: apply ops on tile, then per-row
       WarpShuffleXor for the corresponding reduce
    2. Apply epilogue ops on tile (referencing reduce results as per-row vars)
    """
    ops: list = []
    input_set = set(input_names)
    has_inter = bool(phases.inter_reduce)

    if not has_inter and not phases.epilogue:
        return []

    # Track the "previous" node ID through the op chain.
    prev_id = phases.reduces[0][0]

    # Per-row reduce result variables: reduce_node_id → "rXXX{r}" template
    reduce_row_vars: dict[str, str] = {}

    # Process inter-reduce groups: ops between consecutive reduces.
    # inter_reduce[i] holds ops between reduces[i] and reduces[i+1].
    for ri, inter_ops in enumerate(phases.inter_reduce):
        # The reduce that follows this inter-group (reduces[ri+1])
        reduce_node_id = phases.reduces[ri + 1][0]
        reduce_fn = phases.reduces[ri + 1][1].fn

        # Apply inter-reduce ops on the register tile
        ops.extend(
            _apply_ops_on_register_tile(inter_ops, prev_id, shapes, input_set, thread_m, thread_n, f"ir{ri}", extra_vars=reduce_row_vars)
        )
        prev_id = inter_ops[-1][0] if inter_ops else prev_id

        # Per-row reduce via WarpShuffleXor
        init_val = Literal(-1e30) if reduce_fn == "max" else Literal(0.0)
        acc_fn = "fmaxf" if reduce_fn == "max" else None  # sum uses Accumulate

        for r in range(thread_m):
            rvar = f"r{reduce_fn}{r}"
            ops.append(Alloc(rvar, "float", None, "reg", init_val))
            for c in range(thread_n):
                col_c = Var("bn") + Var("tc") + c
                if acc_fn:
                    ops.append(Guard(col_c.lt(Var("N")), [SetVar(rvar, FuncCall(acc_fn, [Var(rvar), RegAccess("c", [r, c])]))]))
                else:
                    ops.append(Guard(col_c.lt(Var("N")), [Accumulate(rvar, reduce_fn, RegAccess("c", [r, c]))]))
            ops.append(WarpShuffleXor(rvar, reduce_fn))

        reduce_row_vars[reduce_node_id] = f"r{reduce_fn}{{r}}"

    # Apply epilogue ops on the register tile.
    if phases.epilogue:
        if reduce_row_vars:
            # Epilogue references reduce results as per-row variables.
            # Apply per-element with column guard.
            epi_prev = prev_id
            for r in range(thread_m):
                for c in range(thread_n):
                    col_c = Var("bn") + Var("tc") + c
                    # Resolve per-row vars for this specific row
                    row_extra = {k: v.format(r=r) for k, v in reduce_row_vars.items()}
                    epi_ops = _apply_ops_on_register_tile(
                        phases.epilogue, epi_prev, shapes, input_set, 1, 1, f"ep_{r}_{c}", extra_vars=row_extra
                    )
                    fixed: list = []
                    for tile_op in epi_ops:
                        if isinstance(tile_op, SetVar) and tile_op.name == "c00":
                            tile_op = SetVar(f"c{r}{c}", _fixup_reg_refs_expr(tile_op.expr, 0, 0, r, c))
                        fixed.append(tile_op)
                    ops.append(Guard(col_c.lt(Var("N")), fixed))
        else:
            # Simple epilogue (no inter-reduces): apply directly on full tile
            ops.extend(_apply_ops_on_register_tile(phases.epilogue, prev_id, shapes, input_set, thread_m, thread_n, "ep"))

    return ops


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
                    other_val = Var(extra[other].format(r=r))
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
                        idx = Literal(0, "int")
                    elif len(other_shape) <= 1:
                        idx = Var("bn") + Var("tc") + c
                    else:
                        row_e = Var("bm") + Var("tr") + r
                        col_e = Var("bn") + Var("tc") + c
                        idx = row_e * Var("N") + col_e
                    ld_name = f"_{prefix}_{safe}_{r}_{c}"
                    ops.append(Load(ld_name, safe, idx, "global"))
                    if inputs[0] == prev_id:
                        ops.append(SetVar(dst, OpCall(fn, [acc, Var(ld_name)])))
                    else:
                        ops.append(SetVar(dst, OpCall(fn, [Var(ld_name), acc])))
                else:
                    # Unary op on accumulator
                    ops.append(SetVar(dst, OpCall(fn, [acc])))

        prev_id = nid

    return ops


def _fixup_reg_refs_expr(expr: LoopExpr, from_r: int, from_c: int, to_r: int, to_c: int) -> LoopExpr:
    """Replace RegAccess("c", [from_r, from_c]) with RegAccess("c", [to_r, to_c]) in an expression tree."""
    if isinstance(expr, RegAccess) and expr.name == "c" and expr.indices == [from_r, from_c]:
        return RegAccess("c", [to_r, to_c])
    if isinstance(expr, OpCall):
        return OpCall(expr.op, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    if isinstance(expr, FuncCall):
        return FuncCall(expr.name, [_fixup_reg_refs_expr(a, from_r, from_c, to_r, to_c) for a in expr.args])
    return expr


def _contraction_write_ops(
    output: Var,
    thread_m: int,
    thread_n: int,
    is_batched: bool,
    k_splits: int,
    row_base: LoopExpr | None = None,
    col_base: LoopExpr | None = None,
) -> list:
    """Generate LoopIR ops for the contraction write phase.

    ``row_base`` and ``col_base`` are the base expressions for computing
    global row/col indices.  Defaults: ``bm + tr`` and ``bn + tc``
    (CTA-swizzle layout).  Smem uses ``row_base`` and ``col_base`` vars.
    """
    ops: list = []
    M, N = Var("M"), Var("N")  # noqa: N806
    rb = row_base if row_base is not None else Var("bm") + Var("tr")
    cb = col_base if col_base is not None else Var("bn") + Var("tc")

    for r in range(thread_m):
        row = rb + r
        row_body: list = []
        for c_idx in range(thread_n):
            col = cb + c_idx
            if is_batched:
                idx = Var("batch") * (M * N) + row * N + col
            else:
                idx = row * N + col
            row_body.append(Store(output, idx, RegAccess("c", [r, c_idx]), "global", guard=col.lt(N), atomic=k_splits > 1))
        ops.append(Guard(row.lt(M), row_body))

    return ops


# ---------------------------------------------------------------------------
# Contraction lowering
# ---------------------------------------------------------------------------


def _lower_contraction(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
    strategy: str = "naive",
) -> LoopProgram:
    """Lower contraction to LoopIR for any strategy (naive, tma, smem).

    Shared structure: params → grid → accumulators → K-loop → epilogue → write.
    Only the grid setup and K-loop body differ between strategies.
    """
    is_batched = analysis.batch_size > 1
    k_splits = int(hints.get("k_splits", 1))
    phases = analysis.op_phases

    A = Var(_safe(analysis.contraction_a))  # noqa: N806
    B = Var(_safe(analysis.contraction_b))  # noqa: N806
    C = Var(_safe(region.output_names[0]))  # noqa: N806

    # --- Strategy-specific: grid + K-loop + dims ---
    dims = _contraction_dims(analysis, hints)
    if strategy == "tma":
        grid_ops, k_ops, extra = _k_loop_tma(A, B, dims, hints, is_batched, k_splits)
    elif strategy == "smem":
        grid_ops, k_ops, extra = _k_loop_smem(A, B, dims, hints, is_batched, k_splits)
    else:
        grid_ops, k_ops, extra = _k_loop_naive(A, B, dims, is_batched, k_splits)

    # Smem overrides block dims (32,4 vs 32,8)
    if "dims" in extra:
        dims = extra["dims"]
    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    tx, ty = dims["tx"], dims["ty"]

    # --- Shared: params ---
    if strategy == "tma":
        tma_exclude = {A.name, B.name, "M", "N", "K"}
        if is_batched:
            tma_exclude.add("batch_count")
        params = [(dt, nm) for dt, nm in _build_params(region) if nm not in tma_exclude]
        if k_splits > 1:
            params.append(("int", "k_splits"))
    else:
        params = _build_params(region)
        params.extend([("int", "M"), ("int", "N"), ("int", "K")])
        if is_batched:
            params.append(("int", "batch_count"))
        elif k_splits > 1 and strategy == "smem":
            params.append(("int", "k_splits"))

    # --- Shared: assemble body ---
    body: list = []
    body.extend(grid_ops)
    body.extend(k_ops)

    # Epilogue
    body.extend(_contraction_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n))

    # Write
    row_base = extra.get("row_base")
    col_base = extra.get("col_base")
    out = extra.get("out_var", C)
    body.extend(_contraction_write_ops(out, thread_m, thread_n, is_batched, k_splits, row_base=row_base, col_base=col_base))

    return LoopProgram(
        name=name,
        params=params,
        body=body,
        block_size=(tx, ty, 1),
        tile_m=tile_m,
        tile_n=tile_n,
        grid_2d=extra.get("grid_2d", False),
        tma_params=extra.get("tma_params"),
        tma_config=extra.get("tma_config"),
        batched=extra.get("batched", False),
        includes=extra.get("includes"),
    )


# ---------------------------------------------------------------------------
# Strategy-specific K-loop helpers
# ---------------------------------------------------------------------------


def _k_loop_naive(A, B, dims, is_batched, k_splits):  # noqa: N803
    """Naive K-loop: global loads + FMA. Returns (grid_ops, k_ops, extra)."""
    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    M, N, K = Var("M"), Var("N"), Var("K")  # noqa: N806

    grid_ops = list(_cta_swizzle_grid(thread_m, thread_n, tile_m, tile_n, is_batched))

    k_ops: list = []
    if is_batched:
        batch = Var("batch")
        k_ops.append(Let("Ab", A + batch * (M * K), dtype="const float*"))
        k_ops.append(Let("Bb", B + batch * (K * N), dtype="const float*"))
    a_src = Var("Ab") if is_batched else A
    b_src = Var("Bb") if is_batched else B

    k_ops.append(Alloc("c", "float", (thread_m, thread_n), "reg", Literal(0.0)))

    bn, tc, bm, tr, k = Var("bn"), Var("tc"), Var("bm"), Var("tr"), Var("k")
    k_body: list = []
    for c in range(thread_n):
        col = bn + tc + c
        k_body.append(Load(f"b{c}", b_src, k * N + col, "global", guard=col.lt(N)))
    for r in range(thread_m):
        row = bm + tr + r
        k_body.append(Load(f"a{r}", a_src, row * K + k, "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accumulate(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))
    k_ops.append(LoopNest("k", Literal(0, "int"), K, None, k_body))

    return grid_ops, k_ops, {}


def _k_loop_tma(A, B, dims, hints, is_batched, k_splits):  # noqa: N803
    """TMA double-buffered K-loop. Returns (grid_ops, k_ops, extra)."""
    from deplodock.compiler.backend.cuda.tma_ops import TMALoadConfig
    from deplodock.compiler.backend.ir.loop_ir import SmemPipelineKLoop

    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    tx = dims["tx"]
    bk = int(hints.get("block_k", 32))
    a_size = tile_m * bk
    stage = a_size + bk * tile_n

    grid_ops = list(_cta_swizzle_grid(thread_m, thread_n, tile_m, tile_n, is_batched))

    k_ops = [
        SmemPipelineKLoop(
            stages=2,
            tile_m=tile_m,
            tile_n=tile_n,
            block_k=bk,
            a_size=a_size,
            stage_size=stage,
            thread_m=thread_m,
            thread_n=thread_n,
            tx=tx,
            k_splits=k_splits,
            is_batched=is_batched,
        )
    ]

    tma_a_ref = f"&{A.name}_tma[batch]" if is_batched else f"&{A.name}_tma"
    tma_b_ref = f"&{B.name}_tma[batch]" if is_batched else f"&{B.name}_tma"

    extra = {
        "tma_config": TMALoadConfig(a_tma_ref=tma_a_ref, b_tma_ref=tma_b_ref),
        "tma_params": [f"{A.name}_tma", f"{B.name}_tma"],
        "batched": is_batched,
        "includes": ["cuda.h"],
    }
    return grid_ops, k_ops, extra


def _k_loop_smem(A, B, dims, hints, is_batched, k_splits):  # noqa: N803
    """Smem K-tile loop: shared memory for A, global for B. Returns (grid_ops, k_ops, extra)."""
    thread_m = int(hints.get("thread_m", 4))
    thread_n = 4
    bk = int(hints.get("block_k", 32))
    tx, ty = 32, 4
    tile_m = ty * thread_m
    tile_n = tx * thread_n
    smem_stride = bk + 1
    I = "int"  # noqa: E741
    M, N, K = Var("M"), Var("N"), Var("K")  # noqa: N806

    # 2D standard grid
    bidy, bidx = Builtin("blockIdx.y"), Builtin("blockIdx.x")
    tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")

    grid_ops: list = []
    if is_batched:
        grid_ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))
    grid_ops.append(Let("row_base", (bidy * ty + tidy) * thread_m, dtype=I))
    grid_ops.append(Let("col_base", (bidx * tx + tidx) * thread_n, dtype=I))
    grid_ops.append(Let("sr", tidy * thread_m, dtype=I))

    k_ops: list = []
    k_ops.append(Alloc("As", "float", (tile_m * smem_stride,), "smem"))

    # K-range for k_splits
    if k_splits > 1:
        bidz = Builtin("blockIdx.z")
        k_per, k_start = Var("k_per"), Var("k_start")
        k_ops.append(Let("k_per", K / bk / Var("k_splits") * bk, dtype=I))
        k_ops.append(Let("k_start", bidz * k_per, dtype=I))
        k_ops.append(Let("k_end", Ternary(bidz.eq(Literal(k_splits - 1, I)), K, k_start + k_per), dtype=I))
    else:
        k_ops.append(Let("k_start", Literal(0, I), dtype=I))
        k_ops.append(Let("k_end", K, dtype=I))

    k_ops.append(Alloc("c", "float", (thread_m, thread_n), "reg", Literal(0.0)))

    # Batch pointer aliases
    if is_batched:
        batch = Var("batch")
        k_ops.append(Let("Ab", A + batch * (M * K), dtype="const float*"))
        k_ops.append(Let("Bb", B + batch * (K * N), dtype="const float*"))
    a_src = Var("Ab") if is_batched else A
    b_src = Var("Bb") if is_batched else B

    # K-tile loop body
    row_base, col_base = Var("row_base"), Var("col_base")
    sr, tk, kk = Var("sr"), Var("tk"), Var("kk")
    tidx_v = Builtin("threadIdx.x")

    tk_body: list = []
    for r in range(thread_m):
        row_r = row_base + r
        k_col = tk + tidx_v
        tk_body.append(Load(f"As_ld_{r}", a_src, row_r * K + k_col, "global", guard=row_r.lt(M).and_(k_col.lt(K))))
        tk_body.append(Store("As", (sr + r) * smem_stride + tidx_v, Var(f"As_ld_{r}"), "smem"))
    tk_body.append(Barrier())

    kk_body: list = []
    for c in range(thread_n):
        col_c = col_base + c
        kk_body.append(Load(f"b{c}", b_src, (tk + kk) * N + col_c, "global", guard=col_c.lt(N)))
    for r in range(thread_m):
        kk_body.append(Load(f"a{r}", "As", (sr + r) * smem_stride + kk, "smem"))
        for c in range(thread_n):
            kk_body.append(Accumulate(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))
    tk_body.append(LoopNest("kk", Literal(0, I), Literal(bk, I), None, kk_body))
    tk_body.append(Barrier())

    k_ops.append(LoopNest("tk", Var("k_start"), Var("k_end"), Literal(bk, I), tk_body))

    extra: dict = {
        "grid_2d": True,
        "row_base": row_base,
        "col_base": col_base,
        "dims": {"tx": tx, "ty": ty, "thread_m": thread_m, "thread_n": thread_n, "tile_m": tile_m, "tile_n": tile_n},
    }
    return grid_ops, k_ops, extra
