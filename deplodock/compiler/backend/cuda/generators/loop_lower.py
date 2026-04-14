"""Lower FusedRegionOp + TileAnalysis to LoopIR.

``lower_generic()`` is the single entry point: it reads a ``Schedule`` and
emits LoopIR through five phases (grid → accumulators → reductions →
epilogue → write).  No pattern-matching — all structural decisions live
in the Schedule.

All contraction strategies (naive, tma, smem) route through the Schedule.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, _needed_by
from deplodock.compiler.backend.ir.loop_ir import (
    Accum,
    AccumInit,
    Alloc,
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
    ShuffleReduce,
    Store,
    Var,
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
        dim_strides=_build_dim_strides(analysis, region, schedule),
    )


# ---------------------------------------------------------------------------
# Phase 1: Grid setup
# ---------------------------------------------------------------------------


def _emit_grid(schedule: Schedule, analysis: TileAnalysis) -> list:
    grid = schedule.grid
    if grid.type == "1d":
        axis_name = "i" if schedule.accum.shape is None else "row"
        return [ParallelAxis(axis_name, "blockIdx.x", grid.bound)]
    if grid.type == "1d_contraction":
        # Online reduction: 1D grid over M-tiles, N is tiled sequentially.
        # Emit bm, tr, tc but NOT bn (comes from the N-tile loop).
        I = "int"  # noqa: E741
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        tile_m = schedule.tile_m or 64
        tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
        ops: list = []
        if schedule.is_batched:
            ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))
        ops.append(Let("bm", Builtin("blockIdx.x") * tile_m, dtype=I))
        ops.append(Let("tr", tidy * thread_m, dtype=I))
        ops.append(Let("tc", tidx * thread_n, dtype=I))
        ops.append(
            Guard(
                (Var("bm") + Var("tr")).ge(Var("M")),
                [],  # early return
            )
        )
        return ops
    if grid.type == "2d_swizzle":
        return _cta_swizzle_grid(
            schedule.thread_m or 8,
            schedule.thread_n or 4,
            schedule.tile_m or 64,
            schedule.tile_n or 128,
            schedule.is_batched,
        )
    # 2d_standard (smem): row_base/col_base/sr grid
    if grid.type == "2d_standard":
        I = "int"  # noqa: E741
        tx, ty, _ = grid.block_size
        thread_m = schedule.thread_m or 4
        thread_n = schedule.thread_n or 4
        bidy, bidx = Builtin("blockIdx.y"), Builtin("blockIdx.x")
        tidy, tidx = Builtin("threadIdx.y"), Builtin("threadIdx.x")
        ops: list = []
        if schedule.is_batched:
            ops.append(Let("batch", Builtin("blockIdx.z"), dtype=I))
        ops.append(Let("row_base", (bidy * ty + tidy) * thread_m, dtype=I))
        ops.append(Let("col_base", (bidx * tx + tidx) * thread_n, dtype=I))
        ops.append(Let("sr", tidy * thread_m, dtype=I))
        return ops
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
            ops.append(AccumInit(acc_name, op.fn))
        return ops

    # 2D register tile (contraction)
    ops.append(Alloc("c", accum.dtype, accum.shape, "reg", Literal(0.0)))

    # Batch pointer aliases for contraction
    if schedule.is_batched and analysis.contraction_a:
        A = Var(_safe(analysis.contraction_a))  # noqa: N806
        B = Var(_safe(analysis.contraction_b))  # noqa: N806
        batch, M, K, N = Var("batch"), Var("M"), Var("K"), Var("N")  # noqa: N806
        # GQA / broadcast batch: one operand may have fewer batch elements.
        # E.g. 28 Q heads vs 4 KV heads → b_batch_group=7, B uses batch//7.
        a_batch_idx = batch / analysis.a_batch_group if analysis.a_batch_group > 1 else batch
        b_batch_idx = batch / analysis.b_batch_group if analysis.b_batch_group > 1 else batch
        ops.append(Let("Ab", A + a_batch_idx * (M * K), dtype="const float*"))
        ops.append(Let("Bb", B + b_batch_idx * (K * N), dtype="const float*"))

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

    # 2D register tile contraction
    if schedule.grid.type == "1d_contraction":
        # Contraction + multi-reduce: online N-tiled reduction.
        # Handles phases 3-5 (reductions, epilogue, write) internally.
        return _emit_online_contraction_reduce(schedule, analysis, region, shapes)

    # Standard contraction K-loop (single reduce, no multi-reduce)
    if schedule.load_strategy == "tma":
        return _emit_tma_k_loop(schedule, analysis)
    if schedule.load_strategy == "smem":
        return _emit_smem_k_loop(schedule, analysis)
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
        indices = _input_indices(inp, analysis, "i", out_size)
        load_name = f"v_{_safe(inp)}"
        guard_body.append(Load(load_name, _safe(inp), indices, "global"))
        var_map[inp] = Var(load_name)

    _emit_loop_ops(region.region_ops, var_map, "v_", guard_body)

    for out_id in region.output_names:
        val = var_map.get(out_id, Literal(0.0))
        guard_body.append(Store(_safe(out_id), [Var("i")], val, "global"))

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
    guarded: list = []

    # Accumulator allocs were already emitted in phase 2; build var tracking.
    reduce_vars: dict[str, tuple[str, str]] = {}
    for node_id, op, _ in phases.reduces:
        acc_name = f"acc_{_safe(node_id)}"
        reduce_vars[node_id] = (acc_name, op.fn)

    var_map: dict[str, LoopExpr] = {}

    # One tile loop per reduce pass. For single-reduce this is one pass;
    # for multi-reduce (softmax) it's one per reduce with inter-reduce ops between.
    for ri, (node_id, _op, input_ids) in enumerate(phases.reduces):
        acc_name, fn = reduce_vars[node_id]
        pass_body: list = []

        pass_var_map: dict[str, LoopExpr] = {}
        for inp in region.input_names:
            indices = _input_indices(inp, analysis, "j", out_size)
            load_name = f"r{ri}ld_{_safe(inp)}"
            pass_body.append(Load(load_name, _safe(inp), indices, "global"))
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
        pass_body.append(Accum(acc_name, fn, val))

        guarded.append(LoopNest("j", Builtin("threadIdx.x"), Var("cols"), Builtin("blockDim.x"), pass_body))
        guarded.append(ShuffleReduce(acc_name, fn))
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
        k_body.append(Load(f"b{c}", b_src, [k, col], "global", guard=col.lt(N)))

    for r in range(thread_m):
        row = bm + tr + r
        k_body.append(Load(f"a{r}", a_src, [row, k], "global", guard=row.lt(M)))
        for c in range(thread_n):
            k_body.append(Accum(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))

    return [LoopNest("k", Literal(0, "int"), K, None, k_body)]


def _emit_tma_k_loop(schedule: Schedule, analysis: TileAnalysis) -> list:
    """Emit TMA double-buffered K-loop via SmemPipelineKLoop."""
    from deplodock.compiler.backend.ir.loop_ir import SmemPipelineKLoop

    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    tile_m = schedule.tile_m or 64
    tile_n = schedule.tile_n or 128
    bk = schedule.block_k
    a_size = tile_m * bk
    stage = a_size + bk * tile_n

    return [
        SmemPipelineKLoop(
            stages=2,
            tile_m=tile_m,
            tile_n=tile_n,
            block_k=bk,
            a_size=a_size,
            stage_size=stage,
            thread_m=thread_m,
            thread_n=thread_n,
            tx=schedule.grid.block_size[0],
            k_splits=schedule.k_splits,
            is_batched=schedule.is_batched,
        )
    ]


def _emit_smem_k_loop(schedule: Schedule, analysis: TileAnalysis) -> list:
    """Emit smem K-tile loop via SmemPipelineKLoop.expand()."""
    from deplodock.compiler.backend.ir.loop_ir import SmemPipelineKLoop

    A = Var(_safe(analysis.contraction_a))  # noqa: N806
    B = Var(_safe(analysis.contraction_b))  # noqa: N806
    a_buf = "Ab" if schedule.is_batched else A.name
    b_buf = "Bb" if schedule.is_batched else B.name

    pipeline = SmemPipelineKLoop(
        stages=2,
        tile_m=schedule.tile_m or 16,
        tile_n=schedule.tile_n or 128,
        block_k=schedule.block_k,
        a_size=(schedule.tile_m or 16) * schedule.block_k,
        stage_size=(schedule.tile_m or 16) * schedule.block_k + schedule.block_k * (schedule.tile_n or 128),
        thread_m=schedule.thread_m or 4,
        thread_n=schedule.thread_n or 4,
        tx=schedule.grid.block_size[0],
        k_splits=schedule.k_splits,
        is_batched=schedule.is_batched,
        a_buf=a_buf,
        b_buf=b_buf,
    )
    return pipeline.expand()


# ---------------------------------------------------------------------------
# Online contraction + multi-reduce (handles phases 3-5 together)
# ---------------------------------------------------------------------------


def _emit_online_contraction_reduce(
    schedule: Schedule,
    analysis: TileAnalysis,
    region: FusedRegionOp,
    shapes: dict[str, tuple],
) -> list:
    """Emit N-tiled contraction with generic online reduction.

    When a contraction (matmul) is followed by multi-reduce (e.g. softmax),
    we tile over N sequentially and process each reduce in a separate pass
    over the output buffer.  This works for any multi-reduce pattern without
    hardcoded correction factors.

    Structure:
      Loop 1: K-loop + prologue + head reduce + write raw scores
      Loop 2..len(reduces): apply inter_reduce ops, compute next reduce
      Final loop: apply all inter_reduce + epilogue ops, write result

    TODO: When N <= tile_n, a KernelIR optimizer pass could detect
    single-iteration N-loops and fuse them into one in-register pass,
    eliminating the global memory round-trip for raw scores.
    """
    thread_m = schedule.thread_m or 8
    thread_n = schedule.thread_n or 4
    tile_n = schedule.tile_n or 128
    phases = analysis.op_phases
    output = _safe(region.output_names[0])
    input_set = set(region.input_names)

    I = "int"  # noqa: E741
    M, N = Var("M"), Var("N")  # noqa: N806

    # The first reduce (reduces[0]) is the contraction reduce over K —
    # already handled by the K-loop.  The N-axis reduces start at index 1.
    # inter_reduce[0] contains ops between the K-reduce and the first N-reduce
    # (e.g. scale), which we apply on the register tile after the K-loop.
    n_reduces = phases.reduces[1:]  # reduces over N (skip contraction reduce)
    n_inter = phases.inter_reduce  # inter_reduce[i] is between reduces[i] and reduces[i+1]
    is_batched = schedule.is_batched

    def _out_indices(row: LoopExpr, col: LoopExpr) -> list[LoopExpr]:
        """Build output buffer indices, adding batch dim if needed."""
        if is_batched:
            return [Var("batch"), row, col]
        return [row, col]

    def _running_var(node_id: str, r: int) -> Var:
        """Get the per-row running accumulator variable for a reduce node."""
        return Var(f"{reduce_running[node_id]}{r}")

    ops: list = []

    # Running accumulators for each N-axis reduce, one per row of the thread
    # tile.  Each thread handles thread_m rows, so we need thread_m variables
    # per reduce.  reduce_running maps node_id → base name (append {r} for row).
    reduce_running: dict[str, str] = {}  # node_id → base var name (e.g. "running_mx")
    for node_id, r_op, _ in n_reduces:
        base = f"running_{_safe(node_id)}"
        reduce_running[node_id] = base
        for r in range(thread_m):
            ops.append(AccumInit(f"{base}{r}", r_op.fn))

    # ---------------------------------------------------------------
    # Loop 1: K-loop + post-contraction ops + head N-reduce + write raw scores
    # ---------------------------------------------------------------
    loop1_body: list = []
    loop1_body.append(Let("bn", Var("n_tile"), dtype=I))

    # Reset contraction accumulators
    for r in range(thread_m):
        for c in range(thread_n):
            loop1_body.append(SetVar(f"c{r}{c}", Literal(0.0)))

    # K-loop (reuse existing emission based on load strategy)
    if schedule.load_strategy == "tma":
        loop1_body.extend(_emit_tma_k_loop(schedule, analysis))
    elif schedule.load_strategy == "smem":
        loop1_body.extend(_emit_smem_k_loop(schedule, analysis))
    else:
        loop1_body.extend(_emit_contraction_k_loop(schedule, analysis, region, shapes))

    # Apply inter_reduce[0] ops on register tile (ops between contraction
    # reduce and first N-reduce, e.g. scale multiplication).
    contraction_id = phases.reduces[0][0]
    if n_inter:
        loop1_body.extend(
            _apply_ops_on_register_tile(
                n_inter[0],
                contraction_id,
                shapes,
                input_set,
                thread_m,
                thread_n,
                "ir0",
            )
        )

    # Head N-reduce (n_reduces[0], e.g. max) via warp_xor per row
    head_id, head_op, head_inputs = n_reduces[0]
    head_running = reduce_running[head_id]
    for r in range(thread_m):
        tvar = f"tile_{_safe(head_id)}_{r}"
        loop1_body.append(AccumInit(tvar, head_op.fn))
        for c in range(thread_n):
            col_c = Var("bn") + Var("tc") + c
            loop1_body.append(
                Guard(
                    col_c.lt(N),
                    [
                        Accum(tvar, head_op.fn, RegAccess("c", [r, c])),
                    ],
                )
            )
        loop1_body.append(ShuffleReduce(tvar, head_op.fn, kind="warp_xor"))
        # Update running{r} = combine(running{r}, tile)
        rvar_r = f"{head_running}{r}"
        if head_op.fn == "sum":
            loop1_body.append(SetVar(rvar_r, Var(rvar_r) + Var(tvar)))
        else:
            loop1_body.append(SetVar(rvar_r, FuncCall("fmaxf" if head_op.fn == "max" else head_op.fn, [Var(rvar_r), Var(tvar)])))

    # Write raw scores (contraction output after inter_reduce[0] ops, before N-reduces)
    loop1_body.extend(
        _contraction_write_ops(
            Var(output),
            thread_m,
            thread_n,
            schedule.is_batched,
            k_splits=1,
        )
    )

    ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), loop1_body))

    # ---------------------------------------------------------------
    # Loop 2..len(n_reduces): compute subsequent N-axis reduces
    # ---------------------------------------------------------------
    for ri in range(1, len(n_reduces)):
        red_id, red_op, red_inputs = n_reduces[ri]
        red_running = reduce_running[red_id]
        # inter_reduce between n_reduces[ri-1] and n_reduces[ri] is
        # phases.inter_reduce[ri] (offset by 1 because inter_reduce[0]
        # is between the contraction reduce and n_reduces[0])
        inter_idx = ri  # inter_reduce[ri] = ops between reduces[ri] and reduces[ri+1]
        inter_ops = n_inter[inter_idx] if inter_idx < len(n_inter) else []

        loop_body: list = []
        loop_body.append(Let("bn", Var("n_tile"), dtype=I))

        for r in range(thread_m):
            row = Var("bm") + Var("tr") + r
            row_body: list = []

            for c in range(thread_n):
                col = Var("bn") + Var("tc") + c
                ld_name = f"rv{ri}_{r}_{c}"
                row_body.append(Load(ld_name, output, _out_indices(row, col), "global", guard=col.lt(N)))

                # Apply inter_reduce ops (e.g. sub(v, running_max{r}), exp(...))
                val_expr: LoopExpr = Var(ld_name)
                for _op_id, op_obj, op_inputs in inter_ops:
                    if not isinstance(op_obj, ElementwiseOp):
                        continue
                    if op_obj.info.arity == 1:
                        val_expr = OpCall(op_obj.fn, [val_expr])
                    else:
                        other_input = None
                        for inp in op_inputs:
                            if inp in reduce_running:
                                other_input = inp
                                break
                        if other_input is not None:
                            other_var = _running_var(other_input, r)
                            if op_inputs[0] == other_input:
                                val_expr = OpCall(op_obj.fn, [other_var, val_expr])
                            else:
                                val_expr = OpCall(op_obj.fn, [val_expr, other_var])
                        else:
                            val_expr = OpCall(op_obj.fn, [val_expr, Literal(0.0)])

                vname = f"tv{ri}_{r}_{c}"
                row_body.append(Let(vname, val_expr))

            # Reduce across columns via warp_xor
            tvar = f"tile_{_safe(red_id)}_{r}"
            row_body.append(AccumInit(tvar, red_op.fn))
            for c in range(thread_n):
                col = Var("bn") + Var("tc") + c
                row_body.append(
                    Guard(
                        col.lt(N),
                        [
                            Accum(tvar, red_op.fn, Var(f"tv{ri}_{r}_{c}")),
                        ],
                    )
                )
            row_body.append(ShuffleReduce(tvar, red_op.fn, kind="warp_xor"))
            # Update running{r}
            rvar_r = f"{red_running}{r}"
            if red_op.fn == "sum":
                row_body.append(SetVar(rvar_r, Var(rvar_r) + Var(tvar)))
            else:
                row_body.append(SetVar(rvar_r, FuncCall("fmaxf" if red_op.fn == "max" else red_op.fn, [Var(rvar_r), Var(tvar)])))

            loop_body.append(Guard(row.lt(M), row_body))

        ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), loop_body))

    # ---------------------------------------------------------------
    # Final loop: apply all inter_reduce + epilogue ops, write result
    # ---------------------------------------------------------------
    final_body: list = []
    final_body.append(Let("bn", Var("n_tile"), dtype=I))

    for r in range(thread_m):
        row = Var("bm") + Var("tr") + r
        row_body: list = []

        for c in range(thread_n):
            col = Var("bn") + Var("tc") + c
            ld_name = f"fv_{r}_{c}"
            row_body.append(Load(ld_name, output, _out_indices(row, col), "global", guard=col.lt(N)))

            # Apply N-axis inter_reduce ops (skip inter_reduce[0] which is
            # between the contraction reduce and first N-reduce — already
            # applied on the register tile in loop 1).
            val_expr: LoopExpr = Var(ld_name)
            all_inter_ops = [op for group in n_inter[1:] for op in group]
            for _op_id, op_obj, op_inputs in all_inter_ops:
                if not isinstance(op_obj, ElementwiseOp):
                    continue
                if op_obj.info.arity == 1:
                    val_expr = OpCall(op_obj.fn, [val_expr])
                else:
                    other_input = None
                    for inp in op_inputs:
                        if inp in reduce_running:
                            other_input = inp
                            break
                    if other_input is not None:
                        other_var = _running_var(other_input, r)
                        if op_inputs[0] == other_input:
                            val_expr = OpCall(op_obj.fn, [other_var, val_expr])
                        else:
                            val_expr = OpCall(op_obj.fn, [val_expr, other_var])
                    else:
                        val_expr = OpCall(op_obj.fn, [val_expr, Literal(0.0)])

            # Apply epilogue ops
            for _op_id, op_obj, op_inputs in phases.epilogue:
                if not isinstance(op_obj, ElementwiseOp):
                    continue
                if op_obj.info.arity == 1:
                    val_expr = OpCall(op_obj.fn, [val_expr])
                else:
                    other_input = None
                    for inp in op_inputs:
                        if inp in reduce_running:
                            other_input = inp
                            break
                    if other_input is not None:
                        other_var = _running_var(other_input, r)
                        if op_inputs[0] == other_input:
                            val_expr = OpCall(op_obj.fn, [other_var, val_expr])
                        else:
                            val_expr = OpCall(op_obj.fn, [val_expr, other_var])
                    else:
                        val_expr = OpCall(op_obj.fn, [val_expr, Literal(0.0)])

            result_name = f"fr_{r}_{c}"
            row_body.append(Let(result_name, val_expr))
            row_body.append(Store(output, _out_indices(row, col), Var(result_name), "global", guard=col.lt(N)))

        row_body_wrapped: list = []
        row_body_wrapped.append(Guard(row.lt(M), row_body))
        final_body.extend(row_body_wrapped)

    ops.append(LoopNest("n_tile", Literal(0, I), N, Literal(tile_n, I), final_body))

    return ops


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

    if schedule.grid.type == "1d_contraction":
        # Online reduction handles epilogue internally
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
            indices = _input_indices(inp, analysis, "j", out_size)
            load_name = f"epld_{_safe(inp)}"
            epi_body.append(Load(load_name, _safe(inp), indices, "global"))
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
            epi_body.append(Store(_safe(out_id), [Var("row"), Var("j")], val, "global"))

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

    if schedule.grid.type == "1d_contraction":
        # Online reduction handles writes internally
        return []

    if accum.shape != ():
        # Contraction register tile write
        thread_m = schedule.thread_m or 8
        thread_n = schedule.thread_n or 4
        out_id = region.output_names[0]
        rb = Var(schedule.row_base_var) if schedule.row_base_var else None
        cb = Var(schedule.col_base_var) if schedule.col_base_var else None
        return _contraction_write_ops(
            Var(_safe(out_id)), thread_m, thread_n, schedule.is_batched, schedule.k_splits, row_base=rb, col_base=cb
        )

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
                    [Store(_safe(out_id), [Var("row")], val, "global")],
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
                    [Store(_safe(out_id), [Var("row")], val, "global")],
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
                [Store(_safe(out_id), [Var("row")], val, "global")],
            )
        )
    return ops


# ---------------------------------------------------------------------------
# Convenience entry point: build_schedule + lower_generic in one call
# ---------------------------------------------------------------------------


def lower_to_loop_ir(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> tuple[LoopProgram, object]:
    """Lower a FusedRegionOp to LoopIR via ``build_schedule()`` + ``lower_generic()``.

    Returns ``(loop_program, schedule)`` so callers can pass the Schedule
    through to ``loop_ir_to_kernel()``.
    """
    from deplodock.compiler.backend.cuda.schedule import build_schedule

    schedule = build_schedule(analysis, strategy, hints or {})
    return lower_generic(region, name, shapes, analysis, schedule), schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _build_dim_strides(analysis: TileAnalysis, region: FusedRegionOp, schedule) -> dict[str, list[str]]:
    """Build per-buffer stride variable names for multi-dim index flattening.

    Contraction buffers use M/N/K strides.  Reduction buffers use "cols".
    Pointwise buffers use a single flat index (no strides needed).
    """
    strides: dict[str, list[str]] = {}

    if analysis.pattern == "contraction" and analysis.contraction_a:
        a = _safe(analysis.contraction_a)
        b = _safe(analysis.contraction_b)
        strides[a] = ["K"]
        strides[b] = ["N"]
        # Batched pointer aliases
        strides["Ab"] = ["K"]
        strides["Bb"] = ["N"]
        for out_id in region.output_names:
            if schedule.is_batched:
                strides[_safe(out_id)] = ["M * N", "N"]
            else:
                strides[_safe(out_id)] = ["N"]
        # Epilogue external inputs: use N for 2D, nothing for 1D/scalar
        for inp in region.input_names:
            safe = _safe(inp)
            if safe not in strides:
                acc = analysis.input_access.get(inp)
                if acc and acc.is_2d:
                    strides[safe] = ["N"]
    elif analysis.pattern in ("row_reduce", "reduce_broadcast", "multi_reduce"):
        # All 2D buffers use "cols" stride
        for inp in region.input_names:
            acc = analysis.input_access.get(inp)
            if acc and acc.is_2d:
                strides[_safe(inp)] = ["cols"]
        for out_id in region.output_names:
            strides[_safe(out_id)] = ["cols"]

    return strides


def _input_indices(inp: str, analysis: TileAnalysis, idx_var: str, out_size: int = 0) -> list[LoopExpr]:
    """Build per-dimension index expressions for reading an input tensor.

    Returns a list of index expressions, one per logical dimension:
    - scalar → []
    - 1D (pointwise) → [i] or [i % size]
    - row-vector → [j]
    - per-row → [row]
    - 2D → [row, j]
    """
    acc = analysis.input_access[inp]
    j = Var(idx_var)

    if analysis.pattern == "pointwise":
        if acc.is_scalar:
            return []
        if acc.size < out_size:
            return [Var("i") % acc.size]
        return [Var("i")]

    if acc.is_broadcast:
        # Broadcast input: use modulo to wrap the flat index into the
        # smaller input's element range.
        return [Var("row") * Var("cols") + j % acc.size]
    if acc.is_2d:
        return [Var("row"), j]
    if acc.is_per_row:
        return [Var("row")]
    if acc.is_row_vector:
        return [j]
    return []


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
# Contraction helpers
# ---------------------------------------------------------------------------


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
        for r in range(thread_m):
            rvar = f"r{reduce_fn}{r}"
            ops.append(AccumInit(rvar, reduce_fn))
            for c in range(thread_n):
                col_c = Var("bn") + Var("tc") + c
                ops.append(Guard(col_c.lt(Var("N")), [Accum(rvar, reduce_fn, RegAccess("c", [r, c]))]))
            ops.append(ShuffleReduce(rvar, reduce_fn, kind="warp_xor"))

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
                        indices: list = []
                    elif len(other_shape) <= 1:
                        indices = [Var("bn") + Var("tc") + c]
                    else:
                        row_e = Var("bm") + Var("tr") + r
                        col_e = Var("bn") + Var("tc") + c
                        indices = [row_e, col_e]
                    ld_name = f"_{prefix}_{safe}_{r}_{c}"
                    ops.append(Load(ld_name, safe, indices, "global"))
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
                indices = [Var("batch"), row, col]
            else:
                indices = [row, col]
            row_body.append(Store(output, indices, RegAccess("c", [r, c_idx]), "global", guard=col.lt(N), atomic=k_splits > 1))
        ops.append(Guard(row.lt(M), row_body))

    return ops
