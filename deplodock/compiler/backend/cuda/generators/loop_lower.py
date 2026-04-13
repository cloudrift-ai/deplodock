"""Lower FusedRegionOp + TileAnalysis to LoopIR.

Each pattern (pointwise, row_reduce, reduce_broadcast, contraction) has a
dedicated lowering function that produces a LoopProgram.  Strategy variants
(naive, tma_db, smem) are handled within contraction lowering.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, _needed_by
from deplodock.compiler.backend.loop_ir import (
    Accumulate,
    Alloc,
    Compute,
    Guard,
    Load,
    LoopBinOp,
    LoopBuiltin,
    LoopExpr,
    LoopFuncCall,
    LoopLiteral,
    LoopNest,
    LoopProgram,
    LoopVar,
    ParallelAxis,
    RawLoopOp,
    RegAccess,
    Store,
    WarpReduce,
)
from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReshapeOp, TransposeOp


def lower_to_loop_ir(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    *,
    strategy: str = "naive",
    hints: dict | None = None,
) -> LoopProgram:
    """Lower a FusedRegionOp to LoopIR based on its TileAnalysis."""
    if analysis.pattern == "pointwise":
        return _lower_pointwise(region, name, shapes, analysis)

    if analysis.pattern == "contraction":
        if strategy == "smem":
            return _lower_contraction_smem(region, name, shapes, analysis, hints or {})
        if strategy == "tma_db":
            return _lower_contraction_tma(region, name, shapes, analysis, hints or {})
        return _lower_contraction_naive(region, name, shapes, analysis, hints or {})

    # row_reduce or reduce_broadcast
    has_multi_reduce = len(analysis.op_phases.reduces) > 1
    if has_multi_reduce:
        return _lower_multi_reduce(region, name, shapes, analysis)
    return _lower_single_reduce(region, name, shapes, analysis)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def _build_loop_expr(fn: str, a: LoopExpr, b: LoopExpr | None = None) -> LoopExpr | None:
    """Map an ElementwiseOp function name to a LoopExpr."""
    if fn == "mul":
        return LoopBinOp("*", a, b)
    if fn == "add":
        return LoopBinOp("+", a, b)
    if fn == "sub":
        return LoopBinOp("-", a, b)
    if fn == "div":
        return LoopBinOp("/", a, b)
    if fn == "neg":
        return LoopBinOp("-", LoopLiteral(0.0), a)
    if fn == "exp":
        return LoopFuncCall("expf", [a])
    if fn == "rsqrt":
        return LoopFuncCall("rsqrtf", [a])
    if fn == "recip":
        return LoopBinOp("/", LoopLiteral(1.0), a)
    if fn == "relu":
        return LoopFuncCall("fmaxf", [LoopLiteral(0.0), a])
    return None


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

    if analysis.pattern == "pointwise":
        if acc.is_scalar:
            return LoopLiteral(0, "int")
        if acc.size < out_size:
            return LoopBinOp("%", LoopVar("i"), LoopLiteral(acc.size, "int"))
        return LoopVar("i")

    # row_reduce / reduce_broadcast
    if acc.is_2d:
        return LoopBinOp("+", LoopBinOp("*", LoopVar("row"), LoopVar("cols")), LoopVar(idx_var))
    if acc.is_per_row:
        return LoopVar("row")
    if acc.is_row_vector:
        return LoopVar(idx_var)
    if acc.is_scalar:
        return LoopLiteral(0, "int")
    return LoopLiteral(0, "int")


def _emit_loop_ops(
    ops: list,
    var_map: dict[str, LoopExpr],
    prefix: str,
    body: list,
) -> None:
    """Walk ops and emit Compute nodes, updating var_map.

    Appends to `body` in-place.
    """
    for node_id, op, input_ids in ops:
        if isinstance(op, (ReshapeOp, TransposeOp)):
            if input_ids and input_ids[0] in var_map:
                var_map[node_id] = var_map[input_ids[0]]
            continue

        if isinstance(op, ElementwiseOp):
            a = var_map.get(input_ids[0], LoopLiteral(0.0)) if input_ids else LoopLiteral(0.0)
            b = var_map.get(input_ids[1], LoopLiteral(0.0)) if len(input_ids) > 1 else LoopLiteral(0.0)
            expr = _build_loop_expr(op.fn, a, b)
            if expr is None:
                expr = LoopLiteral(0.0)
            var_name = f"{prefix}{_safe(node_id)}"
            body.append(Compute(var_name, op.fn, [a] if len(input_ids) == 1 else [a, b]))
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

    body.append(Guard(LoopBinOp("<", LoopVar("i"), LoopVar("n")), guard_body))

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
                    epi_body.append(Compute(var_name, op.fn, [a] if len(input_ids) == 1 else [a, b]))
                    epi_var_map[node_id] = LoopVar(var_name)

            _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

            for out_id in region.output_names:
                val = epi_var_map.get(out_id, LoopLiteral(0.0))
                idx = LoopBinOp("+", LoopBinOp("*", LoopVar("row"), LoopVar("cols")), LoopVar("j"))
                epi_body.append(Store(_safe(out_id), idx, val, "global"))

            guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), epi_body))
        else:
            # Scalar epilogue (thread 0 only)
            _emit_loop_ops(epilogue_ops, var_map, "e_", guarded)
            for out_id in region.output_names:
                val = var_map.get(out_id, LoopLiteral(0.0))
                guarded.append(
                    Guard(
                        LoopBinOp("==", LoopBuiltin("threadIdx.x"), LoopLiteral(0, "int")),
                        [Store(_safe(out_id), LoopVar("row"), val, "global")],
                    )
                )
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            guarded.append(
                Guard(
                    LoopBinOp("==", LoopBuiltin("threadIdx.x"), LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )

    body.append(Guard(LoopBinOp("<", LoopVar("row"), LoopVar("rows")), guarded))

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
                pass_body.append(Compute(var_name, pop.fn, [a] if len(pinp) == 1 else [a, b]))
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
                epi_body.append(Compute(var_name, pop.fn, [a] if len(pinp) == 1 else [a, b]))
                epi_var_map[pid] = LoopVar(var_name)

        _emit_loop_ops(epilogue_ops, epi_var_map, "e_", epi_body)

        for out_id in region.output_names:
            val = epi_var_map.get(out_id, LoopLiteral(0.0))
            idx = LoopBinOp("+", LoopBinOp("*", LoopVar("row"), LoopVar("cols")), LoopVar("j"))
            epi_body.append(Store(_safe(out_id), idx, val, "global"))

        guarded.append(LoopNest("j", LoopBuiltin("threadIdx.x"), LoopVar("cols"), LoopBuiltin("blockDim.x"), epi_body))
    else:
        # No epilogue — write last reduce result (thread 0 only)
        for out_id in region.output_names:
            val = var_map.get(out_id, LoopLiteral(0.0))
            guarded.append(
                Guard(
                    LoopBinOp("==", LoopBuiltin("threadIdx.x"), LoopLiteral(0, "int")),
                    [Store(_safe(out_id), LoopVar("row"), val, "global")],
                )
            )

    body.append(Guard(LoopBinOp("<", LoopVar("row"), LoopVar("rows")), guarded))

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


def _contraction_epilogue_ops(
    phases,
    shapes: dict[str, tuple],
    input_names: list[str],
    thread_m: int,
    thread_n: int,
) -> list:
    """Generate LoopIR ops for contraction epilogue (bias, activation, etc.).

    Operates on the register array ``c[r][c]`` in-place.  Only loads from
    external inputs (buffers in ``input_names``); internal intermediates
    are already in the accumulator chain.
    """
    if not phases.epilogue:
        return []

    ops: list = []
    prev_id = phases.reduces[0][0]
    input_set = set(input_names)

    for epi_nid, epi_op, epi_inputs in phases.epilogue:
        fn = epi_op.fn
        other = None
        if len(epi_inputs) == 2:
            for inp in epi_inputs:
                if inp != prev_id:
                    other = inp
                    break

        # Only emit loads for external inputs.  If `other` is an internal
        # region node (not a kernel parameter), skip — the value is already
        # in the register tile via the prev_id chain.
        if other is not None and other not in input_set:
            other = None

        for r in range(thread_m):
            for c_idx in range(thread_n):
                acc = RegAccess("c", [r, c_idx])
                dst = f"c{r}{c_idx}"

                if fn in ("add", "sub", "mul", "div") and other is not None:
                    other_shape = shapes.get(other, ())
                    safe = _safe(other)
                    if len(other_shape) <= 1:
                        idx = LoopBinOp("+", LoopBinOp("+", LoopVar("bn"), LoopVar("tc")), LoopLiteral(c_idx, "int"))
                    else:
                        row_idx = LoopBinOp("+", LoopBinOp("+", LoopVar("bm"), LoopVar("tr")), LoopLiteral(r, "int"))
                        col_idx = LoopBinOp("+", LoopBinOp("+", LoopVar("bn"), LoopVar("tc")), LoopLiteral(c_idx, "int"))
                        idx = LoopBinOp("+", LoopBinOp("*", row_idx, LoopVar("N")), col_idx)
                    other_val = LoopVar(f"_ep_{safe}_{r}_{c_idx}")
                    ops.append(Load(other_val.name, safe, idx, "global"))
                    if epi_inputs[0] == prev_id:
                        ops.append(Compute(dst, fn, [acc, other_val]))
                    else:
                        ops.append(Compute(dst, fn, [other_val, acc]))
                elif fn in ("relu", "neg", "exp", "rsqrt", "recip"):
                    ops.append(Compute(dst, fn, [acc]))

        prev_id = epi_nid

    return ops


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

    for r in range(thread_m):
        row = LoopBinOp("+", LoopBinOp("+", LoopVar("bm"), LoopVar("tr")), LoopLiteral(r, "int"))
        row_guard = LoopBinOp("<", row, LoopVar("M"))
        row_body: list = []

        for c_idx in range(thread_n):
            col = LoopBinOp("+", LoopBinOp("+", LoopVar("bn"), LoopVar("tc")), LoopLiteral(c_idx, "int"))
            col_guard = LoopBinOp("<", col, LoopVar("N"))

            if is_batched:
                base = LoopBinOp("*", LoopVar("batch"), LoopBinOp("*", LoopVar("M"), LoopVar("N")))
                idx = LoopBinOp("+", base, LoopBinOp("+", LoopBinOp("*", row, LoopVar("N")), col))
            else:
                idx = LoopBinOp("+", LoopBinOp("*", row, LoopVar("N")), col)

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

    Grid setup and K-loop body use RawLoopOp.  Accumulator declaration,
    epilogue, and write phase use proper LoopIR with register arrays.

    Falls back to the legacy wrapper for contraction+softmax epilogue
    (multi-reduce with in-register warp shuffles).
    """
    dims = _contraction_dims(analysis, hints)
    thread_m, thread_n = dims["thread_m"], dims["thread_n"]
    tile_m, tile_n = dims["tile_m"], dims["tile_n"]
    phases = analysis.op_phases

    # Contraction + softmax (multi-reduce): complex in-register warp shuffles.
    # Fall back to legacy until softmax epilogue is ported to LoopIR.
    if len(phases.reduces) > 1 and analysis.cols <= tile_n:
        from deplodock.compiler.backend.cuda.generators.tiled import _lower_naive

        kernel = _lower_naive(region, name, shapes, analysis, strategy="naive", hints=hints)
        return _kernel_def_to_loop_program(kernel)

    tx, ty = dims["tx"], dims["ty"]
    is_batched = analysis.batch_size > 1
    k_splits = int(hints.get("k_splits", 1))

    a_name = _safe(analysis.contraction_a)
    b_name = _safe(analysis.contraction_b)
    out_id = region.output_names[0]

    # Params
    params = _build_params(region)
    params.extend([("int", "M"), ("int", "N"), ("int", "K")])
    if is_batched:
        params.append(("int", "batch_count"))

    body: list = []

    # Grid setup (CTA-swizzle) — RawLoopOp for now
    batch_decl = "int batch=blockIdx.z;\n" if is_batched else ""
    grid_code = (
        f"{batch_decl}"
        f"int tr=threadIdx.y*{thread_m},tc=threadIdx.x*{thread_n};\n"
        f"const int SWIZ=8;\n"
        f"int ntx=(N+{tile_n - 1})/{tile_n};\n"
        f"int nty=(M+{tile_m - 1})/{tile_m};\n"
        f"int pid=blockIdx.x+blockIdx.y*gridDim.x;\n"
        f"int grp=pid/(ntx*SWIZ);\n"
        f"int rem=pid%(ntx*SWIZ);\n"
        f"int by_s=grp*SWIZ+rem%SWIZ;\n"
        f"int bx_s=rem/SWIZ;\n"
        f"if(by_s>=nty||bx_s>=ntx)return;\n"
        f"int bm=by_s*{tile_m},bn=bx_s*{tile_n};"
    )
    body.append(RawLoopOp(grid_code, "CTA-swizzle grid setup"))

    # Accumulator register array
    body.append(Alloc("c", "float", (thread_m, thread_n), "reg", LoopLiteral(0.0)))

    # K-loop body — RawLoopOp (bounds-checked global loads + FMA)
    if is_batched:
        a_local, b_local = "Ab", "Bb"
        batch_ptrs = f"const float*Ab={a_name}+batch*M*K;\nconst float*Bb={b_name}+batch*K*N;\n"
    else:
        a_local, b_local = a_name, b_name
        batch_ptrs = ""

    naive_a_lines = []
    for r in range(thread_m):
        fma = "".join(f"c{r}{c}+=a{r}*b{c};" for c in range(thread_n))
        naive_a_lines.append(f"float a{r}=(bm+tr+{r}<M)?{a_local}[(bm+tr+{r})*K+k]:0.0f;{fma}")
    naive_fma = "\n        ".join(naive_a_lines)

    k_loop = (
        f"{batch_ptrs}"
        f"for(int k=0;k<K;k++){{\n"
        f"    float b0=(bn+tc<N)?{b_local}[k*N+bn+tc]:0.0f,"
        f"b1=(bn+tc+1<N)?{b_local}[k*N+bn+tc+1]:0.0f,"
        f"b2=(bn+tc+2<N)?{b_local}[k*N+bn+tc+2]:0.0f,"
        f"b3=(bn+tc+3<N)?{b_local}[k*N+bn+tc+3]:0.0f;\n"
        f"        {naive_fma}\n"
        f"}}"
    )
    body.append(RawLoopOp(k_loop, "naive K-loop"))

    # Epilogue — proper LoopIR with register arrays
    epi_ops = _contraction_epilogue_ops(phases, shapes, region.input_names, thread_m, thread_n)
    body.extend(epi_ops)

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
    """Lower contraction (TMA double-buffer) via legacy _lower_naive.

    TMA requires complex inline asm for mbarrier + cp.async.bulk.
    The entire kernel body stays as RawLoopOp until TMA gets proper LoopIR ops.
    """
    from deplodock.compiler.backend.cuda.generators.tiled import _lower_naive

    kernel = _lower_naive(region, name, shapes, analysis, strategy="tma_db", hints=hints)
    return _kernel_def_to_loop_program(kernel)


def _lower_contraction_smem(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (smem strategy) via legacy _lower_smem."""
    from deplodock.compiler.backend.cuda.generators.tiled import _lower_smem

    kernel = _lower_smem(region, name, shapes, analysis, hints=hints)
    return _kernel_def_to_loop_program(kernel)


def _kernel_def_to_loop_program(kernel) -> LoopProgram:
    """Convert a legacy KernelDef to a LoopProgram wrapping the body as RawLoopOp."""
    from deplodock.compiler.backend.codegen import emit_kernel

    source = emit_kernel(kernel)
    brace_start = source.index("{") + 1
    brace_end = source.rindex("}")
    body_source = source[brace_start:brace_end].strip()

    return LoopProgram(
        name=kernel.name,
        params=[(p.dtype, p.name) for p in kernel.params],
        body=[RawLoopOp(body_source, "legacy contraction kernel body")],
        block_size=kernel.block_size,
        tile_m=kernel.tile_m,
        tile_n=kernel.tile_n,
        grid_2d=kernel.grid_2d,
        tma_params=kernel.tma_params,
        batched=getattr(kernel, "batched", False),
        includes=kernel.includes,
        extra_smem_bytes=getattr(kernel, "extra_smem_bytes", 0),
        min_blocks_per_sm=getattr(kernel, "min_blocks_per_sm", 0),
    )
