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
# Contraction — delegate to legacy tiled.py via RawLoopOp for now
# ---------------------------------------------------------------------------


def _lower_contraction_naive(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (naive strategy) via legacy _lower_naive.

    Wraps the entire kernel body as a RawLoopOp temporarily.
    Will be decomposed into proper LoopIR ops in a follow-up.
    """
    from deplodock.compiler.backend.cuda.generators.tiled import _lower_naive

    kernel = _lower_naive(region, name, shapes, analysis, strategy="naive", hints=hints)
    return _kernel_def_to_loop_program(kernel)


def _lower_contraction_tma(
    region: FusedRegionOp,
    name: str,
    shapes: dict[str, tuple],
    analysis: TileAnalysis,
    hints: dict,
) -> LoopProgram:
    """Lower contraction (TMA double-buffer) via legacy _lower_naive."""
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
    # Extract the body between the opening { and closing }
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
