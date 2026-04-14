"""CUDA backend: converts an ExecutionPlan into a runnable Program.

Maps each OpKernel to a .cu template, computes grid/block dimensions,
and produces a Program that can be compiled with nvcc and executed.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.program import CudaLaunch, benchmark_program, run_program
from deplodock.compiler.backend.program import Buffer, Program
from deplodock.compiler.plan import ExecutionPlan, OpKernel

# Instance counter for unique kernel names when the same op appears multiple times.
_name_counters: dict[str, int] = {}


def _unique_name(op: str) -> str:
    """Generate a unique kernel function name for an op."""
    count = _name_counters.get(op, 0)
    _name_counters[op] = count + 1
    return f"{op}_{count}" if count > 0 else op


def _cd(a: int, b: int) -> int:
    return (a + b - 1) // b


class CudaBackend(Backend):
    """CUDA backend: ExecutionPlan → Program → nvcc → GPU."""

    def compile(self, plan: ExecutionPlan) -> Program:
        """Map OpKernels to .cu templates and build a Program."""
        _name_counters.clear()

        def _safe_prod(shape):
            return math.prod(d for d in shape if isinstance(d, int)) if shape else 1

        buffers = [Buffer(name=b.name, size=_safe_prod(b.shape), dtype="float", role=b.role) for b in plan.buffers]

        # Noop ops (reshape, transpose) become buffer aliases instead of
        # empty kernel launches. The output buffer shares the input's
        # device pointer — no allocation, no launch overhead.
        _NOOP_OPS = {"reshape", "transpose", "gather", "scatter"}
        aliases: dict[str, str] = {}
        launches = []
        for op in plan.ops:
            if op.op in _NOOP_OPS and op.inputs and op.outputs:
                # Resolve transitive aliases (reshape of reshape).
                target = op.inputs[0]
                while target in aliases:
                    target = aliases[target]
                aliases[op.outputs[0]] = target
                continue
            launches.extend(_compile_op(op))

        return Program(
            name=plan.name,
            buffers=buffers,
            launches=launches,
            aliases=aliases,
        )

    def run(self, program: Program, input_data: dict[str, list[float]] | None = None) -> ProgramResult:
        result = run_program(program, input_data=input_data)
        return ProgramResult(outputs=result.outputs, time_ms=result.time_ms)

    def benchmark(self, program: Program, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        result = benchmark_program(program, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
        )


# --- Per-op handlers ---


def _build_region_and_shapes(op: OpKernel):
    """Extract FusedRegionOp and shapes from an OpKernel's params."""
    from deplodock.compiler.ops import FusedRegionOp

    region_ops = op.params["_region_ops"]
    shapes = dict(op.params.get("_shapes", {}))
    if not shapes:
        input_shapes = op.params.get("_input_shapes", {})
        shapes.update(input_shapes)
        shapes[op.outputs[0]] = op.params.get("shape", (1,))

    # Use plan-level buffer names for the kernel params.
    # When fusion rewires graph edges, plan-level names (op.inputs) may differ
    # from original names (_input_names). Remap to keep kernel params aligned
    # with the buffers that will be passed as launch args.
    original_input_names = op.params.get("_input_names", list(op.inputs))
    original_output_names = op.params.get("_output_names", list(op.outputs))

    if original_input_names != list(op.inputs) or original_output_names != list(op.outputs):
        input_map = dict(zip(original_input_names, op.inputs, strict=False))
        output_map = dict(zip(original_output_names, op.outputs, strict=False))
        # Reverse output map: plan_name → original_name (for op node IDs)
        all_renames = {**input_map, **output_map}

        rewritten_ops = []
        for rid, rop, inp_ids in region_ops:
            new_inps = [all_renames.get(i, i) for i in inp_ids]
            # Also rename the op node_id if it's an output
            new_rid = all_renames.get(rid, rid)
            rewritten_ops.append((new_rid, rop, new_inps))

        region = FusedRegionOp(
            region_ops=rewritten_ops,
            input_names=list(op.inputs),
            output_names=list(op.outputs),
        )

        for orig, plan_name in {**input_map, **output_map}.items():
            if orig in shapes and plan_name not in shapes:
                shapes[plan_name] = shapes[orig]
    else:
        region = FusedRegionOp(
            region_ops=region_ops,
            input_names=original_input_names,
            output_names=original_output_names,
        )
    return region, shapes


def _select_strategy(op: OpKernel, analysis) -> tuple[str, dict]:
    """Select compilation strategy and hints based on analysis pattern.

    For contractions: loads GPU tuning profile, applies hint overrides,
    clamps thread_m to M, computes k_splits, validates block_k.
    For other patterns: returns ("naive", {}).
    """
    if analysis.pattern != "contraction":
        return "naive", {}

    m = analysis.rows
    n = analysis.cols
    k = analysis.k_dim
    batch_size = analysis.batch_size if hasattr(analysis, "batch_size") else 1
    has_epilogue = len(op.params.get("_region_ops", [])) > 2

    # Determine strategy and tile parameters from hints + tuning profile.
    explicit_hints = op.params.get("_hints", {})

    # Load tuning profile for this GPU if no explicit hints are set.
    matmul_hints: dict = {}
    if not any(hk.startswith("cuda.matmul.") for hk in explicit_hints):
        from deplodock.compiler.backend.cuda.tuning import default_matmul_strategy_map

        strategy_map, _profile = default_matmul_strategy_map()
        # Pick config based on max(M, N) as a size proxy.
        size = max(m, n)
        for threshold, cfg in strategy_map:
            if size <= threshold:
                matmul_hints = dict(cfg)
                break
        else:
            matmul_hints = dict(strategy_map[-1][1])

    # Explicit hints override tuning defaults.
    for hk, hv in explicit_hints.items():
        if hk.startswith("cuda.matmul."):
            matmul_hints[hk[len("cuda.matmul.") :]] = hv

    strategy = matmul_hints.get("strategy", "tma_db")

    # M-aware thread_m: clamp so tile_m does not exceed M.
    if strategy == "smem":
        ty = 4  # smem uses (32, 4) block
    else:
        ty = 8  # TMA/naive use (32, 8) block
    profile_tm = int(matmul_hints.get("thread_m", 8))
    max_tm = max(1, m // ty)
    if profile_tm > max_tm:
        matmul_hints["thread_m"] = max_tm

    tile_m_val = ty * int(matmul_hints.get("thread_m", 8))
    tile_n_val = 128  # fixed in tiled.py

    # Epilogue ops are incompatible with k_splits > 1 (partial sums
    # from each split cannot have epilogue applied before reduction).
    if has_epilogue:
        matmul_hints["k_splits"] = 1

    # M-aware k_splits: when M is small relative to the tile size, the grid
    # has too few blocks to fill the GPU. Increase k_splits to add blocks
    # along the K dimension.
    if not has_epilogue:
        if m <= tile_m_val:
            grid_m = _cd(m, tile_m_val)
            grid_n = _cd(n, tile_n_val)
            grid_blocks = grid_m * grid_n
            target_blocks = 170  # approximate SM count
            bk_est = int(matmul_hints.get("block_k", 32))
            max_ks = k // bk_est if bk_est > 0 else 1
            desired_ks = min(target_blocks // max(grid_blocks, 1), max_ks)
            desired_ks = min(desired_ks, 8)
            if desired_ks > 1:
                matmul_hints["k_splits"] = desired_ks

    # Validate BK and k_splits against actual K dimension.
    bk_val = int(matmul_hints.get("block_k", 32))
    # BK must not exceed K (otherwise TMA K-loop has 0 iterations).
    if bk_val > k:
        bk_val = max(1, k)
        matmul_hints["block_k"] = bk_val
    # K_per_split = (K / BK / k_splits) * BK must be > 0.
    ks = int(matmul_hints.get("k_splits", 1))
    if ks > 1 and k // bk_val < ks:
        matmul_hints["k_splits"] = max(1, k // bk_val)
    # Fall back to naive for very small K (TMA overhead not worth it).
    if k < 32 and strategy == "tma_db":
        strategy = "naive"
        matmul_hints["strategy"] = "naive"
    # Fall back to naive for batched matmuls (program.py TMA setup doesn't
    # support per-batch descriptor arrays yet).
    if batch_size > 1 and strategy == "tma_db":
        strategy = "naive"
        matmul_hints["strategy"] = "naive"
    # Online reduction (contraction + multi-reduce) uses 1D grid which is
    # incompatible with TMA descriptor setup.  Fall back to naive.
    has_multi_reduce = len(analysis.op_phases.reduces) > 1 if hasattr(analysis, "op_phases") else False
    if has_multi_reduce and strategy in ("tma_db", "smem"):
        strategy = "naive"
        matmul_hints["strategy"] = "naive"

    return strategy, matmul_hints


def _compute_grid(kernel_def, analysis, params: dict) -> tuple[tuple, int]:
    """Compute grid dimensions and shared memory bytes for a kernel.

    Returns (grid_tuple, smem_bytes).
    """
    if kernel_def.online_reduce and kernel_def.tile_m:
        # Online reduction: 1D grid over M-tiles only.
        m = params.get("M", analysis.rows)
        tile_m = kernel_def.tile_m
        batch_size = params.get("batch_size", 1)
        grid = (_cd(m, tile_m), 1, batch_size)
        # No TMA smem in online mode (uses naive K-loop for now)
        return grid, 0

    if kernel_def.tile_m and kernel_def.tile_n:
        # Contraction grid.
        m = params.get("M", analysis.rows)
        n = params.get("N", analysis.cols)
        batch_size = params.get("batch_size", 1)
        k_splits = int(params.get("_k_splits", 1))

        tile_m = kernel_def.tile_m
        tile_n = kernel_def.tile_n
        ntx = _cd(n, tile_n)
        nty = _cd(m, tile_m)
        grid_z = batch_size if batch_size > 1 else k_splits

        # smem strategy uses (32, 4) block → standard 2D grid.
        strategy = params.get("_strategy", "")
        if strategy == "smem":
            grid = (ntx, nty, grid_z)
        else:
            # CTA-swizzle grid (shared between naive and TMA).
            grid = (ntx * _cd(nty, 8) * 8, 1, grid_z)
    elif analysis.pattern == "pointwise":
        total = math.prod(d for d in analysis.output_shape if isinstance(d, int))
        bx = kernel_def.block_size[0]
        grid = (_cd(total, bx), 1, 1)
    else:
        # Reduce: row-per-block.
        out_shape = params.get("shape", (1,))
        total = math.prod(d for d in out_shape if isinstance(d, int))
        input_shapes = params.get("_input_shapes", {})
        best_shape = out_shape
        best_size = total
        for inp_shape in input_shapes.values():
            inp_size = math.prod(d for d in inp_shape if isinstance(d, int))
            if inp_size > best_size:
                best_shape = inp_shape
                best_size = inp_size
        if len(best_shape) >= 2:
            rows = math.prod(d for d in best_shape[:-1] if isinstance(d, int))
        else:
            rows = 1
        grid = (rows, 1, 1)

    # Shared memory for TMA kernels.
    smem_bytes = 0
    if kernel_def.tma_params:
        tile_m = kernel_def.tile_m or 64
        tile_n = kernel_def.tile_n or 128
        bk = int(params.get("_block_k", 32))
        a_tile = tile_m * bk
        b_tile = bk * tile_n
        smem_bytes = 2 * (a_tile + b_tile) * 4 + 256

    return grid, smem_bytes


def _build_launch_args(op: OpKernel, analysis) -> list[str]:
    """Build launch arguments for a kernel based on analysis pattern."""
    if analysis.pattern == "contraction":
        # Contraction: inputs + outputs + M, N, K + optional batch/k_splits.
        m = analysis.rows
        n = analysis.cols
        k = analysis.k_dim
        batch_size = analysis.batch_size if hasattr(analysis, "batch_size") else 1
        k_splits = int(op.params.get("_k_splits", 1))

        args = [*op.inputs, *op.outputs, str(m), str(n), str(k)]
        if batch_size > 1:
            args.append(str(batch_size))
        elif k_splits > 1:
            args.append(str(k_splits))
        return args

    # Non-contraction: inputs-not-in-outputs first, then outputs.
    output_set = set(op.outputs)
    buffer_args = [inp for inp in op.inputs if inp not in output_set]
    buffer_args.extend(op.outputs)

    if analysis.pattern == "pointwise":
        total = math.prod(d for d in analysis.output_shape if isinstance(d, int))
        return buffer_args + [str(total)]

    # Reduce: rows, cols.
    out_shape = op.params.get("shape", (1,))
    total = math.prod(d for d in out_shape if isinstance(d, int))
    input_shapes = op.params.get("_input_shapes", {})
    best_shape = out_shape
    best_size = total
    for inp_shape in input_shapes.values():
        inp_size = math.prod(d for d in inp_shape if isinstance(d, int))
        if inp_size > best_size:
            best_shape = inp_shape
            best_size = inp_size
    if len(best_shape) >= 2:
        rows = math.prod(d for d in best_shape[:-1] if isinstance(d, int))
        cols = best_shape[-1] if isinstance(best_shape[-1], int) else 1
    else:
        rows = 1
        cols = total
    return buffer_args + [str(rows), str(cols)]


def _build_tma_descs(kernel_def, op: OpKernel, hints: dict) -> list:
    """Build TMA descriptor specs if the kernel uses TMA."""
    from deplodock.compiler.backend.cuda.program import TmaDescriptorSpec

    if not kernel_def.tma_params:
        return []

    m = op.params.get("M", 1)
    n = op.params.get("N", 1)
    k = op.params.get("K", 1)
    a_buf = op.inputs[0]
    b_buf = op.inputs[1]
    tile_m = kernel_def.tile_m or 64
    tile_n = kernel_def.tile_n or 128
    bk = int(hints.get("block_k", 32))

    return [
        TmaDescriptorSpec(
            param_name=kernel_def.tma_params[0],
            buffer=a_buf,
            dims=[str(k), str(m)],
            strides=[str(k)],
            tile=[bk, tile_m],
        ),
        TmaDescriptorSpec(
            param_name=kernel_def.tma_params[1],
            buffer=b_buf,
            dims=[str(n), str(k)],
            strides=[str(n)],
            tile=[tile_n, bk],
        ),
    ]


def _resolve_contraction_shapes(op: OpKernel, region, shapes: dict):
    """Resolve symbolic shapes for contraction patterns to numeric values.

    Fusion may produce symbolic dims (e.g. 'K') in intermediate shapes.
    For contraction detection, the analyzer needs all-numeric shapes.
    When a contraction pattern (mul + reduce_sum) is detected, rebuild
    shapes with proper numeric M, N, K dims derived from input shapes.
    """
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    region_ops = op.params.get("_region_ops", [])
    if len(region_ops) < 2:
        return region, shapes

    _, op0, inputs0 = region_ops[0]
    _, op1, _ = region_ops[1]

    # Check for contraction core: mul + reduce_sum with two distinct inputs.
    if not (isinstance(op0, ElementwiseOp) and op0.fn == "mul" and isinstance(op1, ReduceOp) and op1.fn == "sum"):
        return region, shapes
    if len(inputs0) != 2 or inputs0[0] == inputs0[1]:
        return region, shapes

    # Same-shape mul (e.g. RMSNorm x*x) is not a contraction.
    input_shapes = op.params.get("_input_shapes", {})
    a_shape = input_shapes.get(inputs0[0])
    b_shape = input_shapes.get(inputs0[1])
    if a_shape and b_shape and a_shape == b_shape:
        return region, shapes

    # Check if shapes have symbolic dims that need resolution.
    has_symbolic = any(not isinstance(d, int) for shape in shapes.values() for d in shape)
    if not has_symbolic:
        return region, shapes

    # Extract M, N, K from input/output shapes (all numeric).
    if not a_shape or not b_shape:
        return region, shapes

    out_shape = op.params.get("shape", (1,))
    n = int(out_shape[-1]) if len(out_shape) >= 1 and isinstance(out_shape[-1], int) else 1
    m = 1
    for d in out_shape[:-1]:
        if isinstance(d, int):
            m *= d
    if m == 0:
        m = 1
    k = int(a_shape[-1]) if len(a_shape) >= 2 and isinstance(a_shape[-1], int) else m

    # Rebuild shapes with numeric values (2D — let analyze detect batch dims).
    a_buf, b_buf = inputs0[0], inputs0[1]
    c_buf = op.outputs[0]
    ew_id = region_ops[0][0]
    red_id = region_ops[1][0]

    new_shapes = dict(shapes)
    new_shapes[a_buf] = (m, k)
    new_shapes[b_buf] = (k, n)
    new_shapes[ew_id] = (m, k, n)
    new_shapes[red_id] = (m, n)
    new_shapes[c_buf] = (m, n)

    # Epilogue shapes.
    for rid, rop, _ in region_ops[2:]:
        if isinstance(rop, ReduceOp):
            new_shapes[rid] = (m, 1)
        else:
            new_shapes[rid] = (m, n)

    # Extra inputs keep their original shapes.
    for inp_name, inp_shape in input_shapes.items():
        if inp_name not in new_shapes:
            new_shapes[inp_name] = inp_shape

    return region, new_shapes


def _compile_single(op: OpKernel) -> CudaLaunch:
    """Unified compilation: compile any OpKernel to a single CudaLaunch.

    Handles contractions (with tuning, TMA, k_splits), pointwise, and
    reduce patterns through a single code path.
    """
    from deplodock.compiler.backend.cuda.generators.analysis import analyze
    from deplodock.compiler.backend.cuda.generators.tiled import lower_tiled
    from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel

    region_ops = op.params.get("_region_ops", [])
    if not region_ops:
        name = _unique_name("fused_region")
        src = f"__global__ void {name}() {{}}"
        return CudaLaunch(kernel_source=src, kernel_name=name, grid=(1, 1, 1), block=(1, 1, 1), args=[])

    # Build region + analyze.
    region, shapes = _build_region_and_shapes(op)
    # Resolve symbolic shapes for contraction patterns.
    region, shapes = _resolve_contraction_shapes(op, region, shapes)
    analysis = analyze(region, shapes)

    # For contractions, populate M/N/K/batch params from analysis.
    if analysis.pattern == "contraction":
        op.params["M"] = analysis.rows
        op.params["N"] = analysis.cols
        op.params["K"] = analysis.k_dim
        if analysis.batch_size > 1:
            op.params["batch_size"] = analysis.batch_size
            op.params["batch_dims"] = analysis.batch_dims

    # Select strategy (contraction → tuning profile; others → naive).
    strategy, hints = _select_strategy(op, analysis)

    # Lower → kernel.
    name = _unique_name(analysis.pattern or "fused_region")
    kernel_def, loop_prog, sched = lower_tiled(region, name, shapes, analysis, strategy=strategy, hints=hints if hints else {})
    source = emit_kernel(kernel_def)

    # Stash computed values for grid/args computation.
    k_splits = int(hints.get("k_splits", 1)) if analysis.pattern == "contraction" else 1
    batch_size = op.params.get("batch_size", 1)
    if batch_size > 1:
        k_splits = 1
    op.params["_k_splits"] = k_splits
    op.params["_strategy"] = strategy
    op.params["_block_k"] = int(hints.get("block_k", 32)) if hints else 32

    # Grid dims.
    grid, smem_bytes = _compute_grid(kernel_def, analysis, op.params)

    # Launch args.
    args = _build_launch_args(op, analysis)

    # TMA descriptors.
    tma_descs = _build_tma_descs(kernel_def, op, hints) if kernel_def.tma_params else []

    return CudaLaunch(
        kernel_source=source,
        kernel_name=name,
        grid=grid,
        block=kernel_def.block_size,
        args=args,
        smem_bytes=smem_bytes,
        tma_descriptors=tma_descs,
        zero_outputs=list(op.outputs) if k_splits > 1 else [],
        loop_ir=loop_prog,
        schedule=sched,
    )


def _split_contraction_softmax(op: OpKernel, analysis) -> list[CudaLaunch] | None:
    """Split a contraction+softmax region when N > tile_n.

    The fused softmax epilogue only works when all columns fit in one CTA
    (N <= tile_n = 128). When N exceeds this, split into:
      1. Matmul kernel (contraction core only, no epilogue)
      2. Softmax kernel (scale + max + sub + exp + sum + div)

    The matmul writes to the output buffer, and the softmax reads from /
    writes back to the same buffer (in-place).

    Returns None if the region doesn't need splitting.
    """
    phases = analysis.op_phases
    tile_n = 128

    # Only split when: contraction + multi-reduce + N > tile_n.
    if len(phases.reduces) <= 1 or analysis.cols <= tile_n:
        return None

    from deplodock.compiler.ops import ReduceOp

    region_ops = op.params.get("_region_ops", [])
    if len(region_ops) < 3:
        return None

    # Find the contraction boundary: first 2 ops are mul + reduce_sum.
    contraction_reduce_id = region_ops[1][0]

    # --- Launch 1: Matmul core (just mul + reduce_sum) ---
    matmul_op = OpKernel(
        op="fused_region",
        inputs=list(op.inputs[:2]),  # A, B only
        outputs=list(op.outputs),
        params={
            **op.params,
            "_region_ops": list(region_ops[:2]),
        },
    )
    matmul_launch = _compile_single(matmul_op)

    # --- Launch 2: Softmax (remaining ops, in-place on matmul output) ---
    # Collect remaining ops (everything after the contraction reduce).
    softmax_ops = list(region_ops[2:])
    out_buf = op.outputs[0]

    # Remap: replace contraction reduce ID with the output buffer name
    # so the softmax kernel reads from the actual allocated buffer.
    remapped_ops = []
    for nid, sop, inp_ids in softmax_ops:
        new_inputs = [out_buf if inp == contraction_reduce_id else inp for inp in inp_ids]
        remapped_ops.append((nid, sop, new_inputs))

    # Collect external inputs for the softmax kernel.
    internal_ids = {nid for nid, _, _ in remapped_ops}
    softmax_inputs = [out_buf]
    for _, _, inp_ids in remapped_ops:
        for inp in inp_ids:
            if inp not in internal_ids and inp not in softmax_inputs:
                softmax_inputs.append(inp)

    # Build shapes for the softmax kernel.
    input_shapes = op.params.get("_input_shapes", {})
    out_shape = op.params.get("shape", (1,))
    softmax_shapes = {out_buf: out_shape}
    for inp in softmax_inputs:
        if inp in input_shapes:
            softmax_shapes[inp] = input_shapes[inp]
        elif inp == out_buf:
            softmax_shapes[inp] = out_shape

    # Intermediate shapes from the original region.
    orig_shapes = op.params.get("_shapes", {})
    for nid, sop, _ in remapped_ops:
        if isinstance(sop, ReduceOp):
            # Row reductions produce (rows, 1) — keep last dim as 1.
            softmax_shapes[nid] = out_shape[:-1] + (1,)
        elif nid in orig_shapes:
            softmax_shapes[nid] = orig_shapes[nid]
        else:
            softmax_shapes[nid] = out_shape

    softmax_op = OpKernel(
        op="fused_region",
        inputs=softmax_inputs,
        outputs=[out_buf],
        params={
            "_region_ops": remapped_ops,
            "_input_names": softmax_inputs,
            "_output_names": [out_buf],
            "_input_shapes": {inp: softmax_shapes.get(inp, (1,)) for inp in softmax_inputs},
            "_shapes": softmax_shapes,
            "shape": out_shape,
        },
    )
    softmax_launch = _compile_single(softmax_op)

    return [matmul_launch, softmax_launch]


def _compile_fused_region(op: OpKernel) -> list[CudaLaunch]:
    """Compile a FusedRegionOp via the unified analysis → tiled generator path.

    Returns a list of CudaLaunch objects (usually one, but multi-reduce
    contractions with N > tile_n are split into matmul + softmax).
    """
    source = op.params.get("kernel_source", "")
    if source:
        # Pre-generated source (from _compile_singleton) — use regex-based path.
        return [_compile_fused_region_from_source(op, source)]

    region_ops = op.params.get("_region_ops", [])
    if not region_ops:
        name = _unique_name("fused_region")
        src = f"__global__ void {name}() {{}}"
        return [CudaLaunch(kernel_source=src, kernel_name=name, grid=(1, 1, 1), block=(1, 1, 1), args=[])]

    # Contraction + multi-reduce (e.g. softmax) is handled by the online
    # reduction path in lower_generic — no split needed.
    return [_compile_single(op)]


def _compile_fused_region_from_source(op: OpKernel, source: str) -> CudaLaunch:
    """Compile a fused region from pre-generated kernel source (regex-based path).

    Used when kernel_source is already set (e.g., from _compile_singleton).
    """
    import re

    name = _unique_name("fused_region")

    # Rename kernel function in source.
    match = re.search(r"__global__.*?void\s+(\w+)", source, re.DOTALL)
    if match:
        old_name = match.group(1)
        source = source.replace(old_name, name)

    # Count params to determine arg count.
    param_match = re.search(r"void\s+\w+\((.*?)\)\s*\{", source, re.DOTALL)
    if param_match:
        param_text = param_match.group(1)
        param_count = len([p.strip() for p in param_text.split(",") if p.strip()])
    else:
        param_count = 0

    # Build buffer args matching kernel param order: inputs-not-in-outputs
    # first, then outputs. This mirrors the param deduplication in tiled.py
    # where in-place buffers (appearing in both inputs and outputs) are
    # emitted only as float* (read-write) in the output position.
    output_set = set(op.outputs)
    buffer_args = [inp for inp in op.inputs if inp not in output_set]
    buffer_args.extend(op.outputs)
    scalar_count = param_count - len(buffer_args)

    out_shape = op.params.get("shape", (1,))
    total = 1
    for d in out_shape:
        if isinstance(d, int):
            total *= d

    input_shapes = op.params.get("_input_shapes", {})
    best_shape = out_shape
    best_size = total
    for inp_shape in input_shapes.values():
        inp_size = 1
        for d in inp_shape:
            if isinstance(d, int):
                inp_size *= d
        if inp_size > best_size:
            best_shape = inp_shape
            best_size = inp_size

    if scalar_count >= 2:
        if len(best_shape) >= 2:
            rows = 1
            for d in best_shape[:-1]:
                if isinstance(d, int):
                    rows *= d
            cols = best_shape[-1] if isinstance(best_shape[-1], int) else 1
        else:
            rows = 1
            cols = total
        grid = (rows, 1, 1)
        block = (256, 1, 1)
        scalar_args = [str(rows), str(cols)]
    elif scalar_count >= 1:
        grid = (_cd(total, 256), 1, 1)
        block = (256, 1, 1)
        scalar_args = [str(total)]
    else:
        grid = (1, 1, 1)
        block = (256, 1, 1)
        scalar_args = []

    args = buffer_args + scalar_args
    return CudaLaunch(kernel_source=source, kernel_name=name, grid=grid, block=block, args=args)


def _compile_singleton(op: OpKernel) -> None:
    """Prepare an unfused elementwise/reduce op for the fused region path.

    Sets up _region_ops, _input_names, _output_names, and _shapes on
    op.params so _compile_fused_region can handle it through the unified path.
    """
    # Reconstruct the op object from the OpKernel tag.
    tag = op.op
    if tag.startswith("elementwise_"):
        from deplodock.compiler.ops import ElementwiseOp

        fn = tag[len("elementwise_") :]
        prim_op = ElementwiseOp(fn=fn)
    elif tag.startswith("reduce_"):
        from deplodock.compiler.ops import ReduceOp

        fn = tag[len("reduce_") :]
        axis = op.params.get("axis", -1)
        prim_op = ReduceOp(fn=fn, axis=axis)
    else:
        raise ValueError(f"_compile_singleton: unsupported op tag {tag!r}")

    # Set up params for the unified path in _compile_fused_region.
    op.params["_region_ops"] = [(op.outputs[0], prim_op, list(op.inputs))]
    op.params["_input_names"] = list(op.inputs)
    op.params["_output_names"] = list(op.outputs)

    # Build shapes.
    out_shape = op.params.get("shape", (1,))
    shapes = {op.outputs[0]: out_shape}
    input_shapes = op.params.get("_input_shapes", {})
    for inp in op.inputs:
        if inp in input_shapes:
            shapes[inp] = input_shapes[inp]
        else:
            shapes[inp] = out_shape
    op.params["_shapes"] = shapes


def _compile_op(op: OpKernel) -> list[CudaLaunch]:
    """Compile a single OpKernel to one or more CUDA launches."""
    if op.op == "fused_region":
        return _compile_fused_region(op)

    # Unfused elementwise/reduce ops — wrap as a FusedRegionOp first.
    if op.op.startswith("elementwise_") or op.op.startswith("reduce_"):
        _compile_singleton(op)  # sets up _region_ops etc. on op.params
        return _compile_fused_region(op)

    raise ValueError(f"Unknown op: {op.op!r}")
