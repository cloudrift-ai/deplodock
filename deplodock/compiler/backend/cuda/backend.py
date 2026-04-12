"""CUDA backend: converts an ExecutionPlan into a runnable Program.

Maps each OpKernel to a .cu template, computes grid/block dimensions,
and produces a Program that can be compiled with nvcc and executed.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.program import Buffer, Launch, Program, benchmark_program, run_program
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
            launch = _compile_op(op)
            launches.append(launch)

        return Program(
            name=plan.name,
            buffers=buffers,
            launches=launches,
            aliases=aliases,
        )

    def run(self, program: Program) -> ProgramResult:
        result = run_program(program)
        return ProgramResult(outputs=result.outputs, time_ms=result.time_ms)

    def benchmark(self, program: Program, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        result = benchmark_program(program, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
        )


# --- Per-op handlers ---


def _compile_matmul(op: OpKernel) -> Launch:
    """Compile a matmul op through the unified tiled generator.

    Defaults to tma_db strategy (TMA double-buffer with mbarrier pipelining).
    Falls back to naive if hints specify it.
    """
    from deplodock.compiler.backend.cuda.codegen import emit_kernel
    from deplodock.compiler.backend.cuda.generators.analysis import analyze
    from deplodock.compiler.backend.cuda.generators.tiled import lower_tiled
    from deplodock.compiler.backend.cuda.program import TmaDescriptorSpec
    from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReduceOp

    m = op.params.get("M", 1)
    n = op.params.get("N", 1)
    k = op.params.get("K", 1)

    # Build a FusedRegionOp for the matmul pattern, including epilogue ops.
    a_buf, b_buf = op.inputs[0], op.inputs[1]
    c_buf = op.outputs[0]
    all_region_ops = op.params.get("_region_ops", [])
    has_epilogue = len(all_region_ops) > 2

    if has_epilogue:
        # Use the full region_ops (contraction core + epilogue).
        region_ops_list = list(all_region_ops)
        # Collect extra external inputs from epilogue ops.
        internal_ids = {rid for rid, _, _ in region_ops_list}
        extra_inputs = []
        for _rid, _eop, epi_inputs in region_ops_list[2:]:
            for inp in epi_inputs:
                if inp not in internal_ids and inp != a_buf and inp != b_buf and inp not in extra_inputs:
                    extra_inputs.append(inp)
        input_names = [a_buf, b_buf] + extra_inputs
        region = FusedRegionOp(
            region_ops=region_ops_list,
            input_names=input_names,
            output_names=[c_buf],
        )
        shapes = {a_buf: (m, k), b_buf: (k, n)}
        # Contraction intermediate shapes.
        ew_id = region_ops_list[0][0]
        red_id = region_ops_list[1][0]
        shapes[ew_id] = (m, k, n)
        shapes[red_id] = (m, n)
        # Epilogue ops preserve matmul output shape.
        input_shapes = op.params.get("_input_shapes", {})
        for rid, _, _ in region_ops_list[2:]:
            shapes[rid] = (m, n)
        for inp in extra_inputs:
            shapes[inp] = input_shapes.get(inp, (n,))
        shapes[c_buf] = (m, n)
    else:
        region = FusedRegionOp(
            region_ops=[
                ("ew", ElementwiseOp("mul"), [a_buf, b_buf]),
                ("red", ReduceOp("sum", axis=1), ["ew"]),
            ],
            input_names=[a_buf, b_buf],
            output_names=[c_buf],
        )
        shapes = {a_buf: (m, k), b_buf: (k, n), "ew": (m, k, n), c_buf: (m, n)}
        extra_inputs = []

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

    # Epilogue ops are incompatible with k_splits > 1 (partial sums
    # from each split cannot have epilogue applied before reduction).
    if has_epilogue:
        matmul_hints["k_splits"] = 1

    # M-aware k_splits: when M is small relative to the tile size, the grid
    # has too few blocks to fill the GPU. Increase k_splits to add blocks
    # along the K dimension.
    if not has_epilogue:
        tile_m_val = 64  # must match tiled.py
        tile_n_val = 128  # must match tiled.py
        if m < tile_m_val:
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

    # Generate kernel through the unified path.
    name = _unique_name("matmul")
    analysis = analyze(region, shapes)
    kernel_def = lower_tiled(region, name, shapes, analysis, strategy=strategy, hints=matmul_hints)
    source = emit_kernel(kernel_def)

    bx, by, bz = kernel_def.block_size
    tile_m = kernel_def.tile_m or 64
    tile_n = kernel_def.tile_n or 128

    # CTA-swizzle grid (shared between naive and TMA).
    k_splits = int(matmul_hints.get("k_splits", 1))
    ntx = _cd(n, tile_n)
    nty = _cd(m, tile_m)
    grid = (ntx * _cd(nty, 8) * 8, 1, k_splits)

    # Build TMA metadata if the kernel uses TMA descriptors.
    tma_descs: list[TmaDescriptorSpec] = []
    smem_bytes = 0

    if kernel_def.tma_params:
        # TMA kernels use M/N/K as compile-time macros in the kernel body.
        source = f"#define M {m}\n#define N {n}\n#define K {k}\n{source}\n#undef M\n#undef N\n#undef K\n"

        bk = int(matmul_hints.get("block_k", 32))  # must match tiled.py
        a_tile = tile_m * bk
        b_tile = bk * tile_n
        smem_bytes = 2 * (a_tile + b_tile) * 4 + 256

        tma_descs = [
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

        # TMA: only C is a regular param. A/B come via descriptors.
        # Epilogue external inputs come before the output (matching kernel param order).
        args = [*extra_inputs, *op.outputs]
        if k_splits > 1:
            args.append(str(k_splits))
    else:
        # Naive: A, B, C, M, N, K as regular params.
        args = [*op.inputs, *op.outputs, str(m), str(n), str(k)]

    return Launch(
        kernel_source=source,
        kernel_name=name,
        grid=grid,
        block=(bx, by, bz),
        args=args,
        smem_bytes=smem_bytes,
        tma_descriptors=tma_descs,
        zero_outputs=list(op.outputs) if k_splits > 1 else [],
    )


def _is_matmul_region(op: OpKernel) -> tuple[bool, int, int, int]:
    """Check if a fused region has a contraction core: Reduce{sum}(Elementwise{mul}(A, B)).

    Allows optional pointwise epilogue ops after the core (e.g. bias add, activation).
    Returns (is_matmul, M, N, K) where M, N, K are inferred from shapes.
    """
    region_ops = op.params.get("_region_ops", [])
    if len(region_ops) < 2:
        return False, 0, 0, 0
    _, op0, inputs0 = region_ops[0]
    _, op1, _ = region_ops[1]
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    if not (isinstance(op0, ElementwiseOp) and op0.fn == "mul" and isinstance(op1, ReduceOp) and op1.fn == "sum"):
        return False, 0, 0, 0

    # The mul must be a broadcast (matmul-style): two distinct inputs whose
    # shapes produce a higher-dimensional output. Self-multiplies like
    # mul(x, x) in RMSNorm are NOT contractions.
    if len(inputs0) != 2 or inputs0[0] == inputs0[1]:
        return False, 0, 0, 0
    input_shapes = op.params.get("_input_shapes", {})
    a_shape = input_shapes.get(inputs0[0])
    b_shape = input_shapes.get(inputs0[1])
    if a_shape and b_shape and a_shape == b_shape:
        return False, 0, 0, 0  # same-shape mul → not a matmul

    # Remaining ops (index 2+) must all be pointwise epilogue.
    for i in range(2, len(region_ops)):
        _, epi_op, _ = region_ops[i]
        if not isinstance(epi_op, ElementwiseOp):
            return False, 0, 0, 0

    # Output shape may be multi-dimensional (batched): (batch, seq_len, N).
    # M = product of all dims except the last, N = last dim.
    shape = op.params.get("shape", (1,))
    n = int(shape[-1]) if len(shape) >= 1 and isinstance(shape[-1], int) else 1
    m = 1
    for d in shape[:-1]:
        if isinstance(d, int):
            m *= d
    if m == 0:
        m = 1

    # K = last dim of A (the reduced/shared dimension).
    input_shapes = op.params.get("_input_shapes", {})
    k = m  # fallback
    if len(inputs0) >= 1 and inputs0[0] in input_shapes:
        a_shape = input_shapes[inputs0[0]]
        if len(a_shape) >= 2 and isinstance(a_shape[-1], int):
            k = int(a_shape[-1])

    return True, m, n, k


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


def _compile_fused_region(op: OpKernel) -> Launch:
    """Compile a FusedRegionOp via the unified analysis → tiled generator path.

    For contractions with non-naive hints (TMA strategies), falls back to
    the dedicated matmul path which supports TMA descriptors.
    """
    source = op.params.get("kernel_source", "")
    if source:
        # Pre-generated source (from _compile_singleton) — use regex-based path.
        return _compile_fused_region_from_source(op, source)

    region_ops = op.params.get("_region_ops", [])
    if not region_ops:
        name = _unique_name("fused_region")
        src = f"__global__ void {name}() {{}}"
        return Launch(kernel_source=src, kernel_name=name, grid=(1, 1, 1), block=(1, 1, 1), args=[])

    # 2-op matmul pattern (mul + sum) goes through the dedicated SGEMM path,
    # which handles batched/transposed shapes and TMA strategies.
    is_matmul, m, n, k = _is_matmul_region(op)
    if is_matmul:
        op.params["M"] = m
        op.params["N"] = n
        op.params["K"] = k
        return _compile_matmul(op)

    from deplodock.compiler.backend.cuda.codegen import emit_kernel
    from deplodock.compiler.backend.cuda.generators.analysis import analyze
    from deplodock.compiler.backend.cuda.generators.tiled import lower_tiled

    region, shapes = _build_region_and_shapes(op)
    analysis = analyze(region, shapes)

    # ALL contraction patterns go through _compile_matmul (handles TMA, k_splits, tuning).
    if analysis.pattern == "contraction":
        op.params["M"] = analysis.rows
        op.params["N"] = analysis.cols
        op.params["K"] = analysis.k_dim
        return _compile_matmul(op)

    # Unified path: analyze → lower_tiled → emit_kernel.
    name = _unique_name("fused_region")
    kernel_def = lower_tiled(region, name, shapes, analysis)
    source = emit_kernel(kernel_def)

    # Build Launch from KernelDef metadata — no regex parsing needed.
    bx, by, bz = kernel_def.block_size
    buffer_args = [*op.inputs, *op.outputs]

    if analysis.pattern == "contraction":
        # CTA-swizzle grid with coarsened tiles (64×128).
        c_tile_m, c_tile_n = 64, 128  # must match tiled.py
        ntx = _cd(analysis.cols, c_tile_n)
        nty = _cd(analysis.rows, c_tile_m)
        grid = (ntx * _cd(nty, 8) * 8, 1, 1)
        block = (bx, by, bz)
        scalar_args = [str(analysis.rows), str(analysis.cols), str(analysis.k_dim)]
    elif analysis.pattern == "pointwise":
        total = math.prod(d for d in analysis.output_shape if isinstance(d, int))
        grid = (_cd(total, bx), 1, 1)
        block = (bx, 1, 1)
        scalar_args = [str(total)]
    else:
        # row_reduce or reduce_broadcast: compute rows/cols from the largest
        # input shape (matching the old regex-based dispatch behavior).
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
        grid = (rows, 1, 1)
        block = (bx, 1, 1)
        scalar_args = [str(rows), str(cols)]

    args = buffer_args + scalar_args
    return Launch(kernel_source=source, kernel_name=name, grid=grid, block=block, args=args)


def _compile_fused_region_from_source(op: OpKernel, source: str) -> Launch:
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

    buffer_args = [*op.inputs, *op.outputs]
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
    return Launch(kernel_source=source, kernel_name=name, grid=grid, block=block, args=args)


def _compile_singleton(op: OpKernel) -> Launch:
    """Compile an unfused elementwise/reduce op by wrapping it as a FusedRegionOp.

    Sets up _region_ops, _input_names, _output_names, and _shapes so
    _compile_fused_region can handle it through the unified path.
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

    return _compile_fused_region(op)


_OP_HANDLERS: dict[str, callable] = {
    "fused_region": _compile_fused_region,
}


def _compile_op(op: OpKernel) -> Launch:
    """Compile a single OpKernel to a CUDA Launch."""
    handler = _OP_HANDLERS.get(op.op)
    if handler is not None:
        return handler(op)

    # Unfused elementwise/reduce ops — wrap in a FusedRegionOp and use kernel_gen.
    if op.op.startswith("elementwise_") or op.op.startswith("reduce_"):
        return _compile_singleton(op)

    raise ValueError(f"Unknown op: {op.op!r}. Known ops: {list(_OP_HANDLERS.keys())}")
