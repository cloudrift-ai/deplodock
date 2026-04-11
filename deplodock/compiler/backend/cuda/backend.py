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

        buffers = [Buffer(name=b.name, size=math.prod(b.shape), dtype="float", role=b.role) for b in plan.buffers]

        launches = []
        for op in plan.ops:
            launch = _compile_op(op)
            launches.append(launch)

        return Program(
            name=plan.name,
            buffers=buffers,
            launches=launches,
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
    """Compile a matmul op using the SGEMM kernel from lower.py.

    Builds a primitive matmul graph and lowers it through the standard path.
    Uses the naive strategy by default. The TMA strategies require
    KernelDef metadata (TMA descriptors, tile sizes) that the Program
    abstraction doesn't support yet — use scripts/bench_matmul.py for
    TMA benchmarking.
    """
    from deplodock.compiler.backend.cuda.codegen import emit_kernel
    from deplodock.compiler.backend.cuda.lower import MatmulConfig, lower_graph
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp

    m = op.params.get("M", 1)
    n = op.params.get("N", 1)
    k = op.params.get("K", 1)

    # Build primitive matmul graph: C = Reduce{sum}(Elementwise{mul}(A, B))
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("AB", (m, k, n)), node_id="ew")
    c = g.add_node(ReduceOp(fn="sum", axis=1), [ew], Tensor("C", (m, n)), node_id="C")
    g.outputs = [c]

    config = MatmulConfig(strategy="naive")
    kernel_def = lower_graph(g, config=config)
    source = emit_kernel(kernel_def)

    bx, by, bz = kernel_def.block_size
    gx = _cd(n, bx)
    gy = _cd(m, by)

    name = _unique_name("matmul")
    # Rename the kernel function in the source.
    source = source.replace(kernel_def.name, name)

    return Launch(
        kernel_source=source,
        kernel_name=name,
        grid=(gx, gy, 1),
        block=(bx, by, bz),
        args=[*op.inputs, *op.outputs, str(m), str(n), str(k)],
    )


def _is_matmul_region(op: OpKernel) -> tuple[bool, int, int, int]:
    """Check if a fused region is Reduce{sum}(Elementwise{mul}(A, B)).

    Returns (is_matmul, M, N, K) where M, N, K are inferred from shapes.
    K is the shared dimension between the two inputs to the mul op:
    A(M, K) @ B(K, N) → C(M, N).
    """
    region_ops = op.params.get("_region_ops", [])
    if len(region_ops) != 2:
        return False, 0, 0, 0
    _, op0, inputs0 = region_ops[0]
    _, op1, _ = region_ops[1]
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    if not (isinstance(op0, ElementwiseOp) and op0.fn == "mul" and isinstance(op1, ReduceOp) and op1.fn == "sum"):
        return False, 0, 0, 0

    # Output shape is (M, N).
    shape = op.params.get("shape", (1,))
    m = int(shape[0]) if len(shape) >= 2 and isinstance(shape[0], int) else 1
    n = int(shape[-1]) if len(shape) >= 1 and isinstance(shape[-1], int) else 1

    # K: the shared (reduced) dimension. Extract from input shapes.
    # A has shape (M, K), B has shape (K, N). K is the last dim of A or first dim of B.
    input_shapes = op.params.get("_input_shapes", {})
    k = m  # fallback
    if len(inputs0) >= 1 and inputs0[0] in input_shapes:
        a_shape = input_shapes[inputs0[0]]
        if len(a_shape) >= 2 and isinstance(a_shape[-1], int):
            k = int(a_shape[-1])
    elif len(inputs0) >= 2 and inputs0[1] in input_shapes:
        b_shape = input_shapes[inputs0[1]]
        if len(b_shape) >= 1 and isinstance(b_shape[0], int):
            k = int(b_shape[0])

    return True, m, n, k


def _compile_fused_region(op: OpKernel) -> Launch:
    """Compile a FusedRegionOp — uses SGEMM for matmul patterns, generated kernel otherwise."""
    is_matmul, m, n, k = _is_matmul_region(op)
    if is_matmul:
        # Inject M, N, K into params for _compile_matmul.
        op.params["M"] = m
        op.params["N"] = n
        op.params["K"] = k
        return _compile_matmul(op)

    source = op.params.get("kernel_source", "")
    name = _unique_name("fused_region")

    # Auto-generate kernel if source wasn't pre-generated.
    if not source:
        region_ops = op.params.get("_region_ops", [])
        if region_ops:
            source = _generate_fused_kernel(op, name)
        else:
            source = f"__global__ void {name}() {{}}"
            return Launch(kernel_source=source, kernel_name=name, grid=(1, 1, 1), block=(1, 1, 1), args=[])

    # Rename kernel function in source to unique name.
    # The source has a __global__ void <something>(...) — replace the function name.
    import re

    match = re.search(r"__global__\s+void\s+(\w+)", source)
    if match:
        old_name = match.group(1)
        source = source.replace(old_name, name)

    # Count actual kernel params from source to determine arg count.
    # The kernel signature is: __global__ void name(params...) {
    import re as _re

    param_match = _re.search(r"__global__\s+void\s+\w+\((.*?)\)", source, _re.DOTALL)
    if param_match:
        param_text = param_match.group(1)
        param_count = len([p.strip() for p in param_text.split(",") if p.strip()])
    else:
        param_count = 0

    # Build args: buffer names for pointers, then scalar dims.
    buffer_args = [*op.inputs, *op.outputs]
    # How many scalar args does the kernel expect?
    scalar_count = param_count - len(buffer_args)

    out_shape = op.params.get("shape", (1,))
    total = 1
    for d in out_shape:
        if isinstance(d, int):
            total *= d

    # For reduction kernels, we need rows/cols from an INPUT shape (which has
    # more elements than the output). The output is smaller after reduction.
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

    # Determine grid from scalar args.
    if scalar_count >= 2:
        # Reduction kernel: (rows, cols) from the pre-reduction shape.
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
        # Pointwise kernel: (n)
        grid = (_cd(total, 256), 1, 1)
        block = (256, 1, 1)
        scalar_args = [str(total)]
    else:
        grid = (1, 1, 1)
        block = (256, 1, 1)
        scalar_args = []

    args = buffer_args + scalar_args
    return Launch(kernel_source=source, kernel_name=name, grid=grid, block=block, args=args)


def _generate_fused_kernel(op: OpKernel, name: str) -> str:
    """Generate kernel source from region_ops stored in the OpKernel params."""
    from deplodock.compiler.backend.cuda.kernel_gen import generate_kernel
    from deplodock.compiler.ops import FusedRegionOp

    region_ops = op.params["_region_ops"]
    # Use shapes stored on the FusedRegionOp (captured during auto_fuse).
    shapes = dict(op.params.get("_shapes", {}))
    # Fallback: merge _input_shapes + output shape if _shapes is empty.
    if not shapes:
        input_shapes = op.params.get("_input_shapes", {})
        shapes.update(input_shapes)
        shapes[op.outputs[0]] = op.params.get("shape", (1,))

    # Use original input/output names from the FusedRegionOp, not the
    # plan-level buffer names. The plan may rename to fused node IDs
    # but the kernel source must use the original region node IDs
    # because region_ops reference them.
    original_input_names = op.params.get("_input_names", list(op.inputs))
    original_output_names = op.params.get("_output_names", list(op.outputs))
    region = FusedRegionOp(
        region_ops=region_ops,
        input_names=original_input_names,
        output_names=original_output_names,
    )
    return generate_kernel(region, name, shapes)


def _compile_singleton(op: OpKernel) -> Launch:
    """Compile an unfused elementwise/reduce op by wrapping it as a FusedRegionOp."""
    from deplodock.compiler.backend.cuda.kernel_gen import generate_kernel
    from deplodock.compiler.ops import FusedRegionOp

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
        return _compile_noop(op)

    # Build a FusedRegionOp with this single op.
    region = FusedRegionOp(
        region_ops=[(op.outputs[0], prim_op, list(op.inputs))],
        input_names=list(op.inputs),
        output_names=list(op.outputs),
    )

    # Build shapes from the OpKernel params.
    out_shape = op.params.get("shape", (1,))
    shapes = {op.outputs[0]: out_shape}
    input_shapes = op.params.get("_input_shapes", {})
    for inp in op.inputs:
        if inp in input_shapes:
            shapes[inp] = input_shapes[inp]
        else:
            shapes[inp] = out_shape  # fallback: assume same shape

    name = _unique_name("singleton")
    source = generate_kernel(region, name, shapes)

    # Delegate to fused_region compilation with the generated source.
    op.params["kernel_source"] = source
    op.params["_region_ops"] = region.region_ops
    op.params["region_ops_count"] = 1
    return _compile_fused_region(op)


def _compile_noop(op: OpKernel) -> Launch:
    """No-op kernel for reshape/transpose/elementwise that don't need computation."""
    name = _unique_name("noop")
    src = f"__global__ void {name}() {{}}"
    return Launch(
        kernel_source=src,
        kernel_name=name,
        grid=(1, 1, 1),
        block=(1, 1, 1),
        args=[],
    )


_OP_HANDLERS: dict[str, callable] = {
    "fused_region": _compile_fused_region,
    "reshape": _compile_noop,
    "transpose": _compile_noop,
    "gather": _compile_noop,
    "scatter": _compile_noop,
}


def _compile_op(op: OpKernel) -> Launch:
    """Compile a single OpKernel to a CUDA Launch."""
    handler = _OP_HANDLERS.get(op.op)
    if handler is not None:
        return handler(op)

    # Unfused elementwise/reduce ops — wrap in a FusedRegionOp and use kernel_gen.
    if op.op.startswith("elementwise_") or op.op.startswith("reduce_"):
        return _compile_singleton(op)

    # Legacy fused ops — noop stub.
    if op.op.startswith("fused_"):
        return _compile_noop(op)

    raise ValueError(f"Unknown op: {op.op!r}. Known ops: {list(_OP_HANDLERS.keys())}")
