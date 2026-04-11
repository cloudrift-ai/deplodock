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

    Uses the naive strategy by default. The TMA strategies require
    KernelDef metadata (TMA descriptors, tile sizes) that the Program
    abstraction doesn't support yet — use scripts/bench_matmul.py for
    TMA benchmarking.
    """
    from deplodock.compiler.backend.cuda.codegen import emit_kernel
    from deplodock.compiler.backend.cuda.lower import MatmulConfig, lower_graph
    from deplodock.compiler.ir import Graph, Tensor
    from deplodock.compiler.ops import InputOp, MatmulOp

    m = op.params.get("M", 1)
    n = op.params.get("N", 1)
    k = op.params.get("K", 1)

    # Build a minimal matmul graph for lowering.
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    c = g.add_node(MatmulOp(), [a, b], Tensor("C", (m, n)), node_id="C")
    g.inputs = [a, b]
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
    """
    region_ops = op.params.get("_region_ops", [])
    if len(region_ops) != 2:
        return False, 0, 0, 0
    _, op0, inputs0 = region_ops[0]
    _, op1, _ = region_ops[1]
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    if not (isinstance(op0, ElementwiseOp) and op0.fn == "mul" and isinstance(op1, ReduceOp) and op1.fn == "sum"):
        return False, 0, 0, 0

    # Infer M, N, K from shape: output is (M, N), mul output is (M, K, N) or similar.
    shape = op.params.get("shape", (1,))
    m = int(shape[0]) if len(shape) >= 2 and isinstance(shape[0], int) else 1
    n = int(shape[-1]) if len(shape) >= 1 and isinstance(shape[-1], int) else 1
    # K: the reduced dimension. We need to get it from the mul's output shape.
    # The mul has 2 inputs: A(M,K) and B(K,N). K = the shared dim.
    # Without access to input shapes here, use output shape and assume K = M (square) as fallback.
    k = m  # heuristic fallback
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

    if not source:
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

    shape = op.params.get("shape", (1,))
    total = 1
    for d in shape:
        if isinstance(d, int):
            total *= d

    # Determine grid from scalar args.
    if scalar_count >= 2 and len(shape) >= 2:
        # Reduction kernel: (rows, cols)
        rows = 1
        for d in shape[:-1]:
            if isinstance(d, int):
                rows *= d
        cols = shape[-1] if isinstance(shape[-1], int) else 1
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

    # Generic elementwise/reduce ops → noop for now.
    if op.op.startswith("elementwise_") or op.op.startswith("reduce_") or op.op.startswith("fused_"):
        return _compile_noop(op)

    raise ValueError(f"Unknown op: {op.op!r}. Known ops: {list(_OP_HANDLERS.keys())}")
