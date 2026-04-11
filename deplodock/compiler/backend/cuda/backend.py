"""CUDA backend: converts an ExecutionPlan into a runnable Program.

Maps each OpKernel to a .cu template, computes grid/block dimensions,
and produces a Program that can be compiled with nvcc and executed.
"""

from __future__ import annotations

import math

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.kernels import load_kernel
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


def _compile_rmsnorm(op: OpKernel) -> Launch:
    rows = op.params["rows"]
    dim = op.params["dim"]
    eps = op.params["eps"]
    name = _unique_name("fused_rmsnorm")
    return Launch(
        kernel_source=load_kernel("rmsnorm", kernel_name=name),
        kernel_name=name,
        grid=(rows, 1, 1),
        block=(256, 1, 1),
        args=[*op.inputs, *op.outputs, str(rows), str(dim), f"{eps}f"],
    )


def _compile_triple_matmul(op: OpKernel) -> Launch:
    m, k = op.params["M"], op.params["K"]
    nq, nk, nv = op.params["Nq"], op.params["Nk"], op.params["Nv"]
    max_n = max(nq, nk, nv)
    name = _unique_name("triple_matmul")
    return Launch(
        kernel_source=load_kernel("matmul_triple", kernel_name=name),
        kernel_name=name,
        grid=(_cd(max_n, 16), _cd(m, 16), 3),
        block=(16, 16, 1),
        args=[*op.inputs, *op.outputs, str(m), str(k), str(nq), str(nk), str(nv)],
    )


def _compile_rope(op: OpKernel) -> Launch:
    batch = op.params["batch"]
    seq_len = op.params["seq_len"]
    q_heads = op.params["q_heads"]
    kv_heads = op.params["kv_heads"]
    head_dim = op.params["head_dim"]
    total = batch * seq_len * q_heads * (head_dim // 2) + batch * seq_len * kv_heads * (head_dim // 2)
    name = _unique_name("fused_rope")
    return Launch(
        kernel_source=load_kernel("rope", kernel_name=name),
        kernel_name=name,
        grid=(_cd(total, 256), 1, 1),
        block=(256, 1, 1),
        args=[*op.outputs, *op.inputs[2:], str(batch), str(seq_len), str(q_heads), str(kv_heads), str(head_dim)],
    )


def _compile_attention_qk(op: OpKernel) -> Launch:
    bh = op.params["batch_heads"]
    s = op.params["seq_len"]
    hd = op.params["head_dim"]
    scale = op.params["scale"]
    name = _unique_name("attention_qk")
    return Launch(
        kernel_source=load_kernel("attention_qk", kernel_name=name),
        kernel_name=name,
        grid=(_cd(s, 16), _cd(s, 16), bh),
        block=(16, 16, 1),
        args=[*op.inputs, *op.outputs, str(bh), str(s), str(hd), f"{scale}f"],
    )


def _compile_attention_softmax(op: OpKernel) -> Launch:
    bh = op.params["batch_heads"]
    s = op.params["seq_len"]
    name = _unique_name("attention_softmax")
    return Launch(
        kernel_source=load_kernel("attention_softmax", kernel_name=name),
        kernel_name=name,
        grid=(bh * s, 1, 1),
        block=(256, 1, 1),
        args=[*op.inputs, str(bh), str(s)],
    )


def _compile_attention_sv(op: OpKernel) -> Launch:
    bh = op.params["batch_heads"]
    s = op.params["seq_len"]
    hd = op.params["head_dim"]
    name = _unique_name("attention_sv")
    return Launch(
        kernel_source=load_kernel("attention_sv", kernel_name=name),
        kernel_name=name,
        grid=(_cd(hd, 16), _cd(s, 16), bh),
        block=(16, 16, 1),
        args=[*op.inputs, *op.outputs, str(bh), str(s), str(hd)],
    )


def _compile_matmul_residual_add(op: OpKernel) -> Launch:
    m, n, k = op.params["M"], op.params["N"], op.params["K"]
    name = _unique_name("matmul_residual_add")
    return Launch(
        kernel_source=load_kernel("matmul_residual_add", kernel_name=name),
        kernel_name=name,
        grid=(_cd(n, 16), _cd(m, 16), 1),
        block=(16, 16, 1),
        args=[*op.inputs, *op.outputs, str(m), str(n), str(k)],
    )


def _compile_dual_matmul_silu_mul(op: OpKernel) -> Launch:
    m, n, k = op.params["M"], op.params["N"], op.params["K"]
    name = _unique_name("dual_matmul_silu_mul")
    return Launch(
        kernel_source=load_kernel("matmul_dual_silu_mul", kernel_name=name),
        kernel_name=name,
        grid=(_cd(n, 16), _cd(m, 16), 1),
        block=(16, 16, 1),
        args=[*op.inputs, *op.outputs, str(m), str(n), str(k)],
    )


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


def _compile_silu_mul(op: OpKernel) -> Launch:
    n = op.params.get("n", 1)
    name = _unique_name("fused_silu_mul")
    return Launch(
        kernel_source=load_kernel("activation", kernel_name=name),
        kernel_name=name,
        grid=(_cd(n, 256), 1, 1),
        block=(256, 1, 1),
        args=[*op.inputs, *op.outputs, str(n)],
    )


def _compile_attention(op: OpKernel) -> Launch:
    """FusedAttentionOp — stub kernel (actual attention needs QK+softmax+SV sub-launches)."""
    name = _unique_name("attention_stub")
    src = f"__global__ void {name}() {{}}"
    return Launch(
        kernel_source=src,
        kernel_name=name,
        grid=(1, 1, 1),
        block=(1, 1, 1),
        args=[],
    )


def _compile_fused_region(op: OpKernel) -> Launch:
    """Compile a FusedRegionOp — kernel source is already generated by kernel_gen."""
    source = op.params.get("kernel_source", "")
    name = _unique_name("fused_region")

    if not source:
        # Fallback: empty kernel if no source was generated.
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
    # Block planner ops (fused, specific kernels).
    "rmsnorm": _compile_rmsnorm,
    "triple_matmul": _compile_triple_matmul,
    "rope": _compile_rope,
    "attention_qk": _compile_attention_qk,
    "attention_softmax": _compile_attention_softmax,
    "attention_sv": _compile_attention_sv,
    "matmul_residual_add": _compile_matmul_residual_add,
    "dual_matmul_silu_mul": _compile_dual_matmul_silu_mul,
    # Graph planner ops (from plan_graph).
    "matmul": _compile_matmul,
    "silu_mul": _compile_silu_mul,
    "attention": _compile_attention,
    "softmax": _compile_noop,  # consumed by attention fusion
    "reshape": _compile_noop,
    "transpose": _compile_noop,
    "gather": _compile_noop,
    "scatter": _compile_noop,
    "fused_region": _compile_fused_region,
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
