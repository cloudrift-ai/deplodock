"""Kernel generator: emit CUDA source from a FusedRegionOp's primitive ops.

Walks the ops in topological order and emits C code directly.
The kernel structure (tiling, reductions) emerges from the ops
and their dimension analysis — no strategy classification.
"""

from __future__ import annotations

import math

from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReduceOp, ReshapeOp, TransposeOp

# Maps ElementwiseOp.fn to C expression templates.
# {a} = first input, {b} = second input (binary ops).
_EXPR: dict[str, str] = {
    "mul": "{a} * {b}",
    "add": "{a} + {b}",
    "sub": "{a} - {b}",
    "div": "{a} / {b}",
    "neg": "-{a}",
    "exp": "expf({a})",
    "rsqrt": "rsqrtf({a})",
    "recip": "1.0f / {a}",
}


def generate_kernel(region: FusedRegionOp, name: str, shapes: dict[str, tuple]) -> str:
    """Generate a CUDA kernel from a FusedRegionOp.

    Args:
        region: The fused region containing primitive ops.
        name: Kernel function name.
        shapes: Map of node_id/buffer_name → shape tuple.

    Returns:
        Complete __global__ function source string.
    """
    # Analyze: does this region have reductions?
    has_reduce = any(isinstance(op, ReduceOp) for _, op, _ in region.region_ops)

    if has_reduce:
        return _gen_reduce_kernel(region, name, shapes)
    return _gen_pointwise_kernel(region, name, shapes)


def _gen_pointwise_kernel(region: FusedRegionOp, name: str, shapes: dict[str, tuple]) -> str:
    """Generate a pointwise kernel — one thread per element."""
    # Determine total elements from output shape.
    # Build parameter list.
    params = []
    for inp in region.input_names:
        params.append(f"const float* __restrict__ {_safe(inp)}")
    for out in region.output_names:
        params.append(f"float* __restrict__ {_safe(out)}")
    params.append("int n")

    # Build body: walk ops, emit expressions.
    body_lines = ["    int i = blockIdx.x * blockDim.x + threadIdx.x;", "    if (i >= n) return;", ""]

    # Map node_id → C variable name.
    var_map: dict[str, str] = {}
    for inp in region.input_names:
        var_map[inp] = f"{_safe(inp)}[i]"

    for node_id, op, input_ids in region.region_ops:
        if isinstance(op, (ReshapeOp, TransposeOp)):
            # Pass-through.
            var_map[node_id] = var_map.get(input_ids[0], f"{_safe(input_ids[0])}[i]")
            continue

        if isinstance(op, ElementwiseOp):
            a = var_map.get(input_ids[0], f"{_safe(input_ids[0])}[i]") if input_ids else "0.0f"
            b = var_map.get(input_ids[1], f"{_safe(input_ids[1])}[i]") if len(input_ids) > 1 else "0.0f"
            expr_template = _EXPR.get(op.fn)
            if expr_template:
                expr = expr_template.format(a=a, b=b)
            else:
                expr = f"/* unknown: {op.fn} */ 0.0f"
            var_name = f"v_{_safe(node_id)}"
            body_lines.append(f"    float {var_name} = {expr};")
            var_map[node_id] = var_name
            continue

    # Write outputs.
    for out_id in region.output_names:
        val = var_map.get(out_id, "0.0f")
        body_lines.append(f"    {_safe(out_id)}[i] = {val};")

    param_str = ",\n    ".join(params)
    body = "\n".join(body_lines)
    return f"""
__global__ void {name}(
    {param_str}
) {{
{body}
}}
"""


def _gen_reduce_kernel(region: FusedRegionOp, name: str, shapes: dict[str, tuple]) -> str:
    """Generate a reduction kernel — one block per row.

    Handles ElementwiseOp + ReduceOp chains. Ops before the first reduce
    are the prologue (fused into the load). Ops after the last reduce
    are the epilogue (fused into the store).
    """
    # Determine output shape.
    out_id = region.output_names[0]
    out_shape = shapes.get(out_id, (1,))
    # Determine row dimensions from output shape.

    # Infer rows and cols from the first input to a reduce.
    first_reduce_input = None
    for _, op, inp_ids in region.region_ops:
        if isinstance(op, ReduceOp):
            first_reduce_input = inp_ids[0]
            break

    inp_shape = shapes.get(first_reduce_input, out_shape) if first_reduce_input else out_shape
    if len(inp_shape) >= 2:
        rows = math.prod(d for d in inp_shape[:-1] if isinstance(d, int))  # noqa: F841 — used in kernel params
        cols = inp_shape[-1] if isinstance(inp_shape[-1], int) else 1  # noqa: F841
    else:
        rows = 1  # noqa: F841
        cols = math.prod(d for d in inp_shape if isinstance(d, int))  # noqa: F841

    # Build parameter list.
    params = []
    for inp in region.input_names:
        params.append(f"const float* __restrict__ {_safe(inp)}")
    for out in region.output_names:
        params.append(f"float* __restrict__ {_safe(out)}")
    params.extend(["int rows", "int cols"])

    # Split ops into phases: before first reduce, reduce, after reduce.
    prologue_ops = []
    reduce_ops = []
    epilogue_ops = []
    phase = "prologue"
    for entry in region.region_ops:
        _, op, _ = entry
        if isinstance(op, ReduceOp):
            reduce_ops.append(entry)
            phase = "epilogue"
        elif phase == "prologue":
            prologue_ops.append(entry)
        else:
            epilogue_ops.append(entry)

    # Generate kernel body.
    body_lines = [
        "    int row = blockIdx.x;",
        "    if (row >= rows) return;",
        "",
    ]

    # --- Prologue + reduction: one pass over cols ---
    # For each reduce, we need an accumulator.
    reduce_vars: dict[str, tuple[str, str]] = {}  # node_id → (var_name, fn)
    for node_id, op, _ in reduce_ops:
        var = f"acc_{_safe(node_id)}"
        init = "-1e30f" if op.fn == "max" else "0.0f"
        body_lines.append(f"    float {var} = {init};")
        reduce_vars[node_id] = (var, op.fn)

    body_lines.append("    for (int j = threadIdx.x; j < cols; j += blockDim.x) {")

    # Build var_map for the load loop.
    var_map: dict[str, str] = {}
    for inp in region.input_names:
        inp_shape = shapes.get(inp, (1,))
        if len(inp_shape) >= 2:
            var_map[inp] = f"{_safe(inp)}[row * cols + j]"
        elif len(inp_shape) == 1:
            # 1D input (e.g., weight vector) — index by j.
            var_map[inp] = f"{_safe(inp)}[j]"
        else:
            var_map[inp] = f"{_safe(inp)}[0]"

    # Emit prologue ops inside the loop.
    for node_id, op, input_ids in prologue_ops:
        if isinstance(op, (ReshapeOp, TransposeOp)):
            var_map[node_id] = var_map.get(input_ids[0], "0.0f")
            continue
        if isinstance(op, ElementwiseOp):
            a = var_map.get(input_ids[0], "0.0f") if input_ids else "0.0f"
            b = var_map.get(input_ids[1], "0.0f") if len(input_ids) > 1 else "0.0f"
            expr = _EXPR.get(op.fn, "0.0f").format(a=a, b=b)
            var = f"p_{_safe(node_id)}"
            body_lines.append(f"        float {var} = {expr};")
            var_map[node_id] = var

    # Emit reduce accumulation.
    for node_id, _op, input_ids in reduce_ops:
        acc_var, fn = reduce_vars[node_id]
        val = var_map.get(input_ids[0], "0.0f")
        if fn == "sum":
            body_lines.append(f"        {acc_var} += {val};")
        elif fn == "max":
            body_lines.append(f"        {acc_var} = fmaxf({acc_var}, {val});")

    body_lines.append("    }")
    body_lines.append("")

    # Warp-shuffle reduction for each accumulator.
    for node_id, (acc_var, fn) in reduce_vars.items():
        body_lines.append(f"    // Reduce {fn} for {acc_var}")
        body_lines.append("    for (int offset = warpSize / 2; offset > 0; offset >>= 1)")
        if fn == "sum":
            body_lines.append(f"        {acc_var} += __shfl_down_sync(0xffffffff, {acc_var}, offset);")
        else:
            body_lines.append(f"        {acc_var} = fmaxf({acc_var}, __shfl_down_sync(0xffffffff, {acc_var}, offset));")

        # Cross-warp via shared memory.
        body_lines.append(f"    __shared__ float warp_{_safe(node_id)}[8];")
        body_lines.append(f"    if (threadIdx.x % warpSize == 0) warp_{_safe(node_id)}[threadIdx.x / warpSize] = {acc_var};")
        body_lines.append("    __syncthreads();")
        body_lines.append(f"    if (threadIdx.x < blockDim.x / warpSize) {acc_var} = warp_{_safe(node_id)}[threadIdx.x];")
        body_lines.append(f"    else {acc_var} = {'-1e30f' if fn == 'max' else '0.0f'};")
        body_lines.append("    for (int offset = warpSize / 2; offset > 0; offset >>= 1)")
        if fn == "sum":
            body_lines.append(f"        {acc_var} += __shfl_down_sync(0xffffffff, {acc_var}, offset);")
        else:
            body_lines.append(f"        {acc_var} = fmaxf({acc_var}, __shfl_down_sync(0xffffffff, {acc_var}, offset));")
        body_lines.append(f"    __shared__ float s_{_safe(node_id)};")
        body_lines.append(f"    if (threadIdx.x == 0) s_{_safe(node_id)} = {acc_var};")
        body_lines.append("    __syncthreads();")
        body_lines.append(f"    {acc_var} = s_{_safe(node_id)};")
        var_map[node_id] = acc_var
        body_lines.append("")

    # --- Epilogue: second pass over cols if needed ---
    if epilogue_ops:
        body_lines.append("    // Epilogue pass")
        body_lines.append("    for (int j = threadIdx.x; j < cols; j += blockDim.x) {")

        # Re-load inputs that the epilogue needs.
        for inp in region.input_names:
            inp_shape = shapes.get(inp, (1,))
            if len(inp_shape) >= 2:
                var_map[inp] = f"{_safe(inp)}[row * cols + j]"
            elif len(inp_shape) == 1:
                var_map[inp] = f"{_safe(inp)}[j]"

        # Re-compute prologue values needed by epilogue.
        for node_id, op, input_ids in prologue_ops:
            if isinstance(op, ElementwiseOp) and node_id in _needed_by(epilogue_ops):
                a = var_map.get(input_ids[0], "0.0f") if input_ids else "0.0f"
                b = var_map.get(input_ids[1], "0.0f") if len(input_ids) > 1 else "0.0f"
                expr = _EXPR.get(op.fn, "0.0f").format(a=a, b=b)
                var = f"p_{_safe(node_id)}"
                body_lines.append(f"        float {var} = {expr};")
                var_map[node_id] = var

        for node_id, op, input_ids in epilogue_ops:
            if isinstance(op, (ReshapeOp, TransposeOp)):
                var_map[node_id] = var_map.get(input_ids[0], "0.0f")
                continue
            if isinstance(op, ElementwiseOp):
                a = var_map.get(input_ids[0], "0.0f") if input_ids else "0.0f"
                b = var_map.get(input_ids[1], "0.0f") if len(input_ids) > 1 else "0.0f"
                expr = _EXPR.get(op.fn, "0.0f").format(a=a, b=b)
                var = f"e_{_safe(node_id)}"
                body_lines.append(f"        float {var} = {expr};")
                var_map[node_id] = var

        # Write outputs.
        for out_id in region.output_names:
            val = var_map.get(out_id, "0.0f")
            body_lines.append(f"        {_safe(out_id)}[row * cols + j] = {val};")

        body_lines.append("    }")
    else:
        # No epilogue — output is the reduction result.
        # Write scalar output per row.
        body_lines.append("    if (threadIdx.x == 0) {")
        for out_id in region.output_names:
            val = var_map.get(out_id, "0.0f")
            body_lines.append(f"        {_safe(out_id)}[row] = {val};")
        body_lines.append("    }")

    param_str = ",\n    ".join(params)
    body = "\n".join(body_lines)
    return f"""
__global__ void {name}(
    {param_str}
) {{
{body}
}}
"""


def _needed_by(ops: list) -> set[str]:
    """Return set of node_ids referenced as inputs by the given ops."""
    needed = set()
    for _, _, input_ids in ops:
        needed.update(input_ids)
    return needed


def _safe(name: str) -> str:
    """Make a node ID safe as a C identifier."""
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")
