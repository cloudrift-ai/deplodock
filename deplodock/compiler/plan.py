"""Backend-agnostic execution plan.

An ExecutionPlan describes WHAT to compute (operations, buffers, data flow)
without specifying HOW (no kernel source, no grid/block, no GPU API calls).
A Backend converts an ExecutionPlan into a runnable Program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.ir import Graph


@dataclass
class BufferSpec:
    """Buffer description: name, shape, dtype, and role."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"


@dataclass
class OpKernel:
    """One operation in the execution plan.

    The ``op`` field is a string tag that the backend looks up in its
    kernel registry (e.g., "rmsnorm" → rmsnorm.cu for CUDA, rmsnorm.hip
    for ROCm).  ``params`` carries op-specific configuration (dimensions,
    epsilon, scale, etc.) that the backend uses to compute grid/block and
    format kernel arguments.
    """

    op: str
    inputs: list[str]
    outputs: list[str]
    params: dict[str, int | float | str] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Ordered sequence of operations on named buffers.

    Backend-agnostic: no kernel source, no grid/block, no GPU specifics.
    """

    name: str
    buffers: list[BufferSpec]
    ops: list[OpKernel]


def plan_graph(graph: Graph, name: str = "graph") -> ExecutionPlan:
    """Walk a compiled graph and produce a backend-agnostic ExecutionPlan.

    Each graph node becomes either a BufferSpec (inputs/constants) or an
    OpKernel (compute ops). Buffer shapes come from node.output.shape.
    Op types are derived from the Op class name.
    """
    import math

    from deplodock.compiler import ops as ops_module

    def _prod(s: tuple) -> int:
        return math.prod(x for x in s if isinstance(x, int)) if s else 1

    buffers: list[BufferSpec] = []
    op_kernels: list[OpKernel] = []
    buf_names: set[str] = set()

    for nid in graph.topological_order():
        node = graph.nodes[nid]
        op = node.op
        op_type = type(op).__name__
        out_name = nid
        shape = node.output.shape
        dtype = node.output.dtype

        # InputOp / ConstantOp → buffer only, no kernel.
        if isinstance(op, ops_module.InputOp):
            buffers.append(BufferSpec(out_name, shape, dtype, role="input"))
            buf_names.add(out_name)
            continue

        if isinstance(op, ops_module.ConstantOp):
            buffers.append(BufferSpec(out_name, shape, dtype, role="constant"))
            buf_names.add(out_name)
            continue

        # Output buffer for this node.
        role = "output" if nid in graph.outputs else "scratch"
        buffers.append(BufferSpec(out_name, shape, dtype, role=role))
        buf_names.add(out_name)

        # Map op type to OpKernel tag + params.
        inputs = list(node.inputs)
        outputs = [out_name]
        params: dict[str, int | float | str] = {}

        if isinstance(op, ops_module.ElementwiseOp):
            tag = f"elementwise_{op.fn}"
        elif isinstance(op, ops_module.ReduceOp):
            tag = f"reduce_{op.fn}"
            params["axis"] = op.axis
        elif isinstance(op, ops_module.TransposeOp):
            tag = "transpose"
        elif isinstance(op, ops_module.ReshapeOp):
            tag = "reshape"
        elif isinstance(op, ops_module.GatherOp):
            tag = "gather"
            params["axis"] = op.axis
        elif isinstance(op, ops_module.ScatterOp):
            tag = "scatter"
            params["axis"] = op.axis
        elif isinstance(op, ops_module.FusedRegionOp):
            tag = "fused_region"
            params["kernel_source"] = op.kernel_source
            params["region_ops_count"] = len(op.region_ops)
            params["_region_ops"] = op.region_ops  # for backend matmul detection
            # Store input shapes so the backend can infer matmul K dimension.
            all_shapes = {inp_id: graph.nodes[inp_id].output.shape for inp_id in node.inputs if inp_id in graph.nodes}
            # Include shapes for intermediate region op nodes (needed by kernel_gen
            # to determine reduction dimensions).
            for rid, _, _ in op.region_ops:
                if rid in graph.nodes:
                    all_shapes[rid] = graph.nodes[rid].output.shape
            params["_input_shapes"] = all_shapes
            # Pass full shapes map (includes intermediates) for kernel generation.
            params["_shapes"] = dict(op.shapes) if op.shapes else all_shapes
            # Preserve original input/output names for kernel generation.
            # The plan uses fused node IDs as buffer names, but the kernel
            # source must use the original names from the region_ops.
            params["_output_names"] = list(op.output_names)
            params["_input_names"] = list(op.input_names)
        else:
            tag = op_type.lower()

        # Add shape info to params for the backend.
        params["shape"] = shape
        # Store input shapes for broadcasting/dimension inference.
        if "_input_shapes" not in params:
            params["_input_shapes"] = {inp_id: graph.nodes[inp_id].output.shape for inp_id in node.inputs if inp_id in graph.nodes}

        op_kernels.append(OpKernel(op=tag, inputs=inputs, outputs=outputs, params=params))

    return ExecutionPlan(name=name, buffers=buffers, ops=op_kernels)
