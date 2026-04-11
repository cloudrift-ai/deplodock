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

        if isinstance(op, ops_module.MatmulOp):
            tag = "matmul"
            a_shape = graph.nodes[inputs[0]].output.shape if inputs and inputs[0] in graph.nodes else ()
            params["M"] = _prod(shape[:-1]) if len(shape) > 1 else (shape[0] if shape else 1)
            params["N"] = shape[-1] if shape else 1
            params["K"] = a_shape[-1] if a_shape else 1
        elif isinstance(op, ops_module.FusedRMSNormOp):
            tag = "rmsnorm"
            params["eps"] = op.eps
            params["rows"] = _prod(shape[:-1]) if len(shape) > 1 else 1
            params["dim"] = shape[-1] if shape else 1
        elif isinstance(op, ops_module.FusedSoftmaxOp):
            tag = "softmax"
            params["axis"] = op.axis
        elif isinstance(op, ops_module.FusedSiLUMulOp):
            tag = "silu_mul"
            params["n"] = _prod(shape)
        elif isinstance(op, ops_module.FusedAttentionOp):
            tag = "attention"
            params["num_heads"] = op.num_heads
            params["head_dim"] = op.head_dim
            params["scale"] = op.scale
        elif isinstance(op, ops_module.ElementwiseOp):
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
        else:
            tag = op_type.lower()

        # Add shape info to params for the backend.
        params["shape"] = shape

        op_kernels.append(OpKernel(op=tag, inputs=inputs, outputs=outputs, params=params))

    return ExecutionPlan(name=name, buffers=buffers, ops=op_kernels)
