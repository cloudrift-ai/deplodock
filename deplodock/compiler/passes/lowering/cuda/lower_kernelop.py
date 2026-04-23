"""Lower every ``KernelOp`` node to a ``CudaOp``.

Renders each ``KernelOp.kernel`` to a ``__global__`` CUDA source string
and replaces ``node.op`` in place with a ``CudaOp`` carrying that string
plus the launch geometry. The node's id, inputs, and output tensor are
preserved.
"""

from __future__ import annotations

from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.ir.kernel import KernelOp, emit_kernel_source


def lower(graph: Graph) -> Graph:
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if not isinstance(node.op, KernelOp):
            continue
        source = emit_kernel_source(node.op.kernel)
        node.op = CudaOp(
            kernel_source=source,
            kernel_name=node.op.kernel_name,
            arg_order=node.op.arg_order,
            grid=node.op.grid,
            block=node.op.block,
            smem_bytes=node.op.smem_bytes,
            zero_outputs=node.op.zero_outputs,
            tma_descriptors=node.op.tma_descriptors,
            comment=node.op.comment,
        )
    return graph
