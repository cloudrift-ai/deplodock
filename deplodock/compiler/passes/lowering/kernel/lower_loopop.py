"""Lower every ``LoopOp`` node to a ``KernelOp``.

For each node whose op is a ``LoopOp``, emit the kernel-level AST
(``GpuKernel``), compute the launch geometry (grid/block), and replace
``node.op`` in place with a ``KernelOp``. The node's id, inputs, and
output tensor are preserved — only the op payload changes.
"""

from __future__ import annotations

from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.ir.kernel import KernelOp, emit_kernel, kernel_name_for, launch_config
from deplodock.compiler.ir.loop import LoopOp


def lower(graph: Graph) -> Graph:
    for idx, nid in enumerate(graph.topological_order()):
        node = graph.nodes[nid]
        if not isinstance(node.op, LoopOp):
            continue
        kname = kernel_name_for(node.op, idx)
        gpu_kernel, arg_order = emit_kernel(node, kname, graph)
        grid, block = launch_config(node)
        node.op = KernelOp(
            kernel=gpu_kernel,
            kernel_name=kname,
            arg_order=tuple(arg_order),
            grid=grid,
            block=block,
        )
    return graph
