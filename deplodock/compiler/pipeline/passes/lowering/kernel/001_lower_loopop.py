"""Lower each ``LoopOp`` node to a ``KernelOp``.

Emits the kernel-level AST (``GpuKernel``) from the loop body and
computes the launch geometry (grid/block), then mutates the node's op
payload in place. The node's id, inputs, and output tensor are
preserved — only the op changes. (The mutation style is required
because ``KernelOp.arg_order`` embeds the node's id as its output
buffer name; splicing in a new node would break that reference.)
"""

from __future__ import annotations

from deplodock.compiler.ir.kernel import KernelOp, emit_kernel, kernel_name_for, launch_config
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline.graph import Graph
from deplodock.compiler.pipeline.matcher import Match, Pattern

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, LoopOp):
        return None
    kname = kernel_name_for(node.op, match.root_node_id)
    gpu_kernel, arg_order = emit_kernel(node, kname, graph)
    grid, block = launch_config(node)
    node.op = KernelOp(
        kernel=gpu_kernel,
        kernel_name=kname,
        arg_order=tuple(arg_order),
        grid=grid,
        block=block,
    )
    return None
