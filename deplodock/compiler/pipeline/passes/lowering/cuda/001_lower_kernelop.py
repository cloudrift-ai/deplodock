"""Lower each ``KernelOp`` node to a ``CudaOp``.

Renders the ``KernelOp.kernel`` AST to a ``__global__`` CUDA source
string and mutates the node's op payload in place. (The mutation style
is required because ``CudaOp.arg_order`` embeds the node's id as its
output buffer name; splicing in a new node would break that reference.)
"""

from __future__ import annotations

from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.kernel import KernelOp, emit_kernel_source
from deplodock.compiler.pipeline.graph import Graph
from deplodock.compiler.pipeline.matcher import Match, Pattern

PATTERN = [Pattern("root", KernelOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, KernelOp):
        return None
    node.op = CudaOp(
        kernel_source=emit_kernel_source(node.op.kernel),
        kernel_name=node.op.kernel_name,
        arg_order=node.op.arg_order,
        grid=node.op.grid,
        block=node.op.block,
        smem_bytes=node.op.smem_bytes,
        zero_outputs=node.op.zero_outputs,
        comment=node.op.comment,
    )
    return None
