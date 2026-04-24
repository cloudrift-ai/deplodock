"""Lower each ``LoopOp`` node to a ``KernelOp``.

Emits the kernel-level AST (``GpuKernel``) from the loop body and
computes the launch geometry (grid/block), then mutates the node's op
payload in place. The node's id, inputs, and output tensor are
preserved — only the op changes. (The mutation style is required
because ``KernelOp.arg_order`` embeds the node's id as its output
buffer name; splicing in a new node would break that reference.)
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._classify import classify
from deplodock.compiler.pipeline.passes.lowering.kernel._common import kernel_name_for
from deplodock.compiler.pipeline.passes.lowering.kernel._emit import emit_kernel, launch_config
from deplodock.compiler.pipeline.passes.lowering.kernel._emit_matmul import emit_matmul_kernel, is_matmul_annotated
from deplodock.compiler.pipeline.passes.lowering.kernel._unified import emit_unified

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, LoopOp):
        return None
    kname = kernel_name_for(node.op, match.root_node_id)
    if is_matmul_annotated(node):
        gpu_kernel, arg_order, grid, block = emit_matmul_kernel(node, kname, graph)
    else:
        sig = classify(node.op)
        if sig is not None:
            gpu_kernel, arg_order, grid, block = emit_unified(node, kname, graph, sig)
        else:
            # Legacy fallback for shapes the unified classifier doesn't recognize
            # yet (e.g. SDPA's flash-style nested-reduce-inside-output-loop).
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
