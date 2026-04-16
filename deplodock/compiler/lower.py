"""Extract ``KernelOp`` nodes from a fully-fused graph.

After the rewriter runs decomposition → optimization → fusion, the graph
contains only ``KernelOp`` + ``InputOp`` + ``ConstantOp`` nodes. This
module extracts the ``KernelOp``s in topological order for backend codegen.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.ops import KernelOp


def extract_kernels(graph: Graph) -> list[KernelOp]:
    """Collect ``KernelOp`` nodes from the graph in topological order."""
    return [graph.nodes[nid].op for nid in graph.topological_order() if isinstance(graph.nodes[nid].op, KernelOp)]
