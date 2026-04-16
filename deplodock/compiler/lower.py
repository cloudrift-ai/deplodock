"""Extract ``KernelOp`` nodes from a fully-fused graph.

After the rewriter runs decomposition → optimization → fusion, the graph
contains only ``KernelOp`` + ``InputOp`` + ``ConstantOp`` nodes. This
module extracts the ``KernelOp``s in topological order for backend codegen,
along with the buffer name mappings that the codegen needs.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir import Graph
from deplodock.compiler.ops import KernelOp


@dataclass
class KernelInfo:
    """A KernelOp plus its external buffer name mappings."""

    kernel: KernelOp
    input_names: list[str]  # graph node inputs → port buffer names
    output_name: str  # graph node ID → output buffer name


def extract_kernels(graph: Graph) -> list[KernelInfo]:
    """Collect ``KernelOp`` nodes with buffer name mappings in topological order."""
    result: list[KernelInfo] = []
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        if isinstance(node.op, KernelOp):
            result.append(KernelInfo(kernel=node.op, input_names=list(node.inputs), output_name=nid))
    return result
