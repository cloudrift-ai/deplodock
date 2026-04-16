"""Wrap standalone IndexMapOps as copy kernels.

An IndexMapOp whose consumers are all already-fused KernelOps gets
wrapped as a single-launch copy kernel.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import (
    IndexMapOp,
    InputOp,
    KernelOp,
    Mux,
    MuxBranch,
    Port,
)

GRAMMAR = [Production("wrap", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, IndexMapOp):
        return None
    consumers = graph.consumers(nid)
    if not all(isinstance(graph.nodes[c].op, KernelOp) for c in consumers if c in graph.nodes):
        return None

    op = node.op

    if len(op.sources) == 1:
        kernel_inputs: tuple = (Port(indexmap=op),)
    else:
        branches = []
        for src in op.sources:
            branches.append(MuxBranch(input=Port(), select=src.select))
        kernel_inputs = (Mux(branches=tuple(branches)),)

    kernel = KernelOp(inputs=kernel_inputs, outputs=(Port(),))

    frag = Graph()
    for inp_id in node.inputs:
        if inp_id not in frag.nodes:
            ext = graph.nodes.get(inp_id)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    input_nids = list(node.inputs)
    out_id = frag.add_node(kernel, input_nids, Tensor(f"kernel_{nid}", tuple(op.out_shape), node.output.dtype))
    frag.outputs = [out_id]
    return frag
