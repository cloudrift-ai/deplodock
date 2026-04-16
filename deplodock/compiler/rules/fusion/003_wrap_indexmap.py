"""Wrap standalone IndexMapOps as copy kernels.

An IndexMapOp whose consumers are all already-fused KernelOps gets
wrapped as a single-launch copy kernel. Runs before assemble_kernels
so that layout-only nodes don't block compute fusion.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import (
    IndexMapOp,
    KernelOp,
    Mux,
    MuxBranch,
    Port,
)

GRAMMAR = [Production("wrap", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, IndexMapOp):
        return graph
    consumers = graph.consumers(nid)
    if not all(isinstance(graph.nodes[c].op, KernelOp) for c in consumers if c in graph.nodes):
        return graph
    return _wrap_indexmap_kernel(graph, node)


def _wrap_indexmap_kernel(graph: Graph, node) -> Graph:
    nid = node.id
    op = node.op
    assert isinstance(op, IndexMapOp)

    if len(op.sources) == 1:
        src_id = node.inputs[op.sources[0].input_idx]
        kernel_inputs: tuple = (Port(buffer_id=src_id, indexmap=op),)
    else:
        branches = []
        for src in op.sources:
            src_id = node.inputs[src.input_idx]
            branches.append(MuxBranch(input=Port(buffer_id=src_id), select=src.select))
        kernel_inputs = (Mux(branches=tuple(branches)),)

    out_port = Port(buffer_id=nid)
    kernel = KernelOp(inputs=kernel_inputs, outputs=(out_port,))

    g = graph.copy()
    input_nids = [kernel_inputs[0].buffer_id if isinstance(kernel_inputs[0], Port) else kernel_inputs[0].branches[0].input.buffer_id]
    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{nid}", shape=tuple(op.out_shape), dtype=node.output.dtype),
    )
    out_port.buffer_id = new_nid
    g.replace_node(nid, new_nid)
    # Rename in existing KernelOps
    for n in g.nodes.values():
        if not isinstance(n.op, KernelOp) or n.id == new_nid:
            continue
        for inp in n.op.inputs:
            if isinstance(inp, Port) and inp.buffer_id == nid:
                inp.buffer_id = new_nid
        for assign in n.op.body:
            if nid in assign.args:
                assign.args = tuple(new_nid if a == nid else a for a in assign.args)
    g.remove_node(nid)
    return g
