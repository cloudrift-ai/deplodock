"""Wrap standalone IndexMapOps as copy kernels.

An IndexMapOp whose consumers are all already-fused LoopOps gets wrapped
as a single-launch copy kernel.

v1 limitation: only single-source IndexMapOps are wrapped. Multi-source
forms (cat / concat) would need a body-level ``Select`` dispatch which
will land with the accumulator/body-shape pass. Until then, a multi-source
IndexMapOp that survives to this rule is left in place; CudaBackend refuses
such kernels at codegen time.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import PLACEHOLDER_PREFIX, Var, substitute
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Axis, LoopOp, Port
from deplodock.compiler.ir.tensor import IndexMapOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("wrap", IndexMapOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    nid = match.root_node_id
    node = graph.nodes[nid]
    if not isinstance(node.op, IndexMapOp):
        return None
    consumers = graph.consumers(nid)
    if not all(isinstance(graph.nodes[c].op, LoopOp) for c in consumers if c in graph.nodes):
        return None

    op = node.op

    # Multi-source IndexMapOps (cat) need a body-level Select which doesn't
    # exist yet. Bail — the IndexMapOp stays as a graph-level node; the
    # numpy LoopBackend handles it via Op.forward, CudaBackend refuses.
    if len(op.sources) != 1:
        return None

    # Copy kernel: one free axis per output dim, Port.index substitutes the
    # coord_map placeholders with axis Vars.
    axes = tuple(Axis(name=f"a{i}", extent=int(d), kind="free") for i, d in enumerate(op.out_shape))
    mapping = {f"{PLACEHOLDER_PREFIX}{i}": Var(a.name) for i, a in enumerate(axes)}
    index = tuple(substitute(c, mapping) for c in op.sources[0].coord_map)

    out_port = Port(index=tuple(Var(a.name) for a in axes))
    kernel = LoopOp(axes=axes, inputs=(Port(index=index),), outputs=(out_port,))

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
