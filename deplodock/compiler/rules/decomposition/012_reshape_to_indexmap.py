"""Decompose ReshapeOp into an identity IndexMapOp (pure buffer alias).

ReshapeOp changes logical shape without moving data. We convert it to an
IndexMapOp whose coord_map is the identity over the output shape — the
backend can then either alias the buffer (if the IndexMapOp is identity)
or emit a copy kernel (if shapes differ in a way that changes stride).
"""

from __future__ import annotations

from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import IndexMapOp, IndexSource, ReshapeOp

GRAMMAR = [Production("root", ReshapeOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    out_shape = tuple(root.op.shape)

    coord_map = tuple(placeholder(d) for d in range(len(out_shape)))
    indexmap = IndexMapOp(
        out_shape=out_shape,
        sources=(IndexSource(input_idx=0, coord_map=coord_map),),
    )

    g = graph.copy()
    new_id = g.add_node(
        op=indexmap,
        inputs=[x_id],
        output=Tensor(root.output.name, out_shape, root.output.dtype),
    )
    g.replace_node(match.root_node_id, new_id)
    g.remove_node(match.root_node_id)
    return g
