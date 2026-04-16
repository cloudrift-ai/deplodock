"""Lower UnsqueezeOp(x, dim=k) → IndexMapOp.

The output has rank input_rank + 1 with a size-1 dim inserted at axis k. The
coord_map for axis i reads the input's coord at axis i (for i < k) or i-1
(for i > k); the inserted axis k contributes nothing to the input read.
"""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend import UnsqueezeOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", UnsqueezeOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    out_shape = tuple(root.output.shape)
    in_shape = tuple(graph.nodes[x_id].output.shape)
    dim = root.op.dim
    norm_dim = dim if dim >= 0 else len(out_shape) + dim

    # coord_map[i] reads input axis (i if i<norm_dim else i-1). The inserted
    # output axis (i==norm_dim) doesn't appear in the coord_map at all.
    coord_map = []
    for i in range(len(in_shape)):
        # Input axis i is read from output axis i if i < norm_dim, else i+1.
        out_axis = i if i < norm_dim else i + 1
        coord_map.append(placeholder(out_axis))

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[x_id],
        output=Tensor(root.output.name, out_shape, root.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
