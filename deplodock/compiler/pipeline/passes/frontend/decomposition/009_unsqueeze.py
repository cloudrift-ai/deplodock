"""Lower UnsqueezeOp(x, dim=k) → IndexMapOp."""

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend.ir import UnsqueezeOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", UnsqueezeOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    out_shape = tuple(root.output.shape)
    in_shape = tuple(graph.nodes[x_id].output.shape)
    dim = root.op.dim
    norm_dim = dim if dim >= 0 else len(out_shape) + dim

    coord_map = [placeholder(i if i < norm_dim else i + 1) for i in range(len(in_shape))]

    frag = open_fragment(graph, [x_id])
    new_id = single_indexmap(frag, x_id, out_shape=out_shape, coord_map=coord_map, name=root.output.name, dtype=root.output.dtype)
    frag.outputs = [new_id]
    return frag
