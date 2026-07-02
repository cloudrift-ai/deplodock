"""Lower UnsqueezeOp(x, dim=k) → IndexMapOp."""

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.expr import placeholder
from emmy.compiler.ir.frontend.ir import UnsqueezeOp
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", UnsqueezeOp)]


def rewrite(match: Match, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    out_shape = tuple(out.shape)
    in_shape = tuple(inp_x.output.shape)
    dim = root.op.dim
    norm_dim = dim if dim >= 0 else len(out_shape) + dim

    coord_map = [placeholder(i if i < norm_dim else i + 1) for i in range(len(in_shape))]

    frag = open_fragment(graph, [inp_x])
    new_node = single_indexmap(frag, inp_x, out_shape=out_shape, coord_map=coord_map, name=out.name)
    frag.outputs = [new_node.id]
    return frag
