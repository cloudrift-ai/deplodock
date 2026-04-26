"""Lower UnsqueezeOp(x, dim=k) → IndexMapOp."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.frontend.ir import UnsqueezeOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", UnsqueezeOp)]


def rewrite(graph: Graph, root: Node, inp_x: Node, out: Tensor) -> Graph | None:
    out_shape = tuple(out.shape)
    in_shape = tuple(inp_x.output.shape)
    dim = root.op.dim
    norm_dim = dim if dim >= 0 else len(out_shape) + dim

    coord_map = [placeholder(i if i < norm_dim else i + 1) for i in range(len(in_shape))]

    frag = open_fragment(graph, [inp_x])
    new_id = single_indexmap(frag, inp_x.id, out_shape=out_shape, coord_map=coord_map, name=out.name, dtype=out.dtype)
    frag.outputs = [new_id]
    return frag
