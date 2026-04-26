"""Lower SliceOp(x, dim, start, end) → IndexMapOp.

Tracer convention: SliceOp.inputs = [tensor, dim_const, start_const, end_const].
After decomposition: IndexMapOp.inputs = [tensor]; dim/start are baked into
the coord_map (orphan constants get cleaned up by the rewriter).
"""

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import SliceOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", SliceOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    x_id = root.inputs[0]
    dim_id = root.inputs[1]
    start_id = root.inputs[2]

    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = tuple(root.output.shape)
    ndim = len(in_shape)

    dim_node = graph.nodes.get(dim_id)
    start_node = graph.nodes.get(start_id)
    if not (isinstance(dim_node.op, ConstantOp) and isinstance(start_node.op, ConstantOp)):
        return None
    if dim_node.op.value is None or start_node.op.value is None:
        return None

    dim = int(dim_node.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim
    start = int(start_node.op.value)

    coord_map = []
    for i in range(ndim):
        if i == norm_dim and start != 0:
            coord_map.append(placeholder(i) + Literal(start, "int"))
        else:
            coord_map.append(placeholder(i))

    frag = open_fragment(graph, [x_id])
    new_id = single_indexmap(frag, x_id, out_shape=out_shape, coord_map=coord_map, name=root.output.name, dtype=root.output.dtype)
    frag.outputs = [new_id]
    return frag
