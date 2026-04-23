"""Lower SliceOp(x, dim, start, end) → IndexMapOp.

Tracer convention: SliceOp.inputs = [tensor, dim_const, start_const, end_const]
where each *_const is a ConstantOp with the literal value.

After decomposition: IndexMapOp.inputs = [tensor]; the dim/start values are
baked into the coord_map.
"""

from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import SliceOp
from deplodock.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from deplodock.compiler.pipeline.graph import Graph, Tensor
from deplodock.compiler.pipeline.matcher import Match, Pattern

PATTERN = [Pattern("root", SliceOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    rs_id = match.root_node_id
    root = graph.nodes[rs_id]
    x_id = root.inputs[0]
    dim_id = root.inputs[1]
    start_id = root.inputs[2]

    rs_node = graph.nodes[rs_id]
    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = tuple(rs_node.output.shape)
    ndim = len(in_shape)

    # Read dim and start from the constant inputs.
    dim_node = graph.nodes.get(dim_id)
    start_node = graph.nodes.get(start_id)
    if not (isinstance(dim_node.op, ConstantOp) and isinstance(start_node.op, ConstantOp)):
        return None
    if dim_node.op.value is None or start_node.op.value is None:
        return None

    dim = int(dim_node.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim
    start = int(start_node.op.value)

    frag = Graph()

    # InputOp sentinel for x (the tensor input only — constant inputs are
    # baked into the coord_map; the rewriter's _remove_orphans handles cleanup).
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    coord_map = []
    for i in range(ndim):
        if i == norm_dim and start != 0:
            coord_map.append(placeholder(i) + Literal(start, "int"))
        else:
            coord_map.append(placeholder(i))

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[x_id],
        output=Tensor(rs_node.output.name, out_shape, rs_node.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
