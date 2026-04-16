"""Lower SliceOp(x, dim, start, end) → IndexMapOp.

Tracer convention: SliceOp.inputs = [tensor, dim_const, start_const, end_const]
where each *_const is a ConstantOp with the literal value.

After decomposition: IndexMapOp.inputs = [tensor]; the dim/start values are
baked into the coord_map.
"""

from deplodock.compiler.backend.ir.expr import Literal
from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ConstantOp, IndexMapOp, IndexSource

PATTERN = "Slice($x, $dim, $start, $end)"


def rewrite(graph: Graph, match: Match) -> Graph:
    rs_id = match.root_node_id
    x_id = match.bindings["x"]
    dim_id = match.bindings["dim"]
    start_id = match.bindings["start"]

    rs_node = graph.nodes[rs_id]
    in_shape = tuple(graph.nodes[x_id].output.shape)
    out_shape = tuple(rs_node.output.shape)
    ndim = len(in_shape)

    # Read dim and start from the constant inputs.
    dim_node = graph.nodes.get(dim_id)
    start_node = graph.nodes.get(start_id)
    if not (isinstance(dim_node.op, ConstantOp) and isinstance(start_node.op, ConstantOp)):
        return graph
    if dim_node.op.value is None or start_node.op.value is None:
        return graph

    dim = int(dim_node.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim
    start = int(start_node.op.value)

    g = graph.copy()
    coord_map = []
    for i in range(ndim):
        if i == norm_dim and start != 0:
            coord_map.append(placeholder(i) + Literal(start, "int"))
        else:
            coord_map.append(placeholder(i))

    new_id = g.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(IndexSource(input_idx=0, coord_map=tuple(coord_map)),)),
        inputs=[x_id],
        output=Tensor(rs_node.output.name, out_shape, rs_node.output.dtype),
    )
    g.replace_node(rs_id, new_id)
    g.remove_node(rs_id)

    # Clean up unused dim/start/end constants if they have no other consumers.
    for cid in (dim_id, start_id, match.bindings.get("end")):
        if cid and cid in g.nodes and not g.consumers(cid):
            g.remove_node(cid)

    return g
