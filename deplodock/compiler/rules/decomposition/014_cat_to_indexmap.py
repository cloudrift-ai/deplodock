"""Lower CatOp([a, b], dim) → IndexMapOp.

Tracer convention: CatOp.inputs = [tensor_a, tensor_b, dim_const]. Only the
2-tensor variant is supported here (covers Qwen rotary's
``cat(neg, slice_1, dim=-1)``); add a 3-tensor rule if a model needs it.

After decomposition: IndexMapOp.inputs = [tensor_a, tensor_b]; the dim is baked
into the source selects and the second source's coord_map offset.
"""

from deplodock.compiler.backend.ir.expr import Literal
from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ConstantOp, IndexMapOp, IndexSource

PATTERN = "Cat($a, $b, $dim)"


def rewrite(graph: Graph, match: Match) -> Graph:
    rs_id = match.root_node_id
    a_id = match.bindings["a"]
    b_id = match.bindings["b"]
    dim_id = match.bindings["dim"]

    rs_node = graph.nodes[rs_id]
    a_shape = tuple(graph.nodes[a_id].output.shape)
    # b_shape would be useful for asserting shape compatibility but is unused
    # here — the cat semantics are encoded entirely via select + offset on the
    # split axis, computed from a_shape's split position.
    out_shape = tuple(rs_node.output.shape)
    ndim = len(out_shape)

    dim_node = graph.nodes.get(dim_id)
    if not (isinstance(dim_node.op, ConstantOp) and dim_node.op.value is not None):
        return graph
    dim = int(dim_node.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

    if not isinstance(a_shape[norm_dim], int):
        return graph
    split = a_shape[norm_dim]

    g = graph.copy()

    # Source A: identity coord_map, select = (out_coord_dim < split)
    src_a = IndexSource(
        input_idx=0,
        coord_map=tuple(placeholder(i) for i in range(ndim)),
        select=placeholder(norm_dim).lt(Literal(split, "int")),
    )
    # Source B: coord_map[norm_dim] = out_coord_dim - split, identity elsewhere.
    # No select on B — it's the default branch in the Ternary chain.
    coord_map_b = []
    for i in range(ndim):
        if i == norm_dim:
            coord_map_b.append(placeholder(i) - Literal(split, "int"))
        else:
            coord_map_b.append(placeholder(i))
    src_b = IndexSource(input_idx=1, coord_map=tuple(coord_map_b))

    new_id = g.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(src_a, src_b)),
        inputs=[a_id, b_id],
        output=Tensor(rs_node.output.name, out_shape, rs_node.output.dtype),
    )
    g.replace_node(rs_id, new_id)
    g.remove_node(rs_id)

    if dim_id in g.nodes and not g.consumers(dim_id):
        g.remove_node(dim_id)

    return g
