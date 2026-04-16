"""Lower CatOp([a, b], dim) → IndexMapOp.

Tracer convention: CatOp.inputs = [tensor_a, tensor_b, dim_const]. Only the
2-tensor variant is supported here (covers Qwen rotary's
``cat(neg, slice_1, dim=-1)``); add a 3-tensor rule if a model needs it.

After decomposition: IndexMapOp.inputs = [tensor_a, tensor_b]; the dim is baked
into the source selects and the second source's coord_map offset.
"""

from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend import CatOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", CatOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    rs_id = match.root_node_id
    root = graph.nodes[rs_id]
    a_id = root.inputs[0]
    b_id = root.inputs[1]
    dim_id = root.inputs[2]

    rs_node = graph.nodes[rs_id]
    a_shape = tuple(graph.nodes[a_id].output.shape)
    out_shape = tuple(rs_node.output.shape)
    ndim = len(out_shape)

    dim_node = graph.nodes.get(dim_id)
    if not (isinstance(dim_node.op, ConstantOp) and dim_node.op.value is not None):
        return None
    dim = int(dim_node.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

    if not isinstance(a_shape[norm_dim], int):
        return None
    split = a_shape[norm_dim]

    frag = Graph()

    # InputOp sentinels for tensor inputs (constant dim input is baked into
    # the coord_map; the rewriter's _remove_orphans handles cleanup).
    for eid in sorted({a_id, b_id}):
        frag.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(graph.nodes[eid].output.name, graph.nodes[eid].output.shape, graph.nodes[eid].output.dtype),
            node_id=eid,
        )

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

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(src_a, src_b)),
        inputs=[a_id, b_id],
        output=Tensor(rs_node.output.name, out_shape, rs_node.output.dtype),
    )

    frag.outputs = [new_id]
    return frag
