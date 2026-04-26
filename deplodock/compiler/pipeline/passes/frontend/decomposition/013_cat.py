"""Lower CatOp([a, b], dim) → IndexMapOp.

Tracer convention: CatOp.inputs = [tensor_a, tensor_b, dim_const]. Only the
2-tensor variant is supported here (covers Qwen rotary's
``cat(neg, slice_1, dim=-1)``); add a 3-tensor rule if a model needs it.

After decomposition: IndexMapOp.inputs = [tensor_a, tensor_b]; the dim is baked
into the source selects and the second source's coord_map offset.
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import CatOp
from deplodock.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", CatOp)]


def rewrite(graph: Graph, inp_a: Node, inp_b: Node, inp_dim: Node, out: Tensor) -> Graph | None:
    a_shape = tuple(inp_a.output.shape)
    out_shape = tuple(out.shape)
    ndim = len(out_shape)

    if not (isinstance(inp_dim.op, ConstantOp) and inp_dim.op.value is not None):
        return None
    dim = int(inp_dim.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

    if not isinstance(a_shape[norm_dim], int):
        return None
    split = a_shape[norm_dim]

    frag = open_fragment(graph, [inp_a, inp_b])

    # Source A: identity coord_map, select = (out_coord_dim < split)
    src_a = IndexSource(
        input_idx=0,
        coord_map=tuple(placeholder(i) for i in range(ndim)),
        select=placeholder(norm_dim).lt(Literal(split, "int")),
    )
    # Source B: coord_map[norm_dim] = out_coord_dim - split, identity elsewhere.
    # No select on B — it's the default branch in the TernaryExpr chain.
    coord_map_b = []
    for i in range(ndim):
        if i == norm_dim:
            coord_map_b.append(placeholder(i) - Literal(split, "int"))
        else:
            coord_map_b.append(placeholder(i))
    src_b = IndexSource(input_idx=1, coord_map=tuple(coord_map_b))

    new_id = frag.add_node(
        op=IndexMapOp(out_shape=out_shape, sources=(src_a, src_b)),
        inputs=[inp_a.id, inp_b.id],
        output=Tensor(out.name, out_shape, out.dtype),
    )

    frag.outputs = [new_id]
    return frag
