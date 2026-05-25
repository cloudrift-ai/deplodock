"""Lower SliceOp(x, dim, start, end) → IndexMapOp.

Tracer convention: SliceOp.inputs = [tensor, dim_const, start_const, end_const].
After decomposition: IndexMapOp.inputs = [tensor]; dim/start are baked into
the coord_map (orphan constants get cleaned up by the rewriter).
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import SliceOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", SliceOp)]


def rewrite(match: Match, inp_x: Node, inp_dim: Node, inp_start: Node, inp_end: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    in_shape = tuple(inp_x.output.shape)
    out_shape = tuple(out.shape)
    ndim = len(in_shape)

    if not (isinstance(inp_dim.op, ConstantOp) and isinstance(inp_start.op, ConstantOp)):
        raise RuleSkipped("slice dim/start must be ConstantOp to bake into coord_map")
    if inp_dim.op.value is None or inp_start.op.value is None:
        raise RuleSkipped("slice dim/start ConstantOp has no value")

    dim = int(inp_dim.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim
    start = int(inp_start.op.value)

    coord_map = []
    for i in range(ndim):
        if i == norm_dim and start != 0:
            coord_map.append(placeholder(i) + Literal(start, "int"))
        else:
            coord_map.append(placeholder(i))

    frag = open_fragment(graph, [inp_x])
    new_node = single_indexmap(frag, inp_x, out_shape=out_shape, coord_map=coord_map, name=out.name)
    frag.outputs = [new_node.id]
    return frag
