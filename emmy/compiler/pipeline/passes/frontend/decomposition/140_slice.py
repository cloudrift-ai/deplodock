"""Lower SliceOp(x, dim, start, end) → IndexMapOp.

dim/start come from ``SliceOp.dim`` / ``SliceOp.start`` (recorded by the
tracer; pre-field IR dumps fall back to the legacy constant-input convention
``inputs = [tensor, dim_const, start_const, end_const]``). After
decomposition: IndexMapOp.inputs = [tensor]; dim/start are baked into the
coord_map (orphan constants get cleaned up by the rewriter). The slice end is
never needed — ``out.shape`` carries the extent, symbolic or static.
"""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.expr import Literal, placeholder
from deplodock.compiler.ir.frontend.ir import SliceOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment, single_indexmap

PATTERN = [Pattern("root", SliceOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_dim: Node | None, inp_start: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    in_shape = tuple(inp_x.output.shape)
    out_shape = tuple(out.shape)
    ndim = len(in_shape)

    # The tracer records dim/start on the op (required when the FX start is
    # ``None`` or the end is a SymInt — those args don't survive as constant
    # inputs). Pre-field IR dumps fall back to the constant-input convention.
    if root.op.dim is not None and root.op.start is not None:
        dim = int(root.op.dim)
        start = int(root.op.start)
    else:
        if inp_dim is None or inp_start is None:
            raise RuleSkipped("slice carries no dim/start fields and no dim/start constant inputs")
        if not (isinstance(inp_dim.op, ConstantOp) and isinstance(inp_start.op, ConstantOp)):
            raise RuleSkipped("slice dim/start must be ConstantOp to bake into coord_map")
        if inp_dim.op.value is None or inp_start.op.value is None:
            raise RuleSkipped("slice dim/start ConstantOp has no value")
        dim = int(inp_dim.op.value)
        start = int(inp_start.op.value)
    norm_dim = dim if dim >= 0 else ndim + dim

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
