"""Decompose MeanOp into sum + div by the reduced dimension size."""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.frontend.ir import MeanOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", MeanOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    """Replace mean(x, axis) with sum(x, axis) / axis_size."""
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]

    axis = root.op.axis
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    x_shape = graph.nodes[x_id].output.shape
    if isinstance(axis, int) and x_shape:
        dim_size = x_shape[axis % len(x_shape)]
    else:
        dim_size = 1
    count_value = float(dim_size) if isinstance(dim_size, int) else 1.0

    frag = open_fragment(graph, [x_id])
    sum_id = frag.add_node(
        op=ReduceOp(op="sum", axis=axis),
        inputs=[x_id],
        output=Tensor(f"{name}_sum", shape, dtype),
    )
    count_bc = const_bc(frag, name=f"{name}_count", value=count_value, target_shape=shape, dtype=dtype)
    div_id = frag.add_node(
        op=ElementwiseOp(op="divide"),
        inputs=[sum_id, count_bc],
        output=Tensor(name, shape, dtype),
    )
    frag.outputs = [div_id]
    return frag
