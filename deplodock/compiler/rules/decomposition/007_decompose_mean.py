"""Decompose MeanOp into sum + div by the reduced dimension size."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, MeanOp, ReduceOp

GRAMMAR = [Production("root", MeanOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace mean(x, axis) with sum(x, axis) / axis_size."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = root.inputs[0]

    axis = root.op.axis
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    x_shape = g.nodes[x_id].output.shape
    if isinstance(axis, int) and x_shape:
        norm_axis = axis % len(x_shape)
        dim_size = x_shape[norm_axis]
    else:
        dim_size = 1
    count_value = float(dim_size) if isinstance(dim_size, int) else 1.0

    sum_id = g.add_node(
        op=ReduceOp(fn="sum", axis=axis),
        inputs=[x_id],
        output=Tensor(f"{name}_sum", shape, dtype),
    )
    count_id = g.add_node(
        op=ConstantOp(name=f"{name}_count", value=count_value),
        inputs=[],
        output=Tensor(f"{name}_count", (1,), dtype),
    )
    div_id = g.add_node(
        op=ElementwiseOp(fn="div"),
        inputs=[sum_id, count_id],
        output=Tensor(name, shape, dtype),
    )

    g.replace_node(match.root_node_id, div_id)
    g.remove_node(match.root_node_id)
    return g
