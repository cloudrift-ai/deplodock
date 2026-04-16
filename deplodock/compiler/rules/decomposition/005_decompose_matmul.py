"""Decompose MatmulOp into mul → reduce_sum [→ add(bias)]."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, MatmulOp, ReduceOp

GRAMMAR = [Production("root", MatmulOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace MatmulOp(a, b [, bias]) with a @ b [+ bias]."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    a_id = root.inputs[0]
    b_id = root.inputs[1]
    bias_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # mul(a, b) → intermediate with extra K dimension
    ew_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a_id, b_id],
        output=Tensor(f"{name}_ew", shape + ("K",), dtype),
    )

    # reduce_sum over K → result
    red_id = g.add_node(
        op=ReduceOp(fn="sum", axis="K"),
        inputs=[ew_id],
        output=Tensor(name, shape, dtype),
    )

    if bias_id:
        add_id = g.add_node(
            op=ElementwiseOp(fn="add"),
            inputs=[red_id, bias_id],
            output=Tensor(f"{name}_bias", shape, dtype),
        )
        g.replace_node(match.root_node_id, add_id)
    else:
        g.replace_node(match.root_node_id, red_id)

    g.remove_node(match.root_node_id)
    return g
