"""Decompose LinearOp into transpose(weight) → mul → reduce_sum [→ add(bias)]."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, LinearOp, ReduceOp, TransposeOp

GRAMMAR = [Production("root", LinearOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace LinearOp(x, w [, b]) with x @ w.T [+ b]."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = root.inputs[0]
    w_id = root.inputs[1]
    b_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # Transpose weight: (out_features, in_features) → (in_features, out_features)
    w_shape = g.nodes[w_id].output.shape
    wt_shape = (w_shape[-1], w_shape[-2]) if len(w_shape) >= 2 else w_shape
    wt_id = g.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[w_id],
        output=Tensor(f"{name}_wt", wt_shape, dtype),
    )

    # mul(x, w_transposed) → intermediate with an extra K dimension
    ew_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x_id, wt_id],
        output=Tensor(f"{name}_ew", shape + ("K",), dtype),
    )

    # reduce_sum over K → result
    red_id = g.add_node(
        op=ReduceOp(fn="sum", axis="K"),
        inputs=[ew_id],
        output=Tensor(name, shape, dtype),
    )

    if b_id:
        # result + bias
        add_id = g.add_node(
            op=ElementwiseOp(fn="add"),
            inputs=[red_id, b_id],
            output=Tensor(f"{name}_bias", shape, dtype),
        )
        g.replace_node(match.root_node_id, add_id)
    else:
        g.replace_node(match.root_node_id, red_id)

    g.remove_node(match.root_node_id)
    return g
