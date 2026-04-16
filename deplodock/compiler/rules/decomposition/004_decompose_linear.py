"""Decompose LinearOp into transpose(weight) → unsqueeze → mul → reduce_sum [→ add(bias)].

Same unsqueeze strategy as matmul decomposition: inputs become
broadcast-compatible via IndexMapOp before the mul.
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, LinearOp, ReduceOp, TransposeOp

GRAMMAR = [Production("root", LinearOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = root.inputs[0]
    w_id = root.inputs[1]
    b_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    w_shape = g.nodes[w_id].output.shape
    wt_shape = (w_shape[-1], w_shape[-2]) if len(w_shape) >= 2 else w_shape
    wt_id = g.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[w_id],
        output=Tensor(f"{name}_wt", wt_shape, dtype),
    )

    # Reuse matmul's unsqueeze logic for x @ wt.
    from deplodock.compiler.rules.decomposition._matmul_helpers import matmul_unsqueeze

    x_shape = tuple(g.nodes[x_id].output.shape)
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(x_shape, tuple(wt_shape))

    a_unsq_id = g.add_node(op=a_unsq, inputs=[x_id], output=Tensor(f"{name}_x_unsq", a_unsq.out_shape, dtype))
    b_unsq_id = g.add_node(op=b_unsq, inputs=[wt_id], output=Tensor(f"{name}_wt_unsq", b_unsq.out_shape, dtype))

    ew_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a_unsq_id, b_unsq_id],
        output=Tensor(f"{name}_ew", mul_shape, dtype),
    )
    red_id = g.add_node(
        op=ReduceOp(fn="sum", axis=k_axis),
        inputs=[ew_id],
        output=Tensor(name, shape, dtype),
    )

    if b_id:
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
