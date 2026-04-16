"""Decompose MatmulOp into unsqueeze(A) * unsqueeze(B) → reduce_sum [→ add(bias)].

Inputs are unsqueezed so the mul is a standard NumPy broadcast:
  A(..., M, K) → A(..., M, K, 1)   via IndexMapOp
  B(..., K, N) → B(..., 1, K, N)   via IndexMapOp
  mul → (..., M, K, N)             broadcast-compatible
  reduce_sum(axis=-2) → (..., M, N)
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ElementwiseOp, MatmulOp, ReduceOp
from deplodock.compiler.rules.decomposition._matmul_helpers import matmul_unsqueeze

GRAMMAR = [Production("root", MatmulOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    a_id = root.inputs[0]
    b_id = root.inputs[1]
    bias_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    a_shape = tuple(g.nodes[a_id].output.shape)
    b_shape = tuple(g.nodes[b_id].output.shape)
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(a_shape, b_shape)

    a_unsq_id = g.add_node(op=a_unsq, inputs=[a_id], output=Tensor(f"{name}_a_unsq", a_unsq.out_shape, dtype))
    b_unsq_id = g.add_node(op=b_unsq, inputs=[b_id], output=Tensor(f"{name}_b_unsq", b_unsq.out_shape, dtype))
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

    if bias_id:
        add_id = g.add_node(op=ElementwiseOp(fn="add"), inputs=[red_id, bias_id], output=Tensor(f"{name}_bias", shape, dtype))
        g.replace_node(match.root_node_id, add_id)
    else:
        g.replace_node(match.root_node_id, red_id)
    g.remove_node(match.root_node_id)
    return g
