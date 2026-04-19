"""Decompose MatmulOp into unsqueeze(A) * unsqueeze(B) → reduce_sum [→ add(bias)].

Inputs are unsqueezed so the mul is a standard NumPy broadcast:
  A(..., M, K) → A(..., M, K, 1)   via IndexMapOp
  B(..., K, N) → B(..., 1, K, N)   via IndexMapOp
  mul → (..., M, K, N)             broadcast-compatible
  reduce_sum(axis=-2) → (..., M, N)
"""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.broadcast import broadcast_to
from deplodock.compiler.ir.frontend import MatmulOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.rules.decomposition._matmul_helpers import matmul_unsqueeze

GRAMMAR = [Production("root", MatmulOp, "1")]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    a_id = root.inputs[0]
    b_id = root.inputs[1]
    bias_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    frag = Graph()

    # InputOp sentinels for all external references.
    ext_ids = {a_id, b_id}
    if bias_id:
        ext_ids.add(bias_id)
    for eid in sorted(ext_ids):
        frag.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(graph.nodes[eid].output.name, graph.nodes[eid].output.shape, graph.nodes[eid].output.dtype),
            node_id=eid,
        )

    a_shape = tuple(graph.nodes[a_id].output.shape)
    b_shape = tuple(graph.nodes[b_id].output.shape)
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(a_shape, b_shape)

    a_unsq_id = frag.add_node(op=a_unsq, inputs=[a_id], output=Tensor(f"{name}_a_unsq", a_unsq.out_shape, dtype))
    b_unsq_id = frag.add_node(op=b_unsq, inputs=[b_id], output=Tensor(f"{name}_b_unsq", b_unsq.out_shape, dtype))
    a_bc = broadcast_to(frag, a_unsq_id, mul_shape)
    b_bc = broadcast_to(frag, b_unsq_id, mul_shape)
    ew_id = frag.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a_bc, b_bc],
        output=Tensor(f"{name}_ew", mul_shape, dtype),
    )
    red_id = frag.add_node(
        op=ReduceOp(fn="sum", axis=k_axis),
        inputs=[ew_id],
        output=Tensor(name, shape, dtype),
    )

    if bias_id:
        bias_bc = broadcast_to(frag, bias_id, shape)
        add_id = frag.add_node(op=ElementwiseOp(fn="add"), inputs=[red_id, bias_bc], output=Tensor(f"{name}_bias", shape, dtype))
        frag.outputs = [add_id]
    else:
        frag.outputs = [red_id]

    return frag
