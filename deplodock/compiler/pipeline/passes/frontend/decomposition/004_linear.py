"""Decompose LinearOp into transpose(weight) → unsqueeze → mul → reduce_sum [→ add(bias)].

Same unsqueeze strategy as matmul decomposition: inputs become
broadcast-compatible via IndexMapOp before the mul.
"""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, TransposeOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to, squeeze_axis

PATTERN = [Pattern("root", LinearOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    w_id = root.inputs[1]
    b_id = root.inputs[2] if len(root.inputs) > 2 else None
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    frag = Graph()

    # InputOp sentinels for all external references.
    ext_ids = {x_id, w_id}
    if b_id:
        ext_ids.add(b_id)
    for eid in sorted(ext_ids):
        frag.add_node(
            op=InputOp(),
            inputs=[],
            output=Tensor(graph.nodes[eid].output.name, graph.nodes[eid].output.shape, graph.nodes[eid].output.dtype),
            node_id=eid,
        )

    w_shape = graph.nodes[w_id].output.shape
    wt_shape = (w_shape[-1], w_shape[-2]) if len(w_shape) >= 2 else w_shape
    wt_id = frag.add_node(
        op=TransposeOp(axes=(-2, -1)),
        inputs=[w_id],
        output=Tensor(f"{name}_wt", wt_shape, dtype),
    )

    # Reuse matmul's unsqueeze logic for x @ wt.
    from deplodock.compiler.pipeline.passes.frontend.decomposition._matmul_helpers import matmul_unsqueeze

    x_shape = tuple(graph.nodes[x_id].output.shape)
    a_unsq, b_unsq, mul_shape, k_axis = matmul_unsqueeze(x_shape, tuple(wt_shape))

    a_unsq_id = frag.add_node(op=a_unsq, inputs=[x_id], output=Tensor(f"{name}_x_unsq", a_unsq.out_shape, dtype))
    b_unsq_id = frag.add_node(op=b_unsq, inputs=[wt_id], output=Tensor(f"{name}_wt_unsq", b_unsq.out_shape, dtype))

    a_bc = broadcast_to(frag, a_unsq_id, mul_shape)
    b_bc = broadcast_to(frag, b_unsq_id, mul_shape)
    ew_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[a_bc, b_bc],
        output=Tensor(f"{name}_ew", mul_shape, dtype),
    )
    # Keepdim reduce + squeeze — keeps the rank-preservation invariant local
    # to the ReduceOp while restoring the Linear's declared output shape.
    reduce_shape = tuple(mul_shape[:k_axis]) + (1,) + tuple(mul_shape[k_axis + 1 :])
    red_id = frag.add_node(
        op=ReduceOp(op="sum", axis=k_axis),
        inputs=[ew_id],
        output=Tensor(f"{name}_reduce", reduce_shape, dtype),
    )

    if b_id:
        sq_id = squeeze_axis(frag, red_id, k_axis)
        bias_bc = broadcast_to(frag, b_id, shape)
        add_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[sq_id, bias_bc],
            output=Tensor(name, shape, dtype),
        )
        frag.outputs = [add_id]
    else:
        sq_id = squeeze_axis(frag, red_id, k_axis, out_name=name)
        frag.outputs = [sq_id]

    return frag
