"""Decompose softmax(x, dim) into max → sub → exp → sum → div.

    m   = max(x, dim, keepdim=True)
    e   = exp(x - m)
    out = e / sum(e, dim, keepdim=True)

Same math as the softmax block inside ``001_sdpa.py``; kept as its
own rule so standalone ``torch.nn.Softmax`` / ``F.softmax`` calls decompose
without going through the SDPA fast-path.
"""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "softmax"})]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    if not root.inputs:
        return None
    x_id = root.inputs[0]

    # aten.softmax.int stores the dim as a scalar constant input (the tracer
    # converts ints to ConstantOps). If absent, default to -1 (softmax's
    # canonical axis for NN uses).
    axis: int = -1
    if len(root.inputs) >= 2:
        dim_node = graph.nodes.get(root.inputs[1])
        if dim_node and isinstance(dim_node.op, ConstantOp) and isinstance(dim_node.op.value, (int, float)):
            axis = int(dim_node.op.value)

    x_t = graph.nodes[x_id].output
    out_shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # Reduction shape keeps the softmax axis at 1 (keepdim=True).
    norm_axis = axis % len(out_shape) if out_shape else 0
    red_shape = tuple(out_shape[:norm_axis]) + (1,) + tuple(out_shape[norm_axis + 1 :]) if out_shape else (1,)

    frag = Graph()
    frag.add_node(op=InputOp(), inputs=[], output=Tensor(x_t.name, x_t.shape, x_t.dtype), node_id=x_id)

    max_id = frag.add_node(
        op=ReduceOp(fn="max", axis=axis),
        inputs=[x_id],
        output=Tensor(f"{name}_max", red_shape, dtype),
    )
    max_bc = broadcast_to(frag, max_id, out_shape)
    sub_id = frag.add_node(
        op=ElementwiseOp(fn="sub"),
        inputs=[x_id, max_bc],
        output=Tensor(f"{name}_shifted", out_shape, dtype),
    )
    exp_id = frag.add_node(
        op=ElementwiseOp(fn="exp"),
        inputs=[sub_id],
        output=Tensor(f"{name}_exp", out_shape, dtype),
    )
    sum_id = frag.add_node(
        op=ReduceOp(fn="sum", axis=axis),
        inputs=[exp_id],
        output=Tensor(f"{name}_sum", red_shape, dtype),
    )
    sum_bc = broadcast_to(frag, sum_id, out_shape)
    out_id = frag.add_node(
        op=ElementwiseOp(fn="div"),
        inputs=[exp_id, sum_bc],
        output=Tensor(name, out_shape, dtype),
    )

    frag.outputs = [out_id]
    return frag
