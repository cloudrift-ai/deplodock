"""Decompose rms_norm(x, weight [, eps]) into x * rsqrt(mean(x*x, -1, keepdim) + eps) * weight."""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.frontend.ir import MeanOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    const_bc,
    open_fragment,
    reduction_shape,
)

PATTERN = [Pattern("root", RmsNormOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    if len(root.inputs) < 2:
        return None
    x_id, w_id = root.inputs[0], root.inputs[1]
    eps_value = root.op.eps

    out_shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    red_shape = reduction_shape(out_shape, -1) if out_shape else (1,)

    frag = open_fragment(graph, [x_id, w_id])

    sq_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x_id, x_id],
        output=Tensor(f"{name}_sq", out_shape, dtype),
    )
    mean_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[sq_id],
        output=Tensor(f"{name}_mean", red_shape, dtype),
    )
    eps_bc = const_bc(frag, name=f"{name}_eps", value=eps_value, target_shape=red_shape, dtype=dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[mean_id, eps_bc],
        output=Tensor(f"{name}_add_eps", red_shape, dtype),
    )
    rsq_id = frag.add_node(
        op=ElementwiseOp(op="rsqrt"),
        inputs=[add_id],
        output=Tensor(f"{name}_rsq", red_shape, dtype),
    )
    rsq_bc = broadcast_to(frag, rsq_id, out_shape)
    norm_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x_id, rsq_bc],
        output=Tensor(f"{name}_norm", out_shape, dtype),
    )
    w_bc = broadcast_to(frag, w_id, out_shape)
    out_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[norm_id, w_bc],
        output=Tensor(name, out_shape, dtype),
    )

    frag.outputs = [out_id]
    return frag
