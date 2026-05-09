"""Decompose rms_norm(x, weight [, eps]) into x * rsqrt(mean(x*x, -1, keepdim) + eps) * weight."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import MeanOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    const_bc,
    open_fragment,
    reduction_shape,
)

PATTERN = [Pattern("root", RmsNormOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    if inp_w is None:
        raise RuleSkipped("rms_norm without weight input is not decomposed")
    red_shape = reduction_shape(out.shape, -1) if out.shape else (1,)

    frag = open_fragment(graph, [inp_x, inp_w])

    sq_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, inp_x],
        output=Tensor(f"{out.name}_sq", out.shape, out.dtype),
    )
    mean_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[sq_id],
        output=Tensor(f"{out.name}_mean", red_shape, out.dtype),
    )
    eps_bc = const_bc(frag, name=f"{out.name}_eps", value=root.op.eps, target_shape=red_shape, dtype=out.dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[mean_id, eps_bc],
        output=Tensor(f"{out.name}_add_eps", red_shape, out.dtype),
    )
    rsq_id = frag.add_node(
        op=ElementwiseOp(op="rsqrt"),
        inputs=[add_id],
        output=Tensor(f"{out.name}_rsq", red_shape, out.dtype),
    )
    rsq_bc = broadcast_to(frag, rsq_id, out.shape)
    norm_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, rsq_bc],
        output=Tensor(f"{out.name}_norm", out.shape, out.dtype),
    )
    w_bc = broadcast_to(frag, inp_w, out.shape)
    out_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[norm_id, w_bc],
        output=Tensor(out.name, out.shape, out.dtype),
    )

    frag.outputs = [out_id]
    return frag
