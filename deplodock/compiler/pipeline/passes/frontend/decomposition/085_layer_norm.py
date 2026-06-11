"""Decompose layer_norm(x [, w [, b]]) into (x - mean(x)) * rsqrt(mean((x - mean(x))^2) + eps) [* w] [+ b]."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.frontend.ir import LayerNormOp, MeanOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import (
    broadcast_to,
    const_bc,
    open_fragment,
    reduction_shape,
)

PATTERN = [Pattern("root", LayerNormOp)]


def rewrite(match: Match, root: Node, inp_x: Node, inp_w: Node | None, inp_b: Node | None, out: Tensor) -> Graph | None:
    graph = match.graph
    red_shape = reduction_shape(out.shape, -1) if out.shape else (1,)

    frag = open_fragment(graph, [n for n in (inp_x, inp_w, inp_b) if n is not None])

    mean_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[inp_x],
        output=Tensor(f"{out.name}_mean", red_shape, out.dtype),
    )
    mean_bc = broadcast_to(frag, mean_id, out.shape)
    xc_id = frag.add_node(
        op=ElementwiseOp(op="subtract"),
        inputs=[inp_x, mean_bc],
        output=Tensor(f"{out.name}_centered", out.shape, out.dtype),
    )
    sq_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[xc_id, xc_id],
        output=Tensor(f"{out.name}_sq", out.shape, out.dtype),
    )
    var_id = frag.add_node(
        op=MeanOp(axis=-1),
        inputs=[sq_id],
        output=Tensor(f"{out.name}_var", red_shape, out.dtype),
    )
    eps_bc = const_bc(frag, name=f"{out.name}_eps", value=root.op.eps, target_shape=red_shape, dtype=out.dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[var_id, eps_bc],
        output=Tensor(f"{out.name}_add_eps", red_shape, out.dtype),
    )
    rsq_id = frag.add_node(
        op=ElementwiseOp(op="rsqrt"),
        inputs=[add_id],
        output=Tensor(f"{out.name}_rsq", red_shape, out.dtype),
    )
    rsq_bc = broadcast_to(frag, rsq_id, out.shape)
    cur_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[xc_id, rsq_bc],
        output=Tensor(out.name if inp_w is None and inp_b is None else f"{out.name}_norm", out.shape, out.dtype),
    )
    if inp_w is not None:
        w_bc = broadcast_to(frag, inp_w, out.shape)
        cur_id = frag.add_node(
            op=ElementwiseOp(op="multiply"),
            inputs=[cur_id, w_bc],
            output=Tensor(f"{out.name}_scaled" if inp_b is not None else out.name, out.shape, out.dtype),
        )
    if inp_b is not None:
        b_bc = broadcast_to(frag, inp_b, out.shape)
        cur_id = frag.add_node(
            op=ElementwiseOp(op="add"),
            inputs=[cur_id, b_bc],
            output=Tensor(out.name, out.shape, out.dtype),
        )

    frag.outputs = [cur_id]
    return frag
