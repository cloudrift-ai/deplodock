"""Decompose rms_norm(x, weight [, eps]) into primitives.

    out = x * rsqrt(mean(x * x, dim=-1, keepdim=True) + eps) * weight

The resulting MeanOp is further lowered to sum + div by 007_mean.
"""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.frontend.ir import MeanOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

PATTERN = [Pattern("root", RmsNormOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    root = graph.nodes[match.root_node_id]
    if len(root.inputs) < 2:
        return None
    x_id, w_id = root.inputs[0], root.inputs[1]
    eps_value = root.op.eps

    x_t = graph.nodes[x_id].output
    w_t = graph.nodes[w_id].output
    out_shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # Reduction output shape keeps the last dim at 1 (keepdim=True).
    red_shape = tuple(out_shape[:-1]) + (1,) if out_shape else (1,)

    frag = Graph()
    frag.add_node(op=InputOp(), inputs=[], output=Tensor(x_t.name, x_t.shape, x_t.dtype), node_id=x_id)
    frag.add_node(op=InputOp(), inputs=[], output=Tensor(w_t.name, w_t.shape, w_t.dtype), node_id=w_id)

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
    eps_id = frag.add_node(
        op=ConstantOp(name=f"{name}_eps", value=eps_value),
        inputs=[],
        output=Tensor(f"{name}_eps", (1,), dtype),
    )
    eps_bc = broadcast_to(frag, eps_id, red_shape)
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
