"""Decompose silu(x) into x * recip(1 + exp(-x)) to enable SiLU+Mul fusion."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc, open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "silu"})]


def rewrite(match: Match, inp_x: Node, out: Tensor) -> Graph | None:
    graph = match.graph
    """Replace silu(x) with x * recip(1 + exp(-x))."""
    frag = open_fragment(graph, [inp_x])

    neg_id = frag.add_node(op=ElementwiseOp(op="negative"), inputs=[inp_x], output=Tensor(f"{out.name}_neg", out.shape, out.dtype))
    exp_id = frag.add_node(op=ElementwiseOp(op="exp"), inputs=[neg_id], output=Tensor(f"{out.name}_exp", out.shape, out.dtype))
    one_bc = const_bc(frag, name=f"{out.name}_one", value=1.0, target_shape=out.shape, dtype=out.dtype)
    add_id = frag.add_node(
        op=ElementwiseOp(op="add"),
        inputs=[one_bc, exp_id],
        output=Tensor(f"{out.name}_denom", out.shape, out.dtype),
    )
    recip_id = frag.add_node(
        op=ElementwiseOp(op="reciprocal"),
        inputs=[add_id],
        output=Tensor(f"{out.name}_sigmoid", out.shape, out.dtype),
    )
    mul_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[inp_x, recip_id],
        output=Tensor(out.name, out.shape, out.dtype),
    )

    frag.outputs = [mul_id]
    return frag
