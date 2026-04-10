"""Decompose silu(x) into x * recip(1 + exp(-x)) to enable SiLU+Mul fusion."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ConstantOp, ElementwiseOp

PATTERN = "Elementwise{silu}($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace silu(x) with x * recip(1 + exp(-x))."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = match.bindings["x"]
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    # Constant: 1
    one_id = g.add_node(
        op=ConstantOp(name=f"{name}_one"),
        inputs=[],
        output=Tensor(f"{name}_one", (1,), dtype),
    )

    # neg(x)
    neg_id = g.add_node(
        op=ElementwiseOp(fn="neg"),
        inputs=[x_id],
        output=Tensor(f"{name}_neg", shape, dtype),
    )

    # exp(-x)
    exp_id = g.add_node(
        op=ElementwiseOp(fn="exp"),
        inputs=[neg_id],
        output=Tensor(f"{name}_exp", shape, dtype),
    )

    # 1 + exp(-x)
    add_id = g.add_node(
        op=ElementwiseOp(fn="add"),
        inputs=[one_id, exp_id],
        output=Tensor(f"{name}_denom", shape, dtype),
    )

    # recip(1 + exp(-x))
    recip_id = g.add_node(
        op=ElementwiseOp(fn="recip"),
        inputs=[add_id],
        output=Tensor(f"{name}_sigmoid", shape, dtype),
    )

    # x * sigmoid(x)
    mul_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x_id, recip_id],
        output=Tensor(name, shape, dtype),
    )

    g.replace_node(match.root_node_id, mul_id)
    g.remove_node(match.root_node_id)
    return g
