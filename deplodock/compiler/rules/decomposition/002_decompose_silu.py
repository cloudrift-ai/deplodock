"""Decompose silu(x) into x * recip(1 + exp(-x)) to enable SiLU+Mul fusion."""

from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [Production("root", ElementwiseOp, "1", {"fn": "silu"})]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    """Replace silu(x) with x * recip(1 + exp(-x))."""
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    # Constant: 1
    one_id = frag.add_node(
        op=ConstantOp(name=f"{name}_one", value=1.0),
        inputs=[],
        output=Tensor(f"{name}_one", (1,), dtype),
    )

    # neg(x)
    neg_id = frag.add_node(
        op=ElementwiseOp(fn="neg"),
        inputs=[x_id],
        output=Tensor(f"{name}_neg", shape, dtype),
    )

    # exp(-x)
    exp_id = frag.add_node(
        op=ElementwiseOp(fn="exp"),
        inputs=[neg_id],
        output=Tensor(f"{name}_exp", shape, dtype),
    )

    # 1 + exp(-x)
    add_id = frag.add_node(
        op=ElementwiseOp(fn="add"),
        inputs=[one_id, exp_id],
        output=Tensor(f"{name}_denom", shape, dtype),
    )

    # recip(1 + exp(-x))
    recip_id = frag.add_node(
        op=ElementwiseOp(fn="recip"),
        inputs=[add_id],
        output=Tensor(f"{name}_sigmoid", shape, dtype),
    )

    # x * sigmoid(x)
    mul_id = frag.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x_id, recip_id],
        output=Tensor(name, shape, dtype),
    )

    frag.outputs = [mul_id]
    return frag
