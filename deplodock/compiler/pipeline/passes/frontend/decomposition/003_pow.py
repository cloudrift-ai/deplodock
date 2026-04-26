"""Decompose pow(x, 2) into mul(x, x) to enable RMSNorm fusion."""

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import ConstantOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Pattern
from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import open_fragment

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "pow"})]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    """Replace pow(x, 2) with mul(x, x) — enables RMSNorm pattern matching."""
    x_id = root.inputs[0]
    exp_id = root.inputs[1]

    exp_node = graph.nodes.get(exp_id)
    if exp_node and isinstance(exp_node.op, ConstantOp) and exp_node.op.value != 2.0:
        return None

    frag = open_fragment(graph, [x_id])
    mul_id = frag.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x_id, x_id],
        output=Tensor(name=root.output.name, shape=root.output.shape, dtype=root.output.dtype),
    )
    frag.outputs = [mul_id]
    return frag
