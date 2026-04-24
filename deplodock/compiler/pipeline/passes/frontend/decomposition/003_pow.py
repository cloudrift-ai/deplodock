"""Decompose pow(x, 2) into mul(x, x) to enable RMSNorm fusion."""

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", ElementwiseOp, {"fn": "pow"})]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    """Replace pow(x, 2) with mul(x, x) — enables RMSNorm pattern matching."""
    root = graph.nodes[match.root_node_id]
    x_id = root.inputs[0]
    exp_id = root.inputs[1]

    # Only decompose pow(x, 2) — check the exponent constant.
    exp_node = graph.nodes.get(exp_id)
    if exp_node and isinstance(exp_node.op, ConstantOp) and exp_node.op.value != 2.0:
        return None  # Not pow(x, 2) — leave unchanged.

    frag = Graph()

    # InputOp sentinel for x.
    frag.add_node(
        op=InputOp(),
        inputs=[],
        output=Tensor(graph.nodes[x_id].output.name, graph.nodes[x_id].output.shape, graph.nodes[x_id].output.dtype),
        node_id=x_id,
    )

    mul_id = frag.add_node(
        op=ElementwiseOp(op="mul"),
        inputs=[x_id, x_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    frag.outputs = [mul_id]
    return frag
