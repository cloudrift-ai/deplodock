"""Decompose pow(x, 2) into mul(x, x) to enable RMSNorm fusion."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import ConstantOp, ElementwiseOp

GRAMMAR = [Production("root", ElementwiseOp, "1", {"fn": "pow"})]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace pow(x, 2) with mul(x, x) — enables RMSNorm pattern matching."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = root.inputs[0]
    exp_id = root.inputs[1]

    # Only decompose pow(x, 2) — check the exponent constant.
    exp_node = g.nodes.get(exp_id)
    if exp_node and isinstance(exp_node.op, ConstantOp) and exp_node.op.value != 2.0:
        return graph  # Not pow(x, 2) — leave unchanged.

    mul_id = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x_id, x_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, mul_id)
    g.remove_node(match.root_node_id)

    # Remove the exponent constant if it has no other consumers.
    if exp_id in g.nodes and not g.consumers(exp_id):
        g.remove_node(exp_id)

    return g
