"""Decompose pow(x, 2) into mul(x, x) to enable RMSNorm fusion."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ElementwiseOp

PATTERN = "Elementwise{pow}($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace pow(x) with mul(x, x) — enables RMSNorm pattern matching."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = match.bindings["x"]

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
    return g
