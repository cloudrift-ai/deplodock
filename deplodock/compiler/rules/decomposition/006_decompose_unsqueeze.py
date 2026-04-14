"""Decompose UnsqueezeOp into ReshapeOp — same data, new shape with an extra dim."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ReshapeOp

PATTERN = "Unsqueeze($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace UnsqueezeOp with ReshapeOp using the captured output shape."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = match.bindings["x"]
    shape = root.output.shape
    dtype = root.output.dtype
    name = root.output.name

    reshape_id = g.add_node(
        op=ReshapeOp(shape=shape),
        inputs=[x_id],
        output=Tensor(name, shape, dtype),
    )

    g.replace_node(match.root_node_id, reshape_id)
    g.remove_node(match.root_node_id)
    return g
