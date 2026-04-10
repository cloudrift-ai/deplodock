"""Fuse softmax chain: exp(x - max(x)) / sum(exp(x - max(x))) → FusedSoftmaxOp."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedSoftmaxOp

# Fully-specified pattern with $x fan-out (same input node 4 times).
# The exp/sub/max subgraph appears twice (numerator and denominator) but
# in the graph they share the same nodes — the matcher handles this correctly.
PATTERN = (
    "Elementwise{div}("
    "  Elementwise{exp}(Elementwise{sub}($x, Reduce{max, $ax2}($x))),"
    "  Reduce{sum, $ax}(Elementwise{exp}(Elementwise{sub}($x, Reduce{max, $ax3}($x))))"
    ")"
)


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace softmax chain with FusedSoftmaxOp."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = match.bindings["x"]
    axis = match.captured_constraints["ax"]

    fused_id = g.add_node(
        op=FusedSoftmaxOp(axis=axis),
        inputs=[x_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, fused_id)

    # Remove consumed nodes.
    _remove_chain(g, match.root_node_id, keep={x_id})

    return g


def _remove_chain(g: Graph, node_id: str, keep: set[str]) -> None:
    """Remove a node and its inputs recursively if they have no other consumers."""
    if node_id not in g.nodes or node_id in keep:
        return
    node = g.nodes[node_id]
    inputs = list(node.inputs)
    if not g.consumers(node_id):
        g.remove_node(node_id)
        for inp_id in inputs:
            _remove_chain(g, inp_id, keep)
