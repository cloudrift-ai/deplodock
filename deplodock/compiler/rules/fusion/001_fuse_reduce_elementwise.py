"""Fuse elementwise op into reduce — avoid materializing intermediate tensor."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedReduceElementwiseOp

PATTERN = "Reduce{$f, $ax}(Elementwise{$g}($A, $B))"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace Reduce(Elementwise(A, B)) with a single FusedReduceElementwiseOp."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    ew_node_id = root.inputs[0]

    fused_op = FusedReduceElementwiseOp(
        reduce_fn=match.captured_constraints["f"],
        elementwise_fn=match.captured_constraints["g"],
        axis=match.captured_constraints["ax"],
    )

    # Inputs come from the elementwise node's inputs (the original A, B).
    ew_node = g.nodes[ew_node_id]
    fused_id = g.add_node(
        op=fused_op,
        inputs=list(ew_node.inputs),
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    # Rewire consumers of the reduce node to the fused node.
    g.replace_node(match.root_node_id, fused_id)

    # Remove old nodes (reduce and elementwise) if no longer referenced.
    g.remove_node(match.root_node_id)
    if not g.consumers(ew_node_id):
        g.remove_node(ew_node_id)

    return g
