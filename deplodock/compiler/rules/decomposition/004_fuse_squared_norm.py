"""Fuse self-mul + reduce_sum into FusedReduceElementwiseOp.

This runs in the decomposition pass (before fusion) to prevent the
matmul rule from consuming squared norm patterns that belong to RMSNorm.
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedReduceElementwiseOp

PATTERN = "Reduce{sum, $ax}(Elementwise{mul}($x, $x))"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace reduce_sum(mul(x, x)) with FusedReduceElementwiseOp."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    x_id = match.bindings["x"]
    ew_node_id = root.inputs[0]

    fused_id = g.add_node(
        op=FusedReduceElementwiseOp(
            reduce_fn="sum",
            elementwise_fn="mul",
            axis=match.captured_constraints["ax"],
        ),
        inputs=[x_id, x_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, fused_id)
    g.remove_node(match.root_node_id)
    if not g.consumers(ew_node_id):
        g.remove_node(ew_node_id)

    return g
