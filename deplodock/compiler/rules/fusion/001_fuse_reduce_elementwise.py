"""Fuse elementwise op into reduce — avoid materializing intermediate tensor."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import MatmulOp, ReduceOp

PATTERN = "Reduce{sum, $ax}(Elementwise{mul}($A, $B))"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace Reduce{sum}(Elementwise{mul}(A, B)) with MatmulOp.

    Self-multiplication (A == B, i.e. squared norm for RMSNorm) is converted
    to ReduceOp("sum_sq") so auto_fuse handles it instead.
    """
    g = graph.copy()
    root = g.nodes[match.root_node_id]
    ew_node_id = root.inputs[0]

    if match.bindings["A"] == match.bindings["B"]:
        # Squared norm: replace with ReduceOp("sum_sq") to avoid re-matching.
        x_id = match.bindings["A"]
        fused_id = g.add_node(
            op=ReduceOp(fn="sum_sq", axis=match.captured_constraints["ax"]),
            inputs=[x_id],
            output=Tensor(name=root.output.name, shape=root.output.shape, dtype=root.output.dtype),
        )
        g.replace_node(match.root_node_id, fused_id)
        g.remove_node(match.root_node_id)
        if not g.consumers(ew_node_id):
            g.remove_node(ew_node_id)
        return g

    fused_op = MatmulOp()

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
