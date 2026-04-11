"""Fuse attention pattern: Matmul(Softmax(Scale(Matmul(Q, K^T))), V) → FusedAttentionOp."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedAttentionOp

PATTERN = "Matmul(FusedSoftmax(Elementwise{mul}(Matmul($Q, $Kt), $scale)), $V)"


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace attention pattern with FusedAttentionOp."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]

    q_id = match.bindings["Q"]
    kt_id = match.bindings["Kt"]
    v_id = match.bindings["V"]
    # Infer dimensions from Q shape.
    q_shape = g.nodes[q_id].output.shape
    head_dim = q_shape[-1] if len(q_shape) >= 2 and isinstance(q_shape[-1], int) else 64
    num_heads = q_shape[-3] if len(q_shape) >= 4 and isinstance(q_shape[-3], int) else 1
    scale = 1.0 / (head_dim**0.5) if isinstance(head_dim, int) else 0.125

    fused_id = g.add_node(
        op=FusedAttentionOp(num_heads=num_heads, head_dim=head_dim, scale=scale),
        inputs=[q_id, kt_id, v_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, fused_id)

    # Remove consumed nodes.
    _remove_chain(g, match.root_node_id, keep={q_id, kt_id, v_id})

    return g


def _remove_chain(g: Graph, node_id: str, keep: set[str]) -> None:
    if node_id not in g.nodes or node_id in keep:
        return
    node = g.nodes[node_id]
    inputs = list(node.inputs)
    if not g.consumers(node_id):
        g.remove_node(node_id)
        for inp_id in inputs:
            _remove_chain(g, inp_id, keep)
