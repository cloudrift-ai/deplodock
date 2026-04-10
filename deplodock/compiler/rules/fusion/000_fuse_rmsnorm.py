"""Fuse RMSNorm chain: (x * rsqrt(sum(x*x) * inv_n + eps)) * weight → FusedRMSNormOp."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedRMSNormOp

PATTERN = (
    "Elementwise{mul}("
    "  Elementwise{mul}("
    "    $x,"
    "    Elementwise{rsqrt}("
    "      Elementwise{add}("
    "        Elementwise{mul}("
    "          Reduce{sum, $ax}(Elementwise{mul}($x, $x)),"
    "          $inv_n"
    "        ),"
    "        $eps"
    "      )"
    "    )"
    "  ),"
    "  $w"
    ")"
)


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace RMSNorm chain with FusedRMSNormOp(x, weight)."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]

    x_id = match.bindings["x"]
    w_id = match.bindings["w"]

    fused_id = g.add_node(
        op=FusedRMSNormOp(eps=0.0),
        inputs=[x_id, w_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, fused_id)

    # Remove consumed nodes (walk the chain, remove if no other consumers).
    _remove_chain(g, match.root_node_id, keep={x_id, w_id, match.bindings.get("inv_n", ""), match.bindings.get("eps", "")})

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
