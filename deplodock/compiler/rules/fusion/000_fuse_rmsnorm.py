"""Fuse RMSNorm chain into FusedRMSNormOp.

Matches: (x * rsqrt(sum(x*x) + eps)) * weight
The $x fan-out (same node 3 times) ensures only RMSNorm matches.

Must run BEFORE the matmul rule (001) to claim the sum(mul(x,x)) sub-pattern.
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedRMSNormOp

# Match both orderings of the outer multiply (weight * norm or norm * weight).
PATTERN = (
    "Elementwise{mul}("
    "  $w,"
    "  Elementwise{mul}("
    "    $x,"
    "    Elementwise{rsqrt}("
    "      Elementwise{add}("
    "        Reduce{sum, $ax}(Elementwise{mul}($x, $x)),"
    "        $eps"
    "      )"
    "    )"
    "  )"
    ")"
    " | "
    "Elementwise{mul}("
    "  Elementwise{mul}("
    "    $x,"
    "    Elementwise{rsqrt}("
    "      Elementwise{add}("
    "        Reduce{sum, $ax}(Elementwise{mul}($x, $x)),"
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
    _remove_chain(g, match.root_node_id, keep={x_id, w_id, match.bindings.get("eps", "")})

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
