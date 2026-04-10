"""Fuse SiLU+mul: silu(gate) * up → FusedSiLUMulOp."""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import FusedSiLUMulOp

# silu(gate) * up = gate * recip(1 + exp(-gate)) * up
# The $gate fan-out (appears twice) makes this specific to SiLU.
PATTERN = (
    "Elementwise{mul}("
    "  Elementwise{mul}("
    "    $gate,"
    "    Elementwise{recip}("
    "      Elementwise{add}("
    "        $one,"
    "        Elementwise{exp}(Elementwise{neg}($gate))"
    "      )"
    "    )"
    "  ),"
    "  $up"
    ")"
)


def rewrite(graph: Graph, match: Match) -> Graph:
    """Replace SiLU+mul chain with FusedSiLUMulOp(gate, up)."""
    g = graph.copy()
    root = g.nodes[match.root_node_id]

    gate_id = match.bindings["gate"]
    up_id = match.bindings["up"]

    fused_id = g.add_node(
        op=FusedSiLUMulOp(),
        inputs=[gate_id, up_id],
        output=Tensor(
            name=root.output.name,
            shape=root.output.shape,
            dtype=root.output.dtype,
        ),
    )

    g.replace_node(match.root_node_id, fused_id)

    # Remove consumed nodes.
    _remove_chain(g, match.root_node_id, keep={gate_id, up_id, match.bindings.get("one", "")})

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
