"""Wrap a standalone fusible op (Elementwise / IndexMap) into a pointwise KernelOp.

Runs after seed-contraction/reduce/softmax so those patterns claim their
ops first. Any remaining ElementwiseOp or IndexMapOp not inside a
KernelOp becomes a singleton pointwise KernelOp(prologue=(node,),
core=None). Absorption rules (050/060) then grow it.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ElementwiseOp, IndexMapOp, KernelOp, Port
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    rewrite_port_references,
    shape_of,
)

PATTERN = "_"  # wildcard; filter to Elementwise in rewrite


def rewrite(graph: Graph, match: Match) -> Graph:
    nid = match.root_node_id
    node = graph.nodes.get(nid)
    if node is None or not isinstance(node.op, (ElementwiseOp, IndexMapOp)):
        return graph
    # Identity IndexMaps are buffer aliases (no codegen) — never wrap.
    if isinstance(node.op, IndexMapOp):
        if node.inputs and node.inputs[0] in graph.nodes:
            inp_shape = tuple(graph.nodes[node.inputs[0]].output.shape)
            if node.op.is_identity(inp_shape):
                return graph
    # Don't re-seed nodes already held by a KernelOp — but KernelOps don't
    # reference external nodes by their Elementwise identity; if the node
    # is still in the outer graph, it's standalone.

    external_shapes = {inp: shape_of(graph, inp) for inp in node.inputs}
    snap = copy_node(node)
    kernel = KernelOp(
        inputs=[Port(buffer_id=inp) for inp in node.inputs],
        outputs=[Port(buffer_id=nid)],
        prologue=(snap,),
        core=None,
        epilogue=(),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    new_id = g.add_node(
        op=kernel,
        inputs=list(node.inputs),
        output=Tensor(
            name=f"fused_{nid}",
            shape=tuple(node.output.shape),
            dtype=node.output.dtype,
        ),
    )
    g.nodes[new_id].hints.merge(node.hints)
    g.replace_node(nid, new_id)
    rewrite_port_references(g, nid, new_id)
    if nid in g.nodes:
        g.remove_node(nid)
    return g
