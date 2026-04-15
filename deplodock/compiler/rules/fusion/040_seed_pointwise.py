"""Wrap a standalone Elementwise node into a pointwise KernelOp.

Runs after seed-contraction/reduce/softmax so those patterns claim their
ops first. Any remaining Elementwise node not inside a KernelOp becomes
a singleton pointwise KernelOp(prologue=(node,), core=None). Absorption
rules (050/060) then grow it.

Not yet wired into DEFAULT_PASS_ORDER.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ElementwiseOp, KernelOp, Port
from deplodock.compiler.rules.fusion._assembly_helpers import copy_node, shape_of

PATTERN = "_"  # wildcard; filter to Elementwise in rewrite


def rewrite(graph: Graph, match: Match) -> Graph:
    nid = match.root_node_id
    node = graph.nodes.get(nid)
    if node is None or not isinstance(node.op, ElementwiseOp):
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
    g.replace_node(nid, new_id)
    if nid in g.nodes:
        g.remove_node(nid)
    return g
