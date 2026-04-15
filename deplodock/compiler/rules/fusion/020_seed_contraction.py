"""Seed a KernelOp with ContractionCore from ``Reduce{sum}(Elementwise{mul}($a, $b))``.

Recognizes the matmul shape directly on the raw primitive graph. Produces
a single KernelOp node with ContractionCore. The mul and sum Nodes move
out of the outer graph into ContractionCore.mul / .reduce (structural
ownership).

Validates ≥2D matching-K shapes before firing.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    KernelOp,
    Port,
    ReduceOp,
)
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    fan_out_of,
    rewrite_port_references,
    shape_of,
    shapes_match_for_contraction,
)

PATTERN = "Reduce{sum,$k_axis}(Elementwise{mul}($a, $b))"


def rewrite(graph: Graph, match: Match) -> Graph:
    reduce_id = match.root_node_id
    mul_id = None
    # Find the mul node via the reduce's inputs.
    reduce_node = graph.nodes.get(reduce_id)
    if reduce_node is None or not isinstance(reduce_node.op, ReduceOp) or reduce_node.op.fn != "sum":
        return graph
    if len(reduce_node.inputs) != 1:
        return graph
    mul_id = reduce_node.inputs[0]
    mul_node = graph.nodes.get(mul_id)
    if mul_node is None or not isinstance(mul_node.op, ElementwiseOp) or mul_node.op.fn != "mul":
        return graph
    if len(mul_node.inputs) != 2:
        return graph
    # The mul must not fan out to other consumers (would duplicate work).
    if fan_out_of(graph, mul_id) != 1:
        return graph

    a_id, b_id = mul_node.inputs[0], mul_node.inputs[1]
    a_shape = shape_of(graph, a_id)
    b_shape = shape_of(graph, b_id)
    ok, _m, _n, _k, _bd, _bs, _abg, _bbg = shapes_match_for_contraction(a_shape, b_shape)
    if not ok:
        return graph

    # Build external-shapes map.
    external_shapes = {a_id: a_shape, b_id: b_shape}

    # Snapshot mul and reduce as detached Nodes (owned by ContractionCore).
    mul_snap = copy_node(mul_node)
    reduce_snap = copy_node(reduce_node)

    core = ContractionCore(
        a=Port(buffer_id=a_id),
        b=Port(buffer_id=b_id),
        k_axis=len(a_shape) - 1,
        mul=mul_snap,
        reduce=reduce_snap,
    )
    kernel = KernelOp(
        inputs=[Port(buffer_id=a_id), Port(buffer_id=b_id)],
        outputs=[Port(buffer_id=reduce_id)],
        # core owns the mul + reduce Nodes exclusively; prologue holds
        # only pre-contraction elementwise ops (none at seed time).
        prologue=(),
        core=core,
        epilogue=(),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    new_id = g.add_node(
        op=kernel,
        inputs=[a_id, b_id],
        output=Tensor(
            name=f"fused_{reduce_id}",
            shape=tuple(reduce_node.output.shape),
            dtype=reduce_node.output.dtype,
        ),
    )
    g.nodes[new_id].hints.merge(reduce_node.hints)
    g.nodes[new_id].hints.merge(mul_node.hints)
    g.replace_node(reduce_id, new_id)
    rewrite_port_references(g, reduce_id, new_id)
    # Remove the original reduce and mul (they're now owned by the core).
    if reduce_id in g.nodes:
        g.remove_node(reduce_id)
    if mul_id in g.nodes and not g.consumers(mul_id):
        g.remove_node(mul_id)
    return g
