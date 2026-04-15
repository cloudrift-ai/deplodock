"""Seed a KernelOp with a single-stage ReduceCore from ``Reduce{$fn, $axis}($x)``.

Fires after 020_seed_contraction so matmul sums are claimed first. The
reduce Node moves into a ReduceStage; no upstream elementwise absorption
yet (that's handled by 060_absorb_prologue).

Not yet wired into DEFAULT_PASS_ORDER.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import KernelOp, Port, ReduceOp, ReduceStage
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    rewrite_port_references,
    shape_of,
)

PATTERN = "Reduce{$fn, $axis}($x)"


def rewrite(graph: Graph, match: Match) -> Graph:
    reduce_id = match.root_node_id
    x_id = match.bindings.get("x")
    if x_id is None:
        return graph
    reduce_node = graph.nodes.get(reduce_id)
    if reduce_node is None or not isinstance(reduce_node.op, ReduceOp):
        return graph

    x_shape = shape_of(graph, x_id)
    external_shapes = {x_id: x_shape}

    red_snap = copy_node(reduce_node)
    stage = ReduceStage(pre_ops=(), reduce=red_snap)
    kernel = KernelOp(
        inputs=[Port(buffer_id=x_id)],
        outputs=[Port(buffer_id=reduce_id)],
        # Flat-prologue convention: keep reduce in prologue too; core is
        # an annotation pointing at it.
        prologue=(red_snap,),
        core=(stage,),
        epilogue=(),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    new_id = g.add_node(
        op=kernel,
        inputs=[x_id],
        output=Tensor(
            name=f"fused_{reduce_id}",
            shape=tuple(reduce_node.output.shape),
            dtype=reduce_node.output.dtype,
        ),
    )
    g.nodes[new_id].hints.merge(reduce_node.hints)
    g.replace_node(reduce_id, new_id)
    rewrite_port_references(g, reduce_id, new_id)
    if reduce_id in g.nodes:
        g.remove_node(reduce_id)
    return g
