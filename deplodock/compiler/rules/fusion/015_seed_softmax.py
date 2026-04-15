"""Seed a KernelOp with 2-stage ReduceCore from the softmax shape.

Pattern: ``Elementwise{div}(Elementwise{exp}(Elementwise{sub}($x, Reduce{max}($x))),
                            Reduce{sum}(Elementwise{exp}(Elementwise{sub}($x, Reduce{max}($x)))))``

The pattern uses backreferences (`$x` appears twice) so the matcher
unifies them. Emits KernelOp with core=(ReduceStage(max), ReduceStage(sub+exp, sum))
and epilogue=(div,).

"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ElementwiseOp, KernelOp, Port, ReduceOp, ReduceStage
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    rewrite_port_references,
    shape_of,
)

# Backrefs on $x require the matcher's PatternVar unification (already supported).
PATTERN = (
    "Elementwise{div}("
    "  Elementwise{exp}(Elementwise{sub}($x, Reduce{max}($x))),"
    "  Reduce{sum}(Elementwise{exp}(Elementwise{sub}($x, Reduce{max}($x))))"
    ")"
)


def rewrite(graph: Graph, match: Match) -> Graph:
    div_id = match.root_node_id
    div_node = graph.nodes.get(div_id)
    if div_node is None or not isinstance(div_node.op, ElementwiseOp) or div_node.op.fn != "div":
        return graph
    x_id = match.bindings.get("x")
    if x_id is None:
        return graph

    # Find the max and sum reduce nodes via the div's inputs.
    # inputs[0] = exp(sub(x, max)); inputs[1] = sum(exp(sub(x, max)))
    if len(div_node.inputs) != 2:
        return graph
    left_exp_id = div_node.inputs[0]
    sum_id = div_node.inputs[1]
    sum_node = graph.nodes.get(sum_id)
    if sum_node is None or not isinstance(sum_node.op, ReduceOp) or sum_node.op.fn != "sum":
        return graph
    # Walk back to find max_reduce in the subtree.
    # Both branches share max(x) by structural backref.
    left_exp_node = graph.nodes.get(left_exp_id)
    if left_exp_node is None or not isinstance(left_exp_node.op, ElementwiseOp) or left_exp_node.op.fn != "exp":
        return graph
    sub_id = left_exp_node.inputs[0]
    sub_node = graph.nodes.get(sub_id)
    if sub_node is None or not isinstance(sub_node.op, ElementwiseOp) or sub_node.op.fn != "sub":
        return graph
    max_id = sub_node.inputs[1]
    max_node = graph.nodes.get(max_id)
    if max_node is None or not isinstance(max_node.op, ReduceOp) or max_node.op.fn != "max":
        return graph

    # Find the sum branch's exp + sub — they should be the same nodes (shared)
    # or duplicates. Either way, just grab them.
    sum_exp_id = sum_node.inputs[0]
    sum_exp_node = graph.nodes.get(sum_exp_id)
    if sum_exp_node is None:
        return graph
    sum_sub_id = sum_exp_node.inputs[0] if sum_exp_node.inputs else None
    graph.nodes.get(sum_sub_id) if sum_sub_id else None

    # Build the 2-stage core.
    max_snap = copy_node(max_node)
    sub_snap = copy_node(sub_node)
    exp_snap = copy_node(left_exp_node)
    sum_snap = copy_node(sum_node)
    div_snap = copy_node(div_node)
    stages = (
        ReduceStage(pre_ops=(), reduce=max_snap),
        ReduceStage(
            pre_ops=(sub_snap, exp_snap),
            reduce=sum_snap,
        ),
    )

    external_shapes = {x_id: shape_of(graph, x_id)}
    kernel = KernelOp(
        inputs=[Port(buffer_id=x_id)],
        outputs=[Port(buffer_id=div_id)],
        # core owns max + sub + exp + sum via stages; div is the post-reduce
        # elementwise → epilogue. prologue is empty at seed time.
        prologue=(),
        core=stages,
        epilogue=(div_snap,),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    new_id = g.add_node(
        op=kernel,
        inputs=[x_id],
        output=Tensor(name=f"fused_{div_id}", shape=tuple(div_node.output.shape), dtype=div_node.output.dtype),
    )
    for src_id in (div_id, sum_id, left_exp_id, sub_id, max_id, sum_exp_id):
        if src_id and src_id in graph.nodes:
            g.nodes[new_id].hints.merge(graph.nodes[src_id].hints)
    g.replace_node(div_id, new_id)
    rewrite_port_references(g, div_id, new_id)
    # Remove all absorbed nodes (if no other consumers).
    for nid in (div_id, sum_id, left_exp_id, sub_id, max_id, sum_exp_id, sum_sub_id):
        if nid and nid in g.nodes and not g.consumers(nid):
            g.remove_node(nid)
    return g
