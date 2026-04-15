"""Absorb a downstream Elementwise into a KernelOp's epilogue.

Pattern: ``Elementwise{$fn}(Kernel(...))`` where the KernelOp has
fan-out=1. The Elementwise is appended to ``kernel.epilogue`` — it runs
after the kernel's core produces its output. For pointwise kernels
(core=None) this still makes sense: epilogue is strictly downstream of
prologue.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import ElementwiseOp, KernelOp, Port
from deplodock.compiler.rules.fusion._assembly_helpers import (
    copy_node,
    fan_out_of,
    kernel_last_node_id,
    rewire_node_input,
    rewrite_port_references,
    shape_of,
)

PATTERN = "_"  # wildcard; filter to Elementwise(Kernel) in rewrite


def rewrite(graph: Graph, match: Match) -> Graph:
    em_id = match.root_node_id
    em_node = graph.nodes.get(em_id)
    if em_node is None or not isinstance(em_node.op, ElementwiseOp):
        return graph
    # The first input must be a KernelOp with fan-out=1.
    if not em_node.inputs:
        return graph
    kid = em_node.inputs[0]
    knode = graph.nodes.get(kid)
    if knode is None or not isinstance(knode.op, KernelOp):
        return graph
    if fan_out_of(graph, kid) != 1:
        return graph

    kernel = knode.op
    snap = copy_node(em_node)
    # The absorbed node's first input pointed at the parent kernel's outer id.
    # Rewire to the kernel's last-internal node id so var_map[id] resolves.
    last = kernel_last_node_id(kernel)
    if last is not None:
        snap = rewire_node_input(snap, kid, last)

    # Append snap to epilogue — it's strictly downstream of the core.
    new_core = kernel.core
    new_kernel_prologue = kernel.prologue
    new_kernel_epilogue = kernel.epilogue + (snap,)

    # Any Elementwise inputs besides inputs[0] are new external inputs.
    new_inputs_list: list[Port] = list(kernel.inputs)
    new_external = dict(kernel.external_shapes)
    for extra in em_node.inputs[1:]:
        if not any(p.buffer_id == extra for p in new_inputs_list):
            new_inputs_list.append(Port(buffer_id=extra))
            new_external[extra] = shape_of(graph, extra)

    new_kernel = KernelOp(
        inputs=new_inputs_list,
        outputs=[Port(buffer_id=em_id)],
        prologue=new_kernel_prologue,
        core=new_core,
        epilogue=new_kernel_epilogue,
        kernel_source=kernel.kernel_source,
        external_shapes=new_external,
    )

    g = graph.copy()
    new_kid = g.add_node(
        op=new_kernel,
        inputs=[p.buffer_id for p in new_inputs_list],
        output=Tensor(
            name=f"fused_{em_id}",
            shape=tuple(em_node.output.shape),
            dtype=em_node.output.dtype,
        ),
    )
    g.nodes[new_kid].hints.merge(graph.nodes[kid].hints)
    g.nodes[new_kid].hints.merge(em_node.hints)
    g.replace_node(em_id, new_kid)
    rewrite_port_references(g, em_id, new_kid)
    if em_id in g.nodes:
        g.remove_node(em_id)
    if kid in g.nodes and not g.consumers(kid):
        g.remove_node(kid)
    return g
