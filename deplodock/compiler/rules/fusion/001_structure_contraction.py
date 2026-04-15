"""Recognize the matmul kernel shape inside a flat-prologue KernelOp.

Matches a KernelOp whose prologue contains ``Reduce{sum}(Elementwise{mul}(a, b))``
with two ≥2D inputs sharing a K dimension. Rewrites the KernelOp so that
``core=ContractionCore(a_port, b_port, k_axis)``, with the mul and sum
removed from ``prologue`` (they're implicit in the ContractionCore).

This rule operates on KernelOps produced by the greedy ``auto_fuse`` pass:
auto_fuse bundles adjacent fusible ops into a flat-prologue KernelOp; this
rule then classifies the contents structurally. Future rules (softmax,
reduce, prologue/epilogue absorption) layer on top.

Pattern: ``_`` (wildcard — filter to KernelOp in rewrite since KernelOps
have variable input counts that can't be expressed in the pattern language).
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    KernelOp,
    Port,
    ReduceOp,
)

PATTERN = "_"


def rewrite(graph: Graph, match: Match) -> Graph:
    node = graph.nodes.get(match.root_node_id)
    if node is None or not isinstance(node.op, KernelOp):
        return graph

    kernel = node.op
    # Already structured — leave it alone.
    if kernel.core is not None:
        return graph

    # Look for the contraction shape in the flat prologue.
    contraction = _detect_contraction(kernel)
    if contraction is None:
        return graph

    a_port, b_port, k_axis, mul_id, sum_id, epilogue_nodes = contraction

    # Build the new KernelOp with ContractionCore. Drop mul and sum from the
    # prologue; any remaining ops (elementwise preceding the mul) stay; ops
    # after the sum move to epilogue.
    new_prologue = tuple(n for n in kernel.prologue if n.id != mul_id and n.id != sum_id and n.id not in {e.id for e in epilogue_nodes})
    new_epilogue = tuple(epilogue_nodes)

    new_kernel = KernelOp(
        inputs=list(kernel.inputs),
        outputs=list(kernel.outputs),
        prologue=new_prologue,
        core=ContractionCore(a=a_port, b=b_port, k_axis=k_axis),
        epilogue=new_epilogue,
        kernel_source=kernel.kernel_source,
        external_shapes=dict(kernel.external_shapes),
    )

    g = graph.copy()
    new_id = g.add_node(
        op=new_kernel,
        inputs=list(node.inputs),
        output=g.nodes[match.root_node_id].output,
    )
    g.replace_node(match.root_node_id, new_id)
    g.remove_node(match.root_node_id)
    return g


def _detect_contraction(kernel: KernelOp):
    """Find ``sum(mul(a, b))`` in the flat prologue.

    Returns ``(a_port, b_port, k_axis, mul_id, sum_id, epilogue_nodes)`` on
    match, or None.
    """
    # Find the first Reduce{sum} in the prologue and the Elementwise{mul}
    # that feeds it.
    sum_node = None
    mul_node = None
    for n in kernel.prologue:
        if isinstance(n.op, ReduceOp) and n.op.fn == "sum":
            sum_node = n
            break
    if sum_node is None:
        return None
    if len(sum_node.inputs) != 1:
        return None
    mul_candidate_id = sum_node.inputs[0]
    for n in kernel.prologue:
        if n.id == mul_candidate_id and isinstance(n.op, ElementwiseOp) and n.op.fn == "mul" and len(n.inputs) == 2:
            mul_node = n
            break
    if mul_node is None:
        return None

    a_id, b_id = mul_node.inputs[0], mul_node.inputs[1]
    # Both inputs must be external Ports (not produced inside the prologue).
    external_ids = {p.buffer_id for p in kernel.inputs}
    if a_id not in external_ids or b_id not in external_ids:
        return None

    a_shape = kernel.external_shapes.get(a_id, ())
    b_shape = kernel.external_shapes.get(b_id, ())
    if len(a_shape) < 2 or len(b_shape) < 2:
        return None

    # A's last dim == B's matching dim (2D: K = a[-1] = b[0]; batched: a[-1] = b[-2]).
    if len(a_shape) >= 2 and len(b_shape) >= 2:
        a_k = a_shape[-1]
        b_k = b_shape[-2] if len(b_shape) > 2 else b_shape[0]
        if a_k != b_k:
            return None
        k_axis = len(a_shape) - 1
    else:
        return None

    # Collect epilogue: ops in prologue that come after sum_node and transitively
    # depend on it.
    mul_sum_ids = {mul_node.id, sum_node.id}
    epilogue: list = []
    produced_in_epilogue: set = {sum_node.id}
    seen_sum = False
    for n in kernel.prologue:
        if n.id == sum_node.id:
            seen_sum = True
            continue
        if not seen_sum:
            continue
        # Only include elementwise ops in epilogue; bail out for non-elementwise
        # (shouldn't happen in flat auto_fuse output for contraction regions).
        if not isinstance(n.op, ElementwiseOp):
            return None
        if any(inp in produced_in_epilogue or inp in mul_sum_ids for inp in n.inputs):
            epilogue.append(n)
            produced_in_epilogue.add(n.id)

    # Port objects for a and b — preserve existing Port identity (with IndexMaps).
    a_port = next(p for p in kernel.inputs if p.buffer_id == a_id)
    b_port = next(p for p in kernel.inputs if p.buffer_id == b_id)
    # Clone Port so we don't alias the kernel.inputs list entry.
    a_port = Port(buffer_id=a_port.buffer_id, indexmap=a_port.indexmap)
    b_port = Port(buffer_id=b_port.buffer_id, indexmap=b_port.indexmap)

    return a_port, b_port, k_axis, mul_node.id, sum_node.id, epilogue
