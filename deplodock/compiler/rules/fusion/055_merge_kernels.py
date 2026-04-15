"""Merge two adjacent KernelOps when shape/reduce-axis compat allows.

Pattern: ``Kernel_b(..., Kernel_a, ...)`` where ``Kernel_a`` has fan-out=1
(consumed only by ``Kernel_b``). Produces a single merged KernelOp that
flattens ``Kernel_a``'s body into ``Kernel_b``'s structure.

Allowed pairs (``a → b``):
  - pointwise + pointwise → pointwise
  - pointwise + reduce    → reduce (a's prologue prepended to b's first stage)
  - reduce + pointwise    → reduce (b's body appended to a's epilogue)
  - reduce + reduce       → reduce (only when all reduces are row-compatible)
  - contraction + pointwise → contraction (b's body appended to a's epilogue)

Rejected:
  - pointwise + contraction (defer to 080_absorb_a_chain when active)
  - contraction + contraction
  - reduce + contraction
  - contraction + reduce

Solves the RMSNorm regression: after 040 wraps every Elementwise as a
singleton kernel, this rule re-merges adjacent kernels into one.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import Match
from deplodock.compiler.ops import (
    ContractionCore,
    KernelOp,
    Port,
    ReduceStage,
)
from deplodock.compiler.rules.fusion._assembly_helpers import (
    fan_out_of,
    flatten_kernel_nodes,
    kernel_kind,
    kernel_last_node_id,
    kernel_reduces_with_input_shapes,
    merged_external_inputs_compat,
    reduces_compatible,
    rewire_node_input,
    rewrite_port_references,
)

PATTERN = "_"  # wildcard; filter to KernelOp consuming KernelOp in rewrite


def rewrite(graph: Graph, match: Match) -> Graph:
    b_id = match.root_node_id
    b_node = graph.nodes.get(b_id)
    if b_node is None or not isinstance(b_node.op, KernelOp):
        return graph
    b_kernel = b_node.op

    # Find an input that is itself a KernelOp with fan-out=1.
    for port_idx, port in enumerate(b_kernel.inputs):
        a_id = port.buffer_id
        a_node = graph.nodes.get(a_id)
        if a_node is None or not isinstance(a_node.op, KernelOp):
            continue
        if fan_out_of(graph, a_id) != 1:
            continue
        a_kernel = a_node.op

        merged = _try_merge(a_id, a_kernel, b_id, b_kernel, port_idx, graph)
        if merged is None:
            continue

        new_kernel, new_external_inputs = merged
        g = graph.copy()
        new_id = g.add_node(
            op=new_kernel,
            inputs=new_external_inputs,
            output=Tensor(
                name=f"merged_{b_id}",
                shape=tuple(b_node.output.shape),
                dtype=b_node.output.dtype,
            ),
        )
        g.nodes[new_id].hints.merge(graph.nodes[a_id].hints)
        g.nodes[new_id].hints.merge(graph.nodes[b_id].hints)
        g.replace_node(b_id, new_id)
        rewrite_port_references(g, b_id, new_id)
        if b_id in g.nodes:
            g.remove_node(b_id)
        if a_id in g.nodes and not g.consumers(a_id):
            g.remove_node(a_id)
        return g
    return graph


def _try_merge(
    a_id: str,
    a: KernelOp,
    b_id: str,
    b: KernelOp,
    port_idx: int,
    graph: Graph,
) -> tuple[KernelOp, list[str]] | None:
    """Attempt to build the merged KernelOp. Return None if not allowed."""
    a_kind = kernel_kind(a)
    b_kind = kernel_kind(b)

    # Reject contraction-conflicting cases up front.
    if b_kind == "contraction":
        # pointwise+contraction belongs to 080_absorb_a_chain; reject.
        return None
    # Allow contraction + reduce only when the reduces are compatible row
    # reductions (e.g. matmul → softmax). reduces_compatible is checked
    # below; if it fails, the merge is rejected anyway.

    # Reduce-axis compat across non-contraction reduces in the merged kernel.
    # Contraction reduces are handled by the K-loop — they don't share a
    # row-reduction axis with row reduces and shouldn't be checked.
    def _non_contraction_reduces(k):
        if isinstance(k.core, ContractionCore):
            return []
        return kernel_reduces_with_input_shapes(k)

    nc = _non_contraction_reduces(a) + _non_contraction_reduces(b)
    if len(nc) > 1:
        base_r, base_s = nc[0]
        for r, s in nc[1:]:
            if not reduces_compatible(base_r, base_s, r, s):
                return None

    # Compose external inputs (b's port[port_idx] becomes internal).
    new_inputs: list[Port] = []
    new_external_shapes: dict = {}
    seen: set[str] = set()

    def add_port(p: Port, shape: tuple) -> None:
        if p.buffer_id in seen:
            return
        seen.add(p.buffer_id)
        new_inputs.append(Port(buffer_id=p.buffer_id, indexmap=p.indexmap))
        new_external_shapes[p.buffer_id] = shape

    for p in a.inputs:
        add_port(p, a.external_shapes.get(p.buffer_id, ()))
    for i, p in enumerate(b.inputs):
        if i == port_idx:
            continue  # this was a's output — now internal
        add_port(p, b.external_shapes.get(p.buffer_id, ()))

    # Shape compat check. Inputs consumed ONLY by IndexMaps inside the
    # merged body are exempt — IndexMaps read via their own coord_map, so
    # they don't share the kernel's broadcast-row indexing.
    from deplodock.compiler.ops import IndexMapOp

    a_body = flatten_kernel_nodes(a)
    b_body = flatten_kernel_nodes(b)
    merged_body = a_body + b_body
    skip_shapes: set[tuple] = set()
    for buf_id, shape in new_external_shapes.items():
        consumers = [n for n in merged_body if buf_id in n.inputs]
        if consumers and all(isinstance(n.op, IndexMapOp) for n in consumers):
            skip_shapes.add(shape)
    is_pure_contraction = a_kind == "contraction" or b_kind == "contraction"
    if not merged_external_inputs_compat(
        list(new_external_shapes.values()),
        is_pure_contraction=is_pure_contraction,
        skip_shapes=skip_shapes,
    ):
        return None

    # Rewire b's body: any input id pointing at a's outer-graph id must now
    # point at a's last-internal node id (the actual producer Node).
    a_last = kernel_last_node_id(a)
    if a_last is None:
        return None

    def rewire(node):
        return rewire_node_input(node, a_id, a_last)

    # Flat-prologue convention (mirrors legacy auto_fuse): merged kernel
    # holds ALL body nodes in a single prologue tuple, in topo order. The
    # ``core`` field is an annotation that references nodes already in
    # prologue — backend codegen reads them via the dedup'd ``region_ops``
    # compat property regardless of which slot they live in.
    a_flat = flatten_kernel_nodes(a)
    b_flat = tuple(rewire(n) for n in flatten_kernel_nodes(b))
    new_prologue = a_flat + b_flat
    new_epilogue: tuple = ()

    # Compose the core annotation. Reuse a's ContractionCore (rewiring is
    # done elsewhere). For tuple cores, concatenate stages with rewiring
    # applied to b's stages so internal Node references resolve correctly.
    a_core = a.core
    b_core = b.core
    if isinstance(b_core, tuple):
        b_core_rewired = tuple(
            ReduceStage(
                pre_ops=tuple(rewire(n) for n in stage.pre_ops),
                reduce=rewire(stage.reduce) if stage.reduce is not None else None,
            )
            for stage in b_core
            if isinstance(stage, ReduceStage)
        )
    elif isinstance(b_core, ContractionCore):
        return None  # already rejected
    else:
        b_core_rewired = ()

    new_core: object = None
    if isinstance(a_core, ContractionCore):
        new_core = a_core
    elif isinstance(a_core, tuple) and a_core:
        new_core = a_core + b_core_rewired if b_core_rewired else a_core
    elif b_core_rewired:
        new_core = b_core_rewired

    new_kernel = KernelOp(
        inputs=new_inputs,
        outputs=list(b.outputs),
        prologue=new_prologue,
        core=new_core,
        epilogue=new_epilogue,
        kernel_source="",
        external_shapes=new_external_shapes,
    )
    return new_kernel, [p.buffer_id for p in new_inputs]
