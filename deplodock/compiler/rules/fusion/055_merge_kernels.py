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
    kernel_kind,
    kernel_last_node_id,
    kernel_reduces_with_input_shapes,
    merged_external_inputs_compat,
    reduces_compatible,
    rewire_node_input,
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
        g.replace_node(b_id, new_id)
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
    if a_kind == "contraction" and b_kind != "pointwise":
        return None
    if b_kind == "contraction":
        # pointwise+contraction belongs to 080_absorb_a_chain; reject.
        return None

    # Reduce-axis compat across all reduces in the merged kernel.
    a_reduces = kernel_reduces_with_input_shapes(a)
    b_reduces = kernel_reduces_with_input_shapes(b)
    all_reduces = [(r, s) for r, s in (a_reduces + b_reduces)]
    if len(all_reduces) > 1:
        # Skip the contraction reduce (matmul K-loop) — it doesn't co-exist
        # with kernel-level reduce stages anyway, but be safe.
        non_contraction = [
            (r, s)
            for r, s in all_reduces
            if not isinstance(r.axis, str)  # symbolic axes mark contraction reduces
        ]
        if len(non_contraction) > 1:
            base_r, base_s = non_contraction[0]
            for r, s in non_contraction[1:]:
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

    # Shape compat check.
    is_pure_contraction = a_kind == "contraction" or b_kind == "contraction"
    if not merged_external_inputs_compat(
        list(new_external_shapes.values()),
        is_pure_contraction=is_pure_contraction,
    ):
        return None

    # Rewire b's body: any input id pointing at a's outer-graph id must now
    # point at a's last-internal node id (the actual producer Node).
    a_last = kernel_last_node_id(a)
    if a_last is None:
        return None

    def rewire(node):
        return rewire_node_input(node, a_id, a_last)

    b_prologue = tuple(rewire(n) for n in b.prologue)
    b_core = b.core
    if isinstance(b_core, tuple):
        new_stages = []
        for stage in b_core:
            new_stages.append(
                ReduceStage(
                    pre_ops=tuple(rewire(n) for n in stage.pre_ops),
                    reduce=rewire(stage.reduce) if stage.reduce is not None else None,
                )
            )
        b_core = tuple(new_stages)
    elif isinstance(b_core, ContractionCore):
        # b is contraction → already rejected above; defensive only.
        return None
    b_epilogue = tuple(rewire(n) for n in b.epilogue)

    # Build merged structure based on kinds.
    if a_kind == "pointwise" and b_kind == "pointwise":
        new_prologue = a.prologue + b_prologue
        new_core = None
        new_epilogue = b_epilogue  # usually empty for pointwise
    elif a_kind == "pointwise" and b_kind == "reduce":
        # a's prologue prepends b's first stage's pre_ops.
        assert isinstance(b_core, tuple) and b_core
        first = b_core[0]
        merged_first = ReduceStage(
            pre_ops=a.prologue + b_prologue + first.pre_ops,
            reduce=first.reduce,
        )
        new_core = (merged_first,) + b_core[1:]
        new_prologue = ()
        new_epilogue = b_epilogue
    elif a_kind == "reduce" and b_kind == "pointwise":
        # b's body becomes a's epilogue (chained after a's epilogue).
        new_prologue = a.prologue
        new_core = a.core
        new_epilogue = a.epilogue + b_prologue + b_epilogue
    elif a_kind == "reduce" and b_kind == "reduce":
        # Concatenate stages. b's first stage's pre_ops gain b's prologue
        # and any of a's epilogue ops as a chain into the new stage. To keep
        # it simple, merge: a.core + (a.epilogue → b.prologue → b.core[0]) +
        # b.core[1:]. We model this by injecting the chain as pre_ops of
        # b.core[0] when a.epilogue + b.prologue is non-empty.
        assert isinstance(a.core, tuple) and isinstance(b_core, tuple) and b_core
        chain = a.epilogue + b_prologue
        if chain:
            first = b_core[0]
            merged_first = ReduceStage(
                pre_ops=chain + first.pre_ops,
                reduce=first.reduce,
            )
            new_core = a.core + (merged_first,) + b_core[1:]
        else:
            new_core = a.core + b_core
        new_prologue = a.prologue
        new_epilogue = b_epilogue
    elif a_kind == "contraction" and b_kind == "pointwise":
        new_prologue = a.prologue
        new_core = a.core
        new_epilogue = a.epilogue + b_prologue + b_epilogue
    else:
        return None

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
