"""Merge two adjacent ``LoopOp``s via graph splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses
them by handing a two-node subgraph to ``splice_graph``. The splicer
handles multiple consumer Loads of the producer and shared external
inputs uniformly (first-seen slot assignment + splice-edge routing).
Splicing refuses patterns it doesn't handle yet (non-trivial σ writer
forms, etc.); those boundaries stay as separate kernels.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import Loop, LoopOp, Stmt, splice_graph
from deplodock.compiler.pipeline.graph import Graph, Tensor
from deplodock.compiler.pipeline.matcher import Match, Pattern

_BLOWUP_FACTOR = 8


def _max_nest(loop_op: LoopOp) -> int:
    """Maximum per-leaf compute cost the emitter will produce.

    The CUDA emitter flattens *all* free axes into the thread id and then
    runs the body per thread — so a reduce Loop sitting above a sibling
    free axis is re-executed once per value of that free axis. The cost
    of a leaf is therefore ``(product of all free axes) * (product of
    reduce axes enclosing the leaf)``. We return the max over leaves.
    """
    reduce_names = loop_op.reduce_axis_names
    free_prod = 1
    for a in loop_op.axes:
        if a.name not in reduce_names:
            free_prod *= int(a.extent)

    best = free_prod

    def walk(stmts: tuple[Stmt, ...], reduce_prod: int) -> None:
        nonlocal best
        for s in stmts:
            if isinstance(s, Loop):
                extent = int(s.axis.extent)
                new_reduce = reduce_prod * extent if s.axis.name in reduce_names else reduce_prod
                walk(s.body, new_reduce)
            else:
                cost = free_prod * reduce_prod
                if cost > best:
                    best = cost

    walk(loop_op.body, 1)
    return best


PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    producer_id = match.nodes["producer"]
    consumer_id = match.nodes["consumer"]

    producer_node = graph.nodes[producer_id]
    consumer_node = graph.nodes[consumer_id]
    if not isinstance(producer_node.op, LoopOp) or not isinstance(consumer_node.op, LoopOp):
        return None
    if producer_id not in consumer_node.inputs:
        return None

    # Build a subgraph: producer, consumer, and their non-producer external
    # inputs as InputOp nodes. ``splice_graph`` classifies each Load via the
    # graph edges (LoopOp→LoopOp is a splice edge; LoopOp→InputOp is external)
    # and assigns merged slots in first-seen order.
    sub = Graph()
    external_ids: list[str] = []
    for ext_id in list(producer_node.inputs) + list(consumer_node.inputs):
        if ext_id == producer_id or ext_id in sub.nodes:
            continue
        external_ids.append(ext_id)
        ext = graph.nodes.get(ext_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        sub.add_node(InputOp(), [], Tensor(ext_id, shape, dtype), node_id=ext_id)
    sub.add_node(producer_node.op, list(producer_node.inputs), producer_node.output, node_id=producer_id)
    sub.add_node(consumer_node.op, list(consumer_node.inputs), consumer_node.output, node_id=consumer_id)
    sub.outputs = [consumer_id]

    result = splice_graph(sub)
    if result is None:
        return None
    merged, merged_inputs = result

    # Compute-blowup guard: refuse merges that catastrophically duplicate work.
    # The CUDA emitter flattens every free axis across every leaf, so a reduce
    # Loop whose free context grows post-fusion re-runs once per new free
    # iteration. Small growth is expected (softmax fuses three 1D-reduce
    # passes with a 2D div → 4× for the reduce, still a win vs materialized
    # intermediates). Large growth (e.g. up_proj + down_proj → ~1000× because
    # the 2048-wide free axis stacks on the 5632-wide reduce) is never a win.
    pre = _max_nest(producer_node.op) + _max_nest(consumer_node.op)
    post = _max_nest(merged)
    if post > _BLOWUP_FACTOR * pre:
        return None

    # Wrap the merged LoopOp in the rule's output fragment, with an InputOp
    # per external slot in the order ``splice_graph`` assigned.
    frag = Graph()
    for inp_id in merged_inputs:
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    out_id = frag.add_node(
        merged,
        merged_inputs,
        Tensor(
            f"merged_{consumer_id}",
            consumer_node.output.shape,
            consumer_node.output.dtype,
        ),
    )
    frag.outputs = [out_id]

    match.output = consumer_id
    match.consumed = {producer_id, consumer_id}
    return frag
