"""Merge two adjacent ``LoopOp``s via graph splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses
them by handing a two-node subgraph to ``splice_graph``. The splicer
handles multiple consumer Loads of the producer and shared external
inputs uniformly (first-seen slot assignment + splice-edge routing).
Splicing refuses patterns it doesn't handle yet (non-trivial σ writer
forms, etc.); those boundaries stay as separate kernels.

Blowup guard: a leaf's cost is ``(free_prod × enclosing_reduce_prod)``.
Fusions that catastrophically nest reduces (e.g. inlining one matmul
inside another's k-loop: outer_K × inner_K × free_numel) multiply this
leaf cost by the inner reduce axis's extent. We refuse merges whose max
leaf cost grows more than ``_BLOWUP_FACTOR`` over the sum of the
producer+consumer leaf costs.

Factor picked empirically — swept 2…1024 on TinyLlama block (seq=32):
2–16 ties at ~4.18ms/18 launches (best), 32–512 shifts to ~4.7ms/17
launches (one harmful silu→down_proj fusion lands), 1024 unlocks the
up_proj→down_proj nesting (~1000×) and the block takes 433ms. 8 is the
middle of the best plateau.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import Loop, LoopOp, Stmt, splice_graph
from deplodock.compiler.pipeline.engine import Match, Pattern

_BLOWUP_FACTOR = 8


def _max_nest(loop_op: LoopOp) -> int:
    """Maximum per-leaf compute cost (``free_prod × reduce_prod``) over the body tree."""
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
