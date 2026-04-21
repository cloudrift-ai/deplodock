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
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import LoopOp, splice_graph
from deplodock.compiler.matcher import ChainMatch, Production

GRAMMAR = [
    Production("producer", LoopOp, "1"),
    Production("consumer", LoopOp, "1"),
]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    producer_ids = match.get("producer")
    consumer_ids = match.get("consumer")
    if not producer_ids or not consumer_ids:
        return None

    producer_id = producer_ids[0]
    consumer_id = consumer_ids[0]

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
