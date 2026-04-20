"""Merge two adjacent ``LoopOp``s via tree splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses them
by splicing the producer body at every consumer ``Load`` that reads the
producer's output (see ``_splice.py``). Splicing refuses patterns it doesn't
handle yet (multi-read, unbound free axis, non-trivial σ); those boundaries
stay as separate kernels.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.rules.fusion._splice import splice_loop_ops

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

    # Identify which consumer source(s) point to the producer. The splicer
    # operates on one source at a time — splice once per connection.
    consumer_sources = [i for i, inp in enumerate(consumer_node.inputs) if inp == producer_id]
    if not consumer_sources:
        return None

    # Initial version: splice a single connection. Multi-source pairings
    # (same producer fed into multiple distinct consumer sources) stay
    # separate until the splicer grows multi-read support.
    if len(consumer_sources) != 1:
        return None
    target_source = consumer_sources[0]

    merged = splice_loop_ops(
        producer_node.op,
        consumer_node.op,
        target_source,
        producer_inputs=list(producer_node.inputs),
        consumer_inputs=list(consumer_node.inputs),
    )
    if merged is None:
        return None

    # Build the merged node's ``inputs`` list in the same order the splicer
    # renumbered sources: producer's inputs first, then consumer's retained
    # inputs (dropping the target source).
    merged_input_ids: list[str] = list(producer_node.inputs)
    merged_input_ids.extend(inp for i, inp in enumerate(consumer_node.inputs) if i != target_source)

    frag = Graph()
    for inp_id in merged_input_ids:
        if inp_id in frag.nodes:
            continue
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(
        merged,
        merged_input_ids,
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
