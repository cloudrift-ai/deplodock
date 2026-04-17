"""Merge two adjacent ``LoopOp``s connected by a single-consumer buffer.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses them
by aligning the producer's write indices with the consumer's read indices via
the σ solver in ``_merge_core``. Refuses merges that would produce a kernel
with two reduce axes (the CUDA backend is single-reduce in v1) or that would
leak a free producer axis into the consumer's iteration space.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.rules.fusion._merge_core import merge_loop_ops

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

    # The consumer must reference the producer exactly once. If multiple ports
    # read the same producer, the chain-mode match still fires because the
    # producer still has a sole consumer, but merging is ambiguous — skip.
    if consumer_node.inputs.count(producer_id) != 1:
        return None
    consumer_port = consumer_node.inputs.index(producer_id)

    merged = merge_loop_ops(
        producer_node.op,
        producer_output=0,
        consumer=consumer_node.op,
        consumer_port=consumer_port,
    )
    if merged is None:
        return None

    merged_input_ids: list[str] = []
    for i, inp_id in enumerate(consumer_node.inputs):
        if i == consumer_port:
            continue
        merged_input_ids.append(inp_id)
    merged_input_ids.extend(producer_node.inputs)

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
