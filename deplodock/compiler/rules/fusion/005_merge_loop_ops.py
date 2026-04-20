"""Merge two adjacent ``LoopOp``s via tree splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses
them by splicing the producer body at every consumer ``Load`` that reads
the producer's output (see ``_splice.py``). When the consumer has
multiple input slots all pointing to the producer — softmax's two-sweep
pattern after prior merges — the extra slots are deduplicated before
splicing so the splicer sees a single source with multiple Loads.
Splicing refuses patterns it doesn't handle yet (non-trivial σ writer
forms, etc.); those boundaries stay as separate kernels.
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Load, Loop, LoopOp, Stmt
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

    consumer_sources = [i for i, inp in enumerate(consumer_node.inputs) if inp == producer_id]
    if not consumer_sources:
        return None

    # If multiple consumer input slots all point at the producer (e.g. a
    # prior merge stitched softmax's two reads into two physical slots of
    # the same buffer), deduplicate: keep the first slot and rewrite every
    # Load reading from a removed slot to target the kept slot instead.
    consumer_op = consumer_node.op
    consumer_inputs = list(consumer_node.inputs)
    if len(consumer_sources) > 1:
        consumer_op, consumer_inputs = _dedupe_consumer_inputs(consumer_op, consumer_inputs, consumer_sources)
    target_source = consumer_inputs.index(producer_id)

    merged = splice_loop_ops(
        producer_node.op,
        consumer_op,
        target_source,
        producer_inputs=list(producer_node.inputs),
        consumer_inputs=consumer_inputs,
    )
    if merged is None:
        return None

    merged_input_ids: list[str] = list(producer_node.inputs)
    merged_input_ids.extend(inp for i, inp in enumerate(consumer_inputs) if i != target_source)

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


def _dedupe_consumer_inputs(op: LoopOp, inputs: list[str], dup_slots: list[int]) -> tuple[LoopOp, list[str]]:
    """Collapse a run of input slots all pointing at the same buffer into
    the first slot, rewriting every body ``Load`` that referenced a dropped
    slot to target the kept slot instead. Other slots compact past the
    removed positions."""
    kept = dup_slots[0]
    removed = set(dup_slots[1:])

    # Old slot → new slot after compaction.
    old_to_new: dict[int, int] = {}
    new_inputs: list[str] = []
    kept_new: int | None = None
    for old_idx, inp in enumerate(inputs):
        if old_idx in removed:
            continue
        old_to_new[old_idx] = len(new_inputs)
        if old_idx == kept:
            kept_new = len(new_inputs)
        new_inputs.append(inp)
    assert kept_new is not None
    for old in removed:
        old_to_new[old] = kept_new

    def remap(stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Loop):
                out.append(Loop(axis=s.axis, body=remap(s.body)))
            elif isinstance(s, Load):
                out.append(Load(name=s.name, source=old_to_new[s.source], index=s.index))
            else:
                out.append(s)
        return tuple(out)

    return LoopOp(body=remap(op.body)), new_inputs
