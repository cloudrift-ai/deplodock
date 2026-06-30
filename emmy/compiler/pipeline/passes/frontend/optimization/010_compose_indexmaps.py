"""Compose a single-source ``IndexMapOp`` into its single-consumer ``IndexMapOp``.

Matmul's decomposition emits chains like ``b → unsqueeze → broadcast → mul``
where the unsqueeze and broadcast are two separate ``IndexMapOp`` nodes.
Each one lifts to its own trivial ``LoopOp`` at fusion time; the unsqueeze
kernel (``b[k,n] → out[0,k,n]``) then fails to merge downstream because
the consumer iterates an outer axis the unsqueeze doesn't know about, so
the splicer can't align depths. Composing the two ``IndexMapOp`` nodes
into one coord_map before lifting sidesteps the problem entirely — the
unsqueeze disappears as an explicit kernel and its layout fold is absorbed
into the broadcast's own coord_map.

The rule fires on ``producer: IndexMapOp`` (single source, fan-out 1) feeding
``consumer: IndexMapOp``. The composed coord_map substitutes the producer's
coord_map into every placeholder reference the consumer makes to the
producer, reindexed to the producer's input.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expr import PLACEHOLDER_PREFIX, Expr
from emmy.compiler.ir.tensor.ir import IndexMapOp, IndexSource
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [
    Pattern("producer", IndexMapOp),
    Pattern("consumer", IndexMapOp),
]


def rewrite(match: Match, producer: Node, consumer: Node) -> Graph | None:
    graph = match.graph
    producer_op = producer.op
    consumer_op = consumer.op
    if not isinstance(producer_op, IndexMapOp) or not isinstance(consumer_op, IndexMapOp):
        raise RuleSkipped("producer or consumer is no longer an IndexMapOp")

    # Only compose when the producer has a single source; multi-source
    # IndexMaps (cat) can't be folded positionally.
    if len(producer_op.sources) != 1:
        raise RuleSkipped(f"producer has {len(producer_op.sources)} sources; only single-source composes")
    p_src = producer_op.sources[0]

    # The producer feeds exactly the consumer sources whose input_idx
    # points at the producer node. Any other consumer source stays as-is.
    producer_input_id = producer.inputs[p_src.input_idx]

    # Build the new consumer sources: substitute the producer's coord_map
    # into any consumer source that reads from the producer.
    new_sources: list[IndexSource] = []
    new_inputs: list[str] = list(consumer.inputs)
    for c_src in consumer_op.sources:
        consumer_src_input = consumer.inputs[c_src.input_idx]
        if consumer_src_input != producer.id:
            new_sources.append(c_src)
            continue

        composed_coord_map = _compose_coord_map(p_src.coord_map, c_src.coord_map)

        # Redirect this source to the producer's input. If that input is
        # already present in ``new_inputs`` (e.g. the consumer reads both
        # the producer and the producer's input directly), reuse the slot;
        # otherwise append.
        if producer_input_id in new_inputs:
            new_input_idx = new_inputs.index(producer_input_id)
        else:
            new_input_idx = len(new_inputs)
            new_inputs.append(producer_input_id)

        new_sources.append(
            IndexSource(
                input_idx=new_input_idx,
                coord_map=composed_coord_map,
                select=c_src.select,
            )
        )

    # Drop the now-unreferenced producer slot from new_inputs.
    referenced = {src.input_idx for src in new_sources}
    kept_positions = sorted(referenced)
    remap = {old: new for new, old in enumerate(kept_positions)}
    compact_inputs = [new_inputs[i] for i in kept_positions]
    compact_sources = tuple(IndexSource(input_idx=remap[src.input_idx], coord_map=src.coord_map, select=src.select) for src in new_sources)

    new_op = IndexMapOp(out_shape=consumer_op.out_shape, sources=compact_sources)

    frag = Graph()
    for inp_id in compact_inputs:
        if inp_id in frag.nodes:
            continue
        ext = graph.nodes.get(inp_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)

    out_id = frag.add_node(
        new_op,
        compact_inputs,
        Tensor(consumer.output.name, consumer.output.shape, consumer.output.dtype),
    )
    frag.outputs = [out_id]

    match.output = consumer.id
    match.consumed = {producer.id, consumer.id}
    return frag


def _compose_coord_map(producer_coord_map: tuple, consumer_coord_map: tuple) -> tuple:
    """Substitute the producer's coord_map through each consumer coord.

    ``consumer_coord_map[k]`` is an ``Expr`` over the consumer's output
    placeholders. When the consumer reads the producer at index
    ``consumer_coord_map``, the producer remaps that to its own input via
    ``producer_coord_map[i] = f_i(out_coord_0, out_coord_1, ...)`` where
    here ``out_coord_j`` refers to the *producer's* output dims — which
    are the values of ``consumer_coord_map[j]``.

    So the composed map reads the original input at
    ``producer_coord_map[i]{out_coord_j → consumer_coord_map[j]}``.
    """
    substitution: dict[str, Expr] = {f"{PLACEHOLDER_PREFIX}{j}": consumer_coord_map[j] for j in range(len(consumer_coord_map))}
    return tuple(e.substitute(substitution) for e in producer_coord_map)
