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
from deplodock.compiler.ir.loop import Accum, Assign, Loop, LoopOp, Stmt, iter_body, splice_graph
from deplodock.compiler.pipeline.engine import Match, Pattern

_BLOWUP_FACTOR = 8


def _max_nest(loop_op: LoopOp) -> int:
    """Maximum per-leaf compute cost (``free_prod × reduce_prod``) over the body tree.

    Both ``free_prod`` and ``reduce_prod`` accumulate from the *actually*
    enclosing Loops at each leaf — sequential (sibling) free axes on
    either side of a reduce do not pile onto its leaf cost.
    """
    reduce_names = loop_op.reduce_axis_names
    best = 1

    def walk(stmts: tuple[Stmt, ...], free_prod: int, reduce_prod: int) -> None:
        nonlocal best
        for s in stmts:
            if isinstance(s, Loop):
                extent = int(s.axis.extent)
                if s.axis.name in reduce_names:
                    walk(s.body, free_prod, reduce_prod * extent)
                else:
                    walk(s.body, free_prod * extent, reduce_prod)
            else:
                cost = free_prod * reduce_prod
                if cost > best:
                    best = cost

    walk(loop_op.body, 1, 1)
    return best


def _is_pure_indexmap(loop_op: LoopOp) -> bool:
    """Body contains only Loops / Loads / Writes — no compute (Assign) or Accum.

    Such a kernel is an ``IndexMapOp`` lifted into Loop IR: broadcast,
    transpose, reshape, slice. Its content is pure coord rewriting +
    copying. Fusing a non-indexmap producer (one with real compute)
    *into* such a consumer forces the producer's body to land inside
    the indexmap's iteration space — materializing any broadcast the
    indexmap was expressing lazily.
    """
    for s in iter_body(loop_op.body):
        if isinstance(s, (Assign, Accum)):
            return False
    return True


def _output_numel(loop_op: LoopOp) -> int:
    reduce_names = loop_op.reduce_axis_names
    n = 1
    for a in loop_op.axes:
        if a.name not in reduce_names:
            n *= int(a.extent)
    return n


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

    # Broadcast-materialization guard: fusing a compute-bearing producer into
    # a pure-indexmap consumer whose output volume exceeds the producer's
    # replicates the producer's body across the extra axes (the indexmap's
    # broadcast stops being lazy). Skip — the indexmap can still fuse the
    # *other* way, into its downstream consumer.
    if (
        _is_pure_indexmap(consumer_node.op)
        and not _is_pure_indexmap(producer_node.op)
        and _output_numel(consumer_node.op) > _output_numel(producer_node.op)
    ):
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
