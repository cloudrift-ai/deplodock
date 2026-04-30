"""Merge two adjacent ``LoopOp``s via graph splicing.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses
them by handing a two-node subgraph to ``splice_graph``. The splicer
handles multiple consumer Loads of the producer and shared external
inputs uniformly (first-seen slot assignment + splice-edge routing).
Splicing refuses patterns it doesn't handle yet (non-trivial σ writer
forms, etc.); those boundaries stay as separate kernels.

Blowup guards: two metrics, both summed over body leaves (max-per-leaf
wasn't enough — a fusion that introduces a second large leaf alongside
an existing one looks free to a max, but the actual runtime work
doubles).

- ``_total_work``: sum over compute leaves (``Assign`` / ``Accum``) of
  ``enclosing_free × enclosing_reduce`` — proxy for arithmetic.
- ``_total_reads``: sum over ``Load`` stmts of the same product — proxy
  for memory traffic. Global reads dominate cost on small-M matmuls
  where arithmetic is bandwidth-bound, so a fusion that grows reads
  without growing work is still a regression.

A fusion is refused if **either** metric grows by more than
``_BLOWUP_FACTOR`` over the producer+consumer sum. In addition, a
``multi-load-of-reducer`` guard refuses fusions where the consumer reads
the producer from multiple ``Load`` stmts **and** the producer contains
any reduce axis — inlining a reduce twice recomputes it, which
``_total_*`` catches *in ratio* but only when producer is big enough
relative to consumer; the guard catches it structurally.

Factor picked empirically — swept 2…1024 on TinyLlama block (seq=32):
2–16 ties at ~4.18ms/18 launches (best), 32–512 shifts to ~4.7ms/17
launches (one harmful silu→down_proj fusion lands), 1024 unlocks the
up_proj→down_proj nesting (~1000×) and the block takes 433ms. 8 is the
middle of the best plateau and still lets the epilogue-fusion cases
through.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import Accum, Assign, Load, Loop, LoopOp, splice_graph
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.pipeline.engine import Match, Pattern, RuleSkipped

_BLOWUP_FACTOR = 8


def _walk_leaf_costs(loop_op: LoopOp):
    """Yield ``(stmt, enclosing_free_prod × enclosing_reduce_prod)`` per body leaf.

    Leaf = any non-``Loop`` stmt. Products accumulate along the actually
    enclosing ``Loop`` chain; sibling free axes don't pile onto a reduce
    leaf's cost (and vice versa), matching the original max-nest semantics.
    """
    reduce_names = loop_op.reduce_axis_names

    def walk(stmts: Body, free_prod: int, reduce_prod: int):
        for s in stmts:
            if isinstance(s, Loop):
                extent = int(s.axis.extent)
                if s.axis.name in reduce_names:
                    yield from walk(s.body, free_prod, reduce_prod * extent)
                else:
                    yield from walk(s.body, free_prod * extent, reduce_prod)
            else:
                yield s, free_prod * reduce_prod

    yield from walk(loop_op.body, 1, 1)


def _total_work(loop_op: LoopOp) -> int:
    """Sum of enclosing-loop iterations over compute leaves (Assign + Accum).

    Counts how many times each arithmetic stmt executes across the full
    iteration space. A fusion that splices a producer's body in twice
    doubles this number — the old max-nest metric couldn't see that.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, (Assign, Accum))) or 1


def _total_reads(loop_op: LoopOp) -> int:
    """Sum of enclosing-loop iterations over ``Load`` stmts.

    Proxy for global-memory traffic (no cache modeling — all Loads
    count). A fusion that multiplies reads by a seq factor shows up as
    a ratio blowup here even when arithmetic stays flat.
    """
    return sum(cost for stmt, cost in _walk_leaf_costs(loop_op) if isinstance(stmt, Load)) or 1


# Producers with more than a handful of ops per output element are "reduce-heavy":
# their output at position p requires non-trivial compute (typically a reduce whose
# body depends on p). Duplicating such a producer's body (multi-load fusion) then
# re-executes the reduce per load site — what softmax-over-matmul (scaled_qk) does
# at scale. Pure-elementwise chains sit at ~1–3 ops/output; softmax's
# (max + exp) sits at ~3; a matmul sits at reduce_extent (head_dim=64). Threshold
# 4 separates the two regimes cleanly.
_REDUCE_HEAVY_WORK_PER_OUTPUT = 4


def _reduce_heavy(op: LoopOp) -> bool:
    return _total_work(op) > _REDUCE_HEAVY_WORK_PER_OUTPUT * _output_numel(op)


def _is_pure_indexmap(loop_op: LoopOp) -> bool:
    """Body contains only Loops / Loads / Writes — no compute (Assign) or Accum.

    Such a kernel is an ``IndexMapOp`` lifted into Loop IR: broadcast,
    transpose, reshape, slice. Its content is pure coord rewriting +
    copying. Fusing a non-indexmap producer (one with real compute)
    *into* such a consumer forces the producer's body to land inside
    the indexmap's iteration space — materializing any broadcast the
    indexmap was expressing lazily.
    """
    for s in loop_op.body.iter():
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


def _count_loads_from(consumer_op: LoopOp, producer_buf: str) -> int:
    """Number of ``Load`` stmts in the consumer body reading producer's output buffer."""
    return sum(1 for ld in consumer_op.body.loads if ld.input == producer_buf)


PATTERN = [
    Pattern("producer", LoopOp),
    Pattern("consumer", LoopOp),
]


def rewrite(graph: Graph, match: Match, producer: Node, consumer: Node) -> Graph | None:
    if not isinstance(producer.op, LoopOp) or not isinstance(consumer.op, LoopOp):
        raise RuleSkipped("producer or consumer is no longer a LoopOp")
    if producer.id not in consumer.inputs:
        raise RuleSkipped(f"producer {producer.id!r} is not an input of consumer {consumer.id!r}")

    # Multi-load-of-reduce-heavy-producer guard: if the consumer references
    # the producer's output via more than one Load stmt AND the producer does
    # more than a few ops per output element (i.e., has a real reduce whose
    # body can't be shared across the consumer's load positions), fusion
    # would duplicate the reduce per load site. Catches SDPA's softmax over
    # matmul — scaled_qk (head_dim reduce) feeds both row-max and exp, and
    # fusing would re-run the matmul head_dim reduce at every output element.
    # Pure-elementwise producers and "cheap" reducers like softmax's
    # (max + exp) — where the reduce collapses to a row scalar the splicer
    # can hoist — stay fuseable.
    if _reduce_heavy(producer.op) and _count_loads_from(consumer.op, producer.id) > 1:
        raise RuleSkipped("reduce-heavy producer feeds consumer through >1 Load — fusion would duplicate the reduce")

    # Build a subgraph: producer, consumer, and their non-producer external
    # inputs as InputOp nodes. ``splice_graph`` classifies each Load via the
    # graph edges (LoopOp→LoopOp is a splice edge; LoopOp→InputOp is external)
    # and assigns merged slots in first-seen order.
    sub = Graph()
    external_ids: list[str] = []
    for ext_id in list(producer.inputs) + list(consumer.inputs):
        if ext_id == producer.id or ext_id in sub.nodes:
            continue
        external_ids.append(ext_id)
        ext = graph.nodes.get(ext_id)
        shape = ext.output.shape if ext is not None else ()
        dtype = ext.output.dtype if ext is not None else "f32"
        sub.add_node(InputOp(), [], Tensor(ext_id, shape, dtype), node_id=ext_id)
    sub.add_node(producer.op, list(producer.inputs), producer.output, node_id=producer.id)
    sub.add_node(consumer.op, list(consumer.inputs), consumer.output, node_id=consumer.id)
    sub.outputs = [consumer.id]

    result = splice_graph(sub)
    if result is None:
        raise RuleSkipped("splice_graph could not align producer/consumer iteration spaces")
    merged, merged_inputs = result
    new_node_id = f"merged_{consumer.id}"
    merged = _rename_write_output(merged, old=consumer.id, new=new_node_id)

    pre_work = _total_work(producer.op) + _total_work(consumer.op)
    pre_reads = _total_reads(producer.op) + _total_reads(consumer.op)
    post_work = _total_work(merged)
    post_reads = _total_reads(merged)
    if post_work > _BLOWUP_FACTOR * pre_work:
        raise RuleSkipped(f"work blowup: post={post_work} > {_BLOWUP_FACTOR}× pre={pre_work}")
    if post_reads > _BLOWUP_FACTOR * pre_reads:
        raise RuleSkipped(f"read blowup: post={post_reads} > {_BLOWUP_FACTOR}× pre={pre_reads}")

    # Broadcast-materialization guard: fusing a compute-bearing producer into
    # a pure-indexmap consumer whose output volume exceeds the producer's
    # replicates the producer's body across the extra axes (the indexmap's
    # broadcast stops being lazy). Skip — the indexmap can still fuse the
    # *other* way, into its downstream consumer.
    if _is_pure_indexmap(consumer.op) and not _is_pure_indexmap(producer.op) and _output_numel(consumer.op) > _output_numel(producer.op):
        raise RuleSkipped(
            f"broadcast materialization: pure-indexmap consumer numel {_output_numel(consumer.op)} > "
            f"compute producer numel {_output_numel(producer.op)}"
        )

    # Wrap the merged LoopOp in the rule's output fragment. The graph node's
    # ``inputs`` list must be in the SAME order as ``merged.input_bufs`` (the
    # buf names appearing on body Loads in first-use order) so the
    # interpreter — which positionally zips ``node.inputs`` against
    # ``input_bufs`` — keys arrays by the right buf name.
    merged_inputs = list(merged.inputs)
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
            consumer.output.name,
            consumer.output.shape,
            consumer.output.dtype,
        ),
        node_id=new_node_id,
    )
    frag.outputs = [out_id]

    match.output = consumer.id
    match.consumed = {producer.id, consumer.id}
    return frag


def _rename_write_output(op: LoopOp, *, old: str, new: str) -> LoopOp:
    """Return ``op`` with every ``Write`` whose ``output == old`` rewritten
    to ``output=new`` (recursively descends into nested Loops). Used by
    fusion to align the spliced root's Writes with the new graph node id.
    """
    from deplodock.compiler.ir.loop import Write

    def fn(s):
        if isinstance(s, Write) and s.output == old:
            return Write(output=new, index=s.index, value=s.value)
        return s

    return LoopOp(body=op.body.map(fn))
