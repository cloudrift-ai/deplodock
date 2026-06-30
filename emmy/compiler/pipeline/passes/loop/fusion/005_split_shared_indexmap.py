"""Fuse a multi-consumer pure-indexmap producer into all of its consumers at once.

``010_merge_loop_ops`` only extends a producer→consumer chain through
**single-consumer** edges (the match-walker stops at a fan-out:
``pipeline.py`` ``nid = consumers[0] if len(consumers) == 1 else None``). So a
pure-indexmap ``LoopOp`` that fans out to ≥2 consumers — e.g. the scalar-constant
broadcasts torch.export folds the attention-mask / RoPE scaffolding into — never
gets absorbed and survives as its own kernel (a ``[1,1,32,128]`` copy of
``0.0``/``1.0`` that PyTorch never materializes).

Anchored on the **producer**, this rule fuses it into *every* consumer in one
rewrite: it inlines the producer's body into each consumer (reusing the same
``splice_loop_ops`` machinery ``merge_loop_ops`` uses), emits one fused node per
consumer, and dissolves the producer — all via a single multi-output
``Graph.splice`` (``output={consumer_id: fused_node_id}``). For a scalar source
each fused consumer ends up loading the ``ConstantOp`` directly, which the
cuda-lowering literal-inline path turns into a ``float x = 0.0f;`` with no buffer.

If the splicer can't inline the producer into a particular consumer (a σ-solve it
doesn't support), that consumer falls back to a private duplicate of the producer
plus a Load-redirected consumer — exactly what un-sharing did — so the producer
still dissolves and ``merge_loop_ops`` can retry that copy later (at worst it stays
a separate copy, never a regression).

Gate = pure-indexmap (broadcast / transpose / reshape / slice / cat) with ≥2
consumers, and not itself a graph output. Single-consumer pure-indexmaps stay
``merge_loop_ops``'s job. Restricting to pure-indexmap keeps the per-consumer
duplication cheap (the splicer's blowup logic doesn't even run here) — duplicating
an *expensive* producer into every consumer would multiply real work.

Terminates: each firing removes one multi-consumer pure-indexmap producer. A fused
node no longer reads that producer, so the inlining marches one level down any
broadcast tree per firing until it reaches the non-indexmap leaves.
"""

from __future__ import annotations

from emmy.compiler.graph import Graph, Node, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.loop import Load, LoopOp, splice_loop_ops
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.fusion._helpers import is_pure_indexmap, rename_write_output

PATTERN = [Pattern("producer", LoopOp)]


def _redirect_loads(op: LoopOp, *, old: str, new: str) -> LoopOp:
    """Return ``op`` with every ``Load`` reading buf ``old`` redirected to ``new``."""

    def fn(s):
        if isinstance(s, Load) and s.input == old:
            return Load(names=s.names, input=new, index=s.index, dtype=s.dtype)
        return s

    return LoopOp(body=op.body.map(fn))


def rewrite(match: Match, producer: Node) -> Graph | None:
    graph = match.graph
    if not (isinstance(producer.op, LoopOp) and is_pure_indexmap(producer.op)):
        raise RuleSkipped("producer is not a pure-indexmap LoopOp")
    if producer.id in graph.outputs:
        raise RuleSkipped("producer is a graph output — must stay materialized")
    consumers = sorted(graph.consumers(producer.id))
    if len(consumers) < 2:
        raise RuleSkipped("producer is not shared (< 2 consumers) — single-consumer folds are merge's job")

    frag = Graph()
    output_map: dict[str, str] = {}  # old consumer id -> fragment output node id
    # (op, inputs, output_tensor, node_id) specs; copies must be added before the
    # fallback consumers that read them, so the two kinds are kept apart.
    copy_specs: list[tuple[LoopOp, list[str], Tensor, str]] = []
    consumer_specs: list[tuple[LoopOp, list[str], Tensor, str]] = []
    referenced: set[str] = set()  # buf ids the fragment nodes read

    for cid in consumers:
        cons = graph.nodes[cid]
        merged = splice_loop_ops(producer.op, cons.op, producer.id)
        if merged is not None:
            # Inline succeeded — one fused node replaces this consumer.
            fused_id = f"fused_{cid}"
            fused_op = rename_write_output(merged, old=cid, new=fused_id)
            inputs = list(fused_op.inputs)
            consumer_specs.append((fused_op, inputs, cons.output, fused_id))
            output_map[cid] = fused_id
            referenced.update(inputs)
        else:
            # Splicer can't inline here — fall back to a private copy + a
            # Load-redirected consumer (the old un-share behavior for this edge).
            copy_id = f"{producer.id}__dup__{cid}"
            copy_op = rename_write_output(producer.op, old=producer.id, new=copy_id)
            copy_specs.append((copy_op, list(producer.inputs), Tensor(copy_id, producer.output.shape, producer.output.dtype), copy_id))
            referenced.update(producer.inputs)

            unshared_id = f"unshared_{cid}"
            new_cons_op = rename_write_output(_redirect_loads(cons.op, old=producer.id, new=copy_id), old=cid, new=unshared_id)
            inputs = list(new_cons_op.inputs)
            consumer_specs.append((new_cons_op, inputs, cons.output, unshared_id))
            output_map[cid] = unshared_id
            referenced.update(inputs)

    # InputOp aliases for every referenced buf produced *outside* the fragment
    # (the producer's own inputs, each consumer's other inputs). Add externals
    # first, then copies, then consumers — ``add_node`` validates inputs exist.
    frag_ids = {nid for *_r, nid in copy_specs} | {nid for *_r, nid in consumer_specs}
    for ext in dict.fromkeys(r for r in referenced if r not in frag_ids):
        e = graph.nodes[ext]
        frag.add_node(InputOp(), [], Tensor(ext, e.output.shape, e.output.dtype), node_id=ext)
    for op, inputs, out_t, nid in (*copy_specs, *consumer_specs):
        frag.add_node(op, inputs, Tensor(out_t.name, out_t.shape, out_t.dtype), node_id=nid)
    frag.outputs = list(output_map.values())

    # Multi-output splice: dissolve the producer and replace every consumer with
    # its fused (or fallback) node in one shot.
    match.consumed = {producer.id, *consumers}
    match.output = output_map
    return frag
