"""Un-share a multi-consumer pure-indexmap producer so ``merge_loop_ops`` can inline it.

``010_merge_loop_ops`` only extends a producer→consumer chain through
**single-consumer** edges (the match-walker stops at a fan-out:
``pipeline.py`` ``nid = consumers[0] if len(consumers) == 1 else None``).
So a pure-indexmap ``LoopOp`` that fans out to ≥2 consumers — e.g. the
scalar-constant broadcasts torch.export folds the attention-mask / RoPE
scaffolding into — never gets absorbed and survives as its own kernel
(a ``[1,1,32,128]`` copy of ``0.0``/``1.0`` that PyTorch never
materializes).

This rule, sorted *before* merge, only un-shares those producers — it
does no fusion itself. Anchored on a consumer, it gives that consumer a
**private duplicate** of every shared (≥2 consumers) pure-indexmap
producer it reads, and rewrites the consumer's Loads to read the copy.
Each firing peels one consumer onto its own copy, dropping the shared
producer's consumer count by one; when it reaches a single consumer the
``>= 2`` guard stops matching it and that last edge is a plain
single-consumer producer→consumer pair that ``merge_loop_ops`` folds
like any other. A private copy is itself single-consumer, so it never
re-triggers the rule. Across the rule's batch + restart scans the net
effect is: ``producer → {c1, c2, c3}`` becomes ``{c1, c2, c3}`` each with
a private copy folded in by merge (the last reusing the original). For a
scalar source the duplication is free — each copy is a constant ``Load``
that the cuda-lowering literal-inline path turns into a ``float x =
0.0f;`` with no buffer.

Gate = pure-indexmap (broadcast / transpose / reshape / slice / cat) with
≥2 consumers. Restricting to pure-indexmap keeps the duplication cheap
and guarantees merge accepts it (merge's blowup guards never trip on a
pure-indexmap producer, and its broadcast-materialization guard only
fires the opposite way — pure-indexmap *consumer* over compute
*producer*). Duplicating an expensive producer would multiply real work,
so it is deliberately excluded.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop import Load, LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.loop.fusion._helpers import is_pure_indexmap, rename_write_output

PATTERN = [Pattern("consumer", LoopOp)]


def rewrite(match: Match, consumer: Node) -> Graph | None:
    graph = match.graph
    # Inputs that are SHARED pure-indexmap producers. Single-consumer ones
    # are already merge's job, so the ``>= 2`` guard skips them (and keeps
    # the rule from re-firing on the private copies it just emitted).
    shared = [
        inp
        for inp in consumer.inputs
        if (p := graph.nodes.get(inp)) is not None
        and isinstance(p.op, LoopOp)
        and is_pure_indexmap(p.op)
        and len(graph.consumers(inp)) >= 2
    ]
    if not shared:
        raise RuleSkipped("consumer reads no shared pure-indexmap producer")

    # Map each shared producer to a private duplicate id (unique per
    # producer+consumer pair) and collect the producers' own inputs — those
    # become external InputOp aliases the copies read from.
    remap: dict[str, str] = {}
    ext_inputs: list[str] = []
    for prod_id in shared:
        remap[prod_id] = f"{prod_id}__dup__{consumer.id}"
        for pin in graph.nodes[prod_id].inputs:
            if pin not in ext_inputs:
                ext_inputs.append(pin)

    # Rewrite the consumer's body Loads to read the private copy. Index is
    # unchanged — only the source buffer name moves to the duplicate.
    def fn(s):
        if isinstance(s, Load) and s.input in remap:
            return Load(names=s.names, input=remap[s.input], index=s.index, dtype=s.dtype)
        return s

    # Rename the consumer's Write.output from its old node id to the new one so
    # the rewritten node upholds the buf-name == node-id invariant. Skipping
    # this leaves the new node writing the *old* buf, which silently breaks the
    # next merge: ``splice_graph`` derives splice edges as ``(node_id, node_id)``
    # — it assumes a producer's Write.output is its node id (splicer.py:205) — so
    # a mismatched buf makes every downstream fold of this node hit
    # ``_NotSupported`` ("no Write with output=<node_id>") and survive as its own
    # kernel.
    unshared_id = f"unshared_{consumer.id}"
    new_consumer_op = rename_write_output(LoopOp(body=consumer.op.body.map(fn)), old=consumer.id, new=unshared_id)

    # Build the fragment: InputOp aliases for every external (producers'
    # inputs + the consumer's non-shared inputs), then the copy producers,
    # then the rewritten consumer. ``frag.add_node`` validates inputs exist,
    # so the topological order (externals → copies → consumer) matters.
    frag = Graph()
    externals = dict.fromkeys([*ext_inputs, *(i for i in consumer.inputs if i not in remap)])
    for ext in externals:
        e = graph.nodes[ext]
        frag.add_node(InputOp(), [], Tensor(ext, e.output.shape, e.output.dtype), node_id=ext)
    for prod_id, copy_id in remap.items():
        prod = graph.nodes[prod_id]
        copy_op = rename_write_output(prod.op, old=prod_id, new=copy_id)
        frag.add_node(copy_op, list(prod.inputs), Tensor(copy_id, prod.output.shape, prod.output.dtype), node_id=copy_id)
    # ``node.inputs`` must follow the body's first-use buf order (the
    # interpreter zips it positionally against ``input_bufs``); read it
    # straight off the rewritten op rather than reconstructing it.
    out_id = frag.add_node(
        new_consumer_op,
        list(new_consumer_op.inputs),
        Tensor(consumer.output.name, consumer.output.shape, consumer.output.dtype),
        node_id=unshared_id,
    )
    frag.outputs = [out_id]

    # Consume only the consumer — the shared producer stays until its last
    # consumer is peeled, when ``remove_orphans`` reaps it.
    match.output = consumer.id
    match.consumed = {consumer.id}
    return frag
