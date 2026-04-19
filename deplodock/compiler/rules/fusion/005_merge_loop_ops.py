"""Merge two adjacent ``LoopOp``s connected by a single-consumer buffer.

Matches a ``LoopOp`` whose sole consumer is another ``LoopOp`` and fuses them
by aligning the producer's write indices with the consumer's read indices via
the σ solver in ``_merge_core``. Refuses merges that would produce a kernel
with two reduce axes (the CUDA backend is single-reduce in v1) or that would
leak a free producer axis into the consumer's iteration space.

Also detects **reduce-axis aliases** across the producer-consumer boundary:
when the producer has an unbound reduce axis that reads the same external
buffer at the same dim as a consumer reduce axis, the two axes are unified,
letting two independent reductions over one data axis collapse into one
kernel (e.g. ``sum(x, -1) + max(x, -1)``).
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinOp, Cast, Var
from deplodock.compiler.ir.graph import Graph, Node, Tensor
from deplodock.compiler.ir.loop_ir import Axis, LoopOp, Write
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

    # Find every consumer port that reads the producer. One producer may feed
    # several reader patterns (e.g. softmax's sum+div consumer reads the
    # max+sub+exp producer at both [a0, a1] for the divide and [a0, a1_1] for
    # the sum sweep). Merge handles multi-port via per-σ_k post-reduce
    # instantiation.
    consumer_ports = [i for i, inp in enumerate(consumer_node.inputs) if inp == producer_id]
    if not consumer_ports:
        return None

    axis_aliases = _detect_reduce_axis_aliases(producer_node, consumer_node, consumer_ports)

    merged = merge_loop_ops(
        producer_node.op,
        producer_output=0,
        consumer=consumer_node.op,
        consumer_port=consumer_ports,
        axis_aliases=axis_aliases,
    )
    if merged is None:
        return None

    consumer_port_set = set(consumer_ports)
    merged_input_ids: list[str] = []
    for i, inp_id in enumerate(consumer_node.inputs):
        if i in consumer_port_set:
            continue
        merged_input_ids.append(inp_id)
    # Producer ports are emitted once per σ_k, so we append producer.inputs
    # once per consumer port that reads this producer.
    for _ in consumer_ports:
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


# ---------------------------------------------------------------------------
# Reduce-axis alias detection
# ---------------------------------------------------------------------------


def _detect_reduce_axis_aliases(
    producer_node: Node,
    consumer_node: Node,
    consumer_ports: list[int],
) -> dict[str, str]:
    """Find producer reduce axes that can be unified with consumer reduce axes.

    A producer reduce axis ``a_p`` aliases to a consumer reduce axis ``a_c``
    when both index the same external buffer at the same dim position. This
    is the information needed to collapse two independent reductions over one
    data axis into a single reduce sweep.

    The check is conservative: both axes must appear as bare ``Var`` entries
    in their respective Ports (no arithmetic), the extents must match, and
    the external buffer id must be identical on both sides. The consumer
    ports in ``consumer_ports`` are excluded from the search because they
    read the producer's output — not a shared external buffer.
    """
    producer_op = producer_node.op
    consumer_op = consumer_node.op
    assert isinstance(producer_op, LoopOp)
    assert isinstance(consumer_op, LoopOp)

    from deplodock.compiler.ir.loop_ir import flatten_body

    write_axis_names: set[str] = set()
    for stmt in flatten_body(producer_op.body):
        if isinstance(stmt, Write) and stmt.output == 0:
            for e in stmt.index:
                _collect_axis_names(e, write_axis_names)

    unbound_reduce = [a for a in producer_op.axes if a.kind == "reduce" and a.name not in write_axis_names]
    if not unbound_reduce:
        return {}

    consumer_reduce_by_name = {a.name: a for a in consumer_op.axes if a.kind == "reduce"}
    if not consumer_reduce_by_name:
        return {}

    consumer_port_set = set(consumer_ports)
    aliases: dict[str, str] = {}
    for a_p in unbound_reduce:
        found = _find_alias_for(a_p, producer_op, producer_node, consumer_op, consumer_node, consumer_port_set, consumer_reduce_by_name)
        if found is not None:
            aliases[a_p.name] = found

    # Fallback: when a consumer port in `consumer_ports` reads the producer's
    # output with a consumer reduce axis at a dim where producer's Write has a
    # producer free axis — i.e., σ_k will bind a producer free axis to a
    # consumer reduce axis — then any remaining unbound producer reduce axis
    # is semantically "nested inside" the consumer reduce sweep and can be
    # aliased to the consumer's reduce axis. This is the softmax pattern: the
    # strict buffer-dim check misses it because the consumer reads the
    # producer's output (not a shared external buffer), but the σ-flow makes
    # the alignment unambiguous.
    unaliased_producer = [a for a in unbound_reduce if a.name not in aliases]
    if not unaliased_producer:
        return aliases

    consumer_reduce_axes = list(consumer_reduce_by_name.values())
    if len(consumer_reduce_axes) != 1:
        return aliases
    c_reduce = consumer_reduce_axes[0]
    if c_reduce.name in aliases.values():
        return aliases

    writes = [s for s in flatten_body(producer_op.body) if isinstance(s, Write) and s.output == 0]
    if len(writes) != 1:
        return aliases
    w = writes[0]

    producer_free_names = {a.name for a in producer_op.axes if a.kind == "free"}
    flow_alias_found = False
    for cp in consumer_ports:
        c_port = consumer_op.inputs[cp]
        for dim, r_expr in enumerate(c_port.index):
            if dim >= len(w.index):
                continue
            if not (isinstance(r_expr, Var) and r_expr.name == c_reduce.name):
                continue
            w_expr = w.index[dim]
            if isinstance(w_expr, Var) and w_expr.name in producer_free_names:
                flow_alias_found = True
                break
        if flow_alias_found:
            break

    if flow_alias_found:
        for a_p in unaliased_producer:
            if int(a_p.extent) == int(c_reduce.extent):
                aliases[a_p.name] = c_reduce.name
    return aliases


def _find_alias_for(
    a_p: Axis,
    producer_op: LoopOp,
    producer_node: Node,
    consumer_op: LoopOp,
    consumer_node: Node,
    consumer_port_set: set[int],
    consumer_reduce_by_name: dict[str, Axis],
) -> str | None:
    """Return the consumer axis name that aliases ``a_p``, or None."""
    for p_idx, port in enumerate(producer_op.inputs):
        if p_idx >= len(producer_node.inputs):
            continue
        p_buf = producer_node.inputs[p_idx]
        for dim, expr in enumerate(port.index):
            if not (isinstance(expr, Var) and expr.name == a_p.name):
                continue
            for c_idx, c_port in enumerate(consumer_op.inputs):
                if c_idx in consumer_port_set:
                    continue
                if c_idx >= len(consumer_node.inputs):
                    continue
                if consumer_node.inputs[c_idx] != p_buf:
                    continue
                if dim >= len(c_port.index):
                    continue
                c_expr = c_port.index[dim]
                if not isinstance(c_expr, Var):
                    continue
                c_axis = consumer_reduce_by_name.get(c_expr.name)
                if c_axis is None:
                    continue
                if int(c_axis.extent) != int(a_p.extent):
                    continue
                return c_axis.name
    return None


def _collect_axis_names(expr, out: set[str]) -> None:
    if isinstance(expr, Var):
        out.add(expr.name)
    elif isinstance(expr, BinOp):
        _collect_axis_names(expr.left, out)
        _collect_axis_names(expr.right, out)
    elif isinstance(expr, Cast):
        _collect_axis_names(expr.expr, out)
