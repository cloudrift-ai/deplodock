"""Assemble primitive ops into structural ``KernelOp`` nodes.

Uses the chain grammar to parse each fan-out-1 path through the
primitive graph. After this rule runs to fixed point, every compute
node in the graph is wrapped in a ``KernelOp`` — the graph contains
only ``KernelOp``, ``InputOp``, and ``ConstantOp`` nodes.

The kernel grammar:

    contraction?  (mul + sum pair, backtracking)
    stage*        (pre_ops* + reduce, repeating)
    epilogue*     (trailing elementwise chain)

Each consumed graph node becomes an ``Assign`` in the kernel body::

    mul = mul(a, b)
    dot = reduce_sum(mul)
    out = add(dot, bias)
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import ChainMatch, Group, Production, parse_chain
from deplodock.compiler.ops import (
    Assign,
    ConstantOp,
    ElementwiseOp,
    IndexMapOp,
    InputOp,
    KernelOp,
    Mux,
    MuxBranch,
    Port,
    ReduceOp,
)

GRAMMAR = [Production("any", (ElementwiseOp, ReduceOp, IndexMapOp), "1")]

_KERNEL_GRAMMAR = [
    Group(
        "contraction",
        [
            Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
            Production("reduce", ReduceOp, "1", {"fn": "sum"}),
        ],
        "?",
    ),
    Group(
        "stage",
        [
            Production("pre_ops", ElementwiseOp, "*"),
            Production("reduce", ReduceOp, "1"),
        ],
        "*",
    ),
    Production("epilogue", ElementwiseOp, "*"),
]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace a primitive chain with a single ``KernelOp`` node."""
    seed = match.root_node_id
    node = graph.nodes[seed]

    if isinstance(node.op, (InputOp, ConstantOp, KernelOp)):
        return graph

    if isinstance(node.op, IndexMapOp):
        consumers = graph.consumers(seed)
        if all(isinstance(graph.nodes[c].op, KernelOp) for c in consumers if c in graph.nodes):
            return _wrap_indexmap_kernel(graph, node)
        return graph

    chain = parse_chain(graph, seed, _KERNEL_GRAMMAR)
    if chain is None or not chain.consumed:
        return graph

    all_consumed = set(chain.consumed)
    external_inputs = _collect_external_inputs(graph, all_consumed)
    inputs, external_shapes, port_map = _build_input_tree(graph, external_inputs, all_consumed)
    body = _build_body(graph, chain, port_map)

    last_nid = _last_node_id(chain)
    last_node = graph.nodes[last_nid]

    kernel = KernelOp(
        inputs=tuple(inputs),
        body=tuple(body),
        outputs=(Port(buffer_id=last_nid),),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    input_nids = [p.buffer_id if isinstance(p, Port) else _first_port_id(p) for p in inputs]
    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{last_nid}", shape=last_node.output.shape, dtype=last_node.output.dtype),
    )
    for nid in all_consumed:
        orig = graph.nodes.get(nid)
        if orig is not None and orig.hints:
            g.nodes[new_nid].hints.merge(orig.hints)

    g.replace_node(last_nid, new_nid)
    for nid in all_consumed:
        if nid in g.nodes and nid != last_nid:
            g.remove_node(nid)
    if last_nid in g.nodes:
        g.remove_node(last_nid)
    return g


# ---------------------------------------------------------------------------
# Body builder — produces Assign statements from the parsed chain
# ---------------------------------------------------------------------------


def _build_body(graph: Graph, chain: ChainMatch, port_map: dict) -> list[Assign]:
    """Build SSA Assigns from the grammar parse.

    Each consumed graph node becomes an Assign: ``name = op(args)``.
    Args reference input Port.buffer_ids or prior Assign names. When a
    Port was absorbed from an IndexMapOp (recorded in ``port_map``), the
    arg name is remapped from the IndexMapOp's output id to the absorbed
    Port's buffer_id.
    """
    assigns: list[Assign] = []
    # Build reverse map: original graph id → absorbed Port buffer_id.
    remap: dict[str, str] = {}
    for orig_id, port_or_mux in port_map.items():
        if isinstance(port_or_mux, Port):
            remap[orig_id] = port_or_mux.buffer_id

    def _remap_args(node: Node) -> tuple[str, ...]:
        return tuple(remap.get(inp, inp) for inp in node.inputs)

    # Contraction group
    for seg in chain.segments:
        if seg.name.startswith("contraction["):
            for nid in seg.node_ids:
                node = graph.nodes[nid]
                assigns.append(Assign(name=nid, op=node.op, args=_remap_args(node)))

    # Stage groups
    for group_segs in chain.get_groups("stage"):
        for seg in group_segs:
            for nid in seg.node_ids:
                node = graph.nodes[nid]
                assigns.append(Assign(name=nid, op=node.op, args=_remap_args(node)))

    # Epilogue
    for nid in chain.get("epilogue"):
        node = graph.nodes[nid]
        assigns.append(Assign(name=nid, op=node.op, args=_remap_args(node)))

    return assigns


def _last_node_id(chain: ChainMatch) -> str:
    for seg in reversed(chain.segments):
        if seg.node_ids:
            return seg.node_ids[-1]
    return chain.root_node_id


# ---------------------------------------------------------------------------
# Input tree construction (backward walk)
# ---------------------------------------------------------------------------


def _collect_external_inputs(graph: Graph, consumed: set[str]) -> list[str]:
    seen: list[str] = []
    for nid in consumed:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        for inp in node.inputs:
            if inp not in consumed and inp not in seen:
                seen.append(inp)
    return seen


def _build_input_tree(
    graph: Graph,
    external_inputs: list[str],
    consumed: set[str],
) -> tuple[list, dict[str, tuple], dict]:
    inputs = []
    external_shapes: dict[str, tuple] = {}
    port_map: dict[str, Port | Mux] = {}

    for buf_id in external_inputs:
        node = graph.nodes.get(buf_id)
        if node is None:
            p = Port(buffer_id=buf_id)
            inputs.append(p)
            port_map[buf_id] = p
            continue

        if isinstance(node.op, IndexMapOp) and len(node.op.sources) == 1 and len(graph.consumers(buf_id)) == 1:
            # Single-source IndexMapOp with fan-out 1: absorb into Port.indexmap.
            consumed.add(buf_id)
            src_id = node.inputs[node.op.sources[0].input_idx]
            external_shapes[src_id] = tuple(graph.nodes[src_id].output.shape)
            p = Port(buffer_id=src_id, indexmap=node.op)
            inputs.append(p)
            port_map[buf_id] = p
        else:
            external_shapes[buf_id] = tuple(node.output.shape)
            p = Port(buffer_id=buf_id)
            inputs.append(p)
            port_map[buf_id] = p

    return inputs, external_shapes, port_map


def _first_port_id(inp) -> str:
    if isinstance(inp, Port):
        return inp.buffer_id
    if isinstance(inp, Mux):
        return _first_port_id(inp.branches[0].input)
    return ""


# ---------------------------------------------------------------------------
# Standalone IndexMapOp wrapping
# ---------------------------------------------------------------------------


def _wrap_indexmap_kernel(graph: Graph, node: Node) -> Graph:
    nid = node.id
    op = node.op
    assert isinstance(op, IndexMapOp)

    external_shapes: dict[str, tuple] = {}
    if len(op.sources) == 1:
        src_id = node.inputs[op.sources[0].input_idx]
        external_shapes[src_id] = tuple(graph.nodes[src_id].output.shape)
        inputs: tuple = (Port(buffer_id=src_id, indexmap=op),)
    else:
        branches = []
        for src in op.sources:
            src_id = node.inputs[src.input_idx]
            external_shapes[src_id] = tuple(graph.nodes[src_id].output.shape)
            branches.append(MuxBranch(input=Port(buffer_id=src_id), select=src.select))
        inputs = (Mux(branches=tuple(branches)),)

    kernel = KernelOp(
        inputs=inputs,
        outputs=(Port(buffer_id=nid),),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    input_nids = [_first_port_id(inputs[0])]
    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{nid}", shape=tuple(op.out_shape), dtype=node.output.dtype),
    )
    g.replace_node(nid, new_nid)
    g.remove_node(nid)
    return g
