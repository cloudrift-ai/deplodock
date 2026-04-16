"""Assemble primitive ops into structural ``KernelOp`` nodes.

Uses the chain grammar to parse each fan-out-1 path through the
primitive graph. After this rule runs to fixed point, every compute
node in the graph is wrapped in a ``KernelOp`` — the graph contains
only ``KernelOp``, ``InputOp``, and ``ConstantOp`` nodes.

The kernel grammar (see ``ops.py`` docstring for analogies):

    contraction?  (mul + sum pair, backtracking)
    stage*        (pre_ops* + reduce, repeating)
    epilogue*     (trailing elementwise chain)

Input-tree construction (backward walk): for each external input,
single-source ``IndexMapOp`` with fan-out 1 folds into ``Port.indexmap``;
multi-source ``IndexMapOp`` builds a ``Mux``.
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import ChainMatch, Group, Production, parse_chain
from deplodock.compiler.ops import (
    Combine,
    ConstantOp,
    ContractionCore,
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

    # Standalone IndexMapOp: only wrap as a copy kernel once all consumers
    # are already KernelOps (so the backward walk had its chance to absorb).
    if isinstance(node.op, IndexMapOp):
        consumers = graph.consumers(seed)
        if all(isinstance(graph.nodes[c].op, KernelOp) for c in consumers if c in graph.nodes):
            return _wrap_indexmap_kernel(graph, node)
        return graph

    chain = parse_chain(graph, seed, _KERNEL_GRAMMAR)
    if chain is None or not chain.consumed:
        return graph

    body_ops = _build_body(graph, chain)

    all_consumed = set(chain.consumed)
    external_inputs = _collect_external_inputs(graph, all_consumed)
    inputs, external_shapes, port_map = _build_input_tree(graph, external_inputs, all_consumed)
    contraction = _build_contraction(graph, chain, port_map)

    last_nid = _last_node_id(chain)
    last_node = graph.nodes[last_nid]

    kernel = KernelOp(
        inputs=tuple(inputs),
        outputs=(Port(buffer_id=last_nid),),
        contraction=contraction,
        body=tuple(body_ops),
        external_shapes=external_shapes,
    )

    g = graph.copy()
    input_nids = [p.buffer_id if isinstance(p, Port) else _first_port_id(p) for p in inputs]
    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(
            name=f"kernel_{last_nid}",
            shape=last_node.output.shape,
            dtype=last_node.output.dtype,
        ),
    )
    # Promote hints from consumed graph nodes to the KernelOp node.
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
# KernelOp field builders
# ---------------------------------------------------------------------------


def _build_contraction(graph: Graph, chain: ChainMatch, port_map: dict) -> ContractionCore | None:
    """Build a ContractionCore, using ``port_map`` to resolve absorbed inputs.

    ``port_map`` maps original buffer_id → absorbed Port (with indexmap
    set if the original id was an IndexMapOp absorbed by the backward walk).
    """
    mul_ids = chain.get("mul")
    if not mul_ids:
        return None
    contraction_reduce_ids = [
        nid for seg in chain.segments if seg.name.startswith("contraction[") and seg.name.endswith("].reduce") for nid in seg.node_ids
    ]
    if not contraction_reduce_ids:
        return None

    mul_node = graph.nodes[mul_ids[0]]
    red_node = graph.nodes[contraction_reduce_ids[0]]

    src_ports = tuple(port_map.get(inp, Port(buffer_id=inp)) for inp in mul_node.inputs)
    operand = Combine(sources=src_ports, ops=(mul_node.op,))

    assert isinstance(red_node.op, ReduceOp)
    return ContractionCore(operand=operand, reduce=red_node.op)


def _build_body(graph: Graph, chain: ChainMatch) -> list:
    """Build the flat body chain from stage groups + epilogue segments."""
    ops: list = []
    for group_segs in chain.get_groups("stage"):
        for seg in group_segs:
            for nid in seg.node_ids:
                ops.append(graph.nodes[nid].op)
    for nid in chain.get("epilogue"):
        ops.append(graph.nodes[nid].op)
    return ops


def _last_node_id(chain: ChainMatch) -> str:
    for seg in reversed(chain.segments):
        if seg.node_ids:
            return seg.node_ids[-1]
    return chain.root_node_id


# ---------------------------------------------------------------------------
# Input tree construction (backward walk)
# ---------------------------------------------------------------------------


def _collect_external_inputs(graph: Graph, consumed: set[str]) -> list[str]:
    """Collect buffer ids of all external inputs referenced by consumed nodes."""
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
    """Build KernelInput trees for each external input.

    Single-source IndexMapOp with fan-out 1 → fold into Port.indexmap.
    Multi-source IndexMapOp → Mux. Otherwise → bare Port.

    Returns ``(inputs, external_shapes, port_map)`` where ``port_map``
    maps original buffer_id → the constructed Port/Mux (so the
    contraction builder can resolve absorbed inputs).
    """
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

        if isinstance(node.op, IndexMapOp) and len(graph.consumers(buf_id)) == 1:
            consumed.add(buf_id)
            if len(node.op.sources) == 1:
                src_id = node.inputs[node.op.sources[0].input_idx]
                external_shapes[src_id] = tuple(graph.nodes[src_id].output.shape)
                p = Port(buffer_id=src_id, indexmap=node.op)
                inputs.append(p)
                port_map[buf_id] = p
            else:
                branches = []
                for src in node.op.sources:
                    src_id = node.inputs[src.input_idx]
                    external_shapes[src_id] = tuple(graph.nodes[src_id].output.shape)
                    branches.append(MuxBranch(input=Port(buffer_id=src_id), select=src.select))
                m = Mux(branches=tuple(branches))
                inputs.append(m)
                port_map[buf_id] = m
        else:
            external_shapes[buf_id] = tuple(node.output.shape)
            p = Port(buffer_id=buf_id)
            inputs.append(p)
            port_map[buf_id] = p

    return inputs, external_shapes, port_map


def _first_port_id(inp) -> str:
    """Extract the first Port.buffer_id from a KernelInput tree."""
    if isinstance(inp, Port):
        return inp.buffer_id
    if isinstance(inp, Mux):
        return _first_port_id(inp.branches[0].input)
    if isinstance(inp, Combine):
        return _first_port_id(inp.sources[0])
    return ""


def _wrap_indexmap_kernel(graph: Graph, node: Node) -> Graph:
    """Wrap a standalone IndexMapOp as a copy KernelOp.

    Single-source → Port(src, indexmap=op). Multi-source → Mux.
    The kernel has no contraction, no reduce_stages, no epilogue —
    it just reads via the indexmap and writes to the output buffer.
    """
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
