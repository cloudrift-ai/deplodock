"""Assemble primitive ops into structural ``KernelOp`` nodes.

The grammar:

    reduce*       all reachable ReduceOps (must share axis + pre-reduce shape)
    elementwise*  all reachable ElementwiseOps
    layout*       same-rank IndexMapOps (for connectivity, become Port.indexmap)
"""

from __future__ import annotations

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import (
    Assign,
    ElementwiseOp,
    IndexMapOp,
    InputOp,
    KernelOp,
    Port,
    ReduceOp,
)


def _same_rank(op, node, graph):
    in_shape = graph.nodes[node.inputs[0]].output.shape
    return len(op.out_shape) <= len(in_shape)


GRAMMAR = [
    Production("reduce", ReduceOp, "*", bind={"input_shape": "pre_shape", "axis": "reduce_axis"}),
    Production("elementwise", ElementwiseOp, "*"),
    Production("layout", IndexMapOp, "*", where=_same_rank),
]


def rewrite(graph: Graph, match: ChainMatch) -> Graph | None:
    compute_ids = match.get("reduce") + match.get("elementwise")
    layout_ids = set(match.get("layout"))

    if not compute_ids:
        return None

    all_consumed = set(compute_ids)
    output_nid = match.output
    assert output_nid is not None

    external_inputs = _collect_external_inputs(graph, all_consumed)
    inputs, port_map = _build_input_tree(graph, external_inputs, all_consumed)
    all_consumed |= layout_ids
    body = _build_body_from_region(graph, all_consumed, port_map)

    if not body:
        return None

    last_node = graph.nodes[output_nid]

    # Build fragment: InputOps for external ports, one KernelOp node.
    frag = Graph()
    input_nids = []
    for p in inputs:
        bid = p.buffer_id if isinstance(p, Port) else _first_port_id(p)
        if bid not in frag.nodes:
            ext = graph.nodes.get(bid)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(bid, shape, dtype), node_id=bid)
        input_nids.append(bid)

    out_port = Port(buffer_id="__out__")
    kernel = KernelOp(inputs=tuple(inputs), body=tuple(body), outputs=(out_port,))

    out_id = frag.add_node(kernel, input_nids, Tensor(f"kernel_{output_nid}", last_node.output.shape, last_node.output.dtype))
    out_port.buffer_id = out_id
    frag.outputs = [out_id]

    # The match.consumed must include all compute + layout + absorbed IndexMapOps.
    match.consumed = all_consumed
    return frag


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body_from_region(graph: Graph, region: set[str], port_map: dict) -> list[Assign]:
    remap: dict[str, str] = {}
    for orig_id, port_or_mux in port_map.items():
        if isinstance(port_or_mux, Port):
            remap[orig_id] = port_or_mux.buffer_id

    def _remap_args(node: Node) -> tuple[str, ...]:
        return tuple(remap.get(inp, inp) for inp in node.inputs)

    assigns: list[Assign] = []
    for nid in graph.topological_order():
        if nid not in region:
            continue
        node = graph.nodes[nid]
        if not isinstance(node.op, (ElementwiseOp, ReduceOp)):
            continue
        assigns.append(Assign(name=nid, op=node.op, args=_remap_args(node)))
    return assigns


# ---------------------------------------------------------------------------
# Input tree construction
# ---------------------------------------------------------------------------


def _collect_external_inputs(graph: Graph, consumed: set[str]) -> list[str]:
    seen: list[str] = []
    for nid in graph.topological_order():
        if nid not in consumed:
            continue
        node = graph.nodes.get(nid)
        if node is None:
            continue
        for inp in node.inputs:
            if inp not in consumed and inp not in seen:
                seen.append(inp)
    return seen


def _build_input_tree(graph: Graph, external_inputs: list[str], consumed: set[str]) -> tuple[list, dict]:
    inputs = []
    port_map: dict[str, Port] = {}

    for buf_id in external_inputs:
        node = graph.nodes.get(buf_id)
        if node is None:
            p = Port(buffer_id=buf_id)
            inputs.append(p)
            port_map[buf_id] = p
            continue

        if isinstance(node.op, IndexMapOp) and len(node.op.sources) == 1:
            chain = [buf_id]
            cur_id = node.inputs[node.op.sources[0].input_idx]
            while cur_id in graph.nodes:
                cur_node = graph.nodes[cur_id]
                if isinstance(cur_node.op, IndexMapOp) and len(cur_node.op.sources) == 1:
                    chain.append(cur_id)
                    cur_id = cur_node.inputs[cur_node.op.sources[0].input_idx]
                else:
                    break
            if cur_id in consumed:
                for nid in chain:
                    consumed.add(nid)
                    port_map[nid] = Port(buffer_id=cur_id)
                continue
            src_id = node.inputs[node.op.sources[0].input_idx]
            if len(graph.consumers(buf_id)) == 1:
                consumed.add(buf_id)
                p = Port(buffer_id=src_id, indexmap=node.op)
                inputs.append(p)
                port_map[buf_id] = p
                continue

        p = Port(buffer_id=buf_id)
        inputs.append(p)
        port_map[buf_id] = p

    return inputs, port_map


def _first_port_id(inp) -> str:
    if isinstance(inp, Port):
        return inp.buffer_id
    from deplodock.compiler.ops import Mux

    if isinstance(inp, Mux):
        return _first_port_id(inp.branches[0].input)
    return ""
