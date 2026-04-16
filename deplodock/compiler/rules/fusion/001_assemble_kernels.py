"""Assemble primitive ops into structural ``KernelOp`` nodes.

Declares a grammar that the region-growing engine matches against the
graph. After this rule runs to fixed point, every compute node is
wrapped in a ``KernelOp`` — the graph contains only ``KernelOp``,
``InputOp``, and ``ConstantOp`` nodes.

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
    Combine,
    ElementwiseOp,
    IndexMapOp,
    KernelOp,
    Mux,
    Port,
    ReduceOp,
)


def _same_rank(op, node, graph):
    """IndexMapOp predicate: pass through broadcasts (same or lower rank)."""
    in_shape = graph.nodes[node.inputs[0]].output.shape
    return len(op.out_shape) <= len(in_shape)


GRAMMAR = [
    Production("reduce", ReduceOp, "*", bind={"input_shape": "pre_shape", "axis": "reduce_axis"}),
    Production("elementwise", ElementwiseOp, "*"),
    Production("layout", IndexMapOp, "*", where=_same_rank),
]


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace a matched region with a single ``KernelOp`` node."""
    compute_ids = match.get("reduce") + match.get("elementwise")
    layout_ids = set(match.get("layout"))

    if not compute_ids:
        return graph

    all_consumed = set(compute_ids)  # layout nodes handled as external by _build_input_tree
    output_nid = match.output
    assert output_nid is not None

    external_inputs = _collect_external_inputs(graph, all_consumed)
    inputs, port_map = _build_input_tree(graph, external_inputs, all_consumed)
    # _build_input_tree absorbs boundary IndexMapOps into consumed via Port.indexmap.
    # Also consume any layout nodes from the match that weren't at the boundary.
    all_consumed |= layout_ids
    body = _build_body_from_region(graph, all_consumed, port_map)

    if not body:
        return graph

    last_node = graph.nodes[output_nid]

    g = graph.copy()
    input_nids = [p.buffer_id if isinstance(p, Port) else _first_port_id(p) for p in inputs]

    out_port = Port(buffer_id=output_nid)
    kernel = KernelOp(
        inputs=tuple(inputs),
        body=tuple(body),
        outputs=(out_port,),
    )

    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{output_nid}", shape=last_node.output.shape, dtype=last_node.output.dtype),
    )
    out_port.buffer_id = new_nid
    for nid in all_consumed:
        orig = graph.nodes.get(nid)
        if orig is not None and orig.hints:
            g.nodes[new_nid].hints.merge(orig.hints)

    g.replace_node(output_nid, new_nid)
    _rename_in_kernels(g, output_nid, new_nid)

    for nid in all_consumed:
        if nid in g.nodes and nid != output_nid:
            g.remove_node(nid)
    if output_nid in g.nodes:
        g.remove_node(output_nid)
    return g


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body_from_region(graph: Graph, region: set[str], port_map: dict) -> list[Assign]:
    """Build SSA Assigns from the fused region in topological order."""
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


def _build_input_tree(
    graph: Graph,
    external_inputs: list[str],
    consumed: set[str],
) -> tuple[list, dict]:
    """Build KernelInput trees and a port_map for absorbed IndexMapOps."""
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
            # Follow IndexMapOp chain to find ultimate non-IndexMapOp source.
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
                # Internal chain — remap to ultimate source, consume chain.
                for nid in chain:
                    consumed.add(nid)
                    port_map[nid] = Port(buffer_id=cur_id)
                continue
            # External source — absorb only this IndexMapOp (single step).
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


# ---------------------------------------------------------------------------
# Kernel rename helpers
# ---------------------------------------------------------------------------


def _rename_in_kernels(graph: Graph, old_id: str, new_id: str) -> None:
    if old_id == new_id:
        return
    for node in graph.nodes.values():
        if not isinstance(node.op, KernelOp):
            continue
        for inp in node.op.inputs:
            _rename_ports(inp, old_id, new_id)
        for assign in node.op.body:
            if old_id in assign.args:
                assign.args = tuple(new_id if a == old_id else a for a in assign.args)


def _rename_ports(inp, old_id: str, new_id: str) -> None:
    if isinstance(inp, Port):
        if inp.buffer_id == old_id:
            inp.buffer_id = new_id
    elif isinstance(inp, Mux):
        for branch in inp.branches:
            _rename_ports(branch.input, old_id, new_id)
    elif isinstance(inp, Combine):
        for src in inp.sources:
            _rename_ports(src, old_id, new_id)


def _first_port_id(inp) -> str:
    if isinstance(inp, Port):
        return inp.buffer_id
    if isinstance(inp, Mux):
        return _first_port_id(inp.branches[0].input)
    return ""
