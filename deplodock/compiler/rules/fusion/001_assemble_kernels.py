"""Assemble primitive ops into structural ``LoopOp`` nodes.

The grammar:

    reduce*       all reachable ReduceOps (must share axis + pre-reduce shape)
    elementwise*  all reachable ElementwiseOps
    layout*       same-rank IndexMapOps (for connectivity, become Port.indexmap)
"""

from __future__ import annotations

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Node, Tensor
from deplodock.compiler.ir.loop import Assign, LoopOp, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp
from deplodock.compiler.matcher import ChainMatch, Production


def _same_rank(op, node, graph):
    """IndexMapOp predicate: pass through single-source same-rank IndexMapOps.

    Multi-source IndexMapOps (cat) have Mux semantics that can't be
    folded into a simple Port.indexmap — they need their own kernel.
    """
    if len(op.sources) > 1:
        return False
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
    ports, remap, input_names = _build_input_tree(graph, external_inputs, all_consumed)
    all_consumed |= layout_ids
    body = _build_body_from_region(graph, all_consumed, remap)

    if not body:
        return None

    last_node = graph.nodes[output_nid]

    # Build fragment: InputOps for external buffers, one LoopOp node.
    frag = Graph()
    for name in input_names:
        if name not in frag.nodes:
            ext = graph.nodes.get(name)
            shape = ext.output.shape if ext else ()
            dtype = ext.output.dtype if ext else "f32"
            frag.add_node(InputOp(), [], Tensor(name, shape, dtype), node_id=name)

    kernel = LoopOp(inputs=tuple(ports), body=tuple(body), outputs=(Port(),))
    out_id = frag.add_node(kernel, input_names, Tensor(f"kernel_{output_nid}", last_node.output.shape, last_node.output.dtype))
    frag.outputs = [out_id]

    match.consumed = all_consumed
    return frag


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body_from_region(graph: Graph, region: set[str], remap: dict[str, str]) -> list[Assign]:
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


def _build_input_tree(graph: Graph, external_inputs: list[str], consumed: set[str]) -> tuple[list[Port], dict[str, str], list[str]]:
    """Build Ports and a remap dict mapping graph node IDs to $N references.

    Returns (ports, remap, input_names) where:
    - ports: list of Port objects (position = $N index)
    - remap: graph_node_id → "$N" for Assign.args
    - input_names: buffer name for each Port (for graph node inputs)
    """
    ports: list[Port] = []
    remap: dict[str, str] = {}
    input_names: list[str] = []
    idx = 0

    for buf_id in external_inputs:
        node = graph.nodes.get(buf_id)

        if node is not None and isinstance(node.op, IndexMapOp) and len(node.op.sources) == 1:
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
                # Internal chain — remap all to the consumed source's $N.
                # Find which $N the source already has.
                src_ref = remap.get(cur_id, cur_id)
                for nid in chain:
                    consumed.add(nid)
                    remap[nid] = src_ref
                continue
            # External source — absorb this IndexMapOp into Port.
            src_id = node.inputs[node.op.sources[0].input_idx]
            if len(graph.consumers(buf_id)) == 1:
                consumed.add(buf_id)
                ref = f"${idx}"
                ports.append(Port(indexmap=node.op))
                input_names.append(src_id)
                remap[buf_id] = ref
                idx += 1
                continue

        # Plain port (no IndexMapOp absorption).
        ref = f"${idx}"
        ports.append(Port())
        input_names.append(buf_id)
        remap[buf_id] = ref
        idx += 1

    return ports, remap, input_names
