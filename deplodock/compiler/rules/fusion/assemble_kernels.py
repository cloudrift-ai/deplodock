"""Assemble primitive ops into structural ``KernelOp`` nodes.

Uses backward-cone region growing to fuse DAG subgraphs of primitive
ops into single kernels. Starting from a seed node, the algorithm:

1. **Forward BFS** — collect all downstream fusable primitives
   (``ElementwiseOp``, ``ReduceOp``), enforcing that reductions form
   a single linear chain (no parallel reductions).
2. **Find output** — last node in topo order with no consumers in
   the forward set.
3. **Backward cone** — from the output, absorb nodes whose consumers
   are all inside the region. This trims nodes with side-outputs.

After this rule runs to fixed point, every compute node in the graph
is wrapped in a ``KernelOp`` — the graph contains only ``KernelOp``,
``InputOp``, and ``ConstantOp`` nodes.
"""

from __future__ import annotations

from collections import deque

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import ChainMatch, Production
from deplodock.compiler.ops import (
    Assign,
    Combine,
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

_FUSABLE = (ElementwiseOp, ReduceOp, IndexMapOp)


def rewrite(graph: Graph, match: ChainMatch) -> Graph:
    """Replace a primitive region with a single ``KernelOp`` node."""
    seed = match.root_node_id
    node = graph.nodes[seed]

    if isinstance(node.op, (InputOp, ConstantOp, KernelOp)):
        return graph

    if isinstance(node.op, IndexMapOp):
        consumers = graph.consumers(seed)
        if all(isinstance(graph.nodes[c].op, KernelOp) for c in consumers if c in graph.nodes):
            return _wrap_indexmap_kernel(graph, node)
        return graph

    result = _grow_region(graph, seed)
    if result is None:
        return graph

    all_consumed, last_nid = result
    external_inputs = _collect_external_inputs(graph, all_consumed)
    inputs, port_map = _build_input_tree(graph, external_inputs, all_consumed)
    body = _build_body_from_region(graph, all_consumed, port_map)

    if not body:
        return graph

    last_node = graph.nodes[last_nid]

    g = graph.copy()
    input_nids = [p.buffer_id if isinstance(p, Port) else _first_port_id(p) for p in inputs]

    out_port = Port(buffer_id=last_nid)
    kernel = KernelOp(
        inputs=tuple(inputs),
        body=tuple(body),
        outputs=(out_port,),
    )

    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{last_nid}", shape=last_node.output.shape, dtype=last_node.output.dtype),
    )
    out_port.buffer_id = new_nid
    for nid in all_consumed:
        orig = graph.nodes.get(nid)
        if orig is not None and orig.hints:
            g.nodes[new_nid].hints.merge(orig.hints)

    g.replace_node(last_nid, new_nid)
    _rename_in_kernels(g, last_nid, new_nid)

    for nid in all_consumed:
        if nid in g.nodes and nid != last_nid:
            g.remove_node(nid)
    if last_nid in g.nodes:
        g.remove_node(last_nid)
    return g


# ---------------------------------------------------------------------------
# Region growing
# ---------------------------------------------------------------------------


def _grow_region(graph: Graph, seed: str) -> tuple[set[str], str] | None:
    """Grow a fusable region from seed via forward BFS + backward cone."""
    topo = graph.topological_order()
    topo_idx = {nid: i for i, nid in enumerate(topo)}

    # Phase 1: forward BFS — collect downstream fusable nodes.
    forward: set[str] = set()
    last_reduce: str | None = None
    queue: deque[str] = deque([seed])

    while queue:
        nid = queue.popleft()
        if nid in forward:
            continue
        node = graph.nodes.get(nid)
        if node is None or not isinstance(node.op, _FUSABLE):
            continue
        if isinstance(node.op, ReduceOp):
            if last_reduce is not None and not _reduce_depends_on(graph, nid, last_reduce, forward):
                continue
            last_reduce = nid
        forward.add(nid)
        for cid in graph.consumers(nid):
            cnode = graph.nodes.get(cid)
            if cnode is not None and isinstance(cnode.op, _FUSABLE):
                queue.append(cid)

    if not forward:
        return None

    # Phase 2: find output — last in topo with no forward consumers.
    sorted_fwd = sorted(forward, key=lambda n: topo_idx[n])
    output: str | None = None
    for nid in reversed(sorted_fwd):
        if not any(c in forward for c in graph.consumers(nid)):
            output = nid
            break
    if output is None:
        return None

    # Phase 3: backward cone — absorb nodes whose consumers are all in region.
    region: set[str] = set()
    for nid in reversed(sorted_fwd):
        if nid == output:
            region.add(nid)
            continue
        consumers = graph.consumers(nid)
        if consumers and all(c in region for c in consumers):
            region.add(nid)

    return (region, output) if region else None


def _reduce_depends_on(graph: Graph, reduce_nid: str, prev_reduce: str, through: set[str]) -> bool:
    """Check if reduce_nid transitively depends on prev_reduce through nodes in ``through``."""
    visited: set[str] = set()
    stack = list(graph.nodes[reduce_nid].inputs)
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)
        if nid == prev_reduce:
            return True
        if nid in through:
            node = graph.nodes.get(nid)
            if node is not None:
                stack.extend(node.inputs)
    return False


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
# Input tree construction (backward walk)
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

        if isinstance(node.op, IndexMapOp) and len(node.op.sources) == 1 and len(graph.consumers(buf_id)) == 1:
            consumed.add(buf_id)
            src_id = node.inputs[node.op.sources[0].input_idx]
            p = Port(buffer_id=src_id, indexmap=node.op)
            inputs.append(p)
            port_map[buf_id] = p
        else:
            p = Port(buffer_id=buf_id)
            inputs.append(p)
            port_map[buf_id] = p

    return inputs, port_map


def _rename_in_kernels(graph: Graph, old_id: str, new_id: str) -> None:
    """Update internal refs in already-fused KernelOps when a node is renamed."""
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


# ---------------------------------------------------------------------------
# Standalone IndexMapOp wrapping
# ---------------------------------------------------------------------------


def _wrap_indexmap_kernel(graph: Graph, node: Node) -> Graph:
    nid = node.id
    op = node.op
    assert isinstance(op, IndexMapOp)

    if len(op.sources) == 1:
        src_id = node.inputs[op.sources[0].input_idx]
        inputs: tuple = (Port(buffer_id=src_id, indexmap=op),)
    else:
        branches = []
        for src in op.sources:
            src_id = node.inputs[src.input_idx]
            branches.append(MuxBranch(input=Port(buffer_id=src_id), select=src.select))
        inputs = (Mux(branches=tuple(branches)),)

    out_port = Port(buffer_id=nid)
    kernel = KernelOp(
        inputs=inputs,
        outputs=(out_port,),
    )

    g = graph.copy()
    input_nids = [_first_port_id(inputs[0])]
    new_nid = g.add_node(
        op=kernel,
        inputs=input_nids,
        output=Tensor(name=f"kernel_{nid}", shape=tuple(op.out_shape), dtype=node.output.dtype),
    )
    out_port.buffer_id = new_nid
    g.replace_node(nid, new_nid)
    _rename_in_kernels(g, nid, new_nid)
    g.remove_node(nid)
    return g
