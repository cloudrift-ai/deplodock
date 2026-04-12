"""Automatic fusion: optimal graph partitioning via MILP.

Partitions a decomposed primitive graph into fusion regions that minimize
total data movement across region boundaries.  Each region becomes one GPU
kernel.  Uses scipy.optimize.milp for provably optimal partitioning.

Constraints:
  1. Convexity — if u and v are in the same region, every node on any
     path between them must also be in that region.
  2. Codegen — reduces on incompatible axes cannot share a region.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from deplodock.compiler.ops import (
    FusedRegionOp,
    ReshapeOp,
    TransposeOp,
)

if TYPE_CHECKING:
    from deplodock.compiler.ir import Graph


def _tensor_size(shape: tuple) -> int:
    """Compute total elements from shape, treating symbolic dims as 1."""
    return math.prod(d for d in shape if isinstance(d, int)) if shape else 1


def _is_fusible_op(op) -> bool:
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    return isinstance(op, (ElementwiseOp, ReduceOp, ReshapeOp, TransposeOp))


def _reduces_compatible(r1_id: str, r2_id: str, graph) -> bool:
    """Can two ReduceOps coexist in one fused region?

    Compatible means the codegen can tile them in the same kernel:
    - Same axis type (both row or both contraction)
    - Same pre-reduction shape (same rows/cols dimensions)
    - Connected by a data path (one feeds into the other)
    """
    from deplodock.compiler.ops import ReduceOp

    r1_node, r2_node = graph.nodes[r1_id], graph.nodes[r2_id]
    r1, r2 = r1_node.op, r2_node.op
    if not isinstance(r1, ReduceOp) or not isinstance(r2, ReduceOp):
        return True

    # Normalize axes: positive int axis on the last dim is equivalent to -1.
    def _is_row_reduce(op, node):
        if op.axis == -1:
            return True
        if isinstance(op.axis, int) and node.inputs:
            inp = node.inputs[0]
            if inp in graph.nodes:
                ndim = len(graph.nodes[inp].output.shape)
                if ndim > 0 and op.axis == ndim - 1:
                    return True
        return False

    is_row_1 = _is_row_reduce(r1, r1_node)
    is_row_2 = _is_row_reduce(r2, r2_node)

    # Different axis types → always incompatible.
    if is_row_1 != is_row_2:
        return False

    # Contractions (symbolic axis) → always incompatible with each other.
    if isinstance(r1.axis, str) and isinstance(r2.axis, str):
        return False

    # Both row reductions: compatible only if their pre-reduction
    # inputs have the same shape (they'll share the same tiled loop).
    if is_row_1 and is_row_2:
        # Compare input shapes. The first input determines the tile dimensions.
        r1_inp = r1_node.inputs[0] if r1_node.inputs else None
        r2_inp = r2_node.inputs[0] if r2_node.inputs else None
        if r1_inp and r2_inp and r1_inp in graph.nodes and r2_inp in graph.nodes:
            s1 = graph.nodes[r1_inp].output.shape
            s2 = graph.nodes[r2_inp].output.shape
            # Same number of dims and same last dim (cols must match for shared tile loop).
            if len(s1) == len(s2) and s1[-1:] == s2[-1:]:
                return True
        return False

    return False


# ---------------------------------------------------------------------------
# MILP solver
# ---------------------------------------------------------------------------


def _solve_fusion(graph: Graph) -> dict[str, int]:
    """Solve optimal fusion via MILP. Returns node_id → region_label."""
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp

    from deplodock.compiler.ops import ReduceOp

    # Collect fusible nodes.
    node_list = sorted(nid for nid, n in graph.nodes.items() if _is_fusible_op(n.op))
    if not node_list:
        return {}
    node_idx = {nid: i for i, nid in enumerate(node_list)}
    n_nodes = len(node_list)

    # Build directed edges between fusible nodes.
    direct_edges = []  # (u_idx, v_idx, tensor_bytes)
    for nid in node_list:
        for inp_id in graph.nodes[nid].inputs:
            if inp_id in node_idx:
                u, v = node_idx[inp_id], node_idx[nid]
                w = _tensor_size(graph.nodes[inp_id].output.shape)
                direct_edges.append((u, v, w))

    # Adjacency + reachability.
    children = [[] for _ in range(n_nodes)]
    for u, v, _ in direct_edges:
        children[u].append(v)

    # All-pairs reachability via reverse-topo BFS.
    in_degree = [0] * n_nodes
    for _u, v, _ in direct_edges:
        in_degree[v] += 1
    queue = [i for i in range(n_nodes) if in_degree[i] == 0]
    topo = []
    while queue:
        u = queue.pop(0)
        topo.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    reachable = [set() for _ in range(n_nodes)]
    for u in reversed(topo):
        reachable[u].add(u)
        for v in children[u]:
            reachable[u] |= reachable[v]

    # Identify incompatible reduce pairs.
    reduce_indices = [i for i, nid in enumerate(node_list) if isinstance(graph.nodes[nid].op, ReduceOp)]
    incompat_pairs = []
    for ii, ri in enumerate(reduce_indices):
        for rj in reduce_indices[ii + 1:]:
            if not _reduces_compatible(node_list[ri], node_list[rj], graph):
                incompat_pairs.append((ri, rj))

    # --- Build pair variables ---
    # We need a binary "same[u,v]" variable for each pair of nodes that
    # could potentially be in the same region. We need variables for:
    #   1. All direct edges (these carry the objective weight).
    #   2. All (u,w) pairs needed for transitivity: u→v→w implies need for (u,w).
    #   3. All pairs involving incompatible reduces and intermediate nodes.

    pair_to_var = {}  # (min_idx, max_idx) → variable_index
    var_weights = []  # objective weight for each variable

    def _get_or_create_var(a: int, b: int, weight: int = 0) -> int:
        key = (min(a, b), max(a, b))
        if key in pair_to_var:
            return pair_to_var[key]
        idx = len(var_weights)
        pair_to_var[key] = idx
        var_weights.append(weight)
        return idx

    # 1. Direct edges — carry the objective weight.
    for u, v, w in direct_edges:
        _get_or_create_var(u, v, w)
    n_direct = len(var_weights)

    # 2. Transitivity triples (u→v→w → need var for (u,w)).
    transitivity = []
    for v_node in range(n_nodes):
        v_parents = [u for u, v, _ in direct_edges if v == v_node]
        v_children = children[v_node]
        for u_node in v_parents:
            for w_node in v_children:
                i_uv = _get_or_create_var(u_node, v_node)
                i_vw = _get_or_create_var(v_node, w_node)
                i_uw = _get_or_create_var(u_node, w_node)
                transitivity.append((i_uv, i_vw, i_uw))

    # 3. Incompatible reduce pairs + all intermediate nodes.
    for ri, rj in incompat_pairs:
        _get_or_create_var(ri, rj)
        # Every node between ri and rj needs separation constraints.
        for v in range(n_nodes):
            if v == ri or v == rj:
                continue
            # v is between ri and rj if ri→...→v→...→rj or rj→...→v→...→ri.
            if (v in reachable[ri] and rj in reachable[v]) or (v in reachable[rj] and ri in reachable[v]):
                _get_or_create_var(ri, v)
                _get_or_create_var(v, rj)

    n_vars = len(var_weights)
    if n_vars == 0:
        return {nid: i for i, nid in enumerate(node_list)}

    # --- Objective: maximize Σ weight × same (only direct edges have weight) ---
    c = np.zeros(n_vars)
    for i in range(n_direct):
        c[i] = -var_weights[i]  # minimize negative = maximize

    integrality = np.ones(n_vars)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)

    # --- Constraints ---
    rows = []

    # 1. Incompatible reduces: same[ri,rj] = 0.
    for ri, rj in incompat_pairs:
        var = pair_to_var[(min(ri, rj), max(ri, rj))]
        ub[var] = 0  # force to 0

        # No node v can bridge ri and rj: same[ri,v] + same[v,rj] <= 1.
        for v in range(n_nodes):
            if v == ri or v == rj:
                continue
            if (v in reachable[ri] and rj in reachable[v]) or (v in reachable[rj] and ri in reachable[v]):
                key_iv = (min(ri, v), max(ri, v))
                key_vj = (min(v, rj), max(v, rj))
                if key_iv in pair_to_var and key_vj in pair_to_var:
                    row = np.zeros(n_vars)
                    row[pair_to_var[key_iv]] = 1
                    row[pair_to_var[key_vj]] = 1
                    rows.append((row, 1.0))

    # 2. Transitivity: same[u,v] + same[v,w] - same[u,w] <= 1.
    for i_uv, i_vw, i_uw in transitivity:
        row = np.zeros(n_vars)
        row[i_uv] = 1
        row[i_vw] = 1
        row[i_uw] = -1
        rows.append((row, 1.0))

    # 3. Convexity: for all u,w where u can reach w, and all v between them:
    #    same[u,w] <= same[u,v] AND same[u,w] <= same[v,w].
    for u in range(n_nodes):
        for w in reachable[u]:
            if u == w:
                continue
            key_uw = (min(u, w), max(u, w))
            if key_uw not in pair_to_var:
                continue
            var_uw = pair_to_var[key_uw]
            for v in reachable[u]:
                if v == u or v == w or w not in reachable[v]:
                    continue
                # v is between u and w.
                key_uv = (min(u, v), max(u, v))
                key_vw = (min(v, w), max(v, w))
                if key_uv in pair_to_var:
                    row = np.zeros(n_vars)
                    row[var_uw] = 1
                    row[pair_to_var[key_uv]] = -1
                    rows.append((row, 0.0))  # same[u,w] - same[u,v] <= 0
                if key_vw in pair_to_var:
                    row = np.zeros(n_vars)
                    row[var_uw] = 1
                    row[pair_to_var[key_vw]] = -1
                    rows.append((row, 0.0))  # same[u,w] - same[v,w] <= 0

    # Solve.
    if rows:
        A = np.vstack([r for r, _ in rows])
        b = np.array([b for _, b in rows])
        constraints = LinearConstraint(A, ub=b)
    else:
        constraints = []

    result = milp(c=c, integrality=integrality, bounds=Bounds(lb=lb, ub=ub), constraints=constraints)

    if not result.success:
        return {nid: i for i, nid in enumerate(node_list)}

    # --- Convert to region labels via connected components ---
    from collections import deque

    same = result.x
    adj = [[] for _ in range(n_nodes)]
    for (a, b), var_idx in pair_to_var.items():
        if var_idx < n_direct and same[var_idx] > 0.5:
            adj[a].append(b)
            adj[b].append(a)

    region_label = [-1] * n_nodes
    label = 0
    for start in range(n_nodes):
        if region_label[start] >= 0:
            continue
        q = deque([start])
        region_label[start] = label
        while q:
            u = q.popleft()
            for v in adj[u]:
                if region_label[v] < 0:
                    region_label[v] = label
                    q.append(v)
        label += 1

    return {node_list[i]: region_label[i] for i in range(n_nodes)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def auto_fuse(graph: Graph) -> Graph:
    """Discover optimal fusion regions via MILP and replace with FusedRegionOp nodes."""
    from deplodock.compiler.ir import Tensor

    g = graph.copy()
    region_labels = _solve_fusion(g)

    if not region_labels:
        return g

    groups: dict[int, set[str]] = {}
    for nid, label in region_labels.items():
        groups.setdefault(label, set()).add(nid)

    fused_groups = [grp for grp in groups.values() if len(grp) > 1]

    for grp in fused_groups:
        topo = [nid for nid in g.topological_order() if nid in grp]

        external_inputs: list[str] = []
        external_outputs: list[str] = []
        seen_inputs: set[str] = set()

        for nid in topo:
            node = g.nodes[nid]
            for inp_id in node.inputs:
                if inp_id not in grp and inp_id not in seen_inputs:
                    external_inputs.append(inp_id)
                    seen_inputs.add(inp_id)

        for nid in topo:
            consumers = g.consumers(nid)
            is_graph_output = nid in g.outputs
            has_external_consumer = any(c not in grp for c in consumers)
            if is_graph_output or has_external_consumer:
                external_outputs.append(nid)

        if not external_outputs:
            continue

        region_ops = []
        for nid in topo:
            node = g.nodes[nid]
            region_ops.append((nid, node.op, list(node.inputs)))

        primary_out = external_outputs[0]
        out_tensor = g.nodes[primary_out].output

        region_shapes: dict[str, tuple] = {}
        for inp_id in external_inputs:
            if inp_id in g.nodes:
                region_shapes[inp_id] = g.nodes[inp_id].output.shape
        for nid in topo:
            region_shapes[nid] = g.nodes[nid].output.shape

        fused_op = FusedRegionOp(
            region_ops=region_ops,
            input_names=external_inputs,
            output_names=external_outputs,
            shapes=region_shapes,
        )
        fused_id = g.add_node(
            op=fused_op,
            inputs=external_inputs,
            output=Tensor(
                name=f"fused_{primary_out}",
                shape=out_tensor.shape,
                dtype=out_tensor.dtype,
            ),
        )

        for nid in topo:
            if nid in graph.nodes:
                g.nodes[fused_id].hints.merge(graph.nodes[nid].hints)

        for out_id in external_outputs:
            g.replace_node(out_id, fused_id)

        for nid in reversed(topo):
            if nid in g.nodes and not g.consumers(nid):
                g.remove_node(nid)

    return g
