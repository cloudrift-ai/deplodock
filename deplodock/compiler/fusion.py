"""Automatic fusion: discover fusion regions from intermediate tensor sizes.

Operates on the decomposed primitive graph. Groups adjacent ops into
fusion regions by analyzing which intermediates are large and benefit
from staying in shared memory / registers.

Uses a greedy merge with codegen validation: merges highest-value edges
first, allowing multi-consumer edges when the result is a valid codegen
region (convex subgraph with compatible reduce axes).
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


class UnionFind:
    """Disjoint set for tracking fusion regions."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
        self._rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: str) -> str:
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def merge(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def members(self, x: str) -> set[str]:
        root = self.find(x)
        return {k for k in self._parent if self.find(k) == root}

    def all_groups(self) -> list[set[str]]:
        groups: dict[str, set[str]] = {}
        for k in self._parent:
            root = self.find(k)
            groups.setdefault(root, set()).add(k)
        return list(groups.values())


def _tensor_size(shape: tuple) -> int:
    """Compute total elements from shape, treating symbolic dims as 1."""
    return math.prod(d for d in shape if isinstance(d, int)) if shape else 1


def _is_fusible_op(op, node=None) -> bool:
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    return isinstance(op, (ElementwiseOp, ReduceOp, ReshapeOp, TransposeOp))


def _is_contraction_op(nid: str, graph) -> bool:
    """Is this node part of a contraction pattern (matmul mul or reduce)?

    Detects both symbolic-dim (traced from PyTorch) and concrete-dim patterns.
    A contraction mul has MORE output dims than any single input (broadcasting).
    """
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    node = graph.nodes[nid]
    op = node.op

    if isinstance(op, ElementwiseOp) and op.fn == "mul":
        # Symbolic contraction: output has symbolic dims.
        if any(isinstance(d, str) for d in node.output.shape):
            return True
        # Concrete contraction: output has more dims than ALL inputs
        # (broadcast mul: A(M,K) × B(K,N) → AB(M,K,N)).
        # Regular broadcast mul (norm * weight) only exceeds one input's ndim.
        out_ndim = len(node.output.shape)
        inp_ndims = [len(graph.nodes[inp_id].output.shape) for inp_id in node.inputs if inp_id in graph.nodes]
        if inp_ndims and all(out_ndim > nd for nd in inp_ndims):
            return True

    if isinstance(op, ReduceOp):
        # Symbolic axis.
        if isinstance(op.axis, str):
            return True
        # Concrete axis: check if input is a contraction mul.
        for inp_id in node.inputs:
            if inp_id in graph.nodes and _is_contraction_op(inp_id, graph):
                return True

    return False


def _reduces_compatible(r1_id: str, r2_id: str, graph) -> bool:
    """Can two ReduceOps coexist in one fused region?"""
    from deplodock.compiler.ops import ReduceOp

    r1_node, r2_node = graph.nodes[r1_id], graph.nodes[r2_id]
    r1, r2 = r1_node.op, r2_node.op
    if not isinstance(r1, ReduceOp) or not isinstance(r2, ReduceOp):
        return True

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

    if is_row_1 != is_row_2:
        return False  # Mixed axes — incompatible.

    if isinstance(r1.axis, str) and isinstance(r2.axis, str):
        return False  # Different contractions — incompatible.

    if is_row_1 and is_row_2:
        # Same-axis row reductions: compatible if same input shape.
        r1_inp = r1_node.inputs[0] if r1_node.inputs else None
        r2_inp = r2_node.inputs[0] if r2_node.inputs else None
        if r1_inp and r2_inp and r1_inp in graph.nodes and r2_inp in graph.nodes:
            s1 = graph.nodes[r1_inp].output.shape
            s2 = graph.nodes[r2_inp].output.shape
            if len(s1) == len(s2) and s1[-1:] == s2[-1:]:
                return True
        return False

    return False


def _can_merge(graph, uf, a_id, b_id) -> bool:
    """Check if merging two regions is valid: convex + compatible reduces."""
    from deplodock.compiler.ops import ReduceOp

    region_a = uf.members(a_id)
    region_b = uf.members(b_id)
    merged = region_a | region_b

    # 1. Convexity: the merged region must be a convex subgraph.
    # If there's a path from any node in merged to another node in merged
    # that goes through a node NOT in merged, the region is non-convex.
    # Check: for every pair (u in merged, v in merged), all nodes on
    # any path u→...→v must also be in merged.
    # Efficient check: compute the set of all nodes reachable from merged
    # that can also reach merged — if any such node is not in merged, fail.
    topo = graph.topological_order()

    # Forward reachability from merged.
    fwd = set(merged)
    for nid in topo:
        if nid in fwd:
            for c in graph.consumers(nid):
                if c in graph.nodes:
                    fwd.add(c)

    # Backward reachability from merged.
    bwd = set(merged)
    for nid in reversed(topo):
        if nid in bwd:
            for inp in graph.nodes[nid].inputs:
                if inp in graph.nodes:
                    bwd.add(inp)

    # Nodes in both fwd and bwd but not in merged are "between" nodes
    # that would make the region non-convex.
    between = (fwd & bwd) - merged
    # Filter to only fusible nodes (InputOp/ConstantOp are shared and don't cause cycles).
    between_fusible = {n for n in between if _is_fusible_op(graph.nodes[n].op, graph.nodes[n])}
    if between_fusible:
        return False  # Non-convex — would create a cycle.

    # 2. Contraction isolation: a contraction region must contain the
    #    contraction mul + reduce pair. Additional ops are allowed only if
    #    they are pointwise epilogue (ElementwiseOp) consuming the contraction
    #    output or external scalars/1D vectors (e.g. bias add, activation).
    has_contraction = any(_is_contraction_op(nid, graph) for nid in merged)
    if has_contraction:
        from deplodock.compiler.ops import ElementwiseOp as _EpiEwOp

        contraction_ids = {nid for nid in merged if _is_contraction_op(nid, graph)}
        non_contraction = [nid for nid in merged if nid not in contraction_ids]
        if non_contraction:
            # All non-contraction ops must be pointwise ElementwiseOp.
            for nid in non_contraction:
                if not isinstance(graph.nodes[nid].op, _EpiEwOp):
                    return False
            # Find contraction output shape (the ReduceOp node's output).
            contraction_out_shape = None
            for cid in contraction_ids:
                if isinstance(graph.nodes[cid].op, ReduceOp):
                    contraction_out_shape = graph.nodes[cid].output.shape
                    break

            # Each non-contraction op may only consume: contraction output,
            # other epilogue ops, or external inputs that are indexable in the
            # epilogue (scalars, 1D vectors, or 2D tensors matching the
            # contraction output shape).
            valid_sources = contraction_ids | set(non_contraction)
            for nid in non_contraction:
                for inp_id in graph.nodes[nid].inputs:
                    if inp_id in valid_sources:
                        continue
                    if inp_id not in graph.nodes:
                        continue  # external input (InputOp) — checked by shape compat
                    inp_shape = graph.nodes[inp_id].output.shape
                    inp_size = _tensor_size(inp_shape)
                    if inp_size <= 1 or len(inp_shape) <= 1:
                        continue  # scalar or 1D vector (bias)
                    # Allow 2D inputs matching contraction output shape only
                    # if they are graph-level inputs (InputOp/ConstantOp), not
                    # intermediate results from other fused regions.
                    if contraction_out_shape and inp_shape == contraction_out_shape:
                        if not _is_fusible_op(graph.nodes[inp_id].op, graph.nodes[inp_id]):
                            continue  # graph-level input (residual connection)
                    return False  # incompatible external in epilogue — reject

    # 3. Multi-reduce: allow compatible row reduces (e.g. softmax max+sum).
    #    Reject if any reduce is a contraction reduce or reduces are incompatible.
    reduce_ids = [nid for nid in merged if isinstance(graph.nodes[nid].op, ReduceOp)]
    if len(reduce_ids) > 1:
        if has_contraction:
            return False  # Contraction + multi-reduce — not yet supported.
        if not all(_reduces_compatible(reduce_ids[0], rid, graph) for rid in reduce_ids[1:]):
            return False

    # 4. Shape compatibility: all 2D+ external inputs must have the same total
    #    size (the kernel uses a single row*cols+j index for all).
    #    Only exempt PURE 2-op contraction regions (exactly mul + reduce)
    #    which use dedicated A/B indexing.
    is_pure_contraction = has_contraction

    # 5. Dimensionality: reject ops whose output has MORE non-trivial dims
    #    than all inputs (broadcast expansion like matmul's A(M,K)×B(K,N)→(M,K,N)).
    #    Same-rank >2D ops are fine — the codegen flattens leading dims into
    #    rows and keeps the last dim as cols (e.g. (1,28,32,32) → rows=896, cols=32).
    for nid in merged:
        node = graph.nodes[nid]
        shape = node.output.shape
        concrete_dims = [d for d in shape if isinstance(d, int) and d > 1]
        if len(concrete_dims) > 2:
            # Check if this op expanded rank beyond its inputs (broadcast).
            inp_max_concrete = 0
            for inp_id in node.inputs:
                if inp_id in graph.nodes:
                    inp_concrete = [d for d in graph.nodes[inp_id].output.shape if isinstance(d, int) and d > 1]
                    inp_max_concrete = max(inp_max_concrete, len(inp_concrete))
            if len(concrete_dims) > inp_max_concrete and inp_max_concrete > 0:
                # Allow matmul broadcasts that expand by exactly one dim.
                # Standard 2D: (M,K)×(K,N)→(M,K,N) — 2D→3D
                # Batched: (B,M,K)×(B,K,N)→(B,M,K,N) — 3D→4D
                if len(concrete_dims) == inp_max_concrete + 1:
                    continue  # Matmul broadcast (expands by 1 dim) — OK
                return False  # Multi-dim expansion — reject

    if not is_pure_contraction:
        sizes_full: set[int] = set()
        for nid in merged:
            node = graph.nodes[nid]
            for inp_id in node.inputs:
                if inp_id not in merged and inp_id in graph.nodes:
                    inp_shape = graph.nodes[inp_id].output.shape
                    inp_size = _tensor_size(inp_shape)
                    if inp_size <= 1:
                        continue
                    if len(inp_shape) == 1:
                        continue  # 1D vector — uses [j] indexing
                    # Per-row scalar: last dim is 1 (e.g., (N,1) or (1,28,32,1)).
                    # These are indexed by row only, not by column.
                    last_dim = inp_shape[-1] if inp_shape else 1
                    if isinstance(last_dim, int) and last_dim == 1:
                        continue
                    sizes_full.add(inp_size)
        if len(sizes_full) > 1:
            return False

    # 5. Single-output: the merged region must have at most one external output.
    # Multi-output fused regions require infrastructure changes in plan.py/backend.py.
    external_outputs = []
    for nid in merged:
        consumers = graph.consumers(nid)
        is_graph_output = nid in graph.outputs
        has_external_consumer = any(c not in merged for c in consumers)
        if is_graph_output or has_external_consumer:
            external_outputs.append(nid)
    if len(external_outputs) > 1:
        return False

    return True


def auto_fuse(graph: Graph) -> Graph:
    """Discover fusion regions and replace them with FusedRegionOp nodes.

    Algorithm:
    1. Score ALL edges (including multi-consumer) by intermediate tensor size.
    2. Greedy merge: highest-score first, merge if convex + reduce-compatible.
    3. Replace each multi-op region with a FusedRegionOp node.
    """
    from deplodock.compiler.ir import Tensor

    g = graph.copy()
    uf = UnionFind()

    for nid in g.nodes:
        if _is_fusible_op(g.nodes[nid].op, g.nodes[nid]):
            uf.add(nid)

    # Score ALL edges between fusible nodes (not just single-consumer).
    edges: list[tuple[int, str, str]] = []
    for nid in g.topological_order():
        node = g.nodes[nid]
        if not _is_fusible_op(node.op, node):
            continue
        for consumer_id in g.consumers(nid):
            if _is_fusible_op(g.nodes[consumer_id].op, g.nodes[consumer_id]):
                score = _tensor_size(node.output.shape)
                edges.append((score, nid, consumer_id))

    # Greedy merge: highest score first. Iterate until no more merges possible
    # (later passes may unlock merges blocked by multi-output constraints).
    sorted_edges = sorted(edges, reverse=True)
    changed = True
    while changed:
        changed = False
        for _score, producer_id, consumer_id in sorted_edges:
            if producer_id not in uf._parent or consumer_id not in uf._parent:
                continue
            if uf.find(producer_id) == uf.find(consumer_id):
                continue
            if _can_merge(g, uf, producer_id, consumer_id):
                uf.merge(producer_id, consumer_id)
                changed = True

    # Build fused regions.
    # Compute the topo order ONCE before any rewiring.
    global_topo = g.topological_order()
    groups = uf.all_groups()
    fused_groups = [grp for grp in groups if len(grp) > 1]

    for grp in fused_groups:
        topo = [nid for nid in global_topo if nid in grp]

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

        region_ops = [(nid, g.nodes[nid].op, list(g.nodes[nid].inputs)) for nid in topo]
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
