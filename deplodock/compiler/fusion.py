"""Automatic fusion: discover fusion regions from intermediate tensor sizes.

Operates on the decomposed primitive graph. Groups adjacent ops into
fusion regions by analyzing which intermediates are large and benefit
from staying in shared memory / registers.
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


def _is_fusible_op(op) -> bool:
    """Is this a primitive op that can participate in auto-fusion?

    Only ElementwiseOp, ReduceOp, ReshapeOp, TransposeOp are fusible.
    All other ops (FusedRegionOp, etc.) stay standalone.
    """
    from deplodock.compiler.ops import ElementwiseOp, ReduceOp

    return isinstance(op, (ElementwiseOp, ReduceOp, ReshapeOp, TransposeOp))


def _can_merge(graph: Graph, uf: UnionFind, a_id: str, b_id: str) -> bool:
    """Check if merging two regions would create a cycle or shape incompatibility."""
    region_a = uf.members(a_id)
    region_b = uf.members(b_id)
    merged = region_a | region_b

    # Shape compatibility: the kernel generator indexes all 2D+ external inputs with the
    # same row*cols+j pattern. If those inputs have different total sizes, the
    # generated kernel will access out of bounds. 1D inputs (weight vectors)
    # and scalars are OK — they use [j] or [0] indexing respectively.
    #
    # Exception: 2-op matmul regions (mul + reduce_sum) are handled by the
    # backend's SGEMM path, not the kernel generator, so different input sizes are fine.
    from deplodock.compiler.ops import ElementwiseOp as _Ew
    from deplodock.compiler.ops import ReduceOp as _Red

    # Count fusible ops in the merged region.
    fusible_ops = [(nid, graph.nodes[nid].op) for nid in merged]
    is_2op_matmul = (
        len(fusible_ops) == 2
        and any(isinstance(op, _Ew) and op.fn == "mul" for _, op in fusible_ops)
        and any(isinstance(op, _Red) and op.fn == "sum" for _, op in fusible_ops)
    )

    if not is_2op_matmul:
        sizes_full: set[int] = set()
        for nid in merged:
            node = graph.nodes[nid]
            for inp_id in node.inputs:
                if inp_id not in merged and inp_id in graph.nodes:
                    inp_shape = graph.nodes[inp_id].output.shape
                    inp_size = _tensor_size(inp_shape)
                    # Skip scalars and per-row scalars (broadcastable).
                    # A (N,1) shape is a per-row scalar — the kernel generator accesses
                    # it as a scalar per row, not with row*cols+j.
                    if inp_size <= 1:
                        continue
                    if len(inp_shape) >= 2 and all((isinstance(d, int) and d == 1) for d in inp_shape[1:]):
                        continue  # shape like (N, 1) or (N, 1, 1) — per-row scalar
                    if len(inp_shape) >= 2:
                        sizes_full.add(inp_size)

        # All 2D+ external inputs (that aren't per-row scalars) must have
        # the same total size — the kernel generator uses one row*cols+j index for all.
        if len(sizes_full) > 1:
            return False

    return True


def auto_fuse(graph: Graph) -> Graph:
    """Discover fusion regions and replace them with FusedRegionOp nodes.

    Algorithm:
    1. Score each single-consumer edge by intermediate tensor size.
    2. Greedy merge: highest-score first, merge if no cycle.
    3. Structural ops (reshape, transpose) always merge with neighbors.
    4. Replace each multi-op region with a FusedRegionOp node.
    """
    from deplodock.compiler.ir import Tensor

    g = graph.copy()
    uf = UnionFind()

    # Initialize: one region per compute node.
    for nid in g.nodes:
        if _is_fusible_op(g.nodes[nid].op):
            uf.add(nid)

    # Score edges: (score, producer_id, consumer_id).
    edges: list[tuple[int, str, str]] = []
    for nid in g.topological_order():
        node = g.nodes[nid]
        if not _is_fusible_op(node.op):
            continue
        consumers = g.consumers(nid)
        if len(consumers) == 1:
            consumer_id = consumers[0]
            if _is_fusible_op(g.nodes[consumer_id].op):
                score = _tensor_size(node.output.shape)
                edges.append((score, nid, consumer_id))

    # Greedy merge: highest score first.
    for _score, producer_id, consumer_id in sorted(edges, reverse=True):
        if producer_id not in uf._parent or consumer_id not in uf._parent:
            continue
        if uf.find(producer_id) == uf.find(consumer_id):
            continue
        if _can_merge(g, uf, producer_id, consumer_id):
            uf.merge(producer_id, consumer_id)

    # Build fused regions.
    groups = uf.all_groups()

    # Only wrap multi-op groups in FusedRegionOps. Singletons stay as
    # individual ops — the backend handles them with inline kernels.
    fused_groups = [grp for grp in groups if len(grp) > 1]

    # For each fused group, replace with FusedRegionOp.
    for grp in fused_groups:
        # Order ops in the group topologically.
        topo = [nid for nid in g.topological_order() if nid in grp]

        # Find external inputs and outputs.
        external_inputs: list[str] = []
        external_outputs: list[str] = []
        seen_inputs: set[str] = set()

        for nid in topo:
            node = g.nodes[nid]
            for inp_id in node.inputs:
                if inp_id not in grp and inp_id not in seen_inputs:
                    external_inputs.append(inp_id)
                    seen_inputs.add(inp_id)

        # External outputs: group nodes consumed by nodes outside the group, or graph outputs.
        for nid in topo:
            consumers = g.consumers(nid)
            is_graph_output = nid in g.outputs
            has_external_consumer = any(c not in grp for c in consumers)
            if is_graph_output or has_external_consumer:
                external_outputs.append(nid)

        if not external_outputs:
            continue  # Dead region, skip.

        # Build region_ops: [(node_id, op, input_ids), ...]
        region_ops = []
        for nid in topo:
            node = g.nodes[nid]
            region_ops.append((nid, node.op, list(node.inputs)))

        # Determine output shape/dtype from the primary external output.
        primary_out = external_outputs[0]
        out_tensor = g.nodes[primary_out].output

        # Collect shapes for all nodes in the region + external inputs.
        region_shapes: dict[str, tuple] = {}
        for inp_id in external_inputs:
            if inp_id in g.nodes:
                region_shapes[inp_id] = g.nodes[inp_id].output.shape
        for nid in topo:
            region_shapes[nid] = g.nodes[nid].output.shape

        # Create FusedRegionOp node.
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

        # Rewire: consumers of the region's outputs now consume the fused node.
        for out_id in external_outputs:
            g.replace_node(out_id, fused_id)

        # Remove internal nodes.
        for nid in reversed(topo):
            if nid in g.nodes and not g.consumers(nid):
                g.remove_node(nid)

    return g
