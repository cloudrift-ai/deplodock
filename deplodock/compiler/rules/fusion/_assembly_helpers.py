"""Shared helpers for rule-based KernelOp assembly.

All fusion rules that need graph-level checks (fan-out, convexity, shape
compat) go through these utilities.
"""

from __future__ import annotations

from collections import deque

from deplodock.compiler.ir import Graph, Node
from deplodock.compiler.shape_utils import is_broadcast_compatible


def fan_out_of(graph: Graph, node_id: str) -> int:
    """Return the number of distinct consumers of ``node_id``."""
    return len(graph.consumers(node_id))


def is_convex_merge(graph: Graph, inside_ids: set[str], producer_id: str) -> bool:
    """Check that absorbing ``producer_id`` into a region ``inside_ids``
    preserves convexity — i.e. no node outside both ``inside_ids`` and the
    producer lies on a path from the producer's inputs to the region.
    """
    if producer_id in inside_ids:
        return True
    new_region = inside_ids | {producer_id}
    # Walk forward from each input of producer; any consumer that is
    # outside new_region but reaches new_region creates a cycle/concavity.
    producer_node = graph.nodes[producer_id]
    for inp_id in producer_node.inputs:
        if inp_id not in graph.nodes:
            continue
        # BFS forward: can we reach new_region through nodes NOT in new_region?
        visited: set[str] = set()
        queue: deque[str] = deque(graph.consumers(inp_id))
        while queue:
            cur = queue.popleft()
            if cur in visited:
                continue
            visited.add(cur)
            if cur in new_region:
                continue
            # cur is outside new_region; if any of its consumers is in new_region,
            # then absorbing producer creates a non-convex region.
            for c in graph.consumers(cur):
                if c in new_region:
                    return False
                queue.append(c)
    return True


def shapes_match_for_contraction(
    a_shape: tuple, b_shape: tuple
) -> tuple[bool, int, int, int, tuple, int, int, int]:
    """Validate matmul shape relationship + extract dim metadata.

    Returns (ok, M, N, K, batch_dims, batch_size, a_batch_group, b_batch_group).
    Ported from the deleted legacy _detect_contraction dim math.
    """
    import math

    if len(a_shape) < 2 or len(b_shape) < 2:
        return False, 0, 0, 0, (), 1, 1, 1

    batch_dims: tuple = ()
    batch_size = 1
    a_batch_group = 1
    b_batch_group = 1
    if len(a_shape) > 2 and len(b_shape) > 2:
        a_batch = a_shape[:-2]
        b_batch = b_shape[:-2]
        if a_batch == b_batch:
            batch_dims = a_batch
        else:
            max_len = max(len(a_batch), len(b_batch))
            a_padded = (1,) * (max_len - len(a_batch)) + a_batch
            b_padded = (1,) * (max_len - len(b_batch)) + b_batch
            merged: list[int] = []
            for ad, bd in zip(a_padded, b_padded, strict=True):
                if not isinstance(ad, int) or not isinstance(bd, int):
                    return False, 0, 0, 0, (), 1, 1, 1
                if ad == bd:
                    merged.append(ad)
                elif ad > bd and bd > 0 and ad % bd == 0:
                    merged.append(ad)
                elif bd > ad and ad > 0 and bd % ad == 0:
                    merged.append(bd)
                else:
                    return False, 0, 0, 0, (), 1, 1, 1
            batch_dims = tuple(merged)
            a_bs = math.prod(d for d in a_padded if isinstance(d, int))
            b_bs = math.prod(d for d in b_padded if isinstance(d, int))
            if a_bs >= b_bs and b_bs > 0:
                b_batch_group = a_bs // b_bs
            elif a_bs > 0:
                a_batch_group = b_bs // a_bs
        batch_size = math.prod(d for d in batch_dims if isinstance(d, int))
        a_k = a_shape[-1]
        b_k = b_shape[-2]
    else:
        a_k = a_shape[-1]
        b_k = b_shape[0]

    if a_k != b_k:
        return False, 0, 0, 0, (), 1, 1, 1

    if batch_dims:
        m = a_shape[-2] if isinstance(a_shape[-2], int) else 1
    else:
        m = (
            math.prod(d for d in a_shape[:-1] if isinstance(d, int))
            if any(isinstance(d, int) for d in a_shape[:-1])
            else 1
        )
    k = a_k if isinstance(a_k, int) else 1
    n = b_shape[-1] if isinstance(b_shape[-1], int) else 1

    return True, m, n, k, batch_dims, batch_size, a_batch_group, b_batch_group


def shape_of(graph: Graph, node_id: str) -> tuple:
    """Return the output shape of ``node_id`` (empty tuple if unknown)."""
    node = graph.nodes.get(node_id)
    if node is None:
        return ()
    return tuple(node.output.shape)


def copy_node(node: Node) -> Node:
    """Deep-copy a Node to detach from its outer-graph identity."""
    from deplodock.compiler.ir import Tensor

    return Node(
        id=node.id,
        op=node.op,
        inputs=list(node.inputs),
        output=Tensor(
            name=node.output.name,
            shape=tuple(node.output.shape),
            dtype=node.output.dtype,
        ),
    )


def broadcast_compat(external_shapes: list[tuple], output_shape: tuple) -> bool:
    """Check every external input broadcasts to output_shape."""
    return all(is_broadcast_compatible(s, output_shape) for s in external_shapes)
