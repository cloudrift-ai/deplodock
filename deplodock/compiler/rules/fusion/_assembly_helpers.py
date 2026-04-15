"""Shared helpers for rule-based KernelOp assembly.

All fusion rules that need graph-level checks (fan-out, convexity, shape
compat) go through these utilities.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Literal

from deplodock.compiler.ir import Graph, Node
from deplodock.compiler.ops import (
    ContractionCore,
    KernelOp,
    ReduceOp,
    ReduceStage,
)
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


def shapes_match_for_contraction(a_shape: tuple, b_shape: tuple) -> tuple[bool, int, int, int, tuple, int, int, int]:
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
        m = math.prod(d for d in a_shape[:-1] if isinstance(d, int)) if any(isinstance(d, int) for d in a_shape[:-1]) else 1
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


# ---------------------------------------------------------------------------
# Reduce-axis compatibility (ported from fusion.py::_can_merge)
# ---------------------------------------------------------------------------


def is_row_reduce(reduce_op: ReduceOp, input_ndim: int) -> bool:
    """True when a ReduceOp targets the last axis of its input (row reduce)."""
    axis = reduce_op.axis
    if axis == -1:
        return True
    if isinstance(axis, int) and input_ndim > 0:
        return axis == input_ndim - 1
    return False


def reduces_compatible(
    r1: ReduceOp,
    r1_input_shape: tuple,
    r2: ReduceOp,
    r2_input_shape: tuple,
) -> bool:
    """Can two ReduceOps coexist in one kernel? (ported from fusion.py)."""
    is_row_1 = is_row_reduce(r1, len(r1_input_shape))
    is_row_2 = is_row_reduce(r2, len(r2_input_shape))

    if is_row_1 != is_row_2:
        return False  # mixed axes — incompatible

    if isinstance(r1.axis, str) and isinstance(r2.axis, str):
        return False  # different contractions — incompatible

    if is_row_1 and is_row_2:
        # Same-axis row reductions: compatible if same trailing dim + same rank.
        if len(r1_input_shape) == len(r2_input_shape) and r1_input_shape[-1:] == r2_input_shape[-1:]:
            return True
        return False

    return False


# ---------------------------------------------------------------------------
# KernelOp introspection (used by 055_merge_kernels)
# ---------------------------------------------------------------------------


KernelKind = Literal["pointwise", "reduce", "contraction"]


def kernel_kind(kernel: KernelOp) -> KernelKind:
    """Classify a KernelOp by its core type."""
    if isinstance(kernel.core, ContractionCore):
        return "contraction"
    if isinstance(kernel.core, tuple) and kernel.core:
        return "reduce"
    return "pointwise"


def kernel_reduces_with_input_shapes(kernel: KernelOp) -> list[tuple[ReduceOp, tuple]]:
    """Collect (ReduceOp, reduce-input-shape) for every reduce inside ``kernel``.

    Walks ContractionCore.reduce and tuple[ReduceStage] cores. The
    reduce-input shape is the shape of the node that flows into the
    reduce (its inputs[0]) — needed for row-reduce/axis compatibility
    checks.
    """
    result: list[tuple[ReduceOp, tuple]] = []
    if isinstance(kernel.core, ContractionCore):
        if kernel.core.reduce is not None:
            r_node = kernel.core.reduce
            r_op = r_node.op
            if isinstance(r_op, ReduceOp):
                # The reduce consumes the mul output.
                in_shape = tuple(kernel.core.mul.output.shape) if kernel.core.mul is not None else tuple(r_node.output.shape)
                result.append((r_op, in_shape))
    elif isinstance(kernel.core, tuple):
        # ReduceStage tuple — input to each reduce is either the last pre_op,
        # the previous stage's reduce output, or the prologue's last node.
        for stage_idx, stage in enumerate(kernel.core):
            if not isinstance(stage, ReduceStage):
                continue
            r_node = stage.reduce
            if r_node is None:
                continue
            r_op = r_node.op
            if not isinstance(r_op, ReduceOp):
                continue
            # Locate the input shape feeding this reduce.
            in_shape: tuple = ()
            if stage.pre_ops:
                in_shape = tuple(stage.pre_ops[-1].output.shape)
            elif stage_idx > 0 and isinstance(kernel.core[stage_idx - 1], ReduceStage):
                prev = kernel.core[stage_idx - 1].reduce
                if prev is not None:
                    in_shape = tuple(prev.output.shape)
            elif kernel.prologue:
                in_shape = tuple(kernel.prologue[-1].output.shape)
            else:
                # External input — guess via reduce node's input id.
                in_shape = tuple(r_node.output.shape)
            result.append((r_op, in_shape))
    return result


def kernel_has_contraction(kernel: KernelOp) -> bool:
    return isinstance(kernel.core, ContractionCore)


def kernel_last_node_id(kernel: KernelOp) -> str | None:
    """Return the id of the node producing this kernel's external output.

    Walks epilogue → core → prologue in priority order. Used by 055 to
    rewire downstream-kernel input references from the producer kernel's
    outer-graph id to the actual final-node id.
    """
    if kernel.epilogue:
        return kernel.epilogue[-1].id
    if isinstance(kernel.core, ContractionCore):
        if kernel.core.reduce is not None:
            return kernel.core.reduce.id
        if kernel.core.mul is not None:
            return kernel.core.mul.id
    if isinstance(kernel.core, tuple) and kernel.core:
        last_stage = kernel.core[-1]
        if isinstance(last_stage, ReduceStage) and last_stage.reduce is not None:
            return last_stage.reduce.id
    if kernel.prologue:
        return kernel.prologue[-1].id
    return None


def rewire_node_input(node: Node, old_id: str, new_id: str) -> Node:
    """Return a copy of ``node`` with one input id remapped."""
    from deplodock.compiler.ir import Tensor

    return Node(
        id=node.id,
        op=node.op,
        inputs=[new_id if i == old_id else i for i in node.inputs],
        output=Tensor(
            name=node.output.name,
            shape=tuple(node.output.shape),
            dtype=node.output.dtype,
        ),
    )


# ---------------------------------------------------------------------------
# Tensor sizing (small helper for shape compat checks)
# ---------------------------------------------------------------------------


def tensor_size(shape: tuple) -> int:
    """Total int-dim element count (symbolic dims treated as 1)."""
    return math.prod(d for d in shape if isinstance(d, int)) if shape else 1


def merged_external_inputs_compat(
    external_shapes: list[tuple],
    *,
    is_pure_contraction: bool = False,
) -> bool:
    """Replicate fusion.py::_can_merge's broadcast/2D-pair shape check.

    Reject when two non-broadcast-compatible 2D+ inputs disagree on size.
    Skip the check entirely for pure-contraction kernels (they use A/B
    indexing).
    """
    if is_pure_contraction:
        return True
    ext: list[tuple[tuple, int]] = []
    for shape in external_shapes:
        sz = tensor_size(shape)
        if sz <= 1:
            continue
        if len(shape) == 1:
            continue
        last_dim = shape[-1] if shape else 1
        if isinstance(last_dim, int) and last_dim == 1:
            continue
        ext.append((shape, sz))
    if not ext:
        return True
    max_size = max(s for _, s in ext)
    max_shape = next(sh for sh, s in ext if s == max_size)
    sizes_full: set[int] = set()
    for shape, sz in ext:
        if sz == max_size:
            sizes_full.add(sz)
        elif is_broadcast_compatible(shape, max_shape):
            continue
        else:
            sizes_full.add(sz)
    return len(sizes_full) <= 1
