"""Shape utilities shared across passes: broadcasting + downstream propagation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from deplodock.compiler.ir import Tensor

if TYPE_CHECKING:
    from deplodock.compiler.ir import Graph


def is_broadcast_compatible(small_shape: tuple, large_shape: tuple) -> bool:
    """Check if small broadcasts to large via NumPy-style right-aligned rules.

    Aligns shapes from the right. Each dim of small must be 1 or match the
    corresponding dim of large. Symbolic (non-int) dims are skipped.
    """
    if len(small_shape) > len(large_shape):
        return False
    offset = len(large_shape) - len(small_shape)
    for i, s in enumerate(small_shape):
        large_dim = large_shape[offset + i]
        if not isinstance(s, int) or not isinstance(large_dim, int):
            continue
        if s != 1 and s != large_dim:
            return False
    return True


def broadcast_shapes(*shapes: tuple, allow_divisible: bool = False) -> tuple:
    """NumPy right-aligned broadcast over multiple shapes.

    Returns the broadcast result shape. Raises if shapes are incompatible.
    Symbolic dims are passed through (the larger of two ints wins; symbolic
    + int picks the int).

    If ``allow_divisible`` is True, GQA-style mismatched dims are accepted
    when one divides the other (the larger wins). This is required for
    contraction-style muls inside SDPA decompositions where Q and K may
    have different head counts.
    """
    if not shapes:
        return ()
    max_rank = max(len(s) for s in shapes)
    padded = [(1,) * (max_rank - len(s)) + tuple(s) for s in shapes]
    result: list[int | str] = []
    for axis in range(max_rank):
        dims = [p[axis] for p in padded]
        out_dim: int | str = 1
        for d in dims:
            if not isinstance(d, int):
                if isinstance(out_dim, int) and out_dim == 1:
                    out_dim = d
                continue
            if d == 1:
                continue
            if isinstance(out_dim, int) and out_dim == 1:
                out_dim = d
            elif out_dim == d:
                continue
            elif allow_divisible and isinstance(out_dim, int) and (d % out_dim == 0 or out_dim % d == 0):
                out_dim = max(out_dim, d)
            else:
                raise ValueError(f"Cannot broadcast shapes {shapes} at axis {axis}: {out_dim} vs {d}")
        result.append(out_dim)
    return tuple(result)


def propagate_shapes(graph: Graph, start_node_ids: Iterable[str]) -> None:
    """Re-derive output shapes for nodes downstream of start_node_ids.

    For every reachable consumer (in topological order), recompute its
    output shape by calling ``node.op.infer_output_shape(input_shapes)``.
    Continues propagation only if the shape changed.

    Inference failures (``NotImplementedError`` for ops that don't support
    it, or ``ValueError`` for inputs the op's shape rule can't handle —
    e.g. GQA-style broadcasts that violate strict NumPy rules) are silently
    skipped: the node keeps its existing shape and propagation does not
    continue past it.
    """
    dirty: set[str] = set(start_node_ids)
    topo = graph.topological_order()
    for nid in topo:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if not any(inp in dirty for inp in node.inputs):
            continue
        input_shapes = [graph.nodes[inp].output.shape for inp in node.inputs if inp in graph.nodes]
        try:
            new_shape = node.op.infer_output_shape(input_shapes)
        except (NotImplementedError, ValueError):
            continue
        if new_shape != node.output.shape:
            node.output = Tensor(name=node.output.name, shape=new_shape, dtype=node.output.dtype)
            dirty.add(nid)
