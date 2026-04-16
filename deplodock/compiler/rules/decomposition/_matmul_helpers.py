"""Shared helpers for matmul decomposition (unsqueeze for broadcast-compatible mul)."""

from __future__ import annotations

from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ops import IndexMapOp, IndexSource


def matmul_unsqueeze(a_shape: tuple, b_shape: tuple) -> tuple[IndexMapOp, IndexMapOp, tuple, int]:
    """Build IndexMapOps that unsqueeze A and B for broadcast-compatible matmul.

    A(..., M, K) → A(..., M, K, 1)   — trailing size-1 dim
    B(..., K, N) → B(..., 1, K, N)   — size-1 dim before K

    Returns (a_indexmap, b_indexmap, broadcast_shape, k_axis).
    """
    ndim_a = len(a_shape)
    ndim_b = len(b_shape)

    # A: add trailing dim.  A(..., M, K) → (..., M, K, 1)
    a_out_shape = tuple(a_shape) + (1,)
    a_coord_map = tuple(placeholder(d) for d in range(ndim_a))
    a_unsq = IndexMapOp(
        out_shape=a_out_shape,
        sources=(IndexSource(input_idx=0, coord_map=a_coord_map),),
    )

    # B: insert size-1 before last two dims.  B(..., K, N) → (..., 1, K, N)
    b_out_shape = tuple(b_shape[:-2]) + (1,) + tuple(b_shape[-2:])
    b_coord_map: list = []
    insert_pos = ndim_b - 2
    out_d = 0
    for inp_d in range(ndim_b):
        if inp_d == insert_pos:
            out_d += 1
        b_coord_map.append(placeholder(out_d))
        out_d += 1
    b_unsq = IndexMapOp(
        out_shape=b_out_shape,
        sources=(IndexSource(input_idx=0, coord_map=tuple(b_coord_map)),),
    )

    # Broadcast shape: max of each dim.
    mul_shape = tuple(
        max(a, b) if isinstance(a, int) and isinstance(b, int) else (a if isinstance(a, int) and a > 1 else b)
        for a, b in zip(a_out_shape, b_out_shape, strict=True)
    )

    return a_unsq, b_unsq, mul_shape, -2
