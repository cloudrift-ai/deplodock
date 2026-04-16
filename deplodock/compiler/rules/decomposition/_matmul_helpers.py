"""Shared helpers for matmul decomposition (unsqueeze for broadcast-compatible mul)."""

from __future__ import annotations

from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.tensor import IndexMapOp, IndexSource


def matmul_unsqueeze(a_shape: tuple, b_shape: tuple) -> tuple[IndexMapOp, IndexMapOp, tuple, int]:
    """Build IndexMapOps that unsqueeze A and B for broadcast-compatible matmul.

    A(..., M, K) → A(..., M, K, 1)   — trailing size-1 dim
    B(..., K, N) → B(..., 1, K, N)   — size-1 dim before K

    When A and B have different ndim, the shorter is left-padded with
    size-1 dims (NumPy right-alignment) before the unsqueeze.

    Returns (a_indexmap, b_indexmap, broadcast_shape, k_axis).
    """
    # Left-pad the shorter shape with 1s so both have the same ndim.
    ndim_max = max(len(a_shape), len(b_shape))
    a_padded = (1,) * (ndim_max - len(a_shape)) + tuple(a_shape)
    b_padded = (1,) * (ndim_max - len(b_shape)) + tuple(b_shape)

    a_pad_count = ndim_max - len(a_shape)
    b_pad_count = ndim_max - len(b_shape)

    # A: add trailing dim.  A_padded(..., M, K) → (..., M, K, 1)
    a_out_shape = a_padded + (1,)
    # coord_map maps each of A's original dims from the padded output.
    # Padded dims map to Literal(0); original dims map to placeholder(pad + d).
    a_coord_map = []
    for d in range(len(a_shape)):
        a_coord_map.append(placeholder(a_pad_count + d))
    a_unsq = IndexMapOp(
        out_shape=a_out_shape,
        sources=(IndexSource(input_idx=0, coord_map=tuple(a_coord_map)),),
    )

    # B: insert size-1 before last two dims.
    # B_padded(..., K, N) → (..., 1, K, N)
    b_out_shape = b_padded[:-2] + (1,) + b_padded[-2:]
    # coord_map for B: maps each of B's original dims from the padded+unsqueezed output.
    # The insertion point in the padded shape is at position ndim_max - 2.
    b_coord_map = []
    out_d = b_pad_count  # start after the padded-1 dims
    insert_pos_in_orig = len(b_shape) - 2
    for inp_d in range(len(b_shape)):
        if inp_d == insert_pos_in_orig:
            out_d += 1  # skip the inserted size-1 dim
        b_coord_map.append(placeholder(out_d))
        out_d += 1
    b_unsq = IndexMapOp(
        out_shape=b_out_shape,
        sources=(IndexSource(input_idx=0, coord_map=tuple(b_coord_map)),),
    )

    # Broadcast shape: max of each dim (they should be compatible now:
    # non-matching dims are size-1 from padding or unsqueeze).
    mul_shape = tuple(
        max(a, b) if isinstance(a, int) and isinstance(b, int) else (a if isinstance(a, int) and a > 1 else b)
        for a, b in zip(a_out_shape, b_out_shape, strict=True)
    )

    return a_unsq, b_unsq, mul_shape, -2
