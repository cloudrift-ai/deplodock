"""Shape utilities shared across passes: NumPy right-aligned broadcasting."""

from __future__ import annotations


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
