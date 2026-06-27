"""Structural algebra predicate over a reduce loop.

The algebra of a kernel is **already in the body** — the carrier (``Accum`` /
``Monoid`` + ``Twist`` = the ⊕ fold and its twist) plus the partial structure
(is the fold fed by a ⊗-product over several contraction operands? = a semiring
contraction). Passes read that structure directly; there is no separate
``AlgebraKind`` tag to keep in sync (it was a derived cache with one consumer, so
it was dropped — the body is the single source of truth).

This module holds the one structural question the schedule still asks: **is this
reduce loop a contraction (matmul-shaped)?** — ≥ 2 distinct K-indexed operands
folded by a carrier. ``lowering/tile/010_recognize`` uses it to keep a matmul's
``Accum`` an ``Accum`` (rather than degenerate-monoidizing it like a plain
reduce); the future mma-atom tier reads the same predicate to pick the
tensor-core cell.
"""

from __future__ import annotations

from deplodock.compiler.ir.stmt.base import ReduceCarrier
from deplodock.compiler.ir.stmt.leaves import Load


def matmul_reduce(loop) -> bool:
    """True iff ``loop`` is a reduce loop whose body matches the matmul
    signature: ≥ 2 distinct buffers with K-indexed Loads (K = ``loop.axis.name``)
    plus at least one :class:`ReduceCarrier` (``Accum`` / ``Monoid`` / ``Mma``).

    Duck-typed on ``.is_reduce`` / ``.axis`` / ``.body`` so it serves Loop-IR
    ``Loop`` / ``StridedLoop`` alike (the caller restricts the type)."""
    if not getattr(loop, "is_reduce", False):
        return False
    k_name = loop.axis.name
    bufs = {ld.input for ld in loop.body.of_type(Load) if k_name in {v for e in ld.index for v in e.free_vars()}}
    if len(bufs) < 2:
        return False
    return any(isinstance(s, ReduceCarrier) for s in loop.body)
