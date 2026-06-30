"""Shared axis-geometry helpers for the kernel materializer + the contraction factorizer.

Tiny, dependency-light functions used by both ``010_materialize`` (scalar / reg / reduce
tiers) and ``_factor`` (the atom-generic mma/scalar factorization), lifted here so the warp
geometry can move out of the materializer without duplicating them. Leading ``_`` so the pass
loader (globs ``*.py``, skips ``_``-prefixed) skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal


def extent_expr(axis: Axis):
    """The axis's extent as an ``Expr`` — a literal int (static) or the symbolic ``Dim`` expr
    (dynamic ``seq_len``)."""
    return Literal(axis.extent.as_static(), "int") if axis.extent.is_static else axis.extent.expr


def shrink_axis(axis: Axis, reg: int) -> Axis:
    """The grid (cell) axis for a register-tiled free axis: ``ceil(E / reg)`` cells, each a
    per-thread ``reg``-wide register sub-tile. ``Dim.ceil_div`` keeps a symbolic extent
    symbolic (``(seq_len+reg-1)//reg``) so the launch grid sizes from the runtime extent."""
    if reg <= 1:
        return axis
    return Axis(name=axis.name, extent=axis.extent.ceil_div(reg), source_axis=axis.source_axis or axis)


def copy_cell(body, sigma, suffix: str, protected) -> list:
    """One copy of a tiled reduce ``body``: σ-substitute its indices (``sigma``) and suffix every
    per-copy SSA name (the shared grid / reduce / lane coordinates in ``protected`` pass through
    unrenamed). This is the **one** replication mechanic shared by the register tile (``_factor``,
    one copy per output cell ``(i, j)`` → ``__c{i}_{j}``) and the ILP register fold (``010_materialize``
    ``_reduce``, one copy per accumulator chain ``r`` → ``__r{r}``); the caller supplies the per-copy
    ``sigma`` (the coordinate offset) and ``suffix`` (the SSA tag)."""
    rename = lambda n: n if n in protected else f"{n}{suffix}"  # noqa: E731
    return [s.rewrite(rename, sigma) for s in body]


__all__ = ["copy_cell", "extent_expr", "shrink_axis"]
