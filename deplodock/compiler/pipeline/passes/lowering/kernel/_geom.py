"""Shared axis-geometry helpers for the kernel materializer + the warp factorizer.

Tiny, dependency-light functions used by both ``010_materialize`` (scalar / reg / reduce
tiers) and ``_warp_factor`` (the warp/mma factorization), lifted here so the warp geometry
can move out of the materializer without duplicating them. Leading ``_`` so the pass loader
(globs ``*.py``, skips ``_``-prefixed) skips this module."""

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


__all__ = ["extent_expr", "shrink_axis"]
