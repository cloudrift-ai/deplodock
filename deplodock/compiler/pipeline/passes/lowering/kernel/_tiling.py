"""Generic tiling-construction layer — ``atomize → register_tile → unit_tile → grid_tile``.

A contraction is lowered by tiling a **leaf atom** (a tensor-core mma cell or a scalar fma cell —
see ``ir/atom``) four ways: GRID block / UNIT / REGISTER / ATOM. The **UNIT** is the atom's
parallel thread footprint (``atom.lanes``) — a warp (32 lanes) for mma, a single thread for scalar —
so the tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only
in ``lanes``. This module owns that nesting (the per-cell coordinate — a per-axis :class:`AxisOffset`
pair, threaded as an :class:`Offset` — the bound ``Tile`` axes, the splice); the atom-specific
codegen is supplied to :func:`grid_tile` as three callables (``state_decls`` / ``reduce_region`` /
``store``) — see ``_atom.reduce_codegen`` (the shared K-loop) + ``_atom.store_sink`` (the per-cell
sink). The geometry (the ``(m, n)`` :class:`~deplodock.compiler.ir.tile.ir.Side` pair — tile width /
mask / block+unit var names — plus ``block_threads``) is read off the
:class:`~deplodock.compiler.ir.tile.ir.Contraction` node, not recomputed here.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips this module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import shrink_axis as _shrink_axis

if TYPE_CHECKING:
    from deplodock.compiler.ir.tile.ir import Side


@dataclass(frozen=True)
class AxisOffset:
    """One output axis's per-register-cell coordinate, accumulated across the tiling levels (atom →
    register → unit → grid). :meth:`base` reproduces ``block·(units·reg·atom) + unit·(reg·atom) +
    r·atom`` once the UNIT level is present (the mma warp tile AND the scalar thread tile both go
    through :func:`unit_tile`), else the bare ``Var(block)·reg + r``."""

    atom_dim: int  # the atom step along this axis
    reg: int = 1  # register sub-cells per unit
    block_var: str | None = None  # the grid-block axis var (set at grid_tile)
    unit_var: str | None = None  # the UNIT-level var — a warp for mma, a thread for scalar
    unit_count: int = 1

    def base(self, r: int) -> Expr:
        """The offset of register cell index ``r`` along this axis."""
        reg_term = Literal(r * self.atom_dim, "int")
        if self.unit_var is not None:  # unit present: block·(units·reg·atom) + unit·(reg·atom) + r·atom
            tile = self.unit_count * self.reg * self.atom_dim
            e = BinaryExpr("*", Var(self.block_var), Literal(tile, "int"))
            e = BinaryExpr("+", e, BinaryExpr("*", Var(self.unit_var), Literal(self.reg * self.atom_dim, "int")))
            return BinaryExpr("+", e, reg_term)
        # no unit level: Var(axis)·reg + r   (atom_dim == 1)
        return BinaryExpr("+", BinaryExpr("*", Var(self.block_var), Literal(self.reg, "int")), reg_term)


class Offset(NamedTuple):
    """The per-cell offset the codegen callables receive — the ``(m, n)`` :class:`AxisOffset` pair.
    Unpacks / zips like the ``(m, n)`` :class:`Side` pair it parallels (``m_off, n_off = offset``;
    ``zip(offset, mn)``), and its fields align: ``offset.m.base(i)`` is the row-cell offset,
    ``offset.n.base(j)`` the col-cell."""

    m: AxisOffset
    n: AxisOffset


@dataclass(frozen=True)
class Tiling:
    """The accumulating tiling state threaded through ``atomize → register_tile → unit_tile →
    grid_tile`` — the per-axis :class:`Offset` pair + the bound ``Tile`` axes (unit → grid) +
    ``block_threads``. Each level ``zip``\\ s the offset pair with the ``(m, n)`` :class:`Side` pair,
    so the two axes never split into ``*_m`` / ``*_n`` locals. ``grid_tile`` (the finalizer) splices
    the codegen callables' state + reduce-region + stores into the ``Tile``."""

    offset: Offset
    axes: tuple[Axis, ...] = ()
    block_threads: int | None = None


def atomize(atoms: tuple[int, int]) -> Tiling:
    """The leaf: a single ``(atom_m, atom_n)`` atom (1×1 for a scalar cell). Seeds the per-axis
    offset with the atom step; the atom-lane offset stays OUT of σ (added at render)."""
    return Tiling(offset=Offset(*(AxisOffset(atom_dim=a) for a in atoms)))


def register_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The REGISTER level: ``m.reg × n.reg`` atoms per thread/warp. Records the cell counts; the
    per-cell ``r·atom_dim`` term is applied at :meth:`AxisOffset.base`."""
    return replace(t, offset=Offset(*(replace(o, reg=s.reg) for o, s in zip(t.offset, mn, strict=True))))


def unit_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The UNIT level: ``m.units × n.units`` parallel units per CTA, where a *unit* is the atom's
    thread footprint — a warp (32 lanes) for an mma atom, a single thread for a scalar atom. (So the
    tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only in
    the atom's ``lanes``; the counts are the warp counts ``WM``/``WN`` for mma, the thread counts
    ``par_m``/``par_n`` for scalar.) Adds the unit term ``unit·(reg·atom)`` to each axis offset and
    the per-axis unit axes."""
    offset = Offset(*(replace(o, unit_var=s.unit, unit_count=s.units) for o, s in zip(t.offset, mn, strict=True)))
    axes = (*t.axes, *(Axis(name=s.unit, extent=s.units) for s in mn))
    return replace(t, offset=offset, axes=axes)


def grid_tile(
    t: Tiling,
    *,
    mn: tuple[Side | None, Side],
    lead_axes: tuple[Axis, ...] = (),
    block_threads: int | None,
    lanes: int = 1,
    state_decls: Callable[[list[tuple[int, int]]], list[Stmt]],
    reduce_region: Callable[..., tuple[list[Stmt], list[Stmt]]],
    store: Callable[..., list[Stmt]],
) -> Tile:
    """The GRID level + finalize: bind the block axes (the shrunk grid), set the per-axis grid
    term ``block·tile``, append any leading (e.g. batch) grid axes verbatim and — when the atom is
    warp-cooperative (``lanes > 1``) — the atom ``_lane`` axis, then splice the codegen callables'
    state + reduce-region + per-cell stores into the ``Tile``. This is the outermost stage — it
    emits. The three callables (atom-specific, from ``_atom.reduce_codegen`` + the ``store`` sink) are the
    only per-atom variation; the splice is shared. The reduce-region / store callables take the
    per-cell :class:`Offset` + the ``mn`` :class:`Side` pair (each axis' mask / extent ride the ``Side``).

    ``mn[0] is None`` is a 1-D output grid (only ``n`` tiled) — no ``m`` block axis is bound.
    ``lead_axes`` are extra outer grid axes (a batched contraction's leading dims) carried through
    untiled. ``lanes == 1`` (scalar) emits no ``_lane`` axis."""
    offset = Offset(*(replace(o, block_var=s.block) if s is not None else o for o, s in zip(t.offset, mn, strict=True)))
    block_axes = tuple(_shrink_axis(Axis(name=s.block, extent=s.axis.extent, source_axis=s.axis), s.tile) for s in mn if s is not None)
    lane_axes = (Axis(name="_lane", extent=lanes),) if lanes > 1 else ()
    axes = (*lead_axes, *block_axes, *t.axes, *lane_axes)

    cells = [(i, j) for i in range(offset.m.reg) for j in range(offset.n.reg)]
    state = state_decls(cells)
    top_decls, kstmts = reduce_region(cells, offset, mn)
    stores = [s for (i, j) in cells for s in store(i, j, offset, mn)]
    return Tile(axes=axes, body=Body((*state, *top_decls, *kstmts, *stores)), block_threads=block_threads)


__all__ = ["AxisOffset", "Offset", "Tiling", "atomize", "grid_tile", "register_tile", "unit_tile"]
