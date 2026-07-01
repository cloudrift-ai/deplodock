"""Generic tiling-construction layer — ``atomize → register_tile → unit_tile → grid_tile``.

A contraction is lowered by tiling a **leaf atom** (a tensor-core mma cell or a scalar fma cell —
see ``ir/atom``) four ways: GRID block / UNIT / REGISTER / ATOM. The **UNIT** is the atom's
parallel thread footprint (``atom.lanes``) — a warp (32 lanes) for mma, a single thread for scalar —
so the tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only
in ``lanes``. This module owns that nesting (the per-cell coordinate :class:`OffsetFn`, the bound
``Tile`` axes, the splice); the atom-specific codegen is supplied to :func:`grid_tile` as three
callables (``state_decls`` / ``reduce_region`` / ``store``) — see ``_factor.reduce_codegen`` (the
shared K-loop) + ``_factor.store_sink`` (the per-cell sink). The geometry
(``tile_m`` / ``mask`` / axis names / ``block_threads`` / …) is
read off the :class:`~deplodock.compiler.ir.tile.structural.Contraction` node, not recomputed here.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips this module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import shrink_axis as _shrink_axis


@dataclass(frozen=True)
class OffsetFn:
    """The per-axis cell-offset coordinate, built up across tiling levels. ``base`` reproduces the
    ``_axis_base`` ``block·(units·reg·atom) + unit·(reg·atom) + r·atom`` once the UNIT level is
    present (the mma warp tile AND the scalar thread tile both go through ``unit_tile``), else the
    bare ``_cell_offset`` ``Var(axis)·reg + r``. Also accumulates the ``Tile`` axes (grid block →
    unit → lane) + ``block_threads``."""

    # Per axis ("m"/"n"): (atom_dim, reg, grid_block_var, unit_var | None, unit_count). ``unit_var``
    # is the UNIT-level axis var (a warp for mma, a thread for scalar).
    levels: dict = field(default_factory=dict)
    axes: tuple[Axis, ...] = ()
    block_threads: int | None = None

    def base(self, which: str, r: int):
        """The offset of register cell index ``r`` along axis ``which`` ("m"/"n")."""
        atom_dim, reg, block_var, unit_var, unit_count = self.levels[which]
        reg_term = Literal(r * atom_dim, "int")
        if unit_var is not None:  # unit present: block·(units·reg·atom) + unit·(reg·atom) + r·atom
            tile = unit_count * reg * atom_dim
            e = BinaryExpr("*", Var(block_var), Literal(tile, "int"))
            e = BinaryExpr("+", e, BinaryExpr("*", Var(unit_var), Literal(reg * atom_dim, "int")))
            return BinaryExpr("+", e, reg_term)
        # no unit level: Var(axis)·reg + r   (atom_dim == 1)
        return BinaryExpr("+", BinaryExpr("*", Var(block_var), Literal(reg, "int")), reg_term)

    def sigma(self, i: int, j: int, m_name: str, n_name: str, *, mask_m: bool, mask_n: bool, m_ext=None, n_ext=None) -> Sigma:
        """σ mapping the output axes to register cell ``(i, j)``'s real coordinate, with a masked
        axis wrapped in-bounds (``% extent``) so an overhanging cell clamp-reads."""
        smap: dict = {}
        bm, bn = self.base("m", i), self.base("n", j)
        smap[m_name] = BinaryExpr("%", bm, m_ext) if mask_m else bm
        smap[n_name] = BinaryExpr("%", bn, n_ext) if mask_n else bn
        return Sigma(smap)


@dataclass(frozen=True)
class Tiling:
    """The accumulating tiling state threaded through ``atomize → register_tile → unit_tile →
    grid_tile`` — the per-cell :class:`OffsetFn` + the register cell counts. ``grid_tile`` (the
    finalizer) splices the codegen callables' state + reduce-region + stores into the ``Tile``."""

    offset: OffsetFn
    reg_m: int = 1
    reg_n: int = 1


def atomize(atom_m: int, atom_n: int) -> Tiling:
    """The leaf: a single atom of ``atom_m × atom_n`` (1×1 for a scalar cell). Seeds the
    per-axis offset with the atom step; the atom-lane offset stays OUT of σ (added at render)."""
    levels = {
        "m": (atom_m, 1, None, None, 1),
        "n": (atom_n, 1, None, None, 1),
    }
    return Tiling(offset=OffsetFn(levels=levels))


def register_tile(t: Tiling, reg_m: int, reg_n: int) -> Tiling:
    """The REGISTER level: ``reg_m × reg_n`` atoms per thread/warp. Records the cell counts; the
    per-cell ``r·atom_dim`` term is applied at ``OffsetFn.base``."""
    levels = dict(t.offset.levels)
    for which, reg in (("m", reg_m), ("n", reg_n)):
        atom_dim, _, block_var, unit_var, unit_count = levels[which]
        levels[which] = (atom_dim, reg, block_var, unit_var, unit_count)
    return replace(t, reg_m=reg_m, reg_n=reg_n, offset=replace(t.offset, levels=levels))


def unit_tile(t: Tiling, um: int, un: int, m_u: str, n_u: str) -> Tiling:
    """The UNIT level: ``um × un`` parallel units per CTA, where a *unit* is the atom's thread
    footprint — a warp (32 lanes) for an mma atom, a single thread for a scalar atom. (So the
    tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only
    in the atom's ``lanes``; ``um``/``un`` are the warp counts ``WM``/``WN`` for mma, the thread
    counts ``par_m``/``par_n`` for scalar.) Adds the unit term ``unit·(reg·atom)`` to each axis
    offset and the ``m_u`` / ``n_u`` unit axes."""
    levels = dict(t.offset.levels)
    for which, count, var in (("m", um, m_u), ("n", un, n_u)):
        atom_dim, reg, block_var, _, _ = levels[which]
        levels[which] = (atom_dim, reg, block_var, var, count)
    axes = (*t.offset.axes, Axis(name=m_u, extent=um), Axis(name=n_u, extent=un))
    return replace(t, offset=replace(t.offset, levels=levels, axes=axes))


def grid_tile(
    t: Tiling,
    masks,
    *,
    n_axis: Axis,
    n_b: str,
    tile_n: int,
    m_axis: Axis | None = None,
    m_b: str = "",
    tile_m: int = 1,
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
    emits. The three callables (atom-specific, from ``_factor.reduce_codegen`` + the ``store`` sink) are the
    only per-atom variation; the splice is shared.

    ``m_axis is None`` is a 1-D output grid (only ``n`` tiled) — no ``m`` block axis is bound.
    ``lead_axes`` are extra outer grid axes (a batched contraction's leading dims) carried through
    untiled. ``lanes == 1`` (scalar) emits no ``_lane`` axis."""
    levels = dict(t.offset.levels)
    levels["n"] = _with_block(levels["n"], n_b)
    grid_axes: tuple[Axis, ...] = lead_axes
    if m_axis is not None:
        levels["m"] = _with_block(levels["m"], m_b)
        grid_axes = (*grid_axes, _shrink_axis(Axis(name=m_b, extent=m_axis.extent, source_axis=m_axis), tile_m))
    grid_axes = (*grid_axes, _shrink_axis(Axis(name=n_b, extent=n_axis.extent, source_axis=n_axis), tile_n))
    lane_axes = (Axis(name="_lane", extent=lanes),) if lanes > 1 else ()
    offset = replace(t.offset, levels=levels, axes=(*grid_axes, *t.offset.axes, *lane_axes), block_threads=block_threads)

    cells = [(i, j) for i in range(t.reg_m) for j in range(t.reg_n)]
    state = state_decls(cells)
    top_decls, kstmts = reduce_region(cells, offset, masks)
    stores = [s for (i, j) in cells for s in store(i, j, offset, masks)]
    return Tile(axes=offset.axes, body=Body((*state, *top_decls, *kstmts, *stores)), block_threads=block_threads)


def _with_block(level: tuple, block_var: str) -> tuple:
    atom_dim, reg, _, unit_var, unit_count = level
    return (atom_dim, reg, block_var, unit_var, unit_count)


__all__ = ["OffsetFn", "Tiling", "atomize", "grid_tile", "register_tile", "unit_tile"]
