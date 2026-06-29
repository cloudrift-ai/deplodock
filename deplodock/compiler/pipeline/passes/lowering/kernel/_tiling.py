"""Generic tiling-construction layer — ``atomize → register_tile → unit_tile → grid_tile``.

The kernel materializer builds a contraction by tiling a **leaf atom** (a tensor-core mma cell or
a scalar fma cell — see ``ir/tile/atom``) four ways: GRID block / UNIT / REGISTER / ATOM. The
**UNIT** is the atom's parallel thread footprint (``atom.lanes``) — a warp (32 lanes) for mma, a
single thread for scalar — so the tensor-core warp tile and the scalar parallel thread-tile are the
same level, differing only in ``lanes``. This module makes that nesting composable and
**unit-generic**: a :class:`Unit` realizes one output cell (its state decl, operands, compute,
store, and reduce-region), and the tiling functions wrap it level by level, building the per-cell
coordinate offset incrementally (one :class:`OffsetFn` reproduces both the warp ``_axis_base`` and
the scalar ``_cell_offset``).

The reuse is real but **moderate**: ``register_tile`` is generic for the cell grid, the offset,
the masks/guards, the state decls, and the stores. The **reduce loop + operand staging live in
the Unit** (``reduce_region``) — staging is CTA-cooperative (the slab fill spans all warps), so
it can't be a per-cell primitive. So each ``Unit`` still owns its reduce-region strategy.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips this module."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import shrink_axis as _shrink_axis


@dataclass(frozen=True)
class Operand:
    """A contraction operand + the free-axis set its index depends on. ``register_tile`` emits
    each operand once per distinct coordinate of ``axes`` — ``{m}`` → once per register row
    ``i`` (shared across the ``n`` cells), ``{n}`` → once per col ``j``, ``{m, n}`` → per cell,
    ``{}`` → once. This one rule is both the warp tier's structural "A per-i / B per-j" and the
    scalar tier's syntactic load dedup."""

    load: Load
    role: str  # "a" | "b"
    axes: frozenset[str]  # the grid output axes the index carries


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
        atom_dim, reg, block_var, warp_var, warp_count = self.levels[which]
        reg_term = Literal(r * atom_dim, "int")
        if warp_var is not None:  # unit present: block·(units·reg·atom) + unit·(reg·atom) + r·atom
            tile = warp_count * reg * atom_dim
            e = BinaryExpr("*", Var(block_var), Literal(tile, "int"))
            e = BinaryExpr("+", e, BinaryExpr("*", Var(warp_var), Literal(reg * atom_dim, "int")))
            return BinaryExpr("+", e, reg_term)
        # scalar tier: Var(axis)·reg + r   (atom_dim == 1)
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
    grid_tile``. ``build`` (called by ``grid_tile``, the finalizer) splices the unit's state +
    reduce-region + stores into the ``Tile``."""

    unit: Unit
    offset: OffsetFn
    reg_m: int = 1
    reg_n: int = 1


class Unit:
    """A leaf cell realization — a tensor-core mma cell or a scalar fma cell. The tiling layer is
    generic over this; two impls live in ``_warp_factor`` (``AtomUnit``) and ``_scalar_factor``
    (``ScalarUnit``)."""

    def state_decls(self, cells: list[tuple[int, int]]) -> list[Stmt]:
        """Per-cell state decls (mma C ``RegFragment``s / scalar ``Init`` accumulators)."""
        raise NotImplementedError

    def operands(self) -> list[Operand]:
        """The contraction operands, each tagged with its free-axis dependence (for dedup)."""
        raise NotImplementedError

    def reduce_region(self, cells: list[tuple[int, int]], offset: OffsetFn, masks) -> tuple[list[Stmt], list[Stmt]]:
        """``(top_decls, kstmts)`` — the CTA-scope decls (staged slabs / descriptors, empty for
        gmem-direct) and the K-loop region (which may carry a cooperative staging prologue). The
        reduce loop + staging are the Unit's strategy; the generic layer never branches on it."""
        raise NotImplementedError

    def store(self, i: int, j: int, offset: OffsetFn, masks) -> list[Stmt]:
        """The per-cell output store stmts (``[RegStore]`` for mma; the guarded projection-tail
        cell for scalar — possibly several stmts). Builds its own cell σ from ``offset`` — the
        warp store uses a raw σ + separate ``m_guard``/``n_guard``, the scalar store a masked
        (``%extent``) σ + a guarded ``Write`` — so the unit owns it."""
        raise NotImplementedError


def atomize(unit: Unit, atom_m: int, atom_n: int) -> Tiling:
    """The leaf: a single atom of ``atom_m × atom_n`` (1×1 for a scalar cell). Seeds the
    per-axis offset with the atom step; the atom-lane offset stays OUT of σ (added at render)."""
    levels = {
        "m": (atom_m, 1, None, None, 1),
        "n": (atom_n, 1, None, None, 1),
    }
    return Tiling(unit=unit, offset=OffsetFn(levels=levels))


def register_tile(t: Tiling, reg_m: int, reg_n: int) -> Tiling:
    """The REGISTER level: ``reg_m × reg_n`` atoms per thread/warp. Records the cell counts; the
    per-cell ``r·atom_dim`` term is applied at ``OffsetFn.base``."""
    levels = dict(t.offset.levels)
    for which, reg in (("m", reg_m), ("n", reg_n)):
        atom_dim, _, block_var, warp_var, warp_count = levels[which]
        levels[which] = (atom_dim, reg, block_var, warp_var, warp_count)
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
) -> Tile:
    """The GRID level + finalize: bind the block axes (the shrunk grid), set the per-axis grid
    term ``block·tile``, append any leading (e.g. batch) grid axes verbatim and — when the atom is
    warp-cooperative (``lanes > 1``) — the atom ``_lane`` axis, then splice the unit's state +
    reduce-region + stores into the ``Tile``. This is the outermost stage — it emits.

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
    state = t.unit.state_decls(cells)
    top_decls, kstmts = t.unit.reduce_region(cells, offset, masks)
    stores = [s for (i, j) in cells for s in t.unit.store(i, j, offset, masks)]
    return Tile(axes=offset.axes, body=Body((*state, *top_decls, *kstmts, *stores)), block_threads=block_threads)


def _with_block(level: tuple, block_var: str) -> tuple:
    atom_dim, reg, _, warp_var, warp_count = level
    return (atom_dim, reg, block_var, warp_var, warp_count)


__all__ = ["OffsetFn", "Operand", "Tiling", "Unit", "atomize", "grid_tile", "register_tile", "unit_tile"]
