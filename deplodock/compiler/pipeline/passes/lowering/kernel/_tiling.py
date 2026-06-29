"""Generic tiling-construction layer — ``atomize → register_tile → warp_tile → grid_tile``.

The kernel materializer builds a contraction by tiling a **leaf unit** (an mma :class:`Atom`
or a **scalar** cell) four ways: GRID block / WARP / REGISTER / ATOM. This module makes that
nesting composable and **unit-generic**: a :class:`Unit` realizes one output cell (its state
decl, operands, compute, store, and reduce-region), and the tiling functions wrap it level by
level, building the per-cell coordinate offset incrementally (one :class:`OffsetFn` reproduces
both the warp ``_axis_base`` and the scalar ``_cell_offset``).

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
    """The per-axis cell-offset coordinate, built up across tiling levels. ``base`` reproduces
    the warp ``_axis_base`` (``block·(WM·FM·atom) + warp·(FM·atom) + r·atom``) when a warp level
    is present, else the scalar ``_cell_offset`` (``Var(axis)·reg + r``). Also accumulates the
    ``Tile`` axes (grid block → warp → lane) + ``block_threads``."""

    # Per axis ("m"/"n"): (atom_dim, reg, grid_block_var, warp_var | None, warp_count).
    levels: dict = field(default_factory=dict)
    axes: tuple[Axis, ...] = ()
    block_threads: int | None = None

    def base(self, which: str, r: int):
        """The offset of register cell index ``r`` along axis ``which`` ("m"/"n")."""
        atom_dim, reg, block_var, warp_var, warp_count = self.levels[which]
        reg_term = Literal(r * atom_dim, "int")
        if warp_var is not None:  # warp tier: block·(WM·FM·atom) + warp·(FM·atom) + r·atom
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
    """The accumulating tiling state threaded through ``atomize → register_tile → warp_tile →
    grid_tile``. ``build`` (called by ``grid_tile``, the finalizer) splices the unit's state +
    reduce-region + stores into the ``Tile``."""

    unit: Unit
    offset: OffsetFn
    reg_m: int = 1
    reg_n: int = 1


class Unit:
    """A leaf cell realization — an mma ``Atom`` or a scalar cell. The tiling layer is generic
    over this; two impls live in ``_warp_factor`` (``AtomUnit``) and the scalar materializer
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

    def store(self, i: int, j: int, offset: OffsetFn, masks) -> Stmt:
        """The per-cell guarded output store (``RegStore`` / guarded ``Write``). Builds its own
        cell σ from ``offset`` — the warp store uses a raw σ + separate ``m_guard``/``n_guard``,
        the scalar store a masked (``%extent``) σ + a guarded ``Write`` — so the unit owns it."""
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


def warp_tile(t: Tiling, wm: int, wn: int, m_w: str, n_w: str) -> Tiling:
    """The WARP level (scalar skips this): ``wm × wn`` warps per CTA. Adds the warp term
    ``warp·(reg·atom)`` to each axis offset and the ``m_w`` / ``n_w`` warp axes."""
    levels = dict(t.offset.levels)
    for which, count, var in (("m", wm, m_w), ("n", wn, n_w)):
        atom_dim, reg, block_var, _, _ = levels[which]
        levels[which] = (atom_dim, reg, block_var, var, count)
    axes = (*t.offset.axes, Axis(name=m_w, extent=wm), Axis(name=n_w, extent=wn))
    return replace(t, offset=replace(t.offset, levels=levels, axes=axes))


def grid_tile(
    t: Tiling,
    masks,
    *,
    m_axis: Axis,
    n_axis: Axis,
    m_b: str,
    n_b: str,
    tile_m: int,
    tile_n: int,
    block_threads: int | None,
    lane: int | None,
) -> Tile:
    """The GRID level + finalize: bind the block axes (the shrunk grid), set the per-axis grid
    term ``block·tile``, optionally append the atom ``_lane`` axis, then splice the unit's state
    + reduce-region + stores into the ``Tile``. This is the outermost stage — it emits."""
    levels = dict(t.offset.levels)
    levels["m"] = _with_block(levels["m"], m_b)
    levels["n"] = _with_block(levels["n"], n_b)
    grid_axes = (
        _shrink_axis(Axis(name=m_b, extent=m_axis.extent, source_axis=m_axis), tile_m),
        _shrink_axis(Axis(name=n_b, extent=n_axis.extent, source_axis=n_axis), tile_n),
    )
    lane_axes = (Axis(name="_lane", extent=lane),) if lane is not None else ()
    offset = replace(t.offset, levels=levels, axes=(*grid_axes, *t.offset.axes, *lane_axes), block_threads=block_threads)

    cells = [(i, j) for i in range(t.reg_m) for j in range(t.reg_n)]
    state = t.unit.state_decls(cells)
    top_decls, kstmts = t.unit.reduce_region(cells, offset, masks)
    stores = [t.unit.store(i, j, offset, masks) for (i, j) in cells]
    return Tile(axes=offset.axes, body=Body((*state, *top_decls, *kstmts, *stores)), block_threads=block_threads)


def _with_block(level: tuple, block_var: str) -> tuple:
    atom_dim, reg, _, warp_var, warp_count = level
    return (atom_dim, reg, block_var, warp_var, warp_count)


__all__ = ["OffsetFn", "Operand", "Tiling", "Unit", "atomize", "grid_tile", "register_tile", "warp_tile"]
