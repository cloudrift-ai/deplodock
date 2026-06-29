"""The one contraction factorizer — atom-generic.

Both contraction arms (``MmaContraction`` / ``ScalarContraction``) expand through the *same*
four-level tiling pipeline (``atomize → register_tile → unit_tile → grid_tile``); they differ only
in the leaf :class:`~._tiling.Unit` their atom selects (``AtomUnit`` for the tensor-core mma cell,
``ScalarUnit`` for the scalar fma cell) and the geometry that unit exposes. :func:`factorize` builds
the right unit and runs the pipeline once — there is no per-atom factorizer.

Every unit exposes the same canonical tiling-geometry interface the pipeline reads: ``atom_m`` /
``atom_n`` (leaf cell), ``reg_m`` / ``reg_n`` (REGISTER sub-tile), ``units_m`` / ``units_n`` +
``m_uvar`` / ``n_uvar`` (the UNIT grid — warps for mma, threads for scalar), ``m_axis`` / ``n_axis``
+ ``m_b`` / ``n_b`` + ``tile_m`` / ``tile_n`` + ``lead_axes`` (the GRID), ``block_threads``,
``lanes`` (``atom.lanes``), and ``mask_m`` / ``mask_n`` / ``m_ext`` / ``n_ext``. Leading ``_`` so the
pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import Contraction, MmaContraction, ScalarContraction
from deplodock.compiler.pipeline.passes.lowering.kernel._scalar_factor import ScalarUnit
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import Unit, atomize, grid_tile, register_tile, unit_tile
from deplodock.compiler.pipeline.passes.lowering.kernel._warp_factor import AtomUnit


def unit_for(c: Contraction) -> Unit:
    """The leaf :class:`~._tiling.Unit` for a contraction, selected by its atom arm."""
    if isinstance(c, MmaContraction):
        return AtomUnit(c)
    if isinstance(c, ScalarContraction):
        return ScalarUnit(c)
    raise TypeError(f"no Unit for contraction {type(c).__name__}")


def factorize(c: Contraction) -> Tile:
    """Expand a :data:`Contraction` into its tiled ``Tile`` — the one pipeline for both atoms. The
    unit (``AtomUnit`` / ``ScalarUnit``) supplies the per-level geometry; the layer owns the offset,
    the axes, and the splice."""
    u = unit_for(c)
    masks = (u.mask_m, u.mask_n, u.m_ext, u.n_ext)
    t = atomize(u, u.atom_m, u.atom_n)
    t = register_tile(t, u.reg_m, u.reg_n)
    t = unit_tile(t, u.units_m, u.units_n, u.m_uvar, u.n_uvar)
    return grid_tile(
        t,
        masks,
        n_axis=u.n_axis,
        n_b=u.n_b,
        tile_n=u.tile_n,
        m_axis=u.m_axis,
        m_b=u.m_b,
        tile_m=u.tile_m,
        lead_axes=u.lead_axes,
        block_threads=u.block_threads,
        lanes=u.lanes,
    )
