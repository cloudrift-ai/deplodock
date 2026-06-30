"""The one contraction factorizer — atom-generic.

Both atoms of a :class:`~...ir.Contraction` (a tensor-core :class:`AtomKind` or the scalar
:class:`ScalarAtom`) expand through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``). :func:`factorize` reads the tiling **geometry straight off
the** ``Contraction`` **node** (``tile_m`` / ``mask_m`` / ``m_b`` / ``m_uvar`` / ``units_m`` /
``block_threads`` / …, derived there from the unit / register widths + the atom) and dispatches the
atom-specific codegen (``mma_codegen`` / ``scalar_codegen``, which synthesize the contraction from
the operands + return the ``state_decls`` / ``reduce_region`` / ``store`` callables ``grid_tile``
splices). Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import Contraction
from deplodock.compiler.ir.tile.atom import AtomKind
from deplodock.compiler.pipeline.passes.lowering.kernel._scalar_factor import scalar_codegen
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from deplodock.compiler.pipeline.passes.lowering.kernel._warp_factor import mma_codegen


def _codegen_for(c: Contraction):
    """The ``(state_decls, reduce_region, store)`` codegen callables for a contraction, selected by
    its ``atom`` — the tensor-core mma cell vs the scalar fma cell."""
    return mma_codegen(c) if isinstance(c.atom, AtomKind) else scalar_codegen(c)


def factorize(c: Contraction) -> Tile:
    """Expand a :class:`Contraction` into its tiled ``Tile`` — the one pipeline for both atoms. The
    node supplies the per-level geometry; the atom's codegen synthesizes the contraction + per-cell
    emission; the layer owns the offset, the axes, and the splice."""
    state_decls, reduce_region, store = _codegen_for(c)
    masks = (c.mask_m, c.mask_n, c.m_ext, c.n_ext)
    t = atomize(c.atom.atom_m, c.atom.atom_n)
    t = register_tile(t, c.reg_m, c.reg_n)
    t = unit_tile(t, c.units_m, c.units_n, c.m_uvar, c.n_uvar)
    return grid_tile(
        t,
        masks,
        n_axis=c.n_axis,
        n_b=c.n_b,
        tile_n=c.tile_n,
        m_axis=c.m_axis,
        m_b=c.m_b,
        tile_m=c.tile_m,
        lead_axes=c.lead_axes,
        block_threads=c.block_threads,
        lanes=c.atom.lanes,
        state_decls=state_decls,
        reduce_region=reduce_region,
        store=store,
    )
