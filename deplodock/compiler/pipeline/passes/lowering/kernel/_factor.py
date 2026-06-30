"""The one contraction factorizer — atom-generic.

Both leaf arms of a ``Contraction`` (``MmaLeaf`` / ``ScalarLeaf``) expand through the *same*
four-level tiling pipeline (``atomize → register_tile → unit_tile → grid_tile``). :func:`factorize`
reads the tiling **geometry straight off the** ``Contraction`` **node** (``tile_m`` / ``mask_m`` /
``m_b`` / ``m_uvar`` / ``units_m`` / ``block_threads`` / ``lanes`` / …, all derived there from the
leaf widths + the skeleton axes — there is no per-atom geometry object) and dispatches the
atom-specific codegen (``mma_codegen`` / ``scalar_codegen``, which return the
``state_decls`` / ``reduce_region`` / ``store`` callables ``grid_tile`` splices). The leaf is the
only thing that varies. Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import Contraction, MmaLeaf, ScalarLeaf
from deplodock.compiler.pipeline.passes.lowering.kernel._scalar_factor import scalar_codegen
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from deplodock.compiler.pipeline.passes.lowering.kernel._warp_factor import mma_codegen


def _codegen_for(c: Contraction):
    """The ``(state_decls, reduce_region, store)`` codegen callables for a contraction, selected by
    its :data:`~...ir.Leaf` arm."""
    if isinstance(c.leaf, MmaLeaf):
        return mma_codegen(c)
    if isinstance(c.leaf, ScalarLeaf):
        return scalar_codegen(c)
    raise TypeError(f"no codegen for contraction leaf {type(c.leaf).__name__}")


def factorize(c: Contraction) -> Tile:
    """Expand a :data:`Contraction` into its tiled ``Tile`` — the one pipeline for both atoms. The
    node supplies the per-level geometry; the leaf's codegen supplies the per-cell emission; the
    layer owns the offset, the axes, and the splice."""
    state_decls, reduce_region, store = _codegen_for(c)
    masks = (c.mask_m, c.mask_n, c.m_ext, c.n_ext)
    t = atomize(c.atom_m, c.atom_n)
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
        lanes=c.lanes,
        state_decls=state_decls,
        reduce_region=reduce_region,
        store=store,
    )
