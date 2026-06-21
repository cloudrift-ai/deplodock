"""Moves: ``TileMap`` on each free axis, ``TileSerial`` on a matmul's reduce axis.

A ``TileMap`` on a map axis contributes a thread-tile factor and a register-tile
factor; the block-tile count is derived from the extent at materialize time
(``MaskMap`` is implicit — a non-divisible / symbolic axis ceil-divides and gets
a store guard). Map axes carry no carrier, so these moves have no algebraic
precondition; legality is purely the resource budget. ``TileSerial`` re-brackets
a ``SEMIRING`` reduce axis into a ``(bk, fk)`` K-chunk + strip-mine — its
precondition (``carrier.associative``) holds for any matmul reduce, so here too
legality reduces to ``bk·fk`` dividing the K extent plus the cell budget.

This module owns the **legal offer set** (the search dimensions) and the knob
param dicts; ``materialize.py`` realizes a complete choice into the tower.
"""

from __future__ import annotations

from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    BK_CHOICES,
    FK_CHOICES,
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    REG_CHOICES,
    TC_ATOM,
    TC_BK,
    TC_REG_CHOICES,
    TC_REG_M,
    TC_REG_N,
    THREAD_CHOICES,
    WARP_CHOICES,
    WARP_M,
    WARP_N,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import MatmulSkeleton, PointwiseSkeleton

# Pointwise is memory-bandwidth bound, so a conservative register-tile menu
# keeps the generative tree small without losing the configs that matter
# (richer reduce / matmul reg menus arrive with their regimes).
_POINTWISE_REG_CHOICES: tuple[int, ...] = (1, 2, 4, 8)


def _axis_thread_choices(extent: int) -> tuple[int, ...]:
    """Thread-tile extents for one axis, clamped to the axis size and deduped
    (so a size-32 axis never offers a 64-wide thread tile)."""
    seen: dict[int, None] = {}
    for c in THREAD_CHOICES:
        seen.setdefault(min(c, extent) if extent >= 1 else c, None)
    return tuple(seen)


# A pointwise CTA wants ~256 threads (8 warps): enough occupancy without
# starving the grid. The cold prior has no weighted feature for the greenfield
# knobs yet (Phase 4 retrain), so emission order is the effective ranking —
# emit the sanest tile first.
_THREAD_TARGET = 256


def thread_offers(skel: PointwiseSkeleton, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` thread tiles within the CTA thread
    budget, best-first (≈``_THREAD_TARGET`` threads, larger to break ties).
    ``thread_m`` is ``1`` for a 1-D pointwise kernel (no M axis)."""
    n_choices = _axis_thread_choices(skel.inner_n.extent)
    m_choices = _axis_thread_choices(skel.outer_m.extent) if skel.outer_m is not None else (1,)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def reg_offers(skel: PointwiseSkeleton, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles within the cell budget, best-first
    (fewest cells — pointwise is bandwidth bound; prefer tiling the contiguous N
    axis on ties)."""
    m_choices = _POINTWISE_REG_CHOICES if skel.outer_m is not None else (1,)
    out = [(r_n, r_m) for r_n in _POINTWISE_REG_CHOICES for r_m in m_choices if budget.cells_ok(r_n * r_m)]
    out.sort(key=lambda rm: (rm[0] * rm[1], -rm[0]))
    return out


def thread_knobs(skel: PointwiseSkeleton, thread: tuple[int, int]) -> dict:
    """Knob delta a thread-tile branch pins."""
    t_n, t_m = thread
    knobs = {MAP_N_THREAD.name: t_n}
    if skel.outer_m is not None:
        knobs[MAP_M_THREAD.name] = t_m
    return knobs


def reg_knobs(skel: PointwiseSkeleton, reg: tuple[int, int]) -> dict:
    """Knob delta a register-tile leaf adds on top of its thread branch."""
    r_n, r_m = reg
    knobs = {MAP_N_REG.name: r_n}
    if skel.outer_m is not None:
        knobs[MAP_M_REG.name] = r_m
    return knobs


# --- Matmul (SEMIRING) moves: TileMap on M/N + TileSerial on K. ---

_CELL_TARGET = 16  # matmul wants a register tile big enough for ILP, small enough for occupancy


def matmul_reduce_offers(skel: MatmulSkeleton) -> list[tuple[int, int]]:
    """Legal ``(bk, fk)`` K-tilings: ``bk·fk`` divides the static K extent (so
    ``K_o = K/(bk·fk)`` is whole). Best-first: deep chunk (large ``bk``), no
    strip-mine (``fk=1``)."""
    out = [(bk, fk) for bk in BK_CHOICES for fk in FK_CHOICES if bk * fk <= skel.k_extent and skel.k_extent % (bk * fk) == 0]
    out.sort(key=lambda bf: (-bf[0], bf[1]))
    return out


def matmul_thread_offers(skel: MatmulSkeleton, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` for a matmul, best-first (≈256 threads)."""
    n_choices = _axis_thread_choices(skel.inner_n.extent)
    m_choices = _axis_thread_choices(skel.outer_m.extent)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def matmul_reg_offers(skel: MatmulSkeleton, budget: Budget, fk: int) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles for a matmul under the cell budget
    (``fk·reg_n·reg_m ≤ max_cells``), best-first (≈``_CELL_TARGET`` cells)."""
    out = [(r_n, r_m) for r_n in REG_CHOICES for r_m in REG_CHOICES if budget.cells_ok(fk * r_n * r_m)]
    out.sort(key=lambda rm: (abs(rm[0] * rm[1] - _CELL_TARGET), -rm[0] * rm[1]))
    return out


def reduce_knobs(reduce: tuple[int, int]) -> dict:
    """Knob delta a reduce-tile branch pins."""
    bk, fk = reduce
    return {RED_BK.name: bk, RED_FK.name: fk}


# --- Tensorize move (warp-tier MMA): gated on atom eligibility. ---

_MAX_WARP_CELLS = 64  # FM·FN per warp-cell
_WARP_TARGET = 4  # warps per CTA (~128 threads)


def eligible_atoms(loop_op, ctx, graph) -> list[Atom]:
    """The atoms whose tensor-core precondition holds for this matmul (compute
    capability, operand dtypes, divisibility, foldable epilogue)."""
    return [a for a in ATOM_REGISTRY.values() if is_atom_eligible(a, loop_op, ctx, graph=graph)]


def warp_offers(skel: MatmulSkeleton, atom: Atom, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(wm, wn)`` warp grids: divide the per-atom cell counts, ≥2 warps
    (single warp can't stage ldmatrix), ``wm·wn·32 ≤`` thread budget."""
    cells_m = skel.outer_m.extent // atom.shape[0]
    cells_n = skel.inner_n.extent // atom.shape[1]
    out = [
        (wm, wn)
        for wm in WARP_CHOICES
        for wn in WARP_CHOICES
        if cells_m % wm == 0 and cells_n % wn == 0 and wm * wn >= 2 and budget.threads_ok(wm * wn * 32)
    ]
    out.sort(key=lambda w: (abs(w[0] * w[1] - _WARP_TARGET), -w[0] * w[1]))
    return out


def warp_reg_offers(skel: MatmulSkeleton, atom: Atom, warp: tuple[int, int]) -> list[tuple[int, int]]:
    """Legal ``(fm, fn)`` register cells per warp: divide the per-warp cell
    counts, ``fm·fn ≤`` the warp-cell budget."""
    wm, wn = warp
    pm = (skel.outer_m.extent // atom.shape[0]) // wm
    pn = (skel.inner_n.extent // atom.shape[1]) // wn
    out = [(fm, fn) for fm in TC_REG_CHOICES for fn in TC_REG_CHOICES if pm % fm == 0 and pn % fn == 0 and fm * fn <= _MAX_WARP_CELLS]
    out.sort(key=lambda r: (abs(r[0] * r[1] - 8), -r[0] * r[1]))
    return out


def warp_bk_offers(skel: MatmulSkeleton, atom: Atom) -> list[int]:
    """Legal ``bk`` (K chunk in atom-K units): divides the atom-K cell count."""
    kc = skel.k_extent // atom.shape[2]
    out = [bk for bk in BK_CHOICES if bk <= kc and kc % bk == 0]
    out.sort(key=lambda bk: -bk)
    return out


def warp_knobs(atom: Atom, warp: tuple[int, int]) -> dict:
    """Knob delta the tensorize+warp branch pins."""
    wm, wn = warp
    return {TC_ATOM.name: atom.name, WARP_M.name: wm, WARP_N.name: wn}


def warp_reg_knobs(reg: tuple[int, int]) -> dict:
    fm, fn = reg
    return {TC_REG_M.name: fm, TC_REG_N.name: fn}


def warp_bk_knobs(bk: int) -> dict:
    return {TC_BK.name: bk}
