"""Moves for the pointwise regime: ``TileMap`` on each free axis.

A ``TileMap`` on a map axis contributes a thread-tile factor and a register-tile
factor; the block-tile count is derived from the extent at materialize time
(``MaskMap`` is implicit — a non-divisible / symbolic axis ceil-divides and gets
a store guard). Map axes carry no carrier, so these moves have no algebraic
precondition; legality is purely the resource budget.

This module owns the **legal offer set** (the search dimensions) and the knob
param dicts; ``materialize.py`` realizes a complete choice into the tower.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    THREAD_CHOICES,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import PointwiseSkeleton

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
