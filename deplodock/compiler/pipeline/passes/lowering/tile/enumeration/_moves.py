"""Moves: ``TileMap`` on each free axis, ``TileSerial`` on a contraction (reduce) axis.

A ``TileMap`` on a map axis contributes a thread-tile factor and a register-tile
factor; the block-tile count is derived from the extent at materialize time
(``MaskMap`` is implicit — a non-divisible / symbolic axis ceil-divides and gets
a store guard). Map axes carry no carrier, so these moves have no algebraic
precondition; legality is purely the resource budget. ``TileSerial`` re-brackets
a ``SEMIRING`` reduce axis into a ``(bk, fk)`` K-chunk + strip-mine — its
precondition (``carrier.associative``) holds for any associative reduce, so here too
legality reduces to ``bk·fk`` dividing the K extent plus the cell budget.

This module owns the **legal offer set** (the search dimensions) and the knob
param dicts; ``materialize.py`` realizes a complete choice into the tower.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Loop, ReduceCarrier, Write
from deplodock.compiler.pipeline.knob import Knob
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, _carrier_of
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    BK_CHOICES,
    FK_CHOICES,
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    MAX_CELLS_PER_THREAD,
    MAX_THREADS_PER_CTA,
    RED_BK,
    RED_FK,
    RED_SPLITK,
    REG_CHOICES,
    SPLITK_CHOICES,
    THREAD_CHOICES,
)

# --- Resource budget (merged from _budget.py) ---


@dataclass(frozen=True)
class Budget:
    """Incremental resource accounting threaded through the move ``offers`` —
    a composition that would blow a per-CTA ceiling is never offered."""

    max_threads: int = MAX_THREADS_PER_CTA
    max_cells: int = MAX_CELLS_PER_THREAD

    def threads_ok(self, threads: int) -> bool:
        return 1 <= threads <= self.max_threads

    def cells_ok(self, cells: int) -> bool:
        return 1 <= cells <= self.max_cells


# --- The carrier-licensed decomposition move (merged from _decompose.py) ---


@dataclass(frozen=True)
class AxisDecomp:
    """A factorization of one reduce ``axis`` into ``factors`` (extent product
    ``== axis.extent`` unless masked), each piece placed by ``placement``.

    The recombine operator and the realization are NOT stored — they are derived:
    the recombine is ``carrier.combine_partials()`` (algebra); the realization
    (atomic / shuffle / tree / mma / serial) is the cost+hardware choice keyed off
    ``placement``."""

    axis: Axis
    factors: tuple[int, ...]
    placement: tuple[Role, ...]


def legal_decomps(
    carrier: ReduceCarrier | None,
    axis: Axis,
    extent: int,
    *,
    factor_menus: Sequence[Sequence[int]],
    placement: Sequence[Role],
    masked: bool,
    allow_split: bool = True,
) -> list[AxisDecomp]:
    """The factorizations the ``carrier`` algebra licenses over ``axis``.

    ``factor_menus`` is one candidate menu per factored piece (e.g. split / chunk
    / strip for a SEMIRING split, or cooperative / chunk / strip for a MONOID reduce);
    ``placement`` names where each piece lands (the same length). The legality is
    a carrier-trait query:

    - **associative** licenses splitting the axis at all. A non-associative
      carrier admits only the trivial all-``1`` factorization (no recombine).
    - **commutative** licenses a *partitioning* factor > 1 whose recombine
      reorders partials (split-K cross-CTA / cooperative-tree) — the FIRST factor
      by convention (``placement[0]`` is the partition: ``BLOCK`` split-K /
      ``THREAD`` cooperative). ``allow_split`` is the orthogonal cost/soundness
      gate the caller supplies (a non-linear epilogue / multi-accumulator reduce
      forbids split-K regardless of algebra).
    - **has_identity** licenses a ``masked`` (ceil-div + identity-fill)
      factorization of a non-divisible / symbolic axis; without it the product
      must divide ``extent`` exactly.

    A **PARALLEL** axis (``carrier is None``) is the degenerate, no-recombine case
    (phase 7): free-axis tiling (block × thread × register) and the tensorize
    atom-block are product decompositions of *independent* work, so every
    factorization is legal (no associativity needed — there is nothing to
    recombine), masking is a plain boundary store-guard (no carrier identity), and
    a partition factor needs no commutativity.

    Returns the legal :class:`AxisDecomp`s unranked — pruning / best-first
    ordering stays with the caller (cost, not algebra)."""
    parallel = carrier is None
    if masked and not parallel and not carrier.has_identity:
        return []  # can't identity-fill a fill-less reduce carrier (a parallel axis masks via a store guard)
    splittable = parallel or carrier.associative
    can_partition = parallel or (allow_split and carrier.commutative)
    placement_t = tuple(placement)

    out: list[AxisDecomp] = []

    def _emit(combo: tuple[int, ...]) -> None:
        product = 1
        for f in combo:
            product *= f
        if product > extent:
            return
        if not masked and extent % product != 0:
            return
        if not splittable and product != 1:
            return
        if combo[0] != 1 and not can_partition:
            return
        out.append(AxisDecomp(axis=axis, factors=combo, placement=placement_t))

    _enumerate(factor_menus, (), _emit)
    return out


def _enumerate(menus: Sequence[Sequence[int]], prefix: tuple[int, ...], emit) -> None:
    """Cartesian product of the per-factor menus, calling ``emit`` per combo."""
    if not menus:
        emit(prefix)
        return
    for v in menus[0]:
        _enumerate(menus[1:], (*prefix, v), emit)


def _pin(knob: Knob) -> int | None:
    """Env override for one search dimension (``DEPLODOCK_<NAME>``) — lets a
    user / test pin a greenfield knob the way the legacy enumerator does."""
    raw = os.environ.get(knob.env)
    return int(raw) if raw not in (None, "") else None


# Pointwise is memory-bandwidth bound, so a conservative register-tile menu
# keeps the generative tree small without losing the configs that matter
# (richer reduce-regime reg menus arrive with their regimes).
_MAP_REG_CHOICES: tuple[int, ...] = (1, 2, 4, 8)


def _axis_thread_choices(extent: int) -> tuple[int, ...]:
    """Thread-tile extents for one axis, clamped to the axis size and deduped
    (so a size-32 axis never offers a 64-wide thread tile)."""
    seen: dict[int, None] = {}
    for c in THREAD_CHOICES:
        seen.setdefault(min(c, extent) if extent >= 1 else c, None)
    return tuple(seen)


# A free-axis (MAP) CTA wants ~256 threads (8 warps): enough occupancy without
# starving the grid. The cold prior has no weighted feature for the greenfield
# knobs yet (Phase 4 retrain), so emission order is the effective ranking —
# emit the sanest tile first.
_THREAD_TARGET = 256


def thread_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` thread tiles within the CTA thread
    budget, best-first (≈``_THREAD_TARGET`` threads, larger to break ties).
    ``thread_m`` is ``1`` for a 1-D nest (no M axis)."""
    n_choices = _axis_thread_choices(dag.inner_n.extent)
    m_choices = _axis_thread_choices(dag.outer_m.extent) if dag.outer_m is not None else (1,)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def map_reg_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles within the cell budget, best-first
    (fewest cells — a MAP nest is bandwidth bound; prefer tiling the contiguous N
    axis on ties)."""
    m_choices = _MAP_REG_CHOICES if dag.outer_m is not None else (1,)
    out = [(r_n, r_m) for r_n in _MAP_REG_CHOICES for r_m in m_choices if budget.cells_ok(r_n * r_m)]
    out.sort(key=lambda rm: (rm[0] * rm[1], -rm[0]))
    return out


def thread_knobs(dag: IterDag, thread: tuple[int, int]) -> dict:
    """Knob delta a thread-tile branch pins."""
    t_n, t_m = thread
    knobs = {MAP_N_THREAD.name: t_n}
    if dag.outer_m is not None:
        knobs[MAP_M_THREAD.name] = t_m
    return knobs


def reg_knobs(dag: IterDag, reg: tuple[int, int]) -> dict:
    """Knob delta a register-tile leaf adds on top of its thread branch."""
    r_n, r_m = reg
    knobs = {MAP_N_REG.name: r_n}
    if dag.outer_m is not None:
        knobs[MAP_M_REG.name] = r_m
    return knobs


# --- Reduce decomposition move (SEMIRING / MONOID): the carrier-licensed
# legal_decomps over a contraction axis, + its compute-bound register menu. ---

_CELL_TARGET = 16  # a reduce regime wants a register tile big enough for ILP, small enough for occupancy


def reduce_offers(dag: IterDag) -> list[tuple[int, int, int]]:
    """Legal ``(bk, fk, splitk)`` K-tilings: ``splitk·bk·fk`` divides the static
    K extent (so ``K_o = K/(splitk·bk·fk)`` is whole). Best-first: no split-K
    (``splitk=1``, a perf opt for small-MN/large-K), deep chunk (large ``bk``),
    no strip-mine (``fk=1``). Each dimension is env-pinnable for tests.

    The legality is the carrier-trait query :func:`legal_decomps` (associative →
    split; commutative → cross-CTA combine); ranking + the split-K soundness gate
    (cost / hardware) stay here."""
    bk_pin, fk_pin, sk_pin = _pin(RED_BK), _pin(RED_FK), _pin(RED_SPLITK)
    bks = (bk_pin,) if bk_pin else BK_CHOICES
    fks = (fk_pin,) if fk_pin else FK_CHOICES
    # Split-K atomic-adds per-CTA partials, sound only for a bare single reduce.
    # A MAP epilogue (a fused scale / residual add) or a multi-accumulator reduce
    # (several same-K reduces) forces SPLITK=1: a non-linear epilogue
    # or a coupled multi-accum over a partial would corrupt the cross-CTA sum.
    has_epilogue = any(not isinstance(s, (Loop, Write)) for s in dag.inner_body)
    n_reduce = sum(1 for s in dag.inner_body if isinstance(s, Loop) and s.is_reduce)
    allow_split = not (has_epilogue or n_reduce > 1)
    sks = (1,) if not allow_split else ((sk_pin,) if sk_pin else SPLITK_CHOICES)
    # Factor order [splitk (BLOCK), bk (STAGE_INNER), fk (REGISTER)] — the
    # partition (splitk) is factor 0, where the commutative-combine gate applies.
    decomps = legal_decomps(
        _carrier_of(dag.k_node.loop),
        dag.k_node.loop.axis,
        dag.k_extent,
        factor_menus=[sks, bks, fks],
        placement=[Role.BLOCK, Role.STAGE_INNER, Role.REGISTER],
        masked=False,
        allow_split=allow_split,
    )
    out = [(d.factors[1], d.factors[2], d.factors[0]) for d in decomps]
    out.sort(key=lambda t: (t[2] != 1, -t[0], t[1], t[2]))
    return out
def reduce_reg_offers(dag: IterDag, budget: Budget, fk: int) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles for a reduce regime under the cell budget
    (``fk·reg_n·reg_m ≤ max_cells``), best-first (≈``_CELL_TARGET`` cells)."""
    out = [(r_n, r_m) for r_n in REG_CHOICES for r_m in REG_CHOICES if budget.cells_ok(fk * r_n * r_m)]
    out.sort(key=lambda rm: (abs(rm[0] * rm[1] - _CELL_TARGET), -rm[0] * rm[1]))
    return out


def reduce_knobs(reduce: tuple[int, int, int]) -> dict:
    """Knob delta a reduce-tile branch pins."""
    bk, fk, sk = reduce
    return {RED_BK.name: bk, RED_FK.name: fk, RED_SPLITK.name: sk}


