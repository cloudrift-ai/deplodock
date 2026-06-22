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

import os

from deplodock.compiler.ir.stmt import Loop, Write
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom
from deplodock.compiler.pipeline.knob import Knob
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.decompose import legal_decomps
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import IterDag, _carrier_of
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    BK_CHOICES,
    BR_CHOICES,
    COOP_BR,
    FK_CHOICES,
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    RED_SPLITK,
    REG_CHOICES,
    SPLITK_CHOICES,
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


def _pin(knob: Knob) -> int | None:
    """Env override for one search dimension (``DEPLODOCK_<NAME>``) — lets a
    user / test pin a greenfield knob the way the legacy enumerator does."""
    raw = os.environ.get(knob.env)
    return int(raw) if raw not in (None, "") else None


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


def thread_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` thread tiles within the CTA thread
    budget, best-first (≈``_THREAD_TARGET`` threads, larger to break ties).
    ``thread_m`` is ``1`` for a 1-D pointwise kernel (no M axis)."""
    n_choices = _axis_thread_choices(dag.inner_n.extent)
    m_choices = _axis_thread_choices(dag.outer_m.extent) if dag.outer_m is not None else (1,)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def reg_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles within the cell budget, best-first
    (fewest cells — pointwise is bandwidth bound; prefer tiling the contiguous N
    axis on ties)."""
    m_choices = _POINTWISE_REG_CHOICES if dag.outer_m is not None else (1,)
    out = [(r_n, r_m) for r_n in _POINTWISE_REG_CHOICES for r_m in m_choices if budget.cells_ok(r_n * r_m)]
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


# --- Matmul (SEMIRING) moves: TileMap on M/N + TileSerial on K. ---

_CELL_TARGET = 16  # matmul wants a register tile big enough for ILP, small enough for occupancy


def matmul_reduce_offers(dag: IterDag) -> list[tuple[int, int, int]]:
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
    # A MAP epilogue (QK^T scale, matmul_add) or a multi-accumulator matmul
    # (gated MLP — several same-K reduces) forces SPLITK=1: a non-linear epilogue
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


def matmul_thread_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` for a matmul, best-first (≈256 threads)."""
    n_choices = _axis_thread_choices(dag.inner_n.extent)
    m_choices = _axis_thread_choices(dag.outer_m.extent)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def matmul_reg_offers(dag: IterDag, budget: Budget, fk: int) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles for a matmul under the cell budget
    (``fk·reg_n·reg_m ≤ max_cells``), best-first (≈``_CELL_TARGET`` cells)."""
    out = [(r_n, r_m) for r_n in REG_CHOICES for r_m in REG_CHOICES if budget.cells_ok(fk * r_n * r_m)]
    out.sort(key=lambda rm: (abs(rm[0] * rm[1] - _CELL_TARGET), -rm[0] * rm[1]))
    return out


def reduce_knobs(reduce: tuple[int, int, int]) -> dict:
    """Knob delta a reduce-tile branch pins."""
    bk, fk, sk = reduce
    return {RED_BK.name: bk, RED_FK.name: fk, RED_SPLITK.name: sk}


# --- Tensorize move (warp-tier MMA): gated on atom eligibility. ---

_MAX_WARP_CELLS = 64  # FM·FN per warp-cell
_WARP_TARGET = 4  # warps per CTA (~128 threads)


def eligible_atoms(loop_op, ctx, graph) -> list[Atom]:
    """The atoms whose tensor-core precondition holds for this matmul (compute
    capability, operand dtypes, divisibility, foldable epilogue)."""
    return [a for a in ATOM_REGISTRY.values() if is_atom_eligible(a, loop_op, ctx, graph=graph)]


def warp_offers(dag: IterDag, atom: Atom, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(wm, wn)`` warp grids: divide the per-atom cell counts, ≥2 warps
    (single warp can't stage ldmatrix), ``wm·wn·32 ≤`` thread budget."""
    cells_m = dag.outer_m.extent // atom.shape[0]
    cells_n = dag.inner_n.extent // atom.shape[1]
    out = [
        (wm, wn)
        for wm in WARP_CHOICES
        for wn in WARP_CHOICES
        if cells_m % wm == 0 and cells_n % wn == 0 and wm * wn >= 2 and budget.threads_ok(wm * wn * 32)
    ]
    out.sort(key=lambda w: (abs(w[0] * w[1] - _WARP_TARGET), -w[0] * w[1]))
    return out


def warp_reg_offers(dag: IterDag, atom: Atom, warp: tuple[int, int]) -> list[tuple[int, int]]:
    """Legal ``(fm, fn)`` register cells per warp: divide the per-warp cell
    counts, ``fm·fn ≤`` the warp-cell budget."""
    wm, wn = warp
    pm = (dag.outer_m.extent // atom.shape[0]) // wm
    pn = (dag.inner_n.extent // atom.shape[1]) // wn
    out = [(fm, fn) for fm in TC_REG_CHOICES for fn in TC_REG_CHOICES if pm % fm == 0 and pn % fn == 0 and fm * fn <= _MAX_WARP_CELLS]
    out.sort(key=lambda r: (abs(r[0] * r[1] - 8), -r[0] * r[1]))
    return out


def warp_bk_offers(dag: IterDag, atom: Atom) -> list[int]:
    """Legal ``bk`` (K chunk in atom-K units): divides the atom-K cell count."""
    kc = dag.k_extent // atom.shape[2]
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


# --- Cooperative-reduce (MONOID) moves: SplitParallel(K) thread-bound. ---

_BR_TARGET = 128  # ~4 warps cooperating per row


def coop_free_threads(dag: IterDag) -> tuple[int, int]:
    """Free-axis THREAD tiles ``(bn, bm)`` for a cooperative reduce, default
    ``(1, 1)`` — the whole-CTA form (one row per CTA, ``BR`` threads reduce it).
    A pinned ``BN``/``BM`` > 1 thread-binds the free rows ALONGSIDE the ``BR``
    cooperative lanes — the **strided-cooperative rows** form (e.g. a per-head
    q/k-norm deploys a ``BN×BR`` CTA instead of a degenerate one). Honors the
    same ``DEPLODOCK_BN`` / ``DEPLODOCK_BM`` pins the legacy planner reads."""
    bn = _pin(MAP_N_THREAD) or 1
    bm = (_pin(MAP_M_THREAD) or 1) if dag.outer_m is not None else 1
    return bn, bm


def coop_reduce_offers(dag: IterDag, *, warp_size: int = 32) -> list[tuple[int, int, int]]:
    """Legal ``(bk, fk, br)`` for a cooperative reduce: ``br`` (the cooperative
    thread count) ≤ 1024. For a static K, ``br·bk·fk`` must divide K; a symbolic
    (masked) K only needs ``br·bk·fk ≤`` the hint — the final partial tile is
    masked (ceil-div). Best-first: ≈``_BR_TARGET`` threads, no chunk / strip-mine.
    Env-pinnable for tests.

    With **strided-cooperative rows** (a pinned ``BN·BM > 1`` free-axis thread
    tile) the cross-thread combine is a SEGMENTED warp shuffle over each row's
    ``BR`` lanes, so ``BR`` clips to a power of two ≤ ``warp_size`` (each row's
    lanes must form an aligned intra-warp segment — matching the legacy
    enumeration and ``cooperative_combine_geometry``), and ``bn·bm·br`` must fit
    the CTA thread budget."""
    bk_pin, fk_pin, br_pin = _pin(RED_BK), _pin(RED_FK), _pin(COOP_BR)
    bks = (bk_pin,) if bk_pin else BK_CHOICES
    fks = (fk_pin,) if fk_pin else FK_CHOICES
    brs = (br_pin,) if br_pin else BR_CHOICES
    masked = dag.k_bound is not None
    # A short reduce (static K < warp_size) stays pure-serial (BR=1): too small to
    # stage a meaningful cross-thread tree-halve combine — matching the legacy
    # WARP_SIZE gate. It still composes (the serial per-row reduce), just without
    # cooperative threads. (A symbolic/masked K tiles at the hint, which is large.)
    br_floor_serial = (not masked) and dag.k_extent < 32
    bn, bm = coop_free_threads(dag)
    strided = bn * bm > 1
    # Same decomposition move as split-K: factor [br (cooperative THREAD), bk
    # (STAGE_INNER), fk (REGISTER)] — the masked-K fill is licensed by the
    # carrier's identity (has_identity), the cooperative partition by commutative.
    decomps = legal_decomps(
        _carrier_of(dag.k_node.loop),
        dag.k_node.loop.axis,
        dag.k_extent,
        factor_menus=[brs, bks, fks],
        placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER],
        masked=masked,
        allow_split=True,
    )
    out = [
        (d.factors[1], d.factors[2], d.factors[0])
        for d in decomps
        if 1 <= d.factors[0] <= 1024
        and (d.factors[0] == 1 or not br_floor_serial)
        and (not strided or _strided_br_ok(d.factors[0], bn * bm, warp_size))
    ]
    out.sort(key=lambda t: (abs(t[2] - _BR_TARGET), t[0], t[1]))
    return out


def _strided_br_ok(br: int, free_threads: int, warp_size: int) -> bool:
    """A strided-cooperative ``BR`` must be a power of two ≤ ``warp_size`` (so the
    segmented shuffle stays intra-warp), and ``free_threads·BR`` must fit the CTA
    thread budget (1024)."""
    if free_threads * br > 1024:
        return False
    return br == 1 or (br <= warp_size and br & (br - 1) == 0)


def coop_reduce_knobs(reduce: tuple[int, int, int]) -> dict:
    """Knob delta a cooperative-reduce leaf pins."""
    bk, fk, br = reduce
    return {RED_BK.name: bk, RED_FK.name: fk, COOP_BR.name: br}


def coop_free_thread_knobs(dag: IterDag) -> dict:
    """Knob delta for the free-axis THREAD tiles (``BN``/``BM``) a cooperative
    reduce binds — default ``1`` (whole-CTA), or the pinned strided-cooperative
    value. ``BM`` only when an outer free axis exists."""
    bn, bm = coop_free_threads(dag)
    knobs = {MAP_N_THREAD.name: bn}
    if dag.outer_m is not None:
        knobs[MAP_M_THREAD.name] = bm
    return knobs
