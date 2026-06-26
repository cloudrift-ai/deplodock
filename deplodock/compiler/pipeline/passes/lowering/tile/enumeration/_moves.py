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

from collections.abc import Sequence
from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import Accum, Loop, ReduceCarrier, Write
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, _carrier_of
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    BK_CHOICES,
    BR_CHOICES,
    FK_CHOICES,
    MAX_CELLS_PER_THREAD,
    MAX_THREADS_PER_CTA,
    REG_CHOICES,
    SPLITK_CHOICES,
    TC_REG_CHOICES,
    THREAD_CHOICES,
    WARP_CHOICES,
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


def thread_offers(dag: IterDag, budget: Budget, *, balanced: bool = False) -> list[tuple[int, int]]:
    """Legal ``(thread_n, thread_m)`` thread tiles within the CTA thread
    budget, best-first (≈``_THREAD_TARGET`` threads, larger to break ties).
    ``thread_m`` is ``1`` for a 1-D nest (no M axis).

    A pinned ``DEPLODOCK_BN``/``BM`` narrows its axis to that single extent (the
    same ``_pin`` idiom :func:`reduce_offers` uses for ``BK``/``FK``/``SPLITK``) so a
    test / golden can force a specific (masked) free-axis tile — without it greedy
    would always take the best-first ≈``_THREAD_TARGET`` offer and silently drop the pin.

    ``balanced`` (the **SEMIRING matmul** regime, set by ``090_thread_tile``) drops
    the degenerate-aspect tiles — one axis collapsed to ``1`` — and leads with a
    square-ish, coalesced tile (``BN >= BM``). The bare ≈``_THREAD_TARGET`` sort ties
    a ``(BN=1, BM=256)`` tile with a balanced ``(16, 16)`` (both 256 threads) and
    emits the degenerate one first on choice order — and emission order *is* the cold
    greedy pick / the search's first trajectory, so the clean search would sample and
    deploy a ``BN=1`` (no-coalescing) / ``BM=1`` (no M-reuse) tile 2–3× off the golden
    band. Only matmul wants this: a MAP (pointwise) nest is bandwidth-bound and keeps
    the wide-N order (``balanced=False``), and a genuinely 1-wide axis (gemv) falls
    back to the bare order so it is never stranded."""
    bn_pin = fam.split_par(dag, dag.inner_n.axis.name)
    bm_pin = fam.split_par(dag, dag.outer_m.axis.name) if dag.outer_m is not None else None
    n_choices = (bn_pin,) if bn_pin else _axis_thread_choices(dag.inner_n.extent)
    if dag.outer_m is None:
        m_choices: tuple[int, ...] = (1,)
    else:
        m_choices = (bm_pin,) if bm_pin else _axis_thread_choices(dag.outer_m.extent)
    out = [(t_n, t_m) for t_n in n_choices for t_m in m_choices if budget.threads_ok(t_n * t_m)]
    if balanced and dag.outer_m is not None:
        non_degenerate = [tm for tm in out if tm[0] > 1 and tm[1] > 1]
        if non_degenerate:
            # near the thread target, then most balanced (low aspect), then the wider
            # extent on the contiguous N axis (BN >= BM), then larger to break ties.
            non_degenerate.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), max(tm) // min(tm), tm[1] > tm[0], -tm[0] * tm[1]))
            return non_degenerate
    out.sort(key=lambda tm: (abs(tm[0] * tm[1] - _THREAD_TARGET), -tm[0] * tm[1]))
    return out


def map_reg_offers(dag: IterDag, budget: Budget) -> list[tuple[int, int]]:
    """Legal ``(reg_n, reg_m)`` register tiles within the cell budget, best-first
    (fewest cells — a MAP nest is bandwidth bound; prefer tiling the contiguous N
    axis on ties). A pinned ``DEPLODOCK_FN``/``FM`` narrows its axis (see
    :func:`thread_offers`)."""
    fn_pin = fam.split_reg(dag, dag.inner_n.axis.name)
    fm_pin = fam.split_reg(dag, dag.outer_m.axis.name) if dag.outer_m is not None else None
    n_choices = (fn_pin,) if fn_pin else _MAP_REG_CHOICES
    if dag.outer_m is None:
        m_choices: tuple[int, ...] = (1,)
    else:
        m_choices = (fm_pin,) if fm_pin else _MAP_REG_CHOICES
    out = [(r_n, r_m) for r_n in n_choices for r_m in m_choices if budget.cells_ok(r_n * r_m)]
    out.sort(key=lambda rm: (rm[0] * rm[1], -rm[0]))
    return out


def thread_knobs(dag: IterDag, thread: tuple[int, int]) -> dict:
    """Knob delta the thread-tile branch pins — the par-only ``SPLIT@<axis>``
    transitional (the register fork completes ``reg`` at :func:`reg_knobs`)."""
    t_n, t_m = thread
    knobs = {fam.split_key(dag.inner_n.axis.name): fam.enc_split(t_n)}
    if dag.outer_m is not None:
        knobs[fam.split_key(dag.outer_m.axis.name)] = fam.enc_split(t_m)
    return knobs


def reg_knobs(dag: IterDag, knobs: dict, reg: tuple[int, int]) -> dict:
    """Knob delta the register-tile leaf adds — it **completes** each free axis's
    ``SPLIT@<axis>`` by reading the par the thread branch stamped and appending the
    register factor (``"P"`` → ``"PxR"``)."""
    r_n, r_m = reg
    par_n = fam.dec_split(knobs[fam.split_key(dag.inner_n.axis.name)])[0]
    out = {fam.split_key(dag.inner_n.axis.name): fam.enc_split(par_n, r_n)}
    if dag.outer_m is not None:
        par_m = fam.dec_split(knobs[fam.split_key(dag.outer_m.axis.name)])[0]
        out[fam.split_key(dag.outer_m.axis.name)] = fam.enc_split(par_m, r_m)
    return out


def free_split_knobs(dag: IterDag, thread: tuple[int, int], reg: tuple[int, int]) -> dict:
    """Complete ``SPLIT@<axis>`` for both free axes at once (par + reg) — the MONOID
    path (cooperative reduce / streaming flash) pins the register factor (1) in the
    same leaf as the thread tile, so there is no separate register fork to complete."""
    t_n, t_m = thread
    r_n, r_m = reg
    out = {fam.split_key(dag.inner_n.axis.name): fam.enc_split(t_n, r_n)}
    if dag.outer_m is not None:
        out[fam.split_key(dag.outer_m.axis.name)] = fam.enc_split(t_m, r_m)
    return out


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
    bk_pin, fk_pin, sk_pin, _ = fam.reduce_fields(dag, dag.k_node.loop.axis.name)
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
    (``fk·reg_n·reg_m ≤ max_cells``), best-first (≈``_CELL_TARGET`` cells). A pinned
    ``DEPLODOCK_FN``/``FM`` narrows its axis (see :func:`thread_offers`)."""
    fn_pin = fam.split_reg(dag, dag.inner_n.axis.name)
    fm_pin = fam.split_reg(dag, dag.outer_m.axis.name) if dag.outer_m is not None else None
    n_choices = (fn_pin,) if fn_pin else REG_CHOICES
    m_choices = (fm_pin,) if fm_pin else REG_CHOICES
    out = [(r_n, r_m) for r_n in n_choices for r_m in m_choices if budget.cells_ok(fk * r_n * r_m)]
    out.sort(key=lambda rm: (abs(rm[0] * rm[1] - _CELL_TARGET), -rm[0] * rm[1]))
    return out


def reduce_knobs(dag: IterDag, reduce: tuple[int, int, int]) -> dict:
    """Knob delta a reduce-tile branch pins — the native ``REDUCE@<primary>`` value
    ``s<serial>/f<fold>/c<cta>``."""
    bk, fk, sk = reduce
    return {fam.reduce_key(dag.k_node.loop.axis.name): fam.enc_reduce(serial=bk, fold=fk, cta=sk)}


# --- Cooperative-reduce (MONOID) moves: the carrier's commutative-licensed
# partition placed on a THREAD axis (``BR`` cooperative lanes per row) instead of
# split-K's BLOCK axis. Same ``legal_decomps`` move; the realization (warp-shuffle /
# tree combine) is the downstream cost choice. ---

_BR_TARGET = 128  # ~4 warps cooperating per row


def coop_free_threads(dag: IterDag) -> tuple[int, int]:
    """Free-axis THREAD tiles ``(bn, bm)`` for a cooperative reduce, default
    ``(1, 1)`` — the whole-CTA form (one row per CTA, ``BR`` threads reduce it).
    A pinned ``BN``/``BM`` > 1 thread-binds the free rows ALONGSIDE the ``BR``
    cooperative lanes — the **strided-cooperative rows** form (e.g. a per-head
    q/k-norm deploys a ``BN×BR`` CTA instead of a degenerate one)."""
    bn = fam.split_par(dag, dag.inner_n.axis.name) or 1
    bm = (fam.split_par(dag, dag.outer_m.axis.name) or 1) if dag.outer_m is not None else 1
    return bn, bm


def coop_reduce_offers(dag: IterDag, *, warp_size: int = 32) -> list[tuple[int, int, int, int]]:
    """Legal ``(bk, fk, br, cta)`` for a cooperative reduce. ``br`` (the cooperative
    thread count) ≤ 1024; ``cta`` is the **cross-CTA split-K** degree (each CTA owns a
    chunk, writes its partial state to a workspace, a sibling combine kernel folds them —
    the cross-CTA producer mirroring the intra-CTA ``br`` partition one level up). For a
    static K, ``br·bk·fk·cta`` must divide K; a symbolic (masked) K only needs
    ``br·bk·fk ≤`` the hint — the final partial tile is masked (ceil-div). Best-first:
    ``cta = 1`` first (no split), then ≈``_BR_TARGET`` threads, no chunk / strip-mine.
    Env-pinnable for tests.

    With **strided-cooperative rows** (a pinned ``BN·BM > 1`` free-axis thread
    tile) the cross-thread combine is a SEGMENTED warp shuffle over each row's
    ``BR`` lanes, so ``BR`` clips to a power of two ≤ ``warp_size`` (each row's
    lanes must form an aligned intra-warp segment — matching
    ``cooperative_combine_geometry``), and ``bn·bm·br`` must fit the CTA thread
    budget.

    **Cross-CTA split-K** (``cta > 1``) is offered ONLY for an **additive ``Accum``**
    carrier over a **static** K with no non-linear epilogue / multi-accumulator reduce
    (the same soundness gate :func:`reduce_offers` applies to the matmul split-K: a
    non-linear epilogue or a coupled multi-accum over a partial would corrupt the
    cross-CTA sum). A non-additive ``Monoid`` carrier (online softmax / flash split-KV)
    opts in separately (the twisted finalize is kernel-only). Cooperative ``br`` and
    cross-CTA ``cta`` compose — K is partitioned across BOTH CTAs and the CTA's threads."""
    bk_pin, fk_pin, sk_pin, br_pin = fam.reduce_fields(dag, dag.k_node.loop.axis.name)
    bks = (bk_pin,) if bk_pin else BK_CHOICES
    fks = (fk_pin,) if fk_pin else FK_CHOICES
    brs = (br_pin,) if br_pin else BR_CHOICES
    masked = dag.k_bound is not None
    # A short reduce (static K < warp_size) stays pure-serial (BR=1): too small to
    # stage a meaningful cross-thread tree-halve combine. It still composes (the
    # serial per-row reduce), just without cooperative threads. (A symbolic/masked
    # K tiles at the hint, which is large.)
    br_floor_serial = (not masked) and dag.k_extent < 32
    bn, bm = coop_free_threads(dag)
    strided = bn * bm > 1
    # Same decomposition move as split-K: factor [br (cooperative THREAD), bk
    # (STAGE_INNER), fk (REGISTER)] — the masked-K fill is licensed by the
    # carrier's identity (has_identity), the cooperative partition by commutative.
    carrier = _carrier_of(dag.k_node.loop)
    decomps = legal_decomps(
        carrier,
        dag.k_node.loop.axis,
        dag.k_extent,
        factor_menus=[brs, bks, fks],
        placement=[Role.THREAD, Role.STAGE_INNER, Role.REGISTER],
        masked=masked,
        allow_split=True,
    )
    # The cross-CTA split-K menu — additive Accum + static K + no epilogue / multi-accum.
    additive = isinstance(carrier, Accum) and carrier.op.name == "add"
    has_epilogue = any(not isinstance(s, (Loop, Write)) for s in dag.inner_body)
    n_reduce = sum(1 for s in dag.inner_body if isinstance(s, Loop) and s.is_reduce)
    allow_cta = additive and not masked and not (has_epilogue or n_reduce > 1)
    sks = ((sk_pin,) if sk_pin else SPLITK_CHOICES) if allow_cta else (1,)
    out: list[tuple[int, int, int, int]] = []
    for d in decomps:
        br, bk, fk = d.factors
        if not (1 <= br <= 1024 and (br == 1 or not br_floor_serial) and (not strided or _strided_br_ok(br, bn * bm, warp_size))):
            continue
        for cta in sks:
            if cta == 1 or (not masked and dag.k_extent % (cta * br * bk * fk) == 0):
                out.append((bk, fk, br, cta))
    out.sort(key=lambda t: (t[3] != 1, abs(t[2] - _BR_TARGET), t[0], t[1]))
    return out


def _strided_br_ok(br: int, free_threads: int, warp_size: int) -> bool:
    """A strided-cooperative ``BR`` must be a power of two ≤ ``warp_size`` (so the
    segmented shuffle stays intra-warp), and ``free_threads·BR`` must fit the CTA
    thread budget (1024)."""
    if free_threads * br > 1024:
        return False
    return br == 1 or (br <= warp_size and br & (br - 1) == 0)


def coop_reduce_knobs(dag: IterDag, reduce: tuple[int, int, int, int]) -> dict:
    """Knob delta a cooperative-reduce leaf pins — ``REDUCE@<primary>`` with the
    cooperative-lane factor in ``t<coop>`` and the cross-CTA split-K degree in ``c<cta>``.
    A ``cta > 1`` emits a **bare** ``c<cta>`` (finalize pending) — the cross-CTA finalize
    fork (``150_cross_cta_finalize``) completes it to ``a`` (atomic) / ``k`` (deferred)."""
    bk, fk, br, cta = reduce
    return {fam.reduce_key(dag.k_node.loop.axis.name): fam.enc_reduce(serial=bk, fold=fk, cta=cta, coop=br)}


def coop_free_thread_knobs(dag: IterDag) -> dict:
    """Knob delta for the free-axis tiles a cooperative reduce binds — the complete
    ``SPLIT@<axis>`` with par = the thread tile (default ``1`` whole-CTA, or the
    pinned strided-cooperative value) and reg forced to ``1`` (one element per
    cell-owner; the MONOID tier has no register fork)."""
    return free_split_knobs(dag, coop_free_threads(dag), (1, 1))


def streaming_br_offers(dag: IterDag) -> list[int]:
    """Cooperative ``BR`` candidates for the streaming-flash reduce axis
    (e.g. flash attention's streaming KV). ``[1]`` (the serial-stream form) unless
    ``DEPLODOCK_BR`` is pinned to a ``br`` (``1 < br ≤ 1024``) that evenly splits the
    **static** streaming extent — then the axis partitions across ``br`` cooperative
    THREAD lanes whose per-lane online-softmax partials merge via the carrier's
    ``combine_states``. Cooperative streaming is opt-in (pin-only), not yet a default
    search dimension; a symbolic streaming axis stays serial."""
    _, _, _, br = fam.reduce_fields(dag, dag.k_node.loop.axis.name)
    if br is None or br <= 1:
        return [1]
    stream = dag.k_node.loop.axis
    if stream.extent.is_static and br <= 1024 and stream.extent.as_static() % br == 0:
        return [br]
    return [1]


def streaming_coop_geometry_ok(br: int, free_threads: int, warp_size: int = 32) -> bool:
    """Whether a cooperative streaming leaf with ``br`` lanes and ``free_threads``
    free-axis THREAD lanes has a legal cross-lane combine (e.g. flash attention's
    cooperative-KV). ``br == 1`` is the serial form (always legal). A **whole-CTA**
    layout (``free_threads == 1``) folds all
    ``br`` lanes via the warp-shuffle (``br ≤ warp_size``) or the smem tree (any
    ``br``). A **strided** layout (``free_threads > 1``) folds each row's lanes via a
    SEGMENTED warp shuffle, so ``br`` must be a power of two ≤ ``warp_size`` (an
    aligned intra-warp segment — matching ``cooperative_combine_geometry``). The CTA
    thread budget (``free_threads · br ≤ 1024``) is enforced by the caller's
    ``Budget``."""
    if br <= 1:
        return True
    if free_threads <= 1:
        return br <= MAX_THREADS_PER_CTA
    return br <= warp_size and br & (br - 1) == 0


# --- Warp-tier (tensor-core ``atomize``) moves: the warp count, the per-warp
# register cells, and the K chunk in atom-K units. Legality is the per-CTA resource
# budget (threads / cells) + the atom-K
# divisibility; the atom eligibility itself is the gate in ``enumeration/_atom``. ---

_MAX_WARP_CELLS = 64  # cells (atom tiles) per warp — the register-file ceiling
_WARP_CELL_TARGET = 16  # a warp wants a register tile big enough for ILP, small enough for occupancy


def warp_offers(dag: IterDag, atom) -> list[tuple[int, int]]:
    """Legal ``(wm, wn)`` warp counts: ``wm·wn ≥ 2`` (single-warp mma.sync is
    pruned — ``ldmatrix`` is smem→register only) and ``wm·wn·group_size`` within
    the CTA thread budget. Best-first: fewest warps, square-ish. ``wm``/``wn`` are
    the par factors of ``SPLIT@<outer_m>`` / ``SPLIT@<inner_n>`` (warp-tier par)."""
    wm_pin = fam.split_par(dag, dag.outer_m.axis.name) if dag.outer_m is not None else None
    wn_pin = fam.split_par(dag, dag.inner_n.axis.name)
    wms = (wm_pin,) if wm_pin else WARP_CHOICES
    wns = (wn_pin,) if wn_pin else WARP_CHOICES
    out = [(wm, wn) for wm in wms for wn in wns if wm * wn >= 2 and wm * wn * atom.group_size <= MAX_THREADS_PER_CTA]
    out.sort(key=lambda wmn: (wmn[0] * wmn[1], abs(wmn[0] - wmn[1])))
    return out


def warp_reg_offers(dag: IterDag, atom) -> list[tuple[int, int]]:  # noqa: ARG001
    """Legal ``(fm, fn)`` per-warp register cells under the cell ceiling
    (``fm·fn ≤ _MAX_WARP_CELLS``), best-first (≈``_WARP_CELL_TARGET`` cells).

    A *fully* pinned ``(DEPLODOCK_FM, DEPLODOCK_FN)`` is authoritative and bypasses
    the ceiling: the cell ceiling is a search-pruning heuristic (don't enumerate
    huge register tiles), not a hardware bound, and a user / test pin is
    authoritative everywhere else (the ``_pin`` doctrine). Honoring an over-ceiling
    pin lets a deliberately-large pinned warp tile (e.g. ``FM=26`` reusing a
    scalar-geometry sweep) reach the build + assemble path — its slabs then exceed
    the smem budget, so the budget-aware ``120_stage`` filter declines staging and
    the operands lower gmem-direct — instead of vanishing into ``no legal warp
    register tile``. The ceiling still prunes the auto-enumerated candidates."""
    fm_pin = fam.split_reg(dag, dag.outer_m.axis.name) if dag.outer_m is not None else None
    fn_pin = fam.split_reg(dag, dag.inner_n.axis.name)
    if fm_pin and fn_pin:
        return [(fm_pin, fn_pin)]
    fms = (fm_pin,) if fm_pin else TC_REG_CHOICES
    fns = (fn_pin,) if fn_pin else TC_REG_CHOICES
    out = [(fm, fn) for fm in fms for fn in fns if fm * fn <= _MAX_WARP_CELLS]
    out.sort(key=lambda fmn: (abs(fmn[0] * fmn[1] - _WARP_CELL_TARGET), -fmn[0] * fmn[1]))
    return out


def warp_bk_offers(dag: IterDag, atom) -> list[int]:
    """Legal ``bk`` K-chunks (in atom-K units): ``bk`` divides the atom-K step
    count ``K / atom_k`` so ``K_o = K/(bk·atom_k)`` is whole. Largest-first
    (``BK_CHOICES`` is descending)."""
    atom_k = atom.shape[2]
    k_atoms = max(1, dag.k_extent // atom_k)
    bk_pin, _, _, _ = fam.reduce_fields(dag, dag.k_node.loop.axis.name)
    cands = (bk_pin,) if bk_pin else BK_CHOICES
    return [bk for bk in cands if bk >= 1 and k_atoms % bk == 0]


def warp_geom_knobs(dag: IterDag, wm: int, wn: int) -> dict:
    """Knob delta the warp-geometry branch pins — the par-only ``SPLIT@<axis>``
    transitional with the warp counts (``wn`` on the inner N axis, ``wm`` on the
    outer M); the register fork completes ``reg`` at :func:`warp_reg_knobs`."""
    out = {fam.split_key(dag.inner_n.axis.name): fam.enc_split(wn)}
    if dag.outer_m is not None:
        out[fam.split_key(dag.outer_m.axis.name)] = fam.enc_split(wm)
    return out


def warp_reg_knobs(dag: IterDag, knobs: dict, fm: int, fn: int) -> dict:
    """Knob delta the warp register-tile branch pins — **completes** each free
    axis's ``SPLIT@<axis>`` by appending the per-warp register cells to the warp
    count the geometry branch stamped."""
    par_n = fam.dec_split(knobs[fam.split_key(dag.inner_n.axis.name)])[0]
    out = {fam.split_key(dag.inner_n.axis.name): fam.enc_split(par_n, fn)}
    if dag.outer_m is not None:
        par_m = fam.dec_split(knobs[fam.split_key(dag.outer_m.axis.name)])[0]
        out[fam.split_key(dag.outer_m.axis.name)] = fam.enc_split(par_m, fm)
    return out


def warp_bk_knobs(dag: IterDag, atom, bk: int) -> dict:
    """Knob delta the warp K-chunk branch pins: the atom-kind stamp + the native
    ``REDUCE@<primary>`` value carrying the K chunk in ``s<bk>``. The v1 ``cta = 1``
    (no cross-CTA split-K — the fragment-store fold relies on it) is the
    ``enc_reduce`` default, so it rides the value implicitly."""
    return {fam.atom_key(fam.MATMUL_CELL): atom.name, fam.reduce_key(dag.k_node.loop.axis.name): fam.enc_reduce(serial=bk)}
