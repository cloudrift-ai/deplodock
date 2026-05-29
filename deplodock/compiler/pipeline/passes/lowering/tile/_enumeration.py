"""Knob-cartesian enumeration for the partition planner.

Owns the planner's tunable knob globals (BN / BM / FM / FN / BK / SPLITK /
BR), the per-mode candidate tuples (matmul / pointwise / reduce), the
pruned cartesian generator over ``(BN, BM, FM, FN, BK, SPLITK, BR)``, and
the per-mode priority/score functions used to rank the resulting
``TileParams`` set.

Three layers:

1. Public ``TileParams`` dataclass — the cartesian product element. One
   ``(bn, bm, fm, fn, bk, splitk, br, overhang)`` variant; frozen for de-dup
   in the ``seen`` set inside the impl.
2. Public ``enumerate_cartesian(...)`` — mode-dispatched wrapper that picks
   the candidate tuples + priority function for ``matmul`` / ``reduce`` /
   ``pointwise``, folds ``DEPLODOCK_<KNOB>`` env pins via ``Knob.narrow``,
   and retries without pins if every candidate was filtered out and any pin
   is set (fallback so peer-kernel pins don't strand a graph with an
   un-lowered LoopOp).
3. Private ``_enumerate_cartesian_impl(...)`` — pure cartesian generator
   over caller-supplied (already-narrowed) tuples; no env reads, no mode
   dispatch. Tests can hit this directly to exercise prune paths.

Kept as a non-pass sibling module (``_`` prefix → Pass loader skips) so
the planner imports the public API without going through
``importlib.import_module`` cross-pass dances. The planner module
(``010_partition_loops.py``) holds the Plan / Materialize / body-building
logic; everything about "which (BN, BM, ...) tuples are admissible and
in what order" lives here.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.knob import Knob, KnobType

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (1, 16, 32, 64, 128, 256)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)
# Cooperative-K thread count. v1: BR > 1 requires BN = BM = 1 (single THREAD
# axis for materializer's _single_thread_var).
_BR_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Cap on per-thread cell-product. NVRTC compile time explodes past this.
_MAX_CELLS_PER_THREAD: int = 128

BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)")
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul N (output) axis")
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")
BR = Knob("BR", KnobType.INT, hints=_BR_CANDIDATES, help="Cooperative-K thread count (1 = pure serial chunked reduce)")

_PLANNER_KNOBS: tuple[Knob, ...] = (BN, BM, FM, FN, BK, SPLITK, BR)


def planner_pin_set() -> bool:
    """True if any planner knob has its ``DEPLODOCK_<NAME>`` env pin set.
    Used by ``enumerate_cartesian`` to gate the peer-kernel fallback."""
    return any(os.environ.get(k.env) is not None for k in _PLANNER_KNOBS)


@dataclass(frozen=True)
class TileParams:
    """One ``(BN, BM, FM, FN, BK, SPLITK, BR)`` variant. Frozen for de-dup in
    the cartesian's ``seen`` set; ``br=1`` default keeps matmul / pointwise
    sites terse.

    ``overhang`` lists output-axis names admitted at a non-divisor BN/BM
    (masked tiles). Empty for clean-divisor variants. The planner stamps the
    same tuple onto the emitted ``TileOp.knobs["OVERHANG"]`` so downstream
    passes (staging, materialization) can guard cooperative loads.
    """

    bn: int
    bm: int
    fm: int
    fn: int
    bk: int
    splitk: int
    br: int = 1
    overhang: tuple[str, ...] = ()


def _divisors_up_to(n: int, cap: int) -> tuple[int, ...]:
    """Divisors of ``n`` ≤ ``cap``, ascending. FM / FN candidate set — a
    divisor of ``E / bm_c`` automatically satisfies the divisibility check."""
    if n < 1 or cap < 1:
        return ()
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _priority_matmul(p: TileParams) -> tuple[int, ...]:
    # High cells/thread (amortize K-loop) capped at 32 (NVRTC compile time),
    # threads near 256, larger BK, smaller SPLITK. Final tiebreaker prefers
    # clean-divisor variants over masked (negative len so fewer-overhang wins
    # under reverse-sorted enumeration).
    threads = p.bn * p.bm
    return (min(p.fm * p.fn, 32), -abs(256 - threads), p.bk, -p.splitk, -len(p.overhang))


def _priority_pointwise(p: TileParams) -> tuple[int, ...]:
    # Memory-bandwidth bound — fewer cells/thread → more CTAs → better
    # SM occupancy. Threads near 256. Final tiebreaker: clean-divisor wins.
    threads = p.bn * p.bm
    return (-(p.fm * p.fn), -abs(256 - threads), -len(p.overhang))


def _priority_reduce(p: TileParams) -> tuple[int, ...]:
    # Warp-sized BR enables warp-shuffle Combine; threads near 256.
    # Static reduce never masks; a symbolic free axis forces masking (every
    # variant then carries the same overhang, so no clean-divisor tiebreak).
    threads = p.bn * p.bm * p.br
    return (min(p.br, 256), -abs(256 - threads), p.bk, -p.splitk)


_NarrowFn = Callable[[tuple[int, ...]], tuple[int, ...]]


def enumerate_cartesian(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    ctx: Context,
    priority_mode: str,
    force_splitk_one: bool = False,
    m_axis_name: str | None = None,
    n_axis_name: str | None = None,
    m_forced_mask: bool = False,
    n_forced_mask: bool = False,
) -> list[TileParams]:
    """Pruned cartesian over ``(BN, BM, FM, FN, BK, SPLITK, BR)``, sorted by
    priority.

    Picks the canonical candidate tuples for ``priority_mode`` (``matmul`` /
    ``reduce`` / ``pointwise``); the choice sets are tightly coupled to the
    kernel class so each one lives here, not at the call site:

        matmul   : BN/BM = _TUNE_AXIS_CHOICES; BK = _BK_CANDIDATES; BR = (1,);
                   SPLITK = _SPLITK_CANDIDATES — clipped to (1,) when
                   ``force_splitk_one`` is set (caller passes True for
                   non-linear post-reduce combines like gated_mlp /
                   sdpa where ``sum_i (c·a_i + r) = c·sum_i a_i + r``
                   doesn't hold, or for the matmul-with-prologue case
                   where the prologue feeds the matmul). min_k_chunks=2.
        reduce   : BN/BM = (1, *_TUNE_AXIS_CHOICES) — the leading 1 enables
                   the cooperative-K v1 constraint (BR>1 ⇒ BN=BM=1, single
                   THREAD axis for the materializer). SPLITK = (1,) — atomic
                   cross-CTA reduce + barrier for the post-reduce epilogue
                   isn't wired up. BR = _BR_CANDIDATES.
        pointwise: BN/BM = _TUNE_AXIS_CHOICES; BK = SPLITK = BR = (1,) — no
                   K loop to chunk or split.

    Env pins (``DEPLODOCK_<KNOB>``, set directly or splatted from
    ``DEPLODOCK_KNOBS`` at ``knob.apply_knobs_env``) are folded into the
    candidate lists via ``Knob.narrow`` here in the wrapper for the five
    static choices, and via ``fm_narrow`` / ``fn_narrow`` callables passed
    into the impl for the per-iteration FM/FN divisor lists. When the
    pinned enumeration is empty *and* any planner pin is set we retry with
    every narrow disabled (raw tuples, ``None`` callables): pins are meant
    to scope the kernel under test, but a graph that fuses peer kernels
    (SDPA = QK^T + P@V; gated MLP at full-model scale) may have peers where
    the pin is invalid by divisibility. Without the fallback those peers
    would ``RuleSkipped`` the planner and leave a ``LoopOp`` in the lowered
    graph, tripping ``CudaBackend``.

    BN/BM clamped to extent + divisibility-checked. FM/FN as divisors of the
    per-thread remainder (auto-divisibility), capped by ``_MAX_CELLS_PER_THREAD``.
    BK/SPLITK divisor-checked against ``per_thread_K = E_K // BR``.
    ``BN·BM·BR ≤ ctx.max_threads_per_cta`` (typically 1024).

    Single-K-iter (per_thread_K == bk) is allowed for pointwise and
    cooperative-reduce, rejected for matmul (``min_k_chunks=2`` — needs ≥ 2
    chunks to amortize K-loop overhead)."""
    # ``_TUNE_AXIS_CHOICES`` already includes ``1`` so tiny output extents
    # (e.g. ``torch.matmul`` of 4×3×2) and global-reduce kernels with a
    # phantom size-1 outer axis survive enumeration. ``_wrap_tower`` drops
    # the resulting size-1 THREAD axes before they reach the IR, so the
    # broader search space only changes behavior for tiny shapes.
    if priority_mode == "matmul":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,) if force_splitk_one else _SPLITK_CANDIDATES
        br_choices: tuple[int, ...] = (1,)
        min_k_chunks = 1
        priority_fn: Callable[[TileParams], tuple[int, ...]] = _priority_matmul
    elif priority_mode == "reduce":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,)
        br_choices = _BR_CANDIDATES
        min_k_chunks = 1
        priority_fn = _priority_reduce
    elif priority_mode == "pointwise":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = (1,)
        splitk_choices = (1,)
        br_choices = (1,)
        min_k_chunks = 1
        priority_fn = _priority_pointwise
    else:
        raise ValueError(f"unknown priority_mode {priority_mode!r}")

    # When the caller passed ``E_M=1`` / ``E_N=1`` only because both free axes
    # were symbolic, the canonical bn*bm splits all collapse to (1, 1) and the
    # planner needs to keep the 1-thread-per-CTA variant rather than skip it.
    # Matmul honors this too so symbolic Q@K^T (M=N=seq_len) can still emit
    # a degenerate single-thread variant.
    allow_empty_threads = E_M == 1 and E_N == 1 and priority_mode in ("pointwise", "matmul")

    # Masked tiles: when a static output extent has no clean divisor in
    # ``_TUNE_AXIS_CHOICES`` (e.g. lm_head vocab=151669), let the planner pick
    # a normal BN/BM and emit per-thread ``if (n < N)`` guards instead of
    # falling back to a degenerate 1-thread CTA. Pointwise / matmul opt in here;
    # a symbolic free axis additionally forces masking in any mode (incl.
    # reduce) via ``m_forced_mask`` / ``n_forced_mask``, tuned at its hint.
    allow_masked = priority_mode in ("pointwise", "matmul")

    def _run(apply_pins: bool) -> list[TileParams]:
        return _enumerate_cartesian_impl(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=BN.narrow(bn_choices) if apply_pins else bn_choices,
            bm_choices=BM.narrow(bm_choices) if apply_pins else bm_choices,
            bk_choices=BK.narrow(bk_choices) if apply_pins else bk_choices,
            splitk_choices=SPLITK.narrow(splitk_choices) if apply_pins else splitk_choices,
            br_choices=BR.narrow(br_choices) if apply_pins else br_choices,
            fm_narrow=FM.narrow if apply_pins else None,
            fn_narrow=FN.narrow if apply_pins else None,
            max_threads_per_cta=ctx.max_threads_per_cta,
            min_k_chunks=min_k_chunks,
            priority_fn=priority_fn,
            allow_empty_threads=allow_empty_threads,
            allow_masked=allow_masked,
            m_axis_name=m_axis_name,
            n_axis_name=n_axis_name,
            m_forced_mask=m_forced_mask,
            n_forced_mask=n_forced_mask,
        )

    result = _run(apply_pins=True)
    if result or not planner_pin_set():
        return result
    return _run(apply_pins=False)


def _enumerate_cartesian_impl(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    bn_choices: tuple[int, ...],
    bm_choices: tuple[int, ...],
    bk_choices: tuple[int, ...],
    splitk_choices: tuple[int, ...],
    br_choices: tuple[int, ...],
    fm_narrow: _NarrowFn | None,
    fn_narrow: _NarrowFn | None,
    max_threads_per_cta: int,
    min_k_chunks: int,
    priority_fn: Callable[[TileParams], tuple[int, ...]],
    allow_empty_threads: bool = False,
    allow_masked: bool = False,
    m_axis_name: str | None = None,
    n_axis_name: str | None = None,
    m_forced_mask: bool = False,
    n_forced_mask: bool = False,
) -> list[TileParams]:
    """Pure cartesian enumeration: caller supplies the (possibly already
    pin-narrowed) choice tuples, the per-iteration FM/FN narrow callables
    (``None`` to skip), the per-thread K-chunk floor, and the sort key.
    No env reads, no mode dispatch.

    When ``allow_masked`` is set, candidates that violate the BN/BM-divides-E
    constraint are also admitted with the offending axis name recorded in
    ``TileParams.overhang``. ``m_forced_mask`` / ``n_forced_mask`` (set for a
    symbolic axis tuned at its hint) admit masking regardless of ``allow_masked``
    AND record the axis in ``overhang`` even when BN/BM divides the hint — the
    runtime extent is unknown, so the masked boundary guard is always needed.
    ``m_axis_name`` / ``n_axis_name`` are the names stamped into ``overhang``
    for the M / N axes respectively (the caller knows them from the LoopOp; we
    don't reconstruct).
    """
    # An axis is maskable if the mode allows masking (static non-divisor case)
    # or it's a forced-mask symbolic axis. ``E > 1`` rules out the size-1
    # phantom / degenerate axes.
    n_maskable = (allow_masked or n_forced_mask) and n_axis_name is not None and E_N > 1
    m_maskable = (allow_masked or m_forced_mask) and m_axis_name is not None and E_M > 1
    seen: set[TileParams] = set()
    ordered: list[TileParams] = []
    for bn in bn_choices:
        bn_c = min(bn, E_N)
        if bn_c < 1:
            continue
        n_nondiv = E_N % bn_c != 0
        if n_nondiv and not n_maskable:
            continue
        n_overhang = n_nondiv or (n_forced_mask and n_maskable)
        for bm in bm_choices:
            bm_c = min(bm, E_M)
            if bm_c < 1:
                continue
            m_nondiv = E_M % bm_c != 0
            if m_nondiv and not m_maskable:
                continue
            m_overhang = m_nondiv or (m_forced_mask and m_maskable)
            if bn_c * bm_c > max_threads_per_cta:
                continue
            # v1 cooperative constraint: BR > 1 ⇒ BN = BM = 1.
            br_eligible: tuple[int, ...] = br_choices if (bn_c == 1 and bm_c == 1) else (1,)
            for br in br_eligible:
                if br < 1 or E_K % br != 0:
                    continue
                if bn_c * bm_c * br > max_threads_per_cta:
                    continue
                # Lowering requires at least one BIND_THREAD axis on the
                # Tile (materializer's _materialize raises otherwise).
                # With bn = bm = br = 1 every output axis lands in BLOCK
                # / REGISTER and the THREAD set is empty — skip. Symbolic
                # all-free-axis kernels opt in to ``allow_empty_threads`` so
                # the planner can still emit a 1-thread-per-CTA variant
                # (slow but correct; perf optimization for strided
                # cooperative work over symbolic axes is M5+ follow-up).
                if bn_c * bm_c * br == 1 and not allow_empty_threads:
                    continue
                per_thread_K = E_K // br
                # FM/FN normally are divisors of the per-axis remainder so
                # the per-thread cell-grid covers cleanly. For an overhang
                # axis the remainder isn't well-defined (non-divisor BM/BN
                # leaves a fractional last tile), so any choice up to
                # _MAX_CELLS_PER_THREAD is admissible — the per-cell guard
                # in the masked Cond handles partial coverage.
                if m_overhang:
                    fm_candidates = tuple(f for f in _TUNE_F_CHOICES if f <= _MAX_CELLS_PER_THREAD)
                else:
                    fm_candidates = _divisors_up_to(E_M // bm_c, _MAX_CELLS_PER_THREAD)
                if fm_narrow is not None:
                    fm_candidates = fm_narrow(fm_candidates)
                for fm in fm_candidates:
                    if n_overhang:
                        fn_candidates = tuple(f for f in _TUNE_F_CHOICES if f <= _MAX_CELLS_PER_THREAD // fm)
                    else:
                        fn_candidates = _divisors_up_to(E_N // bn_c, _MAX_CELLS_PER_THREAD // fm)
                    if fn_narrow is not None:
                        fn_candidates = fn_narrow(fn_candidates)
                    for fn in fn_candidates:
                        for bk in bk_choices:
                            if per_thread_K % bk != 0:
                                continue
                            # Skip when this (bk, per_thread_K) yields fewer than
                            # ``min_k_chunks`` K chunks — matmul uses 2 to amortize
                            # K-loop overhead; reduce/pointwise pass 1 (a no-op
                            # given the divisor check above).
                            if per_thread_K > 1 and per_thread_K < bk * min_k_chunks:
                                continue
                            if bk > per_thread_K:
                                continue
                            k_o_total = per_thread_K // bk
                            for splitk in splitk_choices:
                                if k_o_total % splitk != 0:
                                    continue
                                overhang_axes: tuple[str, ...] = ()
                                if n_overhang and n_axis_name is not None:
                                    overhang_axes = (*overhang_axes, n_axis_name)
                                if m_overhang and m_axis_name is not None:
                                    overhang_axes = (*overhang_axes, m_axis_name)
                                params = TileParams(
                                    bn=bn_c,
                                    bm=bm_c,
                                    fm=fm,
                                    fn=fn,
                                    bk=bk,
                                    splitk=splitk,
                                    br=br,
                                    overhang=overhang_axes,
                                )
                                if params in seen:
                                    continue
                                seen.add(params)
                                ordered.append(params)

    ordered.sort(key=priority_fn, reverse=True)
    return ordered
