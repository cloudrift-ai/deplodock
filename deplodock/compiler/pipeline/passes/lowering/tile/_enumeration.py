"""Knob-cartesian enumeration for the partition planner.

Owns the planner's tunable knob globals (BN / BM / WM / WN / FM / FN / BK /
SPLITK / BR / MMA), the per-mode candidate tuples (matmul / pointwise
/ reduce), the pruned cartesian generator, and the per-mode priority/score
functions used to rank the resulting knob-row set.

Three layers:

1. The variant row — a plain knob dict (see the block comment above the
   priority functions): scalar tier carries {BN, BM, FM, FN, FK, BK,
   SPLITK, BR} (+ FKWIN / OVERHANG), warp tier {WN, WM, FM, FN, BK,
   SPLITK, MMA}; ``"MMA" in row`` discriminates. De-dup keys on
   ``frozenset(row.items())``.
2. Public ``enumerate_cartesian(...)`` — mode-dispatched wrapper that picks
   the candidate tuples + priority function for ``matmul`` / ``reduce`` /
   ``pointwise``, folds ``DEPLODOCK_<KNOB>`` env pins via ``Knob.narrow``,
   and retries without pins if every candidate was filtered out and any pin
   is set (fallback so peer-kernel pins don't strand a graph with an
   un-lowered LoopOp). For ``matmul`` the binding tier can be selected with
   a 2-tuple ``("matmul", "thread")`` / ``("matmul", "warp")`` mode; the
   bare ``"matmul"`` string is an alias for the thread tier.
3. Private ``_enumerate_cartesian_impl(...)`` — pure cartesian generator
   over caller-supplied (already-narrowed) tuples; produces scalar-tier
   scalar-tier knob rows (matmul / reduce / pointwise). A sibling
   ``_enumerate_warp_matmul_impl`` produces warp-tier rows
   (``MMA`` key set).

Kept as a non-pass sibling module (``_`` prefix → Pass loader skips) so
the planner imports the public API without going through
``importlib.import_module`` cross-pass dances. The planner module
(``010_partition_loops.py``) holds the Plan / Materialize / body-building
logic; everything about "which (BN, BM, ...) tuples are admissible and
in what order" lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.knob import Knob, KnobType

if TYPE_CHECKING:
    # Used only in annotations (the warp generator's ``atoms`` param).
    # A runtime import would cycle: _enumeration → ir.tile.ir → tuning →
    # _enumeration. The code only reads attributes off ``Atom`` instances it's
    # handed, never constructs the class, so a type-only import suffices.
    from deplodock.compiler.ir.tile.ir import Atom

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
# Axis-extent + per-cell-owner cell-count candidate tuples for the matmul /
# pointwise / reduce planner. These include the non-power-of-2 midpoints
# (``BM=8`` for the article's 256-thread ``8×32`` layout; ``FM/FN=6, 10, 12,
# 14, 20, 24, 26, 28, 40, 48, 96`` for register-budget-bound matmul tiles)
# that used to sit behind ``DEPLODOCK_WIDE_FM_FN=1``. Greedy now surfaces
# them by default — the score signals introduced alongside this (smem-fit
# safety + TMA-eligibility bonus + SPLITK penalty in
# ``score_tile_geometry``, plus the ``LoweringError`` that ``pipeline.py``
# raises when a chosen variant fails ``validate(ctx)``) keep greedy from
# silently picking a wider variant whose downstream lowering doesn't
# validate; the wider candidate set just adds 18 % more leaves to the
# enumeration cartesian for a couple of extra seconds of autotune wall.
_TUNE_AXIS_CHOICES: tuple[int, ...] = (1, 8, 16, 32, 64, 128, 256)
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 26, 28, 32, 40, 48, 64, 96, 128)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)
# Cooperative-K thread count. v1: BR > 1 requires BN = BM = 1 (single THREAD
# axis for materializer's _single_thread_var).
_BR_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# Cap on per-thread cell-product. NVRTC compile time explodes past this.
_MAX_CELLS_PER_THREAD: int = 128

# Scalar-tier knobs (THREAD-binding). MMA warp-tier rows don't carry these —
# their per-axis warp count lives on WM/WN below.
BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)")
BR = Knob("BR", KnobType.INT, hints=_BR_CANDIDATES, help="Cooperative-K thread count (1 = pure serial chunked reduce)")
# Warp-tier knobs (WARP-binding, MMA matmul only). Plumbed at M1; populated
# with real choices in M3 when the warp enumerator goes live.
_TUNE_WARP_AXIS_CHOICES: tuple[int, ...] = (1, 2, 4, 8)
WN = Knob("WN", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA innermost WARP count along matmul output N")
WM = Knob("WM", KnobType.INT, hints=_TUNE_WARP_AXIS_CHOICES, help="CTA outer WARP count along matmul output M")


def _mma_features(mma: object) -> dict[str, float]:
    """Learned-prior featurizer for the ``MMA`` knob: expand an atom kind into
    physical cell/dtype properties (cell shape, group size, operand bit widths)
    via ``ATOM_REGISTRY`` so a new atom generalizes by geometry/dtype rather than
    a one-hot id. The scalar tier (``0`` / unknown kind) is ``MMA_tier=0``.
    Wired onto the ``MMA`` Knob so ``knob.knob_features`` needs no special-case.
    Lazy import — ``_enumeration`` is imported while the tile IR is still being
    built up."""
    from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY  # noqa: PLC0415

    atom = ATOM_REGISTRY.get(str(mma))
    if atom is None:
        return {"MMA_tier": 0.0}
    m, n, k = atom.shape
    return {
        "MMA_tier": 1.0,
        "MMA_atom_m": float(m),
        "MMA_atom_n": float(n),
        "MMA_atom_k": float(k),
        "MMA_group_size": float(atom.group_size),
        "MMA_a_bits": float(atom.operand_dtype("a").nbytes * 8),
        "MMA_acc_bits": float(atom.operand_dtype("c").nbytes * 8),
    }


MMA = Knob(
    "MMA",
    KnobType.STR,
    hints=(),
    help="Warp-tier MMA control: 0 = scalar-only; 1/true (default) = auto-enumerate; hardware matmul atom kind (MMA/wgmma/...)",
    aliases=("ATOM_KIND",),
    features=_mma_features,
)

_MMA_FALSY = frozenset({"0", "false", "no", "off"})
_MMA_TRUTHY = frozenset({"1", "true", "yes", "on"})


def mma_mode() -> tuple[bool, str | None]:
    """Decode the ``MMA`` knob into ``(enabled, pinned_kind)``.

    Unset / empty / truthy → ``(True, None)`` (auto-enumerate, the default);
    falsy → ``(False, None)`` (scalar-only); any other string is an atom-kind
    name → ``(True, name)``. A name the registry doesn't know simply narrows
    the eligible atoms to nothing and the warp tier falls to scalar. Reads
    through :meth:`Knob.raw`, so the ``ATOM_KIND`` alias spelling decodes
    identically."""
    raw = MMA.raw()
    if raw is None:
        return True, None
    s = raw.strip()
    if not s or s.lower() in _MMA_TRUTHY:
        return True, None
    if s.lower() in _MMA_FALSY:
        return False, None
    return True, s


# Tier-shared knobs (same arithmetic role in both scalar and warp tiers).
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-cell-owner cells along the matmul N (output) axis")
# Reduce-axis register tier (non-matmul reduces only). FK independent
# accumulators strip-mine the K (reduce) axis for multiple-accumulator ILP —
# see ``plans/fk-register-tile-reductions.md``. Matmul gets the same win
# through FM/FN replicating output cells, so it keeps FK=1.
FK = Knob("FK", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread independent accumulators along the reduce (K) axis (1 = single)")
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")

# ``ATOM_KIND`` is deliberately absent: the kind pin rides the ``MMA`` knob
# (``DEPLODOCK_ATOM_KIND`` is read as ``MMA``'s alias via ``Knob.raw``), so
# listing it here would double-count the same pin.
_PLANNER_KNOBS: tuple[Knob, ...] = (BN, BM, FM, FN, FK, BK, SPLITK, BR, WN, WM, MMA)


def planner_pin_set() -> bool:
    """True if any planner knob has its ``DEPLODOCK_<NAME>`` env pin set
    (alias spellings included — ``Knob.raw`` consults them). Used by
    ``enumerate_cartesian`` to gate the peer-kernel fallback."""
    return any(k.raw() is not None for k in _PLANNER_KNOBS)


def planner_pin_snapshot() -> tuple[tuple[str, str | None], ...]:
    """Live ``(name, effective env value)`` snapshot of every planner knob
    pin (``Knob.raw`` — alias spellings resolve to the same entry as the
    primary). Folded into the partition planner's enumeration-memo key so a
    pin flipped mid-process (tests use ``config.set_knob``) lands on a fresh
    key instead of replaying a stale cached enumeration."""
    return tuple((k.name, k.raw()) for k in _PLANNER_KNOBS)


# A planner variant row IS its knob dict — the exact keys the materialized
# TileOp will carry (the planner merges it over the LoopOp's carry-forward
# knobs when stamping / scoring):
#
#   scalar tier  {BN, BM, FM, FN, FK, BK, SPLITK, BR}
#                (+ ``FKWIN``: the fp16 half2 accumulation-window length —
#                disambiguates window ``FK`` from the reduce strip-mine, see
#                ``plans/fk-half2-fp16-matmul.md``; + ``OVERHANG``: tuple of
#                masked output-axis names for non-divisor tiles)
#   warp tier    {WN, WM, FM, FN, BK, SPLITK, MMA}
#                (``"MMA" in row`` discriminates the tier; the ``Atom`` spec
#                is ``ATOM_REGISTRY[row["MMA"]]``)
#
# There is deliberately no row class: one representation flows end-to-end —
# enumeration → fork-tree levels → ``TileOp.lazy_score`` → body builder →
# ``TileOp.knobs`` → DB rows — so any recorded knob dict is directly
# scoreable and materializable. De-dup inside the impls keys on
# ``frozenset(row.items())``.


def _matmul_thread_gate(r: dict) -> bool:
    """The heuristic-plausible band for thread-tier matmul tiles, distilled from
    the measured ``GOLDEN_CONFIGS`` (every recorded golden satisfies it). Used to
    prune the enumeration so the learned prior can't extrapolate its ``mean_score``
    argmax onto an *unbenched* degenerate tile (e.g. ``BN=8, tile_n=8``) and
    override the golden-shaped option-0 — the failure that left greedy-with-prior
    reproducing 0/23 goldens even after a clean tune. Coalesced wide inner axis,
    short outer axis, large K-chunk, light split-K, clean output-column width. The
    caller falls back to the ungated set when this empties (tiny / unusual shapes
    with no in-band candidate), so it only ever *narrows*, never strands a graph."""
    bn, bm = r["BN"], r["BM"]
    threads = bn * bm
    tile_n = bn * r["FN"]
    return (
        16 <= bn <= 64
        and 8 <= bm <= 16
        and bn >= bm
        and r["BK"] >= 32
        and r["SPLITK"] <= 2
        and threads in (128, 256, 512, 1024)
        and tile_n in (32, 64, 128)
    )


def _divisors_up_to(n: int, cap: int) -> tuple[int, ...]:
    """Divisors of ``n`` ≤ ``cap``, ascending. FM / FN candidate set — a
    divisor of ``E / bm_c`` automatically satisfies the divisibility check."""
    if n < 1 or cap < 1:
        return ()
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def _priority_matmul_thread(p: dict) -> tuple[int, ...]:
    # Tiebreaker for the enumeration sort. The Fork-tree builder re-sorts
    # each level by ``score_tile_geometry`` (the MCTS prior) — see
    # ``compiler/pipeline/fork.py`` — so this ordering only affects
    # ties within a score band.
    #
    # Fat-tile regime (macro area ≥ 8192 cells): keep the explicit BK
    # preference but flatten ``{16, 32, 8}`` to a near-tie — earlier tuning
    # runs at 2048³ showed the golden's BK=32 was never visited under
    # patience 200 because BK=16 alone (rank 6) drained the budget. The
    # cuBLAS-band preference (16 first) was a per-shape coincidence, not a
    # universal truth. BK=64 still demotes (cooperative-load granularity
    # too coarse for fp32 lanes), BK=1/2 stay last (the K-loop collapses
    # to ≤2 steps; the -O1/-O3 ranking inversion makes them look
    # artificially fast during tuning).
    threads = p["BN"] * p["BM"]
    macro_area = p["BM"] * p["FM"] * p["BN"] * p["FN"]
    fat_tile = macro_area >= 8192
    if fat_tile:
        _BK_FAT_PREF = {16: 5, 32: 5, 8: 4, 64: 3, 4: 2, 2: 1, 1: 0}
        bk_key = _BK_FAT_PREF.get(p["BK"], -1)
    else:
        bk_key = p["BK"]
    return (
        int(fat_tile),  # fat-tile variants surface first
        min(p["FM"] * p["FN"], 128),  # cells/thread, capped at the register-tile sweet spot
        -abs(256 - threads),  # threads near 256
        bk_key,  # cuBLAS-band BK tier, then large BK otherwise
        -p["SPLITK"],
        # fp16 half2 window FK: ``FK=1`` (scalar fp32 accumulate) ranks first so
        # the greedy default stays byte-identical; among the windowed variants
        # prefer the 2–8 band. Sits below the geometry keys (a low tiebreak).
        int(p["FK"] == 1),
        int(2 <= p["FK"] <= 8),
        -len(p.get("OVERHANG", ())),
    )


def _priority_matmul_warp(p: dict, *, ctx: Context | None = None) -> tuple[int, ...]:
    # Warp-tier ranking for tensor-core MMA matmul (fp16 / bf16 with f32
    # accumulator, swizzled s16816 ``mma_m16n8k16_*`` atom on sm_90+).
    #
    # **Non-TMA arches (sm_70 / sm_80)**: ``cp.async``-staged mma.sync pays
    # per-K_o overhead in cooperative-load instructions, so larger BK
    # (fewer K_o iters) amortizes best and threads-near-256 (8 warps)
    # keeps the per-warp cooperative slice dense. Ranked on cells≈16 +
    # square FM/FN, kept verbatim (sweeps on sm_80 / sm_90 confirmed).
    #
    # **TMA arches (sm_90+, sm_90 Hopper + sm_120 Blackwell)**: a 2026 pinned
    # sweep on RTX 5090 (sm_120) over 1024² / 2048² / 4096² fp16 (each tile
    # paired with ``WARP_SPECIALIZE`` — see ``085_warp_specialize``) found the
    # winner is a **square 64×64 output tile on a 4-warp CTA** with WS on,
    # *not* the cells≈16 / 128×128 shape the pre-2026 prior favored. 2048²:
    #
    #     WM WN FM FN  out-tile  warps  WS  µs    (cuBLAS/eager ≈ 99)
    #      2  2  2  4   64×64       4    1   94   ← winner (1.05× cuBLAS)
    #      1  4  4  2   64×64       4    1   94     (same tile, WM/WN swap)
    #      2  4  2  2   64×64       8    1   97
    #      1  4  4  4   64×128      4    1  102
    #      2  4  4  4  128×128      8    1  111     (old prior's pick)
    #      2  2  2  4   64×64       4    0  110-115 (WS off — ~17% worse)
    #      1  4  2  4   32×128      4    1  142-155 (skewed same-area tile)
    #
    # The 64×64 tile keeps per-lane registers low (≈48-66 vs 166 at 128×128)
    # so 4 such CTAs and WS's extra producer warp fit; skewed same-area tiles
    # (32×128 / 128×32) lose operand reuse and run 50% slower. Encode as:
    # target a 64×64 output tile (``area``), break ties toward a square tile
    # (``|m_tile − n_tile|``), a 4-warp CTA, a balanced warp grid, then BK=2.
    # (Same-tile warp splits — WM2 WN2 vs WM1 WN4, both 64×64 — are perf-equal;
    # which one wins the residual tie is not load-bearing.)
    threads = p["WN"] * p["WM"] * 32
    cells = p["FM"] * p["FN"]
    tma_capable = ctx is not None and ctx.compute_capability >= (9, 0)
    if tma_capable:
        m_tile = p["WM"] * p["FM"] * 16
        n_tile = p["WN"] * p["FN"] * 8
        return (
            -abs(m_tile * n_tile - 64 * 64),  # 64×64 output tile (4096 elems/CTA)
            -abs(m_tile - n_tile),  # square tile beats skewed same-area
            -abs(threads - 128),  # 4-warp CTA
            -abs(p["WM"] - p["WN"]),  # balanced warp grid (avoid WN1/WM1 skew)
            -abs(p["BK"] - 2),
            -p["BK"],
            -p["SPLITK"],
            -len(p.get("OVERHANG", ())),
        )
    return (-abs(cells - 16), -abs(p["FM"] - p["FN"]), -abs(256 - threads), p["BK"], -p["SPLITK"], -len(p.get("OVERHANG", ())))


def _priority_pointwise(p: dict) -> tuple[int, ...]:
    # Memory-bandwidth bound — fewer cells/thread → more CTAs → better
    # SM occupancy. Threads near 256. Final tiebreaker: clean-divisor wins.
    threads = p["BN"] * p["BM"]
    return (-(p["FM"] * p["FN"]), -abs(256 - threads), -len(p.get("OVERHANG", ())))


def _priority_reduce(p: dict) -> tuple[int, ...]:
    # Warp-sized BR enables warp-shuffle Combine; threads near 256.
    # Static reduce never masks; a symbolic free axis forces masking (every
    # variant then carries the same overhang, so no clean-divisor tiebreak).
    #
    # FK ordering, all *below* BR / threads (the geometry that dominates a
    # memory-bound reduce): ``FK=1`` ranks first so the greedy (non-tuned)
    # ``compile`` pick is byte-identical to the pre-FK planner — FK only adds
    # extra leaves the *tuner* explores, never shifting the default. Among the
    # FK>1 leaves the 2–4 ILP sweet spot is preferred so the search visits a
    # couple of multiple-accumulator variants early.
    threads = p["BN"] * p["BM"] * p["BR"]
    return (min(p["BR"], 256), -abs(256 - threads), int(p["FK"] == 1), int(2 <= p["FK"] <= 4), p["BK"], -p["SPLITK"])


_NarrowFn = Callable[[tuple[int, ...]], tuple[int, ...]]


def enumerate_cartesian(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    ctx: Context,
    priority_mode: str | tuple[str, ...],
    force_splitk_one: bool = False,
    m_axis_name: str | None = None,
    n_axis_name: str | None = None,
    m_forced_mask: bool = False,
    n_forced_mask: bool = False,
    atoms: tuple[Atom, ...] = (),
    fp16_window: bool = False,
) -> list[dict]:
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
    # Normalize ``priority_mode`` — accept the bare strings ``"matmul"`` /
    # ``"reduce"`` / ``"pointwise"`` plus the 2-tuple ``("matmul", "thread")``
    # / ``("matmul", "warp")`` per the MMA plan's Design decision 11. ``"matmul"``
    # is an alias for the thread tier (today's only matmul path).
    if isinstance(priority_mode, str):
        mode_kind: str = priority_mode
        mode_tier: str = "thread" if priority_mode == "matmul" else ""
    else:
        mode_kind, mode_tier = priority_mode[0], priority_mode[1]

    # ``_TUNE_AXIS_CHOICES`` already includes ``1`` so tiny output extents
    # (e.g. ``torch.matmul`` of 4×3×2) and global-reduce kernels with a
    # phantom size-1 outer axis survive enumeration. ``_wrap_tower`` drops
    # the resulting size-1 THREAD axes before they reach the IR, so the
    # broader search space only changes behavior for tiny shapes.
    if mode_kind == "matmul" and mode_tier == "warp":
        # Warp-tier matmul — enumerate one knob row per eligible
        # ``Atom``.
        return _enumerate_warp_matmul_impl(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            ctx=ctx,
            force_splitk_one=force_splitk_one,
            atoms=atoms,
            m_axis_name=m_axis_name,
            n_axis_name=n_axis_name,
            m_forced_mask=m_forced_mask,
            n_forced_mask=n_forced_mask,
        )
    # ``sweep_fk`` enables the reduce-axis multiple-accumulator (FK) sweep —
    # only for non-matmul reduces. Matmul gets the ILP win from FM/FN replicating
    # output cells; pointwise has no reduce axis to strip-mine.
    # ``window_fk`` enables the orthogonal fp16 matmul half2-window FK (the
    # window length = stage chunk ``bk``, even), passed via ``fp16_window``.
    sweep_fk = False
    window_fk = False
    if mode_kind == "matmul":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,) if force_splitk_one else _SPLITK_CANDIDATES
        br_choices: tuple[int, ...] = (1,)
        min_k_chunks = 1
        # Order by the offline-learned occupancy-aware heuristic (ranks the
        # recorded goldens at median rank ~7) rather than the old static tiebreak
        # ``_priority_matmul_thread`` (which buried them at ~10⁴ — top picks were
        # degenerate ``BN=1, BM=256``). This sets both the greedy default
        # (option-0) and the order the tuner's inner search visits leaves under
        # patience, so golden-shaped configs actually get benched. Lazy import
        # breaks the ``heuristic → _enumeration`` cycle.
        from deplodock.compiler.pipeline.search.heuristic import score_matmul_thread  # noqa: PLC0415

        def priority_fn(p: dict) -> float:  # type: ignore[misc]  # shape-bound key; other modes assign tuple-returning fns
            return score_matmul_thread(p, E_M, E_N, E_K)

        window_fk = fp16_window
    elif mode_kind == "reduce":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,)
        br_choices = _BR_CANDIDATES
        min_k_chunks = 1
        priority_fn = _priority_reduce
        sweep_fk = True
    elif mode_kind == "pointwise":
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
    allow_empty_threads = E_M == 1 and E_N == 1 and mode_kind in ("pointwise", "matmul")

    # Masked tiles: when a static output extent has no clean divisor in
    # ``_TUNE_AXIS_CHOICES`` (e.g. lm_head vocab=151669), let the planner pick
    # a normal BN/BM and emit per-thread ``if (n < N)`` guards instead of
    # falling back to a degenerate 1-thread CTA. Pointwise / matmul opt in here;
    # a symbolic free axis additionally forces masking in any mode (incl.
    # reduce) via ``m_forced_mask`` / ``n_forced_mask``, tuned at its hint.
    allow_masked = mode_kind in ("pointwise", "matmul")

    # Non-divisor FM/FN candidates (e.g. the article's FM=26 at 2048³) get
    # admitted on masking-eligible axes via the wider ``_TUNE_F_CHOICES``
    # sweep — but only for pure-matmul kernels. The fused-prologue path
    # (SDPA P@V, gated-MLP, anything that sets ``force_splitk_one``) feeds
    # the matmul accumulator through a softmax-style scale per row, and a
    # non-divisor FN over a masked register tile leaves some cells partially
    # scaled (an accuracy bug — SDPA per-head fails the eager-tolerance
    # check). Restrict to clean divisors in those cases. Pointwise stays
    # divisor-sweepable too; only plain matmul gets the wider sweep.
    allow_nondivisor_f_on_mask = mode_kind == "matmul" and not force_splitk_one

    def _run(apply_pins: bool) -> list[dict]:
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
            fk_narrow=FK.narrow if apply_pins else None,
            sweep_fk=sweep_fk,
            window_fk=window_fk,
            max_threads_per_cta=ctx.max_threads_per_cta,
            min_k_chunks=min_k_chunks,
            priority_fn=priority_fn,
            allow_empty_threads=allow_empty_threads,
            allow_masked=allow_masked,
            allow_nondivisor_f_on_mask=allow_nondivisor_f_on_mask,
            m_axis_name=m_axis_name,
            n_axis_name=n_axis_name,
            m_forced_mask=m_forced_mask,
            n_forced_mask=n_forced_mask,
        )

    result = _run(apply_pins=True)
    if not result and planner_pin_set():
        result = _run(apply_pins=False)
    # Thread-tier matmul: narrow to the heuristic-plausible band so neither the
    # tuner's exploration nor the learned prior's greedy argmax wanders onto an
    # unbenched degenerate tile. Fall back to the full set if the gate empties.
    if mode_kind == "matmul" and mode_tier != "warp":
        gated = [r for r in result if _matmul_thread_gate(r)]
        if gated:
            result = gated
    return result


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
    fk_narrow: _NarrowFn | None = None,
    sweep_fk: bool = False,
    window_fk: bool = False,
    max_threads_per_cta: int,
    min_k_chunks: int,
    priority_fn: Callable[[dict], tuple[int, ...]],
    allow_empty_threads: bool = False,
    allow_masked: bool = False,
    allow_nondivisor_f_on_mask: bool = True,
    m_axis_name: str | None = None,
    n_axis_name: str | None = None,
    m_forced_mask: bool = False,
    n_forced_mask: bool = False,
) -> list[dict]:
    """Pure cartesian enumeration: caller supplies the (possibly already
    pin-narrowed) choice tuples, the per-iteration FM/FN narrow callables
    (``None`` to skip), the per-thread K-chunk floor, and the sort key.
    No env reads, no mode dispatch.

    When ``allow_masked`` is set, candidates that violate the BN/BM-divides-E
    constraint are also admitted with the offending axis name recorded in
    the row's ``OVERHANG`` entry. ``m_forced_mask`` / ``n_forced_mask`` (set for a
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
    seen: set[tuple] = set()
    ordered: list[dict] = []
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
                #
                # A pin (``fm_narrow`` returns a 1-tuple authoritatively;
                # see ``Knob.narrow``) may force a non-divisor FM/FN even
                # when BM/BN cleanly divides E_M/E_N. In that case BM·FM
                # / BN·FN doesn't divide E_M / E_N, so the masking guard
                # is needed — we admit the value and flip overhang for
                # that variant.
                # Masking-eligible kernels also enumerate every
                # ``_TUNE_F_CHOICES`` value, not just divisors of
                # ``E_M // bm_c``, so register-budget-bound tiles like the
                # article's FM=26 (non-divisor of 256) surface in the search.
                # Fused-prologue kernels (``allow_nondivisor_f_on_mask=False``)
                # opt out — see the comment in :func:`enumerate_cartesian`.
                if m_overhang or (allow_nondivisor_f_on_mask and m_maskable):
                    fm_candidates = tuple(f for f in _TUNE_F_CHOICES if f <= _MAX_CELLS_PER_THREAD)
                else:
                    fm_candidates = _divisors_up_to(E_M // bm_c, _MAX_CELLS_PER_THREAD)
                if fm_narrow is not None:
                    fm_candidates = fm_narrow(fm_candidates)
                for fm in fm_candidates:
                    if fm < 1 or fm > _MAX_CELLS_PER_THREAD:
                        continue
                    # Pinned FM may violate BM·FM | E_M; force masking when
                    # admissible (allow_masked + named axis).
                    fm_nondiv = E_M % (bm_c * fm) != 0
                    if fm_nondiv and not m_maskable:
                        continue
                    fm_overhang = m_overhang or fm_nondiv
                    if n_overhang or (allow_nondivisor_f_on_mask and n_maskable):
                        fn_candidates = tuple(f for f in _TUNE_F_CHOICES if f <= _MAX_CELLS_PER_THREAD // fm)
                    else:
                        fn_candidates = _divisors_up_to(E_N // bn_c, _MAX_CELLS_PER_THREAD // fm)
                    if fn_narrow is not None:
                        fn_candidates = fn_narrow(fn_candidates)
                    for fn in fn_candidates:
                        if fn < 1 or fm * fn > _MAX_CELLS_PER_THREAD:
                            continue
                        fn_nondiv = E_N % (bn_c * fn) != 0
                        if fn_nondiv and not n_maskable:
                            continue
                        fn_overhang = n_overhang or fn_nondiv
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
                                # FK strip-mines the per-thread serial K loop
                                # (extent ``K_o_ext = k_o_total // splitk``) into
                                # FK independent accumulators. Only the reduce
                                # mode sweeps it (``sweep_fk``); FK must divide
                                # K_o_ext (so the outer serial loop tiles cleanly)
                                # and ``FK · FM · FN`` must stay within the
                                # per-thread register budget. Matmul / pointwise
                                # keep FK=1.
                                K_o_ext = k_o_total // splitk
                                if sweep_fk:
                                    # Reduce path: FK strip-mines the serial K loop
                                    # into independent accumulators (divisor of K_o_ext).
                                    fk_candidates = _divisors_up_to(K_o_ext, _MAX_CELLS_PER_THREAD // (fm * fn))
                                elif window_fk and bk % 2 == 0 and bk >= 2:
                                    # fp16 matmul half2 window: the window length is the
                                    # stage chunk ``bk`` (even). The window is a runtime
                                    # ``for k in 0..bk/2`` loop of ``__hfma2`` packed
                                    # multiply-adds (NOT fully unrolled), so a large bk —
                                    # same staging as the fp32 path — keeps the half2
                                    # throughput win without bloating registers. FK=1
                                    # (scalar fp32 accumulate) stays the default.
                                    fk_candidates = (1, bk)
                                else:
                                    fk_candidates = (1,)
                                if fk_narrow is not None:
                                    fk_candidates = fk_narrow(fk_candidates)
                                for fk in fk_candidates:
                                    # Window FK rides on bk (FK=bk), so it skips the
                                    # K_o_ext-divisor / cell-budget gates that the reduce
                                    # FK obeys; only the reduce FK multiplies output cells.
                                    if fk < 1:
                                        continue
                                    if window_fk and fk > 1 and fk != bk:
                                        # The half2 window length is the stage chunk; an
                                        # FK pin whose value isn't this bk (authoritative
                                        # ``Knob.narrow``) doesn't apply to this variant.
                                        continue
                                    if fk > 1 and not window_fk and (fm * fn * fk > _MAX_CELLS_PER_THREAD or K_o_ext % fk != 0):
                                        continue
                                    overhang_axes: tuple[str, ...] = ()
                                    if fn_overhang and n_axis_name is not None:
                                        overhang_axes = (*overhang_axes, n_axis_name)
                                    if fm_overhang and m_axis_name is not None:
                                        overhang_axes = (*overhang_axes, m_axis_name)
                                    # De-dup on the value tuple (cheaper than
                                    # hashing the row dict's items) BEFORE
                                    # building the row.
                                    key = (bn_c, bm_c, fm, fn, fk, bk, splitk, br, overhang_axes)
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    params = {
                                        "BN": bn_c,
                                        "BM": bm_c,
                                        "FM": fm,
                                        "FN": fn,
                                        "FK": fk,
                                        "BK": bk,
                                        "SPLITK": splitk,
                                        "BR": br,
                                    }
                                    if window_fk and fk > 1:
                                        params["FKWIN"] = fk
                                    if overhang_axes:
                                        params["OVERHANG"] = overhang_axes
                                    ordered.append(params)

    ordered.sort(key=priority_fn, reverse=True)
    return ordered


_MAX_CELLS_PER_WARP_CELL: int = 64  # cells-per-cell-owner cap for the warp tier


def _enumerate_warp_matmul_impl(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    ctx: Context,
    force_splitk_one: bool,
    atoms: tuple[Atom, ...],
    m_axis_name: str | None,  # noqa: ARG001 — overhang plumbing for M9 (skewed shapes)
    n_axis_name: str | None,  # noqa: ARG001
    m_forced_mask: bool,  # noqa: ARG001 — symbolic-axis masking for warp tier lands in M9
    n_forced_mask: bool,  # noqa: ARG001
) -> list[dict]:
    """Pure cartesian enumeration for the warp tier — produces
    warp-tier knob rows. Parallel in spirit to
    :func:`_enumerate_cartesian_impl` but with the warp divisibility +
    per-CTA-thread budget formulae (per Design decision 11 of
    ``plans/mma-fragment-factorization.md``):
    ``E_M % (wm·fm·atom_m) == 0`` / ``wn·wm·32 ≤ max_threads_per_cta``.

    No mask / overhang support at M3 — the warp tier rejects non-divisor
    extents instead of register-tiling a per-cell guard. Symbolic axes
    fall through here too (M9 extends with masked tiles for skewed
    matmul-shapes). BR is forced to 1 (MMA + cooperative-K is incompatible
    in v1; see Failure modes).
    """
    if not atoms:
        return []

    out: list[dict] = []
    seen: set[tuple] = set()
    # Knob-narrow gate: fold ``DEPLODOCK_<KNOB>`` env pins into the candidate
    # tuples here at the warp tier, parallel to ``enumerate_cartesian``'s
    # ``_run(apply_pins=True)`` for the scalar tier. The plan B/C sweep and
    # any CLI repro (``DEPLODOCK_WM=2 DEPLODOCK_WN=2 DEPLODOCK_BK=2 …``) rely
    # on this — without it, warp-tier pins silently fall through to the
    # scalar tier's narrows and the warp tier enumerates the full cartesian
    # regardless. ``MMA=<kind name>`` narrows the caller-supplied ``atoms``
    # (by kind name) so a pin scopes the picker to a single kind.
    wm_choices = WM.narrow(_TUNE_WARP_AXIS_CHOICES)
    wn_choices = WN.narrow(_TUNE_WARP_AXIS_CHOICES)
    bk_choices = BK.narrow(_BK_CANDIDATES)
    # The ``MMA=<kind>`` pin narrows by kind *name*; keep the atoms whose
    # name survives (authoritative like ``Knob.narrow`` — an unknown name
    # keeps nothing and the warp tier falls to scalar).
    _pin = mma_mode()[1]
    _kept = {_pin} if _pin is not None else {a.name for a in atoms}
    atoms = tuple(a for a in atoms if a.name in _kept)
    # v1: MMA + SPLITK > 1 needs atomic-add on MmaStore (cross-CTA
    # accumulation), which isn't wired yet. Force SPLITK=1 so each output
    # cell is written by exactly one CTA. force_splitk_one is honoured
    # but otherwise redundant here. M9 / future plan can enable
    # split-K MMA once atomic MmaStore lands.
    splitk_choices: tuple[int, ...] = (1,)
    for atom in atoms:
        atom_m, atom_n, atom_k = atom.shape
        if E_M % atom_m != 0 or E_N % atom_n != 0 or E_K % atom_k != 0:
            # Outer divisibility — the eligibility predicate already enforced
            # this for the kind it gated, but a different kind in the same
            # registry could disagree (M9 introduces skewed shapes), so
            # double-check structurally here.
            continue
        # Per-axis cell counts available after the atom cell is fixed.
        cells_M = E_M // atom_m
        cells_N = E_N // atom_n
        # K iterations consumed by one mma_sync. Total K cells = E_K / atom_k;
        # BK below is the inner stage_inner trip count (number of mma_syncs
        # per K_o iteration).
        k_cells_total = E_K // atom_k
        for wm in wm_choices:
            if cells_M % wm != 0:
                continue
            for wn in wn_choices:
                if cells_N % wn != 0:
                    continue
                threads = wn * wm * 32
                if threads > ctx.max_threads_per_cta:
                    continue
                cells_M_per_warp = cells_M // wm
                cells_N_per_warp = cells_N // wn
                fm_candidates = _divisors_up_to(cells_M_per_warp, _MAX_CELLS_PER_WARP_CELL)
                fm_candidates = FM.narrow(fm_candidates)
                for fm in fm_candidates:
                    fn_cap = _MAX_CELLS_PER_WARP_CELL // fm if fm > 0 else _MAX_CELLS_PER_WARP_CELL
                    fn_candidates = _divisors_up_to(cells_N_per_warp, fn_cap)
                    fn_candidates = FN.narrow(fn_candidates)
                    for fn in fn_candidates:
                        for bk in bk_choices:
                            # ``bk`` is the inner stage_inner trip count
                            # (mma_syncs per K_o iter); each iter consumes
                            # ``atom_k`` K-elements. ``k_cells_total`` must
                            # divide cleanly by ``bk``.
                            if bk < 1 or k_cells_total % bk != 0:
                                continue
                            k_o_total = k_cells_total // bk
                            for splitk in splitk_choices:
                                if splitk < 1 or k_o_total % splitk != 0:
                                    continue
                                key = (wn, wm, fm, fn, bk, splitk, atom.name)
                                if key in seen:
                                    continue
                                seen.add(key)
                                out.append(
                                    {
                                        "WN": wn,
                                        "WM": wm,
                                        "FM": fm,
                                        "FN": fn,
                                        "BK": bk,
                                        "SPLITK": splitk,
                                        "MMA": atom.name,
                                    }
                                )

    out.sort(key=lambda p: _priority_matmul_warp(p, ctx=ctx), reverse=True)
    return out
