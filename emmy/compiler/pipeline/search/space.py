"""The **search space** — every ``Knob`` declaration plus the enumeration value grids, in one file.

**INVARIANT: every ``Knob`` instance is declared here, and nowhere else.**
:mod:`~emmy.compiler.pipeline.knob` owns the ``Knob`` *descriptor* (the dataclass), the registry,
and the env plumbing; :mod:`~emmy.compiler.pipeline.search.features` owns the featurizers; this
module owns the concrete *declarations* AND the candidate-value generators — so the whole tunable
surface (dimensions × values) is visible in one place. A rule that decides a knob imports it from here
(``from emmy.compiler.pipeline.search.space import VECTORIZE_LOADS``) rather than declaring its
own. ``knob.registry()`` still discovers them via ``knob._walk_modules`` (any package module with
module-level ``Knob`` attributes is walked, and this module is imported at pipeline startup by the
rules that consume its knobs). When adding a knob, declare it here and import it into the owning rule.

Scope note: this module holds the **static** space only — the declared dimensions and their bounded
candidate grids. Per-kernel legality (the warp static-K divisibility check, the stage resolvers, coop
eligibility, the ``_COOP_*`` constants) stays with the scheduler in
``passes/lowering/tile/_schedule.py`` — the legal subset is a function of the node.

Three groups:

- **Schedule codec knobs** (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) — the tile-lowering schedule
  fork points that spell the ir schedule codecs (:mod:`emmy.compiler.ir.schedule`). Decided in
  the ``_schedule`` helper inside ``lowering/tile/010_recognize`` and materialized in
  ``lowering/kernel/010_materialize``. Each is the **ephemeral** codec spelling: it resolves into a
  schedule slice (``ReducePlan`` / ``TilePlan`` / ``Stage`` / ``WarpSpec``) and rides on ``TileOp.knobs``
  so the learned prior featurizes / tunes the decision. ``off=""`` (the conservative serial / per-cell /
  gmem-direct / uniform default) is auto-stamped on kernels the pass doesn't schedule.
- **The structural placement pin** (``PLACE``) — pin-only: where an intermediate edge lives, registers
  (``fuse``) or memory (``cut``), per edge-class element (``PLACE@<element>`` via ``EMMY_KNOBS``).
- **Kernel-lowering policy knobs** (``VECTORIZE_LOADS`` / ``INTERLEAVE_LOADS``) — boolean codegen
  policies recorded on the kernel op (idempotence + env override), on by default and not search
  dimensions (``hints=(True,)``).
"""

from __future__ import annotations

import logging

from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.pipeline.knob import Knob, KnobType

logger = logging.getLogger(__name__)

# --- Schedule codec knobs ---------------------------------------------------

# The reduce-axis partition codec. ``off=""`` = the scalar serial fold.
REDUCE = Knob(
    "REDUCE",
    KnobType.STR,
    help="Reduce-axis partition codec (g<n> cta / b<n> coop / r<n> reg; empty=serial). "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# The free-axis output tile — the **unified output-fragment** knob. A contraction's output tile is
# *either* the scalar register sub-tile (``n<N>[x<M>]`` parallel thread-tile / ``f<fn>[x<fm>]``
# register sub-tile) *or* the tensor-core warp mma tile (``a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>``),
# never both. The value self-discriminates: an ``a:<atom>`` token selects the warp fragment (see
# ``schedule.is_warp_codec``); otherwise the scalar fragment. Only a ``CONTRACTION`` tiles its output
# today; ``off=""`` auto-stamps everything else. The codec is the sole on-dict spelling — the
# learned-prior featurizer (``features.mma_atom`` / ``is_warp`` / ``_free_slots`` / ``tile_signature``)
# parses it directly (no legacy ``WM``/``WN``/``MMA`` keys).
TILE = Knob(
    "TILE",
    KnobType.STR,
    help="Output-fragment codec — scalar tile (n<N>[x<M>]/f<fn>[x<fm>]) OR warp mma tile "
    "(a:<atom>/w<WM>x<WN>/f<FM>x<FN>/k<bk>, selected by the a:<atom> token); empty=per-cell. "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# Operand staging — the reused gmem operands (matmul A/B, a fused prologue's read) ride a
# shared-memory slab + double-buffered producer (``sync`` plain copy / ``cp.async`` / ``tma``) over
# the serial reduce loop, instead of the gmem-direct register baseline. Resolved into the schedule's
# :class:`Stage` (``None`` = gmem-direct). Composes with both fragments of the unified ``TILE`` knob.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/sync|cp|tma[/ring][/p<reg_depth>]; empty=gmem-direct). "
    "Decided in lowering/tile/010_recognize (the _schedule helper), materialized in lowering/kernel/010_materialize.",
    off="",
)

# Warp specialization — the worker-mapping sibling of ``REDUCE``/``TILE``/``STAGE`` and ORTHOGONAL to
# all three: the pipeline (what's staged, the mma tile, the reduce partition) is fixed by those pins;
# ``WSPEC`` only splits the warps that run it into roles (``p<np>`` producer warps drive the ``STAGE``
# load half; the compute warps stay on the mma). ``off=""`` is uniform SIMT (every warp does both
# halves). Resolved into the schedule's :class:`WarpSpec` (``None`` = uniform). **Pin-only, and
# gated** on a warp ``TILE`` + a ``STAGE``; the producer/consumer materialization is a follow-up.
WSPEC = Knob(
    "WSPEC",
    KnobType.STR,
    help="Warp-specialization codec — role→warp split over the fixed pipeline "
    "(p<np> producer[:q<window>], s<ns> sfu, …; compute warps implicit = TilePlan.units; empty=uniform SIMT). "
    "Decided in lowering/tile/010_recognize (the _schedule helper); materialization reserved (TODO).",
    off="",
)


# --- The structural placement pin (PLACE) -----------------------------------
#
# ONE pin-only family controlling structural emission: where an intermediate edge lives — registers
# (``fuse``) or memory (``cut``). Elements are named by the MOVE, not the shape:
#
#   PLACE@cone   producer-cone inlining (the fused producer → matmul edge)
#   PLACE@fold   downstream-fold absorption (flash vs separate softmax + P@V kernels)
#   PLACE@tuple  sibling-fold tupling (online softmax vs two-pass stats)
#
# Vocabulary: ``auto`` (the built-in per-element default) / ``fuse`` / ``cut``. Precedence:
# ``PLACE@<element>`` > bare ``PLACE`` > built-in ``auto`` (read via :func:`place_decision` /
# ``Knob.narrow_at``). ``auto`` never appears in a knob dict — it is pin vocabulary; the stamped
# value is the *resolved* decision (``fuse`` / ``cut``), stamped for ``fold`` / ``cone`` only
# (``tuple`` is pure policy — dominance — and is never stamped). ``fuse`` is a request, not a
# guarantee: a forced fuse on an uncertifiable kernel (e.g. RoPE'd QK, which flash recognition
# rejects) degrades to ``cut`` with a log line — the standard pin-validity rule. Since ``@`` is not
# a valid shell var name character, per-element pins ride the ``EMMY_KNOBS`` aggregate
# (``EMMY_KNOBS="PLACE@fold=cut"``); the bare ``EMMY_PLACE`` env var pins every element.
PLACE = Knob(
    "PLACE",
    KnobType.STR,
    help="Structural placement of an intermediate edge — auto|fuse|cut, per element via "
    "PLACE@cone (producer-cone inlining) / PLACE@fold (flash vs multi-kernel attention) / "
    "PLACE@tuple (online softmax vs two-pass stats); bare PLACE pins every eligible edge. "
    "Pin-only (never enumerated); read in lowering/tile/010_recognize.",
    off="",
)

# The built-in ``auto`` defaults per element — today's emission behavior (fuse everywhere: flash,
# online softmax, and producer-cone inlining are all on when recognizable). Flipping a default is a
# behavior change gated on the validation suite, not a spelling change.
_PLACE_DEFAULTS = {"cone": "fuse", "fold": "fuse", "tuple": "fuse"}


def place_decision(element: str) -> str:
    """The resolved ``PLACE`` decision (``"fuse"`` / ``"cut"``) for ``element`` — the pin
    (``PLACE@<element>`` > bare ``PLACE``, via ``Knob.narrow_at``) with the explicit ``auto`` token
    (and no pin at all) resolving to the built-in per-element default. An unknown pin value degrades
    to the default with a log line — the standard pin-validity rule."""
    default = _PLACE_DEFAULTS[element]
    pin = PLACE.narrow_at(element)
    if pin is None or pin in ("", "auto"):
        return default
    if pin in ("fuse", "cut"):
        return pin
    logger.warning("PLACE@%s pin %r is not auto|fuse|cut; using the built-in %r", element, pin, default)
    return default


# --- Kernel-lowering policy knobs -------------------------------------------
#
# Boolean codegen policies recorded on the kernel op — on by default, not search dimensions
# (``hints=(True,)``); a rule records its knob for idempotence and honors the ``EMMY_<NAME>``
# env override. Consumed by ``lowering/kernel/050_vectorize_loads`` / ``095_interleave_loads``.

VECTORIZE_LOADS = Knob(
    "VECTORIZE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Fold runs of consecutive scalar Loads into one wide vector Load (float4 / __half2).",
    off=False,
)

INTERLEAVE_LOADS = Knob(
    "INTERLEAVE_LOADS",
    KnobType.BOOL,
    hints=(True,),  # on by default; not a search dimension — manual override only via the env var
    help="Sink each Load to just before its first SSA-consumer in flat compute blocks.",
    off=False,
)


# --- Enumeration value grids -------------------------------------------------
#
# The permitted-move catalog: the bounded, legality-guarded candidate values the ``_schedule`` emit
# enumerates into the scheduling fork. Each move is a codec spelling under the node's axis-named key
# (``TILE@<k_axis>`` / ``REDUCE@<axis>`` / ``STAGE@<axis>``) that the existing ``parse`` / ``spell``
# grammar, the prior featurizer, and the perf DB already consume — the move is only the **generator**,
# not new syntax. Two invariants keep a cold greedy compile stable and correct:
#
# - **Conservative option-0.** The per-cell / serial / gmem-direct pick leads every list (the reduce
#   tier deliberately emits its conservative *cooperative* pick first — the option-0 rule is
#   per-family, naming that family's safe default), so the emission-order fallback (no prior loaded)
#   keeps today's behavior.
# - **Static-value legality only.** Guards evaluable from the values alone (the scalar block-thread
#   budget) apply here; per-node guards (warp static-K divisibility, stage resolver eligibility) live
#   with their moves in ``_schedule``. An env pin still wins via ``Knob.narrow`` at the call site —
#   the catalog is the *unpinned* candidate set.

# The scalar block-thread budget (CUDA's 1024-thread/CTA hardware limit); a scalar tile launches
# ``par_n·par_m`` threads (one per parallel output cell). The same limit ``_schedule`` enforces on a
# pinned tile (imported there — one constant, two enforcement points).
MAX_BLOCK_THREADS = 1024

# The scalar register-tile candidate grid: ``(par_n, par_m)`` parallel thread-tile widths ×
# ``(reg_n, reg_m)`` per-thread register sub-tile widths. Bounded and hand-computable — the product the
# structural-coverage test recomputes independently. The parallel widths stay well inside the thread
# budget (``32·16 = 512 ≤ 1024``); the register widths span the square + skewed sub-tiles the prior
# ranks by occupancy / reuse.
_SCALAR_PAR: tuple[tuple[int, int], ...] = ((16, 8), (16, 16), (32, 8), (32, 16))  # (par_n, par_m)
_SCALAR_REG: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (4, 4), (2, 4), (4, 2))  # (reg_n, reg_m)


def scalar_tile_moves() -> list[str]:
    """The scalar-contraction output-tile ``TILE`` codec candidates: per-cell (``""``) first — the
    conservative option-0 — then the register-tile grid (:data:`_SCALAR_PAR` × :data:`_SCALAR_REG`)
    filtered by the ``par_n·par_m ≤ 1024`` thread budget. Each is spelled through :class:`TilePlan`
    so it round-trips the codec grammar exactly."""
    moves = [""]
    for par in _SCALAR_PAR:
        if par[0] * par[1] > MAX_BLOCK_THREADS:
            continue
        for reg in _SCALAR_REG:
            moves.append(TilePlan(units=par, regs=reg).spell())
    return moves


# The warp (tensor-core) tile candidate grid: ``(WM, WN)`` warp counts × ``(FM, FN)`` per-warp
# register fragments × ``bk`` K-chunks, spelled ``a:<atom>/w..x../f..x../k..``. Bounded to shapes the
# golden sweeps have deployed (≤ 4 warps, ``FM·FN ≤ 32`` C-fragment cells, shallow pipelined bk).
# Per-node legality — the atom's operand dtype and the ``_check_warp_static_k`` K-divisibility —
# is the scheduler's (``_schedule``), not the grid's.
_WARP_UNITS: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2), (4, 1), (1, 4), (2, 4), (4, 2), (4, 4), (1, 8))  # (WM, WN)
_WARP_REGS: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (1, 4), (4, 1), (2, 4), (4, 2), (4, 4), (4, 8))  # (FM, FN)
_WARP_BK: tuple[int, ...] = (1, 2, 4, 8)


def warp_tile_moves(atom_names: tuple[str, ...]) -> list[str]:
    """The warp-contraction output-tile ``TILE`` codec candidates over the (already
    dtype-eligible) ``atom_names``: the :data:`_WARP_UNITS` × :data:`_WARP_REGS` × :data:`_WARP_BK`
    grid per atom. No conservative option-0 of its own — these EXTEND :func:`scalar_tile_moves`
    (whose per-cell ``""`` leads the combined list)."""
    from emmy.compiler.ir.atom import ATOM_REGISTRY  # noqa: PLC0415

    moves = []
    for name in atom_names:
        atom = ATOM_REGISTRY[name]
        for units in _WARP_UNITS:
            for regs in _WARP_REGS:
                for bk in _WARP_BK:
                    moves.append(TilePlan(atom=atom, units=units, regs=regs, bk=bk).spell())
    return moves


def stage_moves(*, warp: bool) -> list[str]:
    """The operand-staging ``STAGE`` codec candidates — gmem-direct ``""`` first (the conservative
    option-0), then the transport / depth / double-buffer variants. Both tiers offer the gmem→smem
    prefetch ring depths (the scalar ring lands on the same ``staged_kloop`` phases; its slab
    K-chunk is depth-aware, derived in ``_resolve_scalar_stage``); the ``p2`` smem→register
    double-buffer is an ``ldmatrix`` transform, warp-only. Emission is resolver-gated in
    ``_schedule`` — a candidate is offered only when it RESOLVES against the built node, and the
    row carries the resolved spelling."""
    ring = ["", "d1/cp", "d2/cp/ring", "d3/cp/ring", "d4/cp/ring", "d1/tma", "d2/tma/ring", "d3/tma/ring", "d4/tma/ring"]
    return [*ring, "d2/cp/ring/p2"] if warp else ring


# Cross-CTA split-K widths (the ``REDUCE`` codec's ``g<w>`` field). Divisor / occupancy legality is
# the scheduler's.
SPLITK_WIDTHS: tuple[int, ...] = (2, 4, 8)


def splitk_moves(*, warp: bool) -> list[str]:
    """The cross-CTA split-K ``REDUCE`` codec candidates: deferred-kernel finalize (``g<w>k``) for
    both tiers, in-place atomic (``g<w>a``) for the scalar tier only (an mma C-fragment cannot
    ``atomicAdd`` — ``RegStore`` has no atomic form). These EXTEND the serial ``""`` option-0."""
    moves = [f"g{w}k" for w in SPLITK_WIDTHS]
    if not warp:
        moves += [f"g{w}a" for w in SPLITK_WIDTHS]
    return moves


def coop_reduce_moves() -> list[str]:
    """The cooperative / ILP K-partition ``REDUCE`` codec candidates for a NON-output-tiled
    contraction (``_coop_reduce_spec``'s contract — the per-cell tier folds K across ``b`` coop
    threads / ``r`` ILP register chains). These EXTEND the serial ``""`` option-0."""
    return ["b4", "b8", "r2", "r4", "r2/b4"]
