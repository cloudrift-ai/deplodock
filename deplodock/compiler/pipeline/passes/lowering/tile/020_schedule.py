"""Schedule a lifted ``TileOp`` onto the thread grid (+ pick the reduce partition).

Second of the two tile-lowering steps — ``010_recognize`` lifted the kernel to a
``TileOp`` carrying a typed :class:`~deplodock.compiler.ir.tile.schedule.Kernel` (op-tree
node + a ``*Schedule`` with an UNMAPPED :class:`~...schedule.Placement`). Scheduling binds
the placement's ``free`` axes onto the grid (``Placement.on_grid``) and, for a reduction,
picks the reduce-axis **partition** (:class:`~...schedule.ReducePlan`).

This cut picks a **whole-CTA cooperative** partition for a **static, scalar-output,
degenerate-monoid** reduce (plain ``sum`` / ``max`` / ``mean``) when the reduce axis is
wide and the output grid is small enough to leave the GPU under-occupied — one CTA per
output cell, ``coop`` threads cooperatively folding the reduce axis (the combine is
materialized in ``lowering/kernel``). Everything else (pointwise ``Map``, twisted /
full-row reductions like online-softmax & RMSNorm, contractions, symbolic axes) keeps the
**scalar serial** fold (``ReducePlan()`` — one thread per output cell).

The selection here is **conservative module constants** standing in for the eventual
``REDUCE`` knob + prior-driven choice. ``# TODO``: replace the constants with
``knob.py::_reduce_decomp`` (BR→coop, BK→serial, FK→reg, SPLITK→cta) + the learned /
analytic prior. Strided-cooperative rows (a small whole free axis packed alongside the
coop lanes), the ``reg`` (ILP) fold, the cross-CTA ``cta`` split (``030_split``), the
symbolic-axis cooperative tier, and flash cooperative-KV remain future steps.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from deplodock.compiler.dim import DEFAULT_SEQ_HINT
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt.algebra import Monoid
from deplodock.compiler.ir.tile import MapKernel, MonoidKernel, ReducePlan, SemiringKernel, TileOp, TilePlan
from deplodock.compiler.ir.tile.schedule import Stage, WarpTile
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

# The reduce-axis partition is decided HERE, by the single ``REDUCE`` codec knob — the
# decision hierarchy (env pin via ``Knob.narrow`` > the search/prior fork > the conservative
# constant default below). The knob is **ephemeral**: it's resolved here into the schedule's
# ``ReducePlan`` (the materialized form ``kernel/010_materialize`` reads); the knob value
# also rides on ``TileOp.knobs`` so the learned prior featurizes / tunes the decision.
# ``off=""`` (the scalar serial fold) is auto-stamped on kernels this pass doesn't cooperate.
REDUCE = Knob(
    "REDUCE",
    KnobType.STR,
    help="Reduce-axis partition codec (g<n> cta / b<n> coop / r<n> reg; empty=serial). "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# The free-axis output tile is decided HERE too, by the single ``TILE`` codec knob (the
# ``Scalar``-fragment sibling of ``REDUCE``) — a contraction's per-thread register sub-tile
# (each thread owns ``reg_m·reg_n`` output cells, reusing operands across them). Same decision
# hierarchy: env pin via ``Knob.narrow`` > the (future) prior fork > the per-cell default. Only
# a ``Semiring`` contraction tiles its output today; ``off=""`` auto-stamps everything else.
TILE = Knob(
    "TILE",
    KnobType.STR,
    help="Free-axis output-tile codec (n<N>[xm<M>] parallel thread-tile / f<fn>[xf<fm>] register "
    "sub-tile; empty=per-cell). Decided in lowering/tile/020_schedule, materialized in "
    "lowering/kernel/010_materialize.",
    off="",
)

# The tensor-core warp tile is decided HERE too, by the single ``WARP`` codec knob (the
# ``Warp``-fragment sibling of ``TILE``) — a contraction's mma tile (atom + warps + register
# sub-tile + K-chunk). ``WARP`` is pin-only this cut (the prior auto-fork is a follow-up, as
# ``TILE``'s is): a non-empty ``DEPLODOCK_WARP`` selects the warp tier and wins over ``TILE``
# (the either-ness — a kernel is the scalar register-tile OR the warp tile, never both); empty
# (``off=""``) keeps the scalar path. Only a ``Semiring`` contraction warp-tiles today. The codec
# is exploded into the on-dict ``ATOM@`` / ``WM`` / ``WN`` / ``FM`` / ``FN`` / ``BK`` keys the
# learned-prior featurizer (``knob.py``: ``mma_atom`` / ``is_warp`` / ``_free_slots`` /
# ``tile_signature``) already reads, so the codec stays the pin/display spelling.
WARP = Knob(
    "WARP",
    KnobType.STR,
    help="Tensor-core warp-tile codec (a:<atom>/w<WM>xw<WN>/f<FM>xf<FN>/k<bk>; empty=scalar). "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Operand staging is decided HERE too, by the single ``STAGE`` codec knob — the reused gmem
# operands (matmul A/B, a fused prologue's read) ride a shared-memory slab + double-buffered
# producer (``sync`` plain copy / ``cp.async`` / ``tma``) over the serial reduce loop, instead
# of the gmem-direct register baseline. Resolved here into the schedule's :class:`Stage`
# (``None`` = gmem-direct); the codec also rides on ``TileOp.knobs`` for the prior. ``STAGE`` is
# pin-only this cut (the prior auto-fork is a follow-up, as ``WARP``/``TILE``'s is). Composes
# with both the scalar register-tile (``TILE``) and the warp (``WARP``) tier.
STAGE = Knob(
    "STAGE",
    KnobType.STR,
    help="Operand-staging codec (d<depth>/sync|cp|tma[/ring]; empty=gmem-direct). "
    "Decided in lowering/tile/020_schedule, materialized in lowering/kernel/010_materialize.",
    off="",
)

# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells (under-occupied)


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic (the occupancy heuristic
    sizes a dynamic axis by its hint; the kernel still deploys over the runtime extent)."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _prevpow2(n: int) -> int:
    """The largest power of two ≤ ``n`` (≥ 1)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_coop(extent: int, free: int) -> int:
    """The conservative whole-CTA cooperative-thread count for a reduce of static
    ``extent`` over ``free`` output cells, or ``1`` (stay scalar/serial). Cooperate only on
    a wide reduce (``extent ≥ _COOP_MIN_EXTENT``) feeding a small grid (``free ≤
    _FREE_CAP`` — otherwise the scalar tier already saturates the GPU); the count targets
    ``_SERIAL_TARGET`` serial steps, capped at ``_MAX_COOP``, rounded to a power of two (the
    butterfly / tree reorder)."""
    if extent < _COOP_MIN_EXTENT or free > _FREE_CAP:
        return 1
    coop = min(_prevpow2(extent // _SERIAL_TARGET), _MAX_COOP)
    return coop if coop >= 2 else 1


def _coop_carrier(kernel) -> Monoid | None:
    """The cooperative-eligible reduce carrier of ``kernel``, or ``None`` (keep serial).

    Eligible: any ``MonoidKernel`` over a cleanly-lifted ``Monoid`` carrier — **degenerate**
    (plain ``sum`` / ``max`` / ``mean``) AND **twisted** (online-softmax ``(m, d)``, flash
    ``(m, l, O)``) alike, since the cross-thread combine is carrier-generic (it drives off
    the carrier's ``combine_states``, which a twisted carrier authors). Both **scalar**
    outputs (flash's ``O/l`` per ``(m, d)`` cell — ``d`` is a grid axis) and **full-row**
    outputs (softmax / RMSNorm — the post-reduce sweep is distributed across the coop lanes
    by the materializer) are handled. The reduce axis may be **symbolic** (dynamic
    ``seq_len``): each lane strides it to the runtime extent (the ``< seq_len`` bound is the
    masked tail). A flat-``Map`` fallback (multi / nested-non-flash reduce) is a
    ``MapKernel`` (``reduce_node`` is ``None``), so it isn't eligible and keeps the serial
    fold."""
    if not isinstance(kernel, MonoidKernel):
        return None
    inner = kernel.op.reduce_node
    if not isinstance(inner, Monoid) or inner.axis is None:
        return None
    return inner


def _reduce_specs(kernel, place) -> list[str]:
    """The candidate ``REDUCE`` codec strings for ``kernel``, applying the decision
    hierarchy. A kernel the cooperative tier can't partition (pointwise, or a twisted /
    full-row / contraction reduce) is the lone scalar fold ``[""]`` — the ``REDUCE`` pin is
    ignored there, since it only governs the cooperative reduce tier. An eligible reduce
    offers ``[conservative coop, scalar]`` (a fork the search / prior ranks, option-0 = the
    conservative pick so a cold greedy compile keeps cooperating), with an env pin
    (``DEPLODOCK_REDUCE``) authoritative over the candidates (``Knob.narrow``)."""
    carrier = _coop_carrier(kernel)
    if carrier is None:
        return [""]  # not cooperative-eligible — scalar serial fold; the pin doesn't apply
    # A symbolic reduce axis is sized by its ``Dim`` hint for the conservative pick (the
    # kernel deploys at the hint and strides to the runtime extent); a pin overrides it.
    extent = _hint_extent(carrier.axis)
    # A symbolic free axis (dynamic-grid tier) is sized by its ``Dim`` hint for the occupancy
    # heuristic — the kernel still deploys over the runtime grid.
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    coop = _pick_coop(extent, free)
    cands = [f"b{coop}", ""] if coop > 1 else [""]  # conservative coop first (cold greedy → option-0)
    return list(REDUCE.narrow(cands))


def _option(kernel, place, spec: str, name: str, knobs: dict) -> TileOp:
    """One scheduled ``TileOp``: ``place`` mapped onto the grid + the ``REDUCE`` spec
    resolved into the schedule's ``ReducePlan`` (the ephemeral knob → materialized plan),
    with the spec stamped on ``knobs`` for the prior."""
    sched = replace(kernel.schedule, place=place, reduce=ReducePlan.parse(spec))
    return TileOp(kernel=replace(kernel, schedule=sched), name=name, knobs={**knobs, REDUCE.name: spec})


def _tile_specs(kernel) -> list[str]:
    """Candidate ``TILE`` codec strings for ``kernel`` — only a ``Semiring`` contraction tiles
    its output; everything else is the per-cell tier (``[""]``, the pin doesn't apply). The env
    pin ``DEPLODOCK_TILE`` is authoritative (``Knob.narrow``); the default is the per-cell tier
    (the auto reg-tile fork is a follow-up, wired through the prior alongside the codec)."""
    if not isinstance(kernel, SemiringKernel):
        return [""]
    return list(TILE.narrow([""]))


def _semiring_reduce_spec() -> str:
    """The ``REDUCE`` spec a ``Semiring`` contraction honors today — only a **cross-CTA split**
    (``g``) pin (split-K). The K-axis coop / reg reduce tiers aren't built for the scalar
    contraction, so a non-``g`` ``REDUCE`` pin is ignored here; the split partition is
    orthogonal to the output ``TILE``. Returns the pinned spec when it carries a GRID stage,
    else ``""`` (no split). A pin in another tier's codec (e.g. the warp/MMA ``s``/``c`` split
    spelling) doesn't parse as ``g``/``b``/``r`` — it's not ours, so ignore it rather than fail."""
    pinned = REDUCE.narrow([""])[0]
    try:
        plan = ReducePlan.parse(pinned)
    except ValueError:
        return ""
    return pinned if plan.needs_split else ""


def _warp_spec(kernel) -> str:
    """The pinned ``WARP`` codec for ``kernel`` — only a ``Semiring`` contraction warp-tiles
    (everything else is ``""``, the pin doesn't apply). Pin-only this cut: returns the
    authoritative ``DEPLODOCK_WARP`` pin (``Knob.narrow``) or ``""`` (no warp tier — the
    scalar ``TILE`` path runs instead)."""
    if not isinstance(kernel, SemiringKernel):
        return ""
    return WARP.narrow([""])[0]


def _stage_spec(kernel) -> str:
    """The pinned ``STAGE`` codec for ``kernel`` — only a ``Semiring`` contraction stages its
    operands today (everything else is ``""``, the pin doesn't apply). Pin-only this cut:
    returns the authoritative ``DEPLODOCK_STAGE`` pin (``Knob.narrow``) or ``""`` (gmem-direct,
    ``stage=None``). A pin that doesn't parse as the ``STAGE`` codec (e.g. a legacy operand
    binmask ``"11"``) is **structurally invalid** for this tier, so it degrades to ``""``
    (gmem-direct) rather than failing the lowering — the same pin-validity rule the other
    codecs follow."""
    if not isinstance(kernel, SemiringKernel):
        return ""
    pinned = STAGE.narrow([""])[0]
    if not pinned:
        return ""
    try:
        Stage.parse(pinned)
    except ValueError:
        return ""
    return pinned


def _warp_option(kernel, place, spec: str, name: str, knobs: dict, stage_spec: str = "") -> TileOp:
    """One scheduled warp-tier contraction ``TileOp``: ``place`` mapped onto the grid + the
    ``WARP`` spec resolved into the schedule's ``WarpTile`` (the ``Warp`` fragment), plus an
    optional operand ``STAGE`` resolved into the schedule's :class:`Stage`. The codec is
    exploded into the on-dict ``ATOM@`` / ``WM`` / ``WN`` / ``FM`` / ``FN`` / ``BK`` keys the
    learned-prior featurizer already reads (the codec stays the pin/display spelling)."""
    wt = WarpTile.parse(spec)
    stage = Stage.parse(stage_spec) if stage_spec else None
    sched = replace(kernel.schedule, place=place, warp_tile=wt, stage=stage)
    (wm, wn), (fm, fn) = wt.warps, wt.reg
    stamped = {**knobs, WARP.name: spec, "ATOM@out": wt.atom.name, "WM": wm, "WN": wn, "FM": fm, "FN": fn, "BK": wt.bk}
    if stage_spec:
        stamped[STAGE.name] = stage_spec
    return TileOp(kernel=replace(kernel, schedule=sched), name=name, knobs=stamped)


def _tile_option(kernel, place, spec: str, name: str, knobs: dict, reduce_spec: str = "", stage_spec: str = "") -> TileOp:
    """One scheduled contraction ``TileOp``: ``place`` mapped onto the grid + the ``TILE`` spec
    resolved into the schedule's ``TilePlan`` (and an optional split-K ``REDUCE`` spec into the
    orthogonal ``ReducePlan``, an optional operand ``STAGE`` into the :class:`Stage`), the specs
    stamped on ``knobs`` for the prior."""
    stage = Stage.parse(stage_spec) if stage_spec else None
    sched = replace(kernel.schedule, place=place, tile=TilePlan.parse(spec), reduce=ReducePlan.parse(reduce_spec), stage=stage)
    stamped = {**knobs, TILE.name: spec}
    if reduce_spec:
        stamped[REDUCE.name] = reduce_spec
    if stage_spec:
        stamped[STAGE.name] = stage_spec
    return TileOp(kernel=replace(kernel, schedule=sched), name=name, knobs=stamped)


def rewrite(match: Match, root: Node) -> list[TileOp] | TileOp | None:
    tile: TileOp = root.op
    if tile.kernel is None or tile.kernel.schedule.place.is_mapped:
        # Already mapped (grid set), or nothing to map (a scalar-output kernel materializes
        # on an empty grid), or a placeholder node — leave it for materialize.
        raise RuleSkipped("schedule already mapped")
    kernel = tile.kernel
    place = kernel.schedule.place.on_grid()
    # A pointwise / non-cooperative kernel has no reduce decision — just map the grid (the
    # off-default stamps ``REDUCE=""``). A reduction offers its ``REDUCE`` candidate(s): a
    # single option applies directly; multiple options fork for the search / prior to rank.
    if isinstance(kernel, MapKernel):
        return TileOp(kernel=replace(kernel, schedule=replace(kernel.schedule, place=place)), name=tile.name)
    # A contraction picks its free-axis output tile (``TILE``); a reduction picks its reduce
    # partition (``REDUCE``). Each offers its candidate(s): one applies directly, multiple fork.
    # A ``Semiring`` ALSO honors a cross-CTA split-K (``g``) ``REDUCE`` pin — orthogonal to the
    # output tile (``reduce`` = the K partition, consumed by ``030_split``).
    # A ``WARP`` pin selects the tensor-core warp tier and wins over ``TILE`` (the either-ness);
    # without it the scalar register-tile (``TILE``) path runs.
    if isinstance(kernel, SemiringKernel):
        stage_spec = _stage_spec(kernel)
        warp = _warp_spec(kernel)
        if warp:
            return _warp_option(kernel, place, warp, tile.name, tile.knobs, stage_spec)
        reduce_spec = _semiring_reduce_spec()
        return [_tile_option(kernel, place, spec, tile.name, tile.knobs, reduce_spec, stage_spec) for spec in _tile_specs(kernel)]
    specs = _reduce_specs(kernel, place)
    return [_option(kernel, place, spec, tile.name, tile.knobs) for spec in specs]
