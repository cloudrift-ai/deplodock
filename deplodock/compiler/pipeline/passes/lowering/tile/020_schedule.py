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
from deplodock.compiler.ir.tile import MapKernel, MonoidKernel, ReducePlan, TileOp, reduce_node
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

# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells (under-occupied)


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
    inner = reduce_node(kernel.op)
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
    ext = carrier.axis.extent
    extent = ext.as_static() if ext.is_static else (ext.hint or DEFAULT_SEQ_HINT)
    free = prod(ax.extent.as_static() for ax in place.free) if place.free else 1
    coop = _pick_coop(extent, free)
    cands = [f"b{coop}", ""] if coop > 1 else [""]  # conservative coop first (cold greedy → option-0)
    return list(REDUCE.narrow(cands))


def _option(kernel, place, spec: str, name: str, knobs: dict) -> TileOp:
    """One scheduled ``TileOp``: ``place`` mapped onto the grid + the ``REDUCE`` spec
    resolved into the schedule's ``ReducePlan`` (the ephemeral knob → materialized plan),
    with the spec stamped on ``knobs`` for the prior."""
    sched = replace(kernel.schedule, place=place, reduce=ReducePlan.parse(spec))
    return TileOp(kernel=replace(kernel, schedule=sched), name=name, knobs={**knobs, REDUCE.name: spec})


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
    specs = _reduce_specs(kernel, place)
    return [_option(kernel, place, spec, tile.name, tile.knobs) for spec in specs]
