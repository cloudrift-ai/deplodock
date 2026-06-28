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
symbolic-axis cooperative tier, and flash cooperative-KV remain future steps
(``plans/cooperative-reduction-tile-ir.md``).
"""

from __future__ import annotations

from dataclasses import replace
from math import prod

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Loop
from deplodock.compiler.ir.stmt.algebra import Map, Monoid
from deplodock.compiler.ir.tile import MapKernel, MonoidKernel, ReducePlan, TileOp, reduce_node
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

# Conservative cooperative-reduce selection constants (placeholders for the REDUCE knob).
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


def _has_loop(stmts) -> bool:
    """Any ``Loop`` reachable in ``stmts`` (deep) — a projection that sweeps the reduce
    axis (a full-row output like softmax / RMSNorm) carries one, marking a non-scalar
    output the whole-CTA scalar-write path can't cover this cut."""
    for s in stmts:
        if isinstance(s, Loop):
            return True
        if any(_has_loop(list(b)) for b in s.nested()):
            return True
    return False


def _coop_carrier(kernel) -> Monoid | None:
    """The cooperative-eligible reduce carrier of ``kernel``, or ``None`` (keep serial).

    Eligible this cut: a ``MonoidKernel`` over a **degenerate** carrier (``as_accums`` —
    plain ``sum`` / ``max`` / ``mean``; twisted online-softmax / flash are future) reducing
    a **static** axis, producing a **scalar** output (a bare ``Monoid`` or a projection
    ``Map`` whose body doesn't sweep the reduce axis — full-row outputs stay serial)."""
    if not isinstance(kernel, MonoidKernel):
        return None
    inner = reduce_node(kernel.op)
    if not isinstance(inner, Monoid) or inner.as_accums() is None:
        return None
    if inner.axis is None or not inner.axis.extent.is_static:
        return None
    op = kernel.op
    if isinstance(op, Map) and op.source is not None and _has_loop(list(op.body)):
        return None  # projection sweeps the reduce axis → non-scalar output, keep serial
    return inner


def _schedule_for(kernel):
    """Map the free axes onto the grid and, for an eligible reduction, pick the cooperative
    reduce partition (else the scalar serial fold). Returns the kernel's same ``*Schedule``
    type with ``place`` mapped (and ``reduce`` set for a cooperative reduce)."""
    place = kernel.schedule.place.on_grid()
    if isinstance(kernel, MapKernel):
        return replace(kernel.schedule, place=place)
    carrier = _coop_carrier(kernel)
    if carrier is not None:
        extent = carrier.axis.extent.as_static()
        free = prod(ax.extent.as_static() for ax in place.free) if place.free else 1
        coop = _pick_coop(extent, free)
        if coop > 1:
            return replace(kernel.schedule, place=place, reduce=ReducePlan.of(coop=coop))
    return replace(kernel.schedule, place=place)


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    if tile.kernel is None or tile.kernel.schedule.place.is_mapped:
        # Already mapped (grid set), or nothing to map (a scalar-output kernel materializes
        # on an empty grid), or a placeholder node — leave it for materialize.
        raise RuleSkipped("schedule already mapped")
    return TileOp(kernel=replace(tile.kernel, schedule=_schedule_for(tile.kernel)), name=tile.name)
