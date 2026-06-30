"""Tile *skeleton* — the recognized, axis-typed structure the scheduler enumerates over.

The skeleton is the schedule-side **index** of a :class:`~.ir.TileOp`: a nested tree of
:class:`Scope`\\ s mirroring the carrier (the op tree in ``ir/stmt/algebra``), each scope owning
its parallel (free / grid) axes and at most one reduce axis. It is computed **once at
recognition** (``lowering/tile/_skeleton.build_skeleton``) and carried alongside the op tree the
way :class:`~.schedule.Placement` is, so the scheduler (``020_schedule``) reads recognized facts
— axis roles, the reduce carrier, cooperative eligibility, the operand→role binding — instead of
re-deriving them from the algebra on every call.

The op tree stays the single source of truth for the *combine* / lowering (``op_cache_key``
digests ``lower(op.op)``, never the skeleton); the skeleton only records the *structure* a
schedule is chosen over. The split is deliberate: structural facts live here, while the menus,
occupancy heuristics, and legality filters stay in the scheduler.

**The normalization.** A contraction's reduce (K) axis carries ``carrier = Semiring.as_monoid()``
— the carrier-algebra fact that a SEMIRING is a MONOID with a ``⊗`` lift. So a contraction's K is
structurally identical to any :class:`~.ir.stmt.algebra.Monoid` reduce axis, and the
Semiring↔Monoid duality never reaches scheduling.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.ir.axis import Axis

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.algebra import AlgebraNode, Monoid
    from deplodock.compiler.ir.tile.binding import AtomBinding


class AxisRole(enum.Enum):
    """The hardware role an axis plays in the iteration space."""

    PARALLEL = "parallel"  # a free / output grid axis
    REDUCE = "reduce"  # a fold axis (a Monoid axis, or a Semiring's contraction K)


@dataclass(frozen=True)
class ReduceAxis:
    """The single reduce axis of a :class:`Scope` — the fold dimension plus the recognized facts
    a schedule needs to partition it.

    ``carrier`` is the fold algebra: a :class:`~.ir.stmt.algebra.Monoid` directly, or
    ``Semiring.as_monoid()`` for a contraction (the K-as-reduce normalization). The twist family
    (``"id"`` plain / ``"exp"`` softmax) is read off ``carrier.twist.family`` — no separate field.
    ``axis`` holds the extent-bearing :class:`~.ir.axis.Axis` (read live, not snapshotted, so the
    static-vs-symbolic extent stays single-sourced on the axis)."""

    axis: Axis
    carrier: Monoid  # the fold algebra — Semiring.as_monoid() for a contraction
    contraction: bool  # True iff this axis came from a Semiring (the K axis)
    coop_eligible: bool  # the cooperative-reduce predicate, evaluated once at recognition
    binding: AtomBinding | None = None  # contraction operand→A/B (None: not a contraction, OR unbindable)


@dataclass(frozen=True)
class Scope:
    """One carrier level of the skeleton: the algebra ``node`` it realizes, the parallel axes
    introduced here, its single reduce axis (``None`` for a pure :class:`~.ir.stmt.algebra.Map`),
    and any nested carrier scopes (flash's inner ``Semiring`` rides a child scope)."""

    node: AlgebraNode  # the Map / Monoid / Semiring this scope realizes
    parallel: tuple[Axis, ...] = ()  # free / output axes introduced at this scope (() for inner scopes)
    reduce: ReduceAxis | None = None  # the single reduce axis at this scope
    children: tuple[Scope, ...] = ()  # nested carrier scopes


@dataclass(frozen=True)
class Skeleton:
    """The recognized iteration-space skeleton of a :class:`~.ir.TileOp` — a tree of
    :class:`Scope`\\ s rooted at the kernel's outermost carrier."""

    root: Scope


__all__ = ["AxisRole", "ReduceAxis", "Scope", "Skeleton"]
