"""Tile schedule type system — how a kernel's axes bind to the hardware.

Split out of :mod:`.ir`. The whole layer's thesis is that the **schedule is separate
from the combine**: the combine (the ⊕) lives in the op tree
(:mod:`deplodock.compiler.ir.stmt.algebra`), and the schedule — which axes are parallel,
how the reduce axis partitions across hardware levels — lives here, on a typed
``*Schedule`` paired with the op-tree node in a ``*Kernel``.

A reduction's only freedom is **how the reduce axis is partitioned across hardware
levels** (:class:`ReducePlan`); the combine *mechanism* at each level is **derived** from
the level (:meth:`ReduceStage.combine`), and the combine *algebra* rides the carrier (the
``Twist``). So the same op + the same materializer extend across kernel kinds — only the
carrier and the partition change.

The schedule is **flat, typed by the outermost algebra kind**: a ``Map`` →
:class:`MapSchedule`, a reduction ``Monoid`` → :class:`MonoidSchedule`, a contraction
``Semiring`` → :class:`SemiringSchedule` (flash — a ``Monoid`` over a nested partial
``Semiring`` — is a :class:`MonoidSchedule`; the op tree nests, the schedule does not).
The schedule of a reducing kind is **either** that kind's uniform (SIMT) schedule **or**
:class:`WarpSpec` (the warp-role pipeline, reserved). The pairing in the ``*Kernel``
types makes a ``Monoid``-with-``MapSchedule`` mismatch unrepresentable.

This module builds only the **uniform** arm; ``WarpTile`` / ``Stage`` / ``Channel`` /
``WarpSpec`` are reserved slots (``# TODO``) — see ``plans/cooperative-reduction-tile-ir.md``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt.algebra import Map


class Level(enum.Enum):
    """One hardware level the reduce axis can be partitioned across, coarse→fine."""

    GRID = "grid"  # across CTAs (split-K) — emitted by 030_split, never the in-kernel walk
    BLOCK = "block"  # cooperative threads within a CTA (warp shuffle / smem tree)
    REG = "reg"  # ILP register-fold accumulators
    SERIAL = "serial"  # the per-thread serial remainder (never spelled — derived)


class Fold(enum.Enum):
    """The per-level combine *mechanism* — derived from the :class:`Level`, not tuned."""

    SERIAL = "serial"  # no cross-unit combine (the serial / reg remainder)
    REG = "reg"  # register tree (ILP) — TODO(reg)
    SHFL = "shfl"  # lane-level ``__shfl_xor_sync`` butterfly
    SMEM = "smem"  # cross-warp / block-wide smem tree-halve
    ATOMIC = "atomic"  # cross-CTA ``atomicAdd`` finalize — TODO(cta), 030_split only


@dataclass(frozen=True)
class ReduceStage:
    """One level's **tuned** partition: a ``width`` of partials at a hardware ``level``.

    The combine *mechanism* is **derived** (:meth:`combine`), not stored — the level
    implies the fold, and a BLOCK width derives warp-shuffle vs hierarchical-smem from the
    warp size. ``width`` is power-of-two for BLOCK (the butterfly / tree reorder)."""

    level: Level
    width: int = 1

    def combine(self, *, warp_size: int, segmented: bool = False) -> tuple[Fold, ...]:
        """The derived per-level combine fold(s), fine→coarse within this stage.

        - ``SERIAL`` / ``REG`` → ``()`` (no cross-unit combine; REG-fold is TODO(reg)).
        - ``GRID`` → ``(ATOMIC,)`` (the split-K finalize — emitted by ``030_split``).
        - ``BLOCK`` → the intra-CTA hierarchy: a lone ``SHFL`` when ``segmented`` (the
          per-row segmented butterfly for strided-cooperative rows) or ``width ≤ warp``
          (one warp); ``(SHFL, SMEM)`` when ``width`` is a clean warp multiple (lanes then
          the cross-warp tree); else a standalone ``(SMEM,)`` block tree. Power-of-two
          ``width`` required."""
        if self.level in (Level.SERIAL, Level.REG):
            return ()
        if self.level is Level.GRID:
            return (Fold.ATOMIC,)
        # BLOCK.
        w = self.width
        if w & (w - 1):
            raise ValueError(f"BLOCK reduce width must be a power of two, got {w}")
        if segmented or w <= warp_size:
            return (Fold.SHFL,)
        if w % warp_size == 0:
            return (Fold.SHFL, Fold.SMEM)
        return (Fold.SMEM,)


@dataclass(frozen=True)
class ReducePlan:
    """The kernel's single reduce partition — the **tuned widths only**, coarse→fine.

    There is one reduce carrier per kernel (1:1 and singular — the carrier owns the axis),
    so the plan holds no axis; the per-thread ``serial`` remainder is derived by the
    materializer as ``ceil(extent / parallel)``. ``stages=()`` is the scalar serial fold
    (today's one-thread-per-cell tier)."""

    stages: tuple[ReduceStage, ...] = ()

    @classmethod
    def of(cls, *, cta: int = 1, coop: int = 1, reg: int = 1) -> ReducePlan:
        """Build a plan from per-level widths (1 = absent). Order is coarse→fine:
        GRID (cta) → BLOCK (coop) → REG (reg)."""
        stages: list[ReduceStage] = []
        if cta > 1:
            stages.append(ReduceStage(Level.GRID, cta))
        if coop > 1:
            stages.append(ReduceStage(Level.BLOCK, coop))
        if reg > 1:
            stages.append(ReduceStage(Level.REG, reg))
        return cls(tuple(stages))

    @classmethod
    def parse(cls, spec: str | None) -> ReducePlan:
        """Decode the ``REDUCE`` knob codec (the schedule's single reduce-partition knob,
        decided in ``020_schedule``) into a plan: ``/``-separated level-named tokens,
        coarse→fine — ``g<n>[a|k]`` (GRID cross-CTA split + finalize letter), ``b<n>``
        (BLOCK cooperative threads), ``r<n>`` (REG ILP fold). Empty / ``None`` = the scalar
        serial fold. (The ``serial`` remainder is never spelled — it's derived as
        ``ceil(extent / parallel)``.)"""
        spec = (spec or "").strip()
        if not spec:
            return cls()
        cta = coop = reg = 1
        for raw in spec.split("/"):
            tok = raw.strip()
            if not tok:
                continue
            kind, num = tok[0], tok[1:]
            if kind == "g":
                if num and num[-1] in "ak":  # the finalize letter (atomic / kernel) — 030_split
                    num = num[:-1]
                cta = int(num)
            elif kind == "b":
                coop = int(num)
            elif kind == "r":
                reg = int(num)
            else:
                raise ValueError(f"bad REDUCE token {tok!r} (expect g<n> / b<n> / r<n>)")
        return cls.of(cta=cta, coop=coop, reg=reg)

    def spell(self) -> str:
        """The ``REDUCE`` codec string for this plan (inverse of :meth:`parse`); ``""`` for
        the scalar serial fold."""
        letter = {Level.GRID: "g", Level.BLOCK: "b", Level.REG: "r"}
        return "/".join(f"{letter[s.level]}{s.width}" for s in self.stages if s.level in letter)

    @property
    def parallel(self) -> int:
        """The total parallel degree = ∏ stage widths (the lane/CTA fan-out the serial
        remainder divides into)."""
        p = 1
        for s in self.stages:
            p *= s.width
        return p

    @property
    def needs_split(self) -> bool:
        """True iff any stage is a cross-CTA GRID split (``030_split`` territory)."""
        return any(s.level is Level.GRID for s in self.stages)

    def _width(self, level: Level) -> int:
        for s in self.stages:
            if s.level is level:
                return s.width
        return 1

    @property
    def coop(self) -> int:
        """The BLOCK (cooperative-thread) width, or 1 if no BLOCK stage."""
        return self._width(Level.BLOCK)

    @property
    def cta(self) -> int:
        """The GRID (cross-CTA split) width, or 1 if no GRID stage."""
        return self._width(Level.GRID)

    @property
    def reg(self) -> int:
        """The REG (ILP fold) width, or 1 if no REG stage."""
        return self._width(Level.REG)

    @property
    def block_stage(self) -> ReduceStage | None:
        """The single BLOCK :class:`ReduceStage`, or ``None`` (scalar serial)."""
        for s in self.stages:
            if s.level is Level.BLOCK:
                return s
        return None


@dataclass(frozen=True)
class Placement:
    """Kind-neutral free-axis → grid binding (the parallel output axes and their grid
    mapping). ``010_recognize`` builds an UNMAPPED placement (just ``free``);
    ``020_schedule`` maps every free axis onto ``grid`` (the per-cell tier)."""

    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()

    @property
    def is_mapped(self) -> bool:
        """True once the free axes are bound (``grid`` set) — or there were none to bind
        (a scalar-output kernel materializes on an empty grid)."""
        return bool(self.grid) or not self.free

    def on_grid(self) -> Placement:
        """The scalar-tier mapping: bind every free axis onto the thread grid."""
        return Placement(free=self.free, grid=self.free)


# --------------------------------------------------------------------------- #
# Reserved slots — the tensor-core tile, operand pipelining, warp specialization.
# Defined so the type system is complete (the schedule fields reference them); not
# constructed by this cut. See ``plans/cooperative-reduction-tile-ir.md``.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WarpTile:
    """TODO(warp): the tensor-core mma tile — a **shared** descriptor on both
    :class:`MonoidSchedule` (flash's inner QK/PV) and :class:`SemiringSchedule` (matmul).
    A contraction is mma-tiled onto the one mapping whether it is the top node or nested;
    never a nested/second schedule. Reserved — no fields built this cut."""


@dataclass(frozen=True)
class Stage:
    """TODO(pipelining): one operand-transport pipeline over the serial reduce loop —
    one ``Stage`` per reduce loop (a ``Monoid`` ⇒ one reduce axis ⇒ one pipeline).
    Reserved; the scalar/cooperative tier loads gmem-direct (``depth=1``, ``sync``)."""

    depth: int = 1  # pipeline stages over the serial reduce loop (1 = no prefetch)
    transport: str = "sync"  # sync | cp.async | tma
    smem: tuple[str, ...] = ()  # operands staged through smem (empty = register-only)
    ring: bool = False  # ring buffer vs static double-buffer


@dataclass(frozen=True)
class Channel:
    """TODO(warp-spec): a shared smem ring connecting a warp-spec producer/consumer."""

    name: str
    depth: int


@dataclass(frozen=True)
class WarpRole:
    """TODO(warp-spec): one warp role (producer / mma / reducer); its ``schedule`` is
    itself a uniform schedule scoped to that role's warps, carrying its own :class:`Stage`."""

    stage_node: object
    warps: int
    schedule: object  # MapSchedule | MonoidSchedule | SemiringSchedule
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class WarpSpec:
    """TODO(warp-spec): the warp-role pipeline — ONE shared struct (no ``*WarpSpec*``
    per-kind variants). Appears only at the top CTA-level schedule; roles bottom out in
    uniform schedules. Reserved this cut."""

    place: Placement
    channels: tuple[Channel, ...] = ()
    roles: tuple[WarpRole, ...] = ()


# --------------------------------------------------------------------------- #
# The three uniform (SIMT) schedules — one thread / block / warp mapping per kind.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MapSchedule:
    """A pointwise kernel's schedule — just the free-axis grid binding (no reduce, no
    warp-spec: pointwise never warp-specializes)."""

    place: Placement


@dataclass(frozen=True)
class MonoidSchedule:
    """A reduction (``Monoid``) kernel's schedule.

    Three **orthogonal** reduce-axis fields (kept distinct so they don't become a
    grab-bag): ``reduce`` (the :class:`ReducePlan` partition), ``warp_tile`` (the mma
    operand tile — ``# TODO(warp)``, flash's inner QK/PV), ``stage`` (operand transport —
    ``# TODO(pipelining)``). ``block`` are free axes resident in the CTA alongside the
    cooperative lanes (strided-cooperative rows)."""

    place: Placement
    block: tuple[Axis, ...] = ()
    reduce: ReducePlan = field(default_factory=ReducePlan)
    warp_tile: WarpTile | None = None  # TODO(warp)
    stage: Stage | None = None  # TODO(pipelining)


@dataclass(frozen=True)
class SemiringSchedule:
    """A contraction (``Semiring``) kernel's schedule — the same orthogonal reduce-axis
    fields as :class:`MonoidSchedule` (``warp_tile`` is the matmul's mma tile)."""

    place: Placement
    block: tuple[Axis, ...] = ()
    reduce: ReducePlan = field(default_factory=ReducePlan)
    warp_tile: WarpTile | None = None  # TODO(warp)
    stage: Stage | None = None  # TODO(pipelining)


# --------------------------------------------------------------------------- #
# The op + schedule pairs — the schedule is EITHER the kind's uniform schedule OR
# WarpSpec (the union at the field IS the either; no wrapper class).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MapKernel:
    """A pointwise kernel: a ``Map`` op + its :class:`MapSchedule` (no warp-spec arm)."""

    op: object  # a Map (pure pointwise; the kernel root)
    schedule: MapSchedule


@dataclass(frozen=True)
class MonoidKernel:
    """A reduction kernel: a ``Monoid`` op (or a projection ``Map`` *over* one) + its
    :class:`MonoidSchedule` or :class:`WarpSpec`."""

    op: object  # a Monoid, or a Map(source=Monoid) projection
    schedule: MonoidSchedule | WarpSpec


@dataclass(frozen=True)
class SemiringKernel:
    """A contraction kernel: a ``Semiring`` op (or a projection ``Map`` *over* one) + its
    :class:`SemiringSchedule` or :class:`WarpSpec`."""

    op: object  # a Semiring, or a Map(source=Semiring) projection
    schedule: SemiringSchedule | WarpSpec


#: A scheduled kernel — keyed by the op kind (no ``classify_algebra`` tag). The pairing
#: makes a kind/schedule mismatch unrepresentable.
Kernel = MapKernel | MonoidKernel | SemiringKernel


def reduce_node(op):
    """The nested ``Monoid`` / ``Semiring`` a kernel reduces over, peeling a projection
    ``Map`` (``project ∘ reduce``) to its ``source``; ``None`` for a pure pointwise
    ``Map`` (or a flat ``Map`` whose reduces stay as loop-IR inside its body)."""
    inner = op.source if isinstance(op, Map) and op.source is not None else op
    from deplodock.compiler.ir.stmt.algebra import Monoid, Semiring  # noqa: PLC0415

    return inner if isinstance(inner, (Monoid, Semiring)) else None


def kernel_for(node, place: Placement) -> Kernel:
    """Wrap a lifted op-tree ``node`` + its :class:`Placement` in the matching ``*Kernel``,
    keyed by the (peeled) op kind — a bare reduction or a projection ``Map`` over one is a
    ``Monoid`` / ``Semiring`` kernel; anything else is a ``MapKernel``."""
    from deplodock.compiler.ir.stmt.algebra import Monoid, Semiring  # noqa: PLC0415

    inner = reduce_node(node)
    if isinstance(inner, Monoid):
        return MonoidKernel(op=node, schedule=MonoidSchedule(place=place))
    if isinstance(inner, Semiring):
        return SemiringKernel(op=node, schedule=SemiringSchedule(place=place))
    return MapKernel(op=node, schedule=MapSchedule(place=place))


__all__ = [
    "Channel",
    "Fold",
    "Kernel",
    "Level",
    "MapKernel",
    "MapSchedule",
    "MonoidKernel",
    "MonoidSchedule",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "SemiringKernel",
    "SemiringSchedule",
    "Stage",
    "WarpRole",
    "WarpSpec",
    "WarpTile",
    "kernel_for",
    "reduce_node",
]
