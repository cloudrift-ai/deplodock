"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` records the *schedule* in a
:class:`Schedule` — the parallel (free) axes and how they bind to the hardware
grid (one GPU thread per output cell at the scalar tier) — while the *combine*
lives entirely in the ``op`` tree (``ir/stmt/algebra``): a pointwise ``Map`` of
leaf compute (optionally OVER a nested node — a ``Map`` over a ``Monoid`` is the φ
projection ``project ∘ reduce``), a ``Monoid`` folding through its carrier
(``Twist``), or a ``Semiring`` contraction. The algebra is **not stored as a tag**;
the carriers and partial structure are read directly where a pass needs them, per
the project's "the op tree is the single source of truth" rule. There is no stored
per-cell body — ``lower(op)`` generates it at materialize (and on demand for the
dump / cache key).

The schedule is a property of the **root** kernel (which axes are parallel, how
they map onto threads), NOT of any individual carrier — so it lives on the
``TileOp`` via :class:`Schedule`, never on a ``Map`` / ``Monoid`` / ``Semiring``.
Because the combine is in the op tree and the schedule is on the ``TileOp``, the
SAME op and the SAME materializer extend across kernel kinds — only the carrier
(the ⊕) changes, never the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import Op


@dataclass(frozen=True)
class Schedule:
    """How a kernel's parallel (free) axes bind to the hardware grid — the **root
    schedule**, kept separate from the combine (which lives in the op tree).

    - ``free`` — the parallel axes (one output cell each), as read off the kernel's
      iteration space by ``010_recognize``.
    - ``grid`` — those of ``free`` bound onto the linear thread grid (one thread per
      output cell). Empty until ``020_schedule`` maps them.

    ``010_recognize`` builds an UNMAPPED schedule (just ``free``); ``020_schedule``
    maps every free axis onto ``grid`` (the per-cell scalar tier). Richer tiers
    (cooperative / split-K) will map the same ``free`` axes onto more roles here —
    extending the schedule, never the op tree. ``free``/``grid`` coincide at the scalar
    tier but are distinct concepts (a cooperative axis would be ``free`` but not
    ``grid``)."""

    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()

    @property
    def is_mapped(self) -> bool:
        """True once the free axes are bound (``grid`` set) — or there were none to bind
        (a scalar-output kernel materializes on an empty grid). ``020_schedule`` skips a
        mapped schedule, so the rule is idempotent."""
        return bool(self.grid) or not self.free

    def on_grid(self) -> Schedule:
        """The scalar-tier mapping: bind every free axis onto the thread grid."""
        return Schedule(free=self.free, grid=self.free)


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds exactly the op tree (``op``) and the :class:`Schedule` — not a pre-lowered
    body, and not a ``BodyOp``. ``op`` is a single
    :class:`~deplodock.compiler.ir.stmt.algebra.Map` (a pointwise per-cell body that
    carries its own ``Write``, or a projection ``Map`` over a reduction) or a reduction —
    a :class:`~deplodock.compiler.ir.stmt.algebra.Monoid` / ``Semiring`` whose ``out`` is
    the carried value. ``schedule`` carries the parallel axes and their grid binding.

    There is **no stored body and no stored output store**: the per-cell loop-IR body
    is generated at materialize time by ``lower(op)``, and the ``Write`` that binds a
    reduction's output value to the kernel's output buffer at the grid cell is *glue*
    generated there too, from ``schedule.grid`` + the graph node's output buffer (see
    ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk. ``pretty_body`` lowers ``op``
    on demand for dumps (the cache key lowers it likewise in ``search/keys``)."""

    op: object = None  # Map | Monoid | Semiring — the op tree; None for placeholder nodes
    schedule: Schedule = field(default_factory=Schedule)
    name: str = ""

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from deplodock.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        return "\n".join(pretty(self.op, "    ")) if self.op is not None else ""
