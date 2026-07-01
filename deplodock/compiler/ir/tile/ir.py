"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` holds the structural-IR root ``op``
(the *combine*, in ``ir/stmt/algebra`` + ``ir/tile/structural``) directly,
plus a thin set of **root-global schedule fields** — the free-axis → grid
:class:`~.schedule.Placement` (``place``) and the warp split (``workers``). The
per-node schedule slices ride the structural nodes themselves (a
:class:`~.structural.Contraction`'s ``tile``, a :class:`~.structural.Reduction`'s
``reduce``); the residual root fields (``reduce`` / ``tier`` / ``stage``)
hold the schedule for the not-yet-nodified forms (a non-tiled
contraction's split-K, the pin-only ``STAGE`` / ``WSPEC``; flash is now a
``Map(source=Reduction(source=Contraction))`` node tree, so its partition rides
the node). There is no per-kind kernel/schedule type: the algebra is read
structurally off the axes' :class:`~deplodock.compiler.ir.axis.AxisRole`
(``ops.axis_role``), so MAP / MONOID / SEMIRING all ride the same ``TileOp``.

The combine lives entirely in the ``op`` wrapper (``ir/stmt/algebra`` +
``ir/tile/structural``): a :class:`~deplodock.compiler.ir.tile.structural.Map` /
:class:`~.structural.Reduction` / :class:`~.structural.Contraction` whose per-cell
loop nest carries the role (``AxisRole``) + the decoupled ``Carrier`` (the ⊕
algebra). The algebra is **not stored as a node kind**; the role/carrier are read
off the node / annotated loop where a pass needs them (``ops.axis_role`` /
``ops.reduce_loop``). ``lower(op)`` flattens the structural tree back to the loop
nest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.tile.schedule import Placement, ReducePlan, Stage, TilePlan, WarpSpec

#: Back-compat alias: the old two-field ``Schedule`` (``free`` / ``grid``) is now
#: :class:`~.schedule.Placement`. Kept re-exported during the transition.
Schedule = Placement


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds the structural-IR root ``op`` (a :class:`~.structural.Map` /
    :class:`~.structural.Reduction` / :class:`~.structural.Contraction`, or ``None`` for a
    placeholder node) plus the schedule fields — not a pre-lowered body. The per-cell loop-IR
    body is generated at materialize time by ``lower(op)``, and a bare reduction / contraction's
    output ``Write`` is glue generated there too (from ``place.grid`` + the graph node's output
    buffer; see ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk.

    Schedule fields (all defaulted, so a fresh / placeholder node is well-formed):

    - ``place`` — the free-axis → grid binding (:class:`~.schedule.Placement`); root-global.
    - ``workers`` — the warp-specialization split (:class:`~.schedule.WarpSpec`); root-global, ``None`` =
      uniform SIMT.
    - ``reduce`` — the reduce-axis partition (:class:`~.schedule.ReducePlan`) for a not-yet-nodified
      reduce (a non-tiled contraction's split-K); a ``Reduction`` node (a plain reduce, softmax, **or
      flash** — now a ``Map(source=Reduction)``) carries its own partition (read via
      ``ops.reduce_plan``, which falls back here).
    - ``tier`` — the output fragment (:class:`~.schedule.TilePlan`) for a non-tiled / split-partial
      contraction; a tiled contraction rides its ``tile`` on the ``Contraction`` node. ``None`` = per-cell.
    - ``stage`` — the operand smem pipeline (:class:`~.schedule.Stage`); ``None`` = gmem-direct (pin-only).

    The contraction operand→role binding is not a ``TileOp`` field — a tiled contraction carries its
    A/B operands / accumulator / epilogue on its ``Contraction`` node (``op``), the single source of
    truth; ``_schedule._contraction_node`` resolves them via ``_atomize.semiring_binding``."""

    op: object = None
    name: str = ""
    place: Placement = field(default_factory=Placement)
    reduce: ReducePlan = field(default_factory=ReducePlan)
    tier: TilePlan | None = None
    stage: Stage | None = None
    workers: WarpSpec | None = None

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from deplodock.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        if self.op is None:
            return ""
        return "\n".join(pretty(self.op, "    "))
