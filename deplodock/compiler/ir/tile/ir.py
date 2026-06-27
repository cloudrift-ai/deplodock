"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` records the *schedule* —

- ``grid_axes`` — the parallel (free) axes tiled onto the thread grid (one GPU
  thread per output cell).

— while the *combine* lives entirely in the ``body`` (the leaf compute:
``Load`` / ``Assign`` / ``Write``, plus a reduce ``Loop`` wrapping a
``ReduceCarrier`` for the fold ⊕). The algebra is **not stored**; it is read back
from the body via :attr:`algebra_kind` (``classify_algebra``), per the project's
"algebra is a derived cache, never a second source of truth" rule.

Because the combine is in the body and the schedule is in ``grid_axes``, the
SAME op and the SAME materializer extend across kernel kinds — only the carrier
(the ⊕) changes, never the schedule. The skeleton currently *builds* the no-fold
kind; the kinds that carry a combine schedule later by supplying it, not new
lowering code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.algebra import AlgebraKind, classify_algebra
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt.blocks import Loop
from deplodock.compiler.ir.stmt.ir import BodyOp


@dataclass
class TileOp(BodyOp):
    """One scheduled map/reduce kernel (see module docstring).

    ``body`` (inherited) is the per-cell program in the scalar sublanguage;
    ``grid_axes`` are the parallel axes mapped onto the thread grid. ``inputs``
    / ``outputs`` are seeded from body Loads / Writes by :class:`BodyOp`."""

    grid_axes: tuple[Axis, ...] = field(default_factory=tuple)

    @property
    def algebra_kind(self) -> AlgebraKind:
        """Read the kernel's algebraic kind back from the body — ``MAP`` when
        there is no reduce carrier, else the kind of the first reduce ``Loop``
        (``MONOID`` / ``SEMIRING``). Derived, never stored, so it can never
        contradict the body's carrier (see ``ir/algebra.py``)."""
        for s in self.body.iter():
            if isinstance(s, Loop) and s.is_reduce:
                return classify_algebra(s)
        return AlgebraKind.MAP
