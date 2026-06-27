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
``ReduceCarrier`` for the fold ⊕). The algebra is **not stored**; the body's
carriers (``Accum`` / ``Monoid`` + ``Twist``) and partial structure are read
directly where a pass needs them, per the project's "the body is the single
source of truth" rule.

Because the combine is in the body and the schedule is in ``grid_axes``, the
SAME op and the SAME materializer extend across kernel kinds — only the carrier
(the ⊕) changes, never the schedule. The skeleton currently *builds* the no-fold
kind; the kinds that carry a combine schedule later by supplying it, not new
lowering code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt.ir import BodyOp


@dataclass
class TileOp(BodyOp):
    """One scheduled map/reduce kernel (see module docstring).

    ``body`` (inherited) is the per-cell program in the scalar sublanguage;
    ``grid_axes`` are the parallel axes mapped onto the thread grid. ``inputs``
    / ``outputs`` are seeded from body Loads / Writes by :class:`BodyOp`."""

    grid_axes: tuple[Axis, ...] = field(default_factory=tuple)
