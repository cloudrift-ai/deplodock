"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` records the *schedule* —

- ``grid_axes`` — the parallel (free) axes tiled onto the thread grid (one GPU
  thread per output cell).

— while the *combine* lives entirely in the ``op`` tree (``ir/stmt/algebra``): a
pointwise ``Map`` of leaf compute (optionally OVER a nested node — a ``Map`` over a
``Monoid`` is the φ projection ``project ∘ reduce``), a ``Monoid`` folding through its
carrier (``Twist``), or a ``Semiring`` contraction. The algebra is **not stored as a
tag**; the carriers
and partial structure are read directly where a pass needs them, per the project's
"the op tree is the single source of truth" rule. There is no stored per-cell body —
``lower(op)`` generates it at materialize (and on demand for the dump / cache key).

Because the combine is in the op tree and the schedule is in ``grid_axes``, the SAME
op and the SAME materializer extend across kernel kinds — only the carrier (the ⊕)
changes, never the schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import Op


@dataclass
class TileOp(Op):
    """One scheduled map/reduce kernel (see module docstring).

    Holds exactly the op tree (``op``) and the schedule (``grid_axes``) — not a
    pre-lowered body, and not a ``BodyOp``. ``op`` is a single
    :class:`~deplodock.compiler.ir.stmt.algebra.Map` (a pointwise per-cell body that
    carries its own ``Write``, or a projection ``Map`` over a reduction) or a reduction —
    a :class:`~deplodock.compiler.ir.stmt.algebra.Monoid` / ``Semiring`` whose ``out`` is
    the carried value. ``grid_axes`` are the parallel axes mapped onto the thread grid.

    There is **no stored body and no stored output store**: the per-cell loop-IR body
    is generated at materialize time by ``lower(op)``, and the ``Write`` that binds a
    reduction's output value to the kernel's output buffer at the grid cell is *glue*
    generated there too, from ``grid_axes`` + the graph node's output buffer (see
    ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk. ``pretty_body`` lowers ``op``
    on demand for dumps (the cache key lowers it likewise in ``search/keys``)."""

    op: object = None  # Map | Reduce — the op tree; None for placeholder nodes
    grid_axes: tuple[Axis, ...] = field(default_factory=tuple)
    name: str = ""

    def pretty_body(self) -> str:
        """Render the ``op`` tree structurally (the dump view) — no lowering."""
        from deplodock.compiler.ir.tile.ops import pretty  # noqa: PLC0415

        return "\n".join(pretty(self.op, "    ")) if self.op is not None else ""
