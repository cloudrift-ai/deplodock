"""Tile IR — a map/reduce kernel with its *schedule* made explicit.

One :class:`TileOp` is the article's reduction skeleton — ``project ∘
reduce(⊕, e) ∘ map(f)`` — scheduled but not yet bound to hardware threads.
It sits between Loop IR (pure iteration) and Kernel IR (threads / smem):

    Loop IR ──lowering/tile──▶ Tile IR ──lowering/kernel──▶ Kernel IR

The whole point of the layer is the article's thesis: **the schedule is
separate from the combine.** A ``TileOp`` records the *schedule* —

- ``grid_axes`` — the parallel (free) axes tiled onto the thread grid (one GPU
  thread per output cell).

— while the *combine* lives entirely in the ``op`` tree (``ir/stmt/algebra.Map`` /
``ir/tile/ops.Reduce``): a pointwise ``Map`` of leaf compute, or a ``Reduce`` folding
through a carrier (``Accum`` / ``Monoid`` + ``Twist``) whose ``finalize`` φ projects
the final state to the output. The algebra is **not stored as a tag**; the carriers
and partial structure are read directly where a pass needs them, per the project's
"the op tree is the single source of truth" rule. The per-cell ``body`` is *derived*
from ``op`` by ``lower`` for the matcher / cache-key / dump machinery.

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
    carries its own ``Write``) or a :class:`~deplodock.compiler.ir.tile.ops.Reduce` (a
    fold whose carrier ``finalize``\\ s the output value). ``grid_axes`` are the parallel
    axes mapped onto the thread grid.

    There is **no stored output store**: the ``Write`` that binds a ``Reduce``'s output
    value to the kernel's output buffer at the grid cell is *glue*, generated at
    materialize time from ``grid_axes`` + the graph node's output buffer (see
    ``lowering/kernel/010_materialize``). ``inputs`` / ``outputs`` come from the base
    :meth:`Op.populate_io` (graph edges) — no body walk. The ``body`` property derives
    the per-cell compute from ``op`` (sans glue) for the cache-key / dump machinery."""

    op: object = None  # Map | Reduce — the op tree; None for placeholder nodes
    grid_axes: tuple[Axis, ...] = field(default_factory=tuple)
    name: str = ""

    @property
    def body(self):
        """The per-cell compute as loop-IR stmts, derived from ``op`` (the source of
        truth) — NO output-store ``Write`` (that glue is generated at materialize).
        Empty when ``op`` is unset. Read by the cache key / dumps."""
        from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415  (tile.ir loads mid ir.stmt import)
        from deplodock.compiler.ir.tile.ops import lower  # noqa: PLC0415

        return Body(lower(self.op)) if self.op is not None else Body(())

    def pretty_body(self) -> str:
        from deplodock.compiler.ir.stmt.base import pretty_body as _pretty  # noqa: PLC0415

        return "\n".join(_pretty(self.body, "    "))
