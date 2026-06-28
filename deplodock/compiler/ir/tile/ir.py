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
from deplodock.compiler.ir.stmt.ir import BodyOp


@dataclass
class TileOp(BodyOp):
    """One scheduled map/reduce kernel (see module docstring).

    The kernel's compute is the op tree in ``op`` — a single
    :class:`~deplodock.compiler.ir.stmt.algebra.Map` (a pointwise per-cell body) or
    :class:`~deplodock.compiler.ir.tile.ops.Reduce` (a fold whose carrier finalizes to
    the output value). ``out`` is the output store for a ``Reduce`` op (a ``Map``
    carries its own ``Write``); ``grid_axes`` are the parallel axes mapped onto the
    thread grid. The per-cell ``body`` (inherited) is **derived** from ``op`` by
    ``lower`` — the op tree is the source of truth — so the matcher / cache-key / dump
    machinery that reads ``body`` keeps working. (Constructing with ``body=`` directly
    is still accepted for tests / hand-built guardrail nodes.)"""

    op: object = None  # Map | Reduce — the op tree (source of truth); None when body= is supplied directly
    out: object = None  # TensorRef | None — output store for a Reduce op (a Map carries its own Write)
    grid_axes: tuple[Axis, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.op is not None and not self.body:
            # Local imports: tile.ir is loaded while ir.stmt is mid-import (normalize
            # pulls Stage from here), so module-level tile.ops / stmt-package imports
            # would re-enter a partially-built package. Construction-time is safe.
            from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415
            from deplodock.compiler.ir.stmt.leaves import Write  # noqa: PLC0415
            from deplodock.compiler.ir.tile.ops import lower  # noqa: PLC0415

            stmts = list(lower(self.op))
            if self.out is not None:  # a Reduce: store the carrier-finalized output value
                stmts.append(Write(output=self.out.buf, index=self.out.index, value=self.op.out))
            self.body = Body(stmts)
        super().__post_init__()
