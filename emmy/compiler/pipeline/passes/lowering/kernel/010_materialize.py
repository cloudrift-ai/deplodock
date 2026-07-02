"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the schedule's grid axes to GPU threads and realizes the reduce partition through the **one**
node-kind dispatcher, ``_factor.factorize`` — every ``TileOp`` root (a tiled ``Contraction``, a
cooperative / ILP reduce, or a pointwise / scalar cell) lowers through that single emitter, which
reads the node kind + role + reduce plan off ``tile.op`` and picks the tier:

- **Tiled ``CONTRACTION``** (warp / register tile) — the high-level :class:`Contraction` node was
  built recognize-side (``_schedule._contraction_node``, seam #1); ``factorize`` synthesizes its bare
  grid-``Write`` (needs ``root.output``, so it can't ride the node) and expands it (mma → the
  ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore`` fragment soup; scalar → the
  per-thread register cell tile) through the shared four-level tiling layer (in ``_factor``).
- **Reduce tier** — a ``PLANAR`` / ``TWISTED`` reduce (or a non-output-tiled ``CONTRACTION``) whose
  ``ReducePlan`` carries a BLOCK ``coop`` and/or a REG ``reg`` stage: the reduce axis is partitioned
  ``coop`` ways across the CTA's threads (cooperation) and ``reg`` ways across per-thread register
  accumulators (ILP), then a REG-tree fold, the cross-thread combine, and the projection.
- **Scalar tier** — one thread per output cell (``lower(op)`` + an output-store glue).

The op tree + ``lower`` are shared across kinds; only the schedule's partition changes — the
article's "schedule separate from combine" thesis. The tier machinery all lives in ``_factor``.
"""

from __future__ import annotations

from emmy.compiler.graph import Node
from emmy.compiler.ir.kernel import KernelOp
from emmy.compiler.ir.stmt import Body
from emmy.compiler.ir.tile import TileOp
from emmy.compiler.ir.tile.ops import reduce_plan
from emmy.compiler.pipeline import Match, Pattern
from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    # By the kernel pass, ``030_split`` has consumed every cross-CTA ``GRID`` stage (the
    # partial's plan is stripped, the finalize is a fresh ``ReducePlan``). A surviving split
    # request is a bug — the materializer only lowers single-launch kernels.
    rplan = reduce_plan(tile) if tile.op is not None else None
    assert rplan is None or not rplan.needs_split, "materialize: a GRID split stage survived 030_split"
    return KernelOp(body=Body((factorize(tile, root),)), name=tile.name)
