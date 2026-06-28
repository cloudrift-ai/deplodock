"""Cross-CTA split-reduce (``cta`` tier) — RESERVED SLOT (not built this cut).

A reduce partition with a GRID stage (``ReducePlan.needs_split``) splits the reduce axis
across CTAs: a **partial** kernel and a **finalize** kernel. This pass would emit them; it
is a documented stub until the ``cta`` tier lands.

The key point recorded here is that **splits are expressible in the algebra tile IR** — the
finalize is just another reduce node:

- **partial kernel** — the ``cta`` stage becomes an extra grid axis; each CTA reduces its
  slice of the reduce axis and writes its carrier *state* (not the projected output) to a
  ``ws[cta, *free]`` workspace, seeded by the fold ``Accum``\\ s (``op.identity``).
- **finalize kernel** — an ordinary reduce over the split (``cta``) axis, built from
  ``carrier.as_state_merge(ws_partials)`` (the cross-partition combine rendered through the
  SAME streaming-merge machinery), reading the workspace and projecting the output. An
  additive carrier may instead finalize via an in-place ``atomicAdd`` (``Fold.ATOMIC``); the
  twisted flash ``(m, l, O)`` carrier is kernel-finalize-only (the ``e^{Δm}`` rescale can't
  be an atomic).

Until then this rule never fires (no schedule carries a GRID stage — ``020_schedule`` only
picks BLOCK cooperation), and a graph that somehow reached here with ``needs_split`` raises
loudly rather than miscompiling.
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile import TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    sched = tile.kernel.schedule if tile.kernel is not None else None
    reduce = getattr(sched, "reduce", None)
    if reduce is None or not reduce.needs_split:
        raise RuleSkipped("no cross-CTA split stage — nothing to split")
    raise NotImplementedError(
        "cross-CTA split-reduce (cta tier) is not built yet — see 030_split module docstring "
        "(partial kernel → ws workspace; finalize "
        "kernel = a reduce over the split axis via carrier.as_state_merge)."
    )
