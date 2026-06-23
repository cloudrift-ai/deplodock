"""transport pass (enumeration fork) — ``promote_transport`` (R5).

``plans/tile-ir-block-dag.md`` R5: ``promote_transport(read, →TMA)`` writes the
``Schedule.staged[edge]`` value (``SYNC`` → ``TMA``); it edits **no body**. The smem
slab's swizzle + the double-buffered ``cp.async.bulk.tensor`` ring are synthesized later
by ``assembly/_slab`` from the annotation, and ``assembly/020_peel`` software-pipelines
the K loop. This pass is the genuine fork: ``TMA ∈ {True, False}`` is a ranked knob the
search benches.

Pre-assemble, over the fully-staged stored algorithm: it reads the prospective smem
``Source``s ``assemble`` would build (``assembly/_slab.prospective_sources`` — a derived
projection, no tower) and the TMA-eligibility oracle (``enumeration/_transport.tma_eligible``:
sm_90+, affine box ≤ 256 / 16 B-aligned, a ringable K loop), then writes the chosen
``Transport.TMA`` straight into ``Schedule.staged``.

R5 scope: the **warp-tier** ``mma.sync`` matmul (an ``Atom`` is pinned). Greedy stays
byte-identical to today — the SYNC decision is offered first (option-0), so a cold compile
keeps SYNC staging; the TMA variant is the second offer the tuner explores (or a
``DEPLODOCK_TMA=1`` pin selects directly). Scalar / cooperative-reduce / pointwise tiers
skip here (``RuleSkipped`` → ``apply_off_defaults`` stamps ``TMA=False``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileGraphOp, Transport
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import mma_atom
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import prospective_sources
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import STAGE, TMA
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._transport import tma_eligible

PATTERN = [Pattern("root", TileGraphOp)]

_MIN_CAPABILITY = (9, 0)


def rewrite(ctx: Context, root: Node, match) -> list[TileGraphOp]:  # noqa: ARG001
    op: TileGraphOp = root.op
    if STAGE.name not in op.knobs or TMA.name in op.knobs:
        raise RuleSkipped("transport runs once, after stage decided the staged read-sites (idempotence via the TMA knob)")
    if mma_atom(op.knobs) is None:
        raise RuleSkipped("R5 promote_transport applies to the warp-tier mma matmul (scalar tiers stay SYNC)")
    if not op.tilegraph.schedule.staged:
        raise RuleSkipped("nothing staged — no read-site to promote")

    eligible = ctx.compute_capability >= _MIN_CAPABILITY and tma_eligible(op.tilegraph, prospective_sources(op.tilegraph), ctx)
    pin = TMA.raw()
    if pin is not None:
        # A pin is authoritative; an ineligible pinned-on shape declines gracefully to SYNC.
        decisions = [TMA.parse(pin) and eligible]
    else:
        # Greedy-safe: SYNC first (option-0, byte-identical to today), TMA second when eligible.
        decisions = [False, *([True] if eligible else [])]

    out: list[TileGraphOp] = []
    for use in decisions:
        if use:
            staged = {e: Transport.TMA for e in op.tilegraph.schedule.staged}
            tg = replace(op.tilegraph, schedule=replace(op.tilegraph.schedule, staged=staged))
        else:
            tg = op.tilegraph
        out.append(replace(op, tilegraph=tg, knobs={**op.knobs, TMA.name: use}))
    return out
