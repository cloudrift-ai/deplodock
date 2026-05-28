"""Emit a WS=0/1 Fork for kernels eligible for warp specialization.

Pattern-matches a ``TileOp`` whose body contains a TMA ``StageBundle``
with ``pipeline_depth == 2`` inside a ``SerialTile(serial_outer)`` —
i.e. the post-``080_pipeline_stages`` shape of a pipelined matmul or
attention kernel. Emits a 2-child ``Fork``: ``WS=0`` returns the input
unchanged, ``WS=1`` returns a ``TileOp`` with ``knobs["WS"] = 1`` (read
by ``100_materialize_tile``'s ``_materialize_ws`` path).

This is the first non-planner pass to emit a ``Fork`` — every other
strategy pass (``050_use_tma`` / ``060_use_async_copy`` / ``080_pipeline_stages``)
makes a deterministic ``RuleSkipped``-or-rewrite decision. The pipeline
driver supports ``Fork`` emission anywhere; we're just the first
strategy pass to use it.

Eligibility is intentionally narrow for the initial cut:

- TMA policy only — cp.async kernels can't use the producer/consumer
  split (no mbarrier-based cross-warp coordination).
- ``pipeline_depth == 2`` — the depth-1 path has no producer/consumer
  division to exploit.
- No cooperative ``Accum`` in the body — consumer-tid remap for
  SDPA-style cooperative reductions is pending (Phase D Stage 5);
  cooperative kernels fall through to ``RuleSkipped`` rather than
  emitting a fork that the materializer would reject.

``DEPLODOCK_WS=0`` / ``DEPLODOCK_WS=1`` env pins narrow the fork to a
single child via ``WS.narrow`` (the standard knob env-pin mechanism).
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import SerialTile, StageBundle, StagePolicy, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.pipeline import Fork

PATTERN = [Pattern("root", TileOp)]


WS = Knob(
    "WS",
    KnobType.BOOL,
    hints=(0, 1),
    help="Warp-specialize TMA staging: producer warps issue TMA, consumer warps wait + reduce",
)


def _eligible(op: TileOp) -> bool:
    """True iff the body has a TMA ``StageBundle`` with ``pipeline_depth == 2``
    inside a ``SerialTile(serial_outer)``."""
    for stmt in op.body.iter():
        if isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
            for inner in stmt.body:
                if (
                    isinstance(inner, StageBundle)
                    and inner.policy == StagePolicy.TMA
                    and inner.pipeline_depth == 2
                ):
                    return True
    return False


def _has_cooperative_accum(op: TileOp) -> bool:
    """True iff any Accum in the body has cooperative axes that would
    need consumer-tid remap during WS materialization (Phase D Stage 5).
    Returns False (skip WS) on kernels that would currently fail in
    ``_materialize_ws``'s NotImplementedError gate."""
    escape = op.body.coordination
    return any(escape.accum_cooperative_axes.values())


def rewrite(ctx: Context, root: Node) -> TileOp | Fork | list[Fork] | None:
    op: TileOp = root.op

    # Idempotence: once WS is stamped (by either fork branch) the rule
    # has done its job — the engine re-fires after every rewrite, so a
    # bare ``RuleSkipped`` is the canonical no-op signal.
    if "WS" in op.knobs:
        raise RuleSkipped("WS knob already set")

    if not _eligible(op):
        raise RuleSkipped("no TMA StageBundle with pipeline_depth=2 inside serial_outer")

    if _has_cooperative_accum(op):
        raise RuleSkipped("cooperative Accum present — consumer-tid remap pending (Phase D Stage 5)")

    # Narrow choices via ``DEPLODOCK_WS=0`` / ``DEPLODOCK_WS=1`` env pin.
    ws_choices = WS.narrow((0, 1))
    if not ws_choices:
        raise RuleSkipped("DEPLODOCK_WS env pin removed all WS choices")

    def _stamp(ws: int) -> TileOp:
        new_knobs = dict(op.knobs)
        new_knobs["WS"] = ws
        return TileOp(body=op.body, name=op.name, knobs=new_knobs)

    # Single-choice (env-pinned) short-circuit: emit the chosen variant
    # directly without a fork wrapper.
    if len(ws_choices) == 1:
        return _stamp(ws_choices[0])

    # Two-child fork: autotuner explores both. Scores left at 0 — the
    # tuner refines via measured per-op time, no prior to encode here.
    return [
        Fork(
            knobs={WS.name: ws},
            expand=(lambda ws=ws: [_stamp(ws)]),
            score=0.0,
            is_leaf=True,
        )
        for ws in ws_choices
    ]
