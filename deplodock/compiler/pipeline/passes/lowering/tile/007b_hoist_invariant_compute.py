"""Hoist invariant compute (stage-wrap-body refactor: stubbed).

Pre-refactor: walked the reduce body, identified invariant compute
cones, and either inline-fused them into producer Stages (default) or
hoisted them into a sibling ComputeStage (FUSED_PIPELINE=True).

Post-refactor (Phase C bucket 7): same logic but operating on the new
wrap-body shape. The walker descends into Stage.body (no longer
"opaque" pass-through). The fused output writes new Stage instances
with multiple Sources (inline-fuse) or a sibling ComputeStage carrying
its own compute body.

Stubbed (always RuleSkipped) until Phase C bucket 7 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    raise RuleSkipped("007b_hoist_invariant_compute: stubbed during stage-wrap-body refactor")


# Tests reference these internal helpers. Stubbed-out to raise so the
# bucket-7 rewrite (when it happens) can re-supply the real implementations.
def _maybe_rewrite(body, *, fused_pipeline: bool):  # pragma: no cover — stub
    raise NotImplementedError("007b internal helper stubbed during refactor")
