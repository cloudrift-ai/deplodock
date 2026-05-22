"""K-outer pipelining (stage-wrap-body refactor: stubbed).

Pre-refactor: transformed a double-buffered async K-outer loop from
sync-style to prologue/main/epilogue temporal pipelining via direct
σ-substitution + emission of AsyncWait Stmts.

Post-refactor (Phase C bucket 10): renamed
``015_lower_pipelined_async_stage.py``; consumes
``AsyncBufferedStage(pipeline_depth>1)`` inside a Loop(SERIAL_OUTER)
and emits the prologue/main/epilogue expansion. The wrap-body Stage
contains the consumer subtree; σ_first/next/last are applied to the
stage's Source origins and the body separately. No AsyncWait Stmt is
emitted — the wait is implicit at the lowered Stage boundary.

Stubbed (always RuleSkipped) until Phase C bucket 10 rewrites it as
the new lowering pass.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("015_pipeline_k_outer: stubbed during stage-wrap-body refactor")
