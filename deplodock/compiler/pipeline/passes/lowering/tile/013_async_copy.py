"""cp.async promotion (stage-wrap-body refactor: stubbed).

Pre-refactor: promoted ``BufferedStage`` → ``AsyncBufferedStage`` and
appended an explicit ``AsyncWait(keep=0)`` after each stage so the
synchronous-style invariant held.

Post-refactor (Phase C bucket 11 — kernel emission): the wait is
implicit at the wrap boundary; AsyncBufferedStage with pipeline_depth=1
materializes to CpAsyncCopy + CpAsyncCommit + CpAsyncWait(0) + Sync
inside the materializer. This pass becomes a pure "promote
BufferedStage → AsyncBufferedStage when eligible" rewrite.

Stubbed (always RuleSkipped) until Phase C bucket 11 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("013_async_copy: stubbed during stage-wrap-body refactor")
