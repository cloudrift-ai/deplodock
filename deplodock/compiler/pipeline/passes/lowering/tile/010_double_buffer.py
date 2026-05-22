"""Double-buffering (stage-wrap-body refactor: stubbed).

Pre-refactor: replaced ``Stage`` with ``BufferedStage(buffer_count=2,
phase=K_o%2)`` for K-outer staged matmul kernels.

Post-refactor (Phase C bucket 11 — kernel emission): same logic, but
operates on the new Stage shape with sources tuple. The promotion
swaps Stage → BufferedStage and stamps buffer_count/phase on the
wrapping Stage; per-source pad on Sources is unaffected.

Stubbed (always RuleSkipped) until Phase C bucket 11 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("010_double_buffer: stubbed during stage-wrap-body refactor")
