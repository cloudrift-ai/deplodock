"""Bank-pad detection for staged smem (stage-wrap-body refactor: stubbed).

Pre-refactor: detected bank conflicts in body Loads of staged buffers
and applied ``+1`` padding to a cache-axis extent. Operated on per-Stage
``pad: tuple[int, ...]``.

Post-refactor (Phase C bucket 11 — kernel emission): pad lives per-Source.
The pass walks each Stage's sources and applies pad per-source to break
conflicts in Loads against that source's smem buffer.

Stubbed (always RuleSkipped) until Phase C bucket 11 rewrites it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    raise RuleSkipped("014_pad_smem: stubbed during stage-wrap-body refactor")
