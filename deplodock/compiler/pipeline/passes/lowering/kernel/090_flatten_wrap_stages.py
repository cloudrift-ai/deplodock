"""Vestigial pass — no-op under the StageBundle IR.

The wrap-body Stage chain is gone (replaced by StageBundle). The
materializer dispatches off ``bundle.policy`` directly so there is
nothing to flatten here. This shim is kept (rather than deleted) so the
pipeline registry's pass-id numbering stays stable across the
refactor; it always raises ``RuleSkipped``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> None:
    raise RuleSkipped("StageBundle IR — no wrap-body to flatten")
