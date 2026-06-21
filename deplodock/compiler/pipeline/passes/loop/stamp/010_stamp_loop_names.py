"""Stamp ``LoopOp.name`` with a provenance-derived kernel label.

Runs last in the loop dialect — in the ``loop/stamp`` pass, after ``loop/fusion``
has settled the body AND ``loop/recognize`` has minted any pattern-specialized
kernels (e.g. the fused flash attention nest) — so the structural hash baked into
the name (see :func:`provenance.name_for`) reflects the final form and every
kernel, recognized or not, is labeled. The Tile-dialect partition planner
(``lowering/tile/010_partition_loops``) reads ``loop_op.name`` directly
and forwards it to the emitted ``TileOp``; every subsequent dialect copies
it through. Stamping here means ``--ir loop`` dumps already render
semantic labels (``k_rms_norm_3f2a1b`` etc.) instead of empty strings.

Idempotent: a LoopOp that already carries a name is skipped, so the
rewrite engine reaches fixed-point in one pass over the graph.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler import provenance
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if root.op.name:
        raise RuleSkipped("LoopOp already named")
    name = provenance.name_for(root.op, root.id, provenance.get(root), provenance.totals(match.graph))
    return replace(root.op, name=name)
