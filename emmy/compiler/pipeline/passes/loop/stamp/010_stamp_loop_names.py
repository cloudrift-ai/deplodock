"""Stamp ``LoopOp.name`` with a provenance-derived kernel label.

Runs last in the loop dialect — in the ``loop/stamp`` pass, after ``loop/fusion``
has settled the body — so the structural hash baked into the name (see
:func:`provenance.name_for`) reflects the fused form and every kernel is labeled.
(Pattern recognition — flash / online-softmax — now runs downstream in
``lowering/tile/010_recognize``, so its minted kernels fall back to the node-id
name; re-stamping recognized kernels is future work.) The Tile-dialect partition planner
(``lowering/tile/010_partition_loops``) reads ``loop_op.name`` directly
and forwards it to the emitted ``TileOp``; every subsequent dialect copies
it through. Stamping here means ``--ir loop`` dumps already render
semantic labels (``k_rms_norm_3f2a1b`` etc.) instead of empty strings.

Idempotent: a LoopOp that already carries a name is skipped, so the
rewrite engine reaches fixed-point in one pass over the graph.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.loop.stamp._stamp import name_for_loop

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if root.op.name:
        raise RuleSkipped("LoopOp already named")
    return replace(root.op, name=name_for_loop(root.op, root, match.graph))
