"""Stamp ``LoopOp.knobs`` with the kernel's structural features.

Runs LAST in the loop dialect, after fusion has settled the body (and after
``030_stamp_loop_names``) — the same rationale as the name stamp: the features
must reflect the final fused form. Stamping here, rather than at the head of
Tile-IR lowering, matters for consistency: the loop-dialect ``op_cache_key``
includes knobs, so a mid-lowering stamp would give the same logical kernel two
identities (pre- and post-stamp), splitting the tune DB's effort / perf keyings
and any search-tree comparison that crosses the stamp point.

The structural features (:func:`~deplodock.compiler.ir.features.structure_features`)
are an ``S_``-prefixed flat dict: a stmt/op histogram (the extent-free skeleton)
plus the loop-axis extents. Stamped into the knobs and carried forward by the
engine's knob-merge, they ARE the kernel's structural identity: the partition
planner's score cache keys on ``(ctx, frozenset(merged knobs))`` by VALUE
(``_score_variant``), so structurally identical kernels (the same layer repeated
through a model) carry the same ``S_*`` knobs and share entries with no
object-identity bookkeeping — and ``knob.knob_features`` turns the whole knob
dict into the planner-prior feature vector. The reserved ``S_`` prefix keeps
them out of the tuning-knob view (``format_tuning_knobs``).

Idempotent: a LoopOp that already carries any ``S_`` knob is skipped, so the
rewrite engine reaches fixed-point in one pass over the graph.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.features import STRUCT_PREFIX, structure_features
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if any(k.startswith(STRUCT_PREFIX) for k in root.op.knobs):
        raise RuleSkipped("LoopOp already carries structural features")
    feats = structure_features(root.op.body, match.graph)
    return replace(root.op, knobs={**root.op.knobs, **feats})
