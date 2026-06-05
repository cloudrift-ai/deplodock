"""Stamp ``LoopOp.knobs["SID"]`` with the kernel's structural identity.

Runs LAST in the loop dialect, after fusion has settled the body (and after
``030_stamp_loop_names``) — the same rationale as the name stamp: identity
must reflect the final fused form. Stamping here, rather than at the head
of Tile-IR lowering, matters for consistency: the loop-dialect
``op_cache_key`` includes knobs, so a mid-lowering stamp would give the
same logical kernel two identities (pre- and post-stamp), splitting the
tune DB's effort / perf keyings and any search-tree comparison that
crosses the stamp point.

The SID (:func:`~deplodock.compiler.pipeline.search.keys.loop_structural_id`)
digests the canonical body plus the operand dtypes. Once it rides the knobs
dict, a knob dict alone fully identifies a tile variant: the partition
planner's score cache keys on ``(SID, ctx, row)`` by VALUE, structurally
identical kernels (the same layer repeated through a model) share entries
with no object-identity bookkeeping, and the SID flows forward with every
knob merge so DB rows and dumped configs self-identify the kernel they were
measured on.

Idempotent: a LoopOp that already carries a SID is skipped, so the rewrite
engine reaches fixed-point in one pass over the graph.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.search.keys import loop_structural_id

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if "SID" in root.op.knobs:
        raise RuleSkipped("LoopOp already carries a SID")
    sid = loop_structural_id(root.op, match.graph)
    return replace(root.op, knobs={**root.op.knobs, "SID": sid})
