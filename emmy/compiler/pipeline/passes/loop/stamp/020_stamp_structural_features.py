"""Stamp ``LoopOp.knobs`` with the kernel's structural features.

Runs LAST in the loop dialect — in the ``loop/stamp`` pass, after ``loop/fusion``
and ``loop/recognize`` have settled the body (and after ``010_stamp_loop_names``) —
the same rationale as the name stamp: the features must reflect the final fused /
recognized form. Stamping here, rather than at the head of
Tile-IR lowering, matters for consistency: the loop-dialect ``op_cache_key``
includes knobs, so a mid-lowering stamp would give the same logical kernel two
identities (pre- and post-stamp), splitting the tune DB's effort / perf keyings
and any search-tree comparison that crosses the stamp point.

The work lives in :func:`stamp._stamp.structure_features`: it walks a ``LoopOp``
body and emits an ``S_``-prefixed (``STRUCT_PREFIX``, declared in
``pipeline/knob.py``) flat ``dict[str, float]``: a stmt-type / op-multiset
histogram (the extent-free "skeleton") plus the loop-axis extents (``S_ext_*`` —
the M/N/K shapes). Stamped into the knobs and carried forward by the engine's
knob-merge, they ARE the kernel's structural identity: ``knob.knob_features``
turns the whole knob dict (the row knobs plus these ``S_*`` features) into the
learned-prior feature vector, so structurally identical kernels (the same layer
repeated through a model) featurize alike and share the prior's rows. The
reserved ``S_`` prefix keeps them out of the tuning-knob view
(``format_tuning_knobs``).

This pass stamps during lowering; callers that build LoopOps outside the
pipeline re-stamp via ``stamp._stamp.restamp_structural_features``. Idempotent:
a LoopOp already carrying any ``S_`` knob is skipped, so the rewrite engine
reaches fixed-point in one pass.

Known limitation: the histogram is extent-free except for ``S_ext_*`` — two
kernels with identical histograms but different write-index coalescing layouts
featurize identically. Accepted for now (such kernels are near-identical); the
first feature to add if it proves too coarse.
"""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Node
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.pipeline import Match, Pattern, RuleSkipped
from emmy.compiler.pipeline.knob import STRUCT_PREFIX
from emmy.compiler.pipeline.passes.loop.stamp._stamp import structure_features

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if any(k.startswith(STRUCT_PREFIX) for k in root.op.knobs):
        raise RuleSkipped("LoopOp already carries structural features")
    feats = structure_features(root.op.body, match.graph)
    return replace(root.op, knobs={**root.op.knobs, **feats})
