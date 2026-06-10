"""Stamp ``LoopOp.knobs`` with the kernel's structural features.

Runs LAST in the loop dialect, after fusion has settled the body (and after
``991_stamp_loop_names``) — the same rationale as the name stamp: the features
must reflect the final fused form. Stamping here, rather than at the head of
Tile-IR lowering, matters for consistency: the loop-dialect ``op_cache_key``
includes knobs, so a mid-lowering stamp would give the same logical kernel two
identities (pre- and post-stamp), splitting the tune DB's effort / perf keyings
and any search-tree comparison that crosses the stamp point.

:func:`structure_features` walks a ``LoopOp`` body and emits an ``S_``-prefixed
(``STRUCT_PREFIX``, declared in ``pipeline/knob.py``) flat ``dict[str, float]``:
a stmt-type / op-multiset histogram (the extent-free "skeleton") plus the
loop-axis extents (``S_ext_*`` — the M/N/K shapes). Stamped into the knobs and
carried forward by the engine's knob-merge, they ARE the kernel's structural
identity: ``knob.knob_features`` turns the whole knob dict (the row knobs plus
these ``S_*`` features) into the learned-prior feature vector, so structurally
identical kernels (the same layer repeated through a model) featurize alike and
share the prior's rows. The reserved ``S_`` prefix keeps them out of the
tuning-knob view (``format_tuning_knobs``).

This pass is the ONLY producer of the structural features — callers that need a
finalized identity stamp run it (idempotent: a LoopOp already carrying any
``S_`` knob is skipped, so the rewrite engine reaches fixed-point in one pass).

Known limitation: the histogram is extent-free except for ``S_ext_*`` — two
kernels with identical histograms but different write-index coalescing layouts
featurize identically. Accepted for now (such kernels are near-identical); the
first feature to add if it proves too coarse.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from math import prod
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import STRUCT_PREFIX

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.stmt.body import Body

PATTERN = [Pattern("root", LoopOp)]


def rewrite(match: Match, root: Node) -> LoopOp | None:
    if any(k.startswith(STRUCT_PREFIX) for k in root.op.knobs):
        raise RuleSkipped("LoopOp already carries structural features")
    feats = structure_features(root.op.body, match.graph)
    return replace(root.op, knobs={**root.op.knobs, **feats})


def structure_features(body: Body, graph: Graph | None = None) -> dict[str, float]:
    """Flat ``S_``-prefixed structural feature dict for a LoopOp ``body``:
    the extent-free skeleton merged with the ``S_ext_*`` loop extents.

    ``graph`` supplies operand dtypes for the ``S_dtype_*`` multiset; omit it
    (e.g. ad-hoc callers without a surrounding graph) to skip dtype features.
    Values are floats so the dict drops straight into the numeric knob row."""
    return {**_skeleton(body, graph), **_extents(body)}


def _skeleton(body: Body, graph: Graph | None) -> dict[str, float]:
    """Extent-free histogram: stmt-type counts + pointwise/reduce op multisets
    + loop-nest roles/depth + operand dtype multiset."""
    from deplodock.compiler.ir.stmt.blocks import Cond  # noqa: PLC0415
    from deplodock.compiler.ir.stmt.leaves import Assign, Mma  # noqa: PLC0415

    feats: Counter[str] = Counter()
    loads = body.loads
    feats["S_n_load"] = len(loads)
    feats["S_n_distinct_input"] = len({ld.input for ld in loads})
    feats["S_n_write"] = len(body.writes)
    feats["S_n_accum"] = len(body.accums)
    feats["S_n_mma"] = len(body.iter_of_type(Mma))
    feats["S_n_cond"] = len(body.iter_of_type(Cond))
    assigns = body.iter_of_type(Assign)
    feats["S_n_assign"] = len(assigns)
    for s in assigns:
        feats[f"S_pw_{s.op.name}"] += 1
    for s in body.accums:
        feats[f"S_reduce_{s.op.name}"] += 1
    loops = body.loops
    feats["S_n_loop"] = len(loops)
    feats["S_n_reduce_loop"] = sum(1 for loop in loops if loop.is_reduce)
    feats["S_n_free_loop"] = sum(1 for loop in loops if not loop.is_reduce)
    feats["S_loop_depth"] = _loop_depth(body)
    if graph is not None:
        for ld in loads:
            node = graph.nodes.get(ld.input)
            dt = str(node.output.dtype) if node is not None else "?"
            feats[f"S_dtype_{dt}"] += 1
    return {k: float(v) for k, v in feats.items()}


def _loop_depth(body: Body) -> int:
    """Max ``Loop`` nesting depth along any path (non-Loop wrappers like
    ``Cond`` recurse without incrementing)."""
    from deplodock.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

    best = 0
    for s in body:
        if isinstance(s, Loop):
            best = max(best, 1 + _loop_depth(s.body))
        else:
            for nested in s.nested():
                best = max(best, _loop_depth(nested))
    return best


def _extents(body: Body) -> dict[str, float]:
    """Continuous ``S_ext_*`` loop extents, split by free vs reduce axis
    (``Loop.is_reduce``). Symbolic axes (non-static extent) are excluded from
    the products and counted in ``S_ext_n_symbolic_axis``."""
    free: list[int] = []
    reduce_: list[int] = []
    n_symbolic = 0
    for loop in body.loops:
        ext = loop.axis.extent
        if not ext.is_static:
            n_symbolic += 1
            continue
        (reduce_ if loop.is_reduce else free).append(ext.as_static())
    return {
        "S_ext_n_free_axis": float(len(free)),
        "S_ext_free_prod": float(prod(free)) if free else 1.0,
        "S_ext_free_max": float(max(free)) if free else 0.0,
        "S_ext_n_reduce_axis": float(len(reduce_)),
        "S_ext_reduce_prod": float(prod(reduce_)) if reduce_ else 1.0,
        "S_ext_reduce_max": float(max(reduce_)) if reduce_ else 0.0,
        "S_ext_n_symbolic_axis": float(n_symbolic),
    }
