"""Structural feature extraction — describe a kernel's loop-level computation
as a flat numeric dict for the (future) learned planner prior.

:func:`structure_features` walks a ``LoopOp`` body and emits an ``S_``-prefixed
``dict[str, float]``: a stmt-type / op-multiset histogram (the extent-free
"skeleton") plus the loop-axis extents (``S_ext_*`` — the M/N/K shapes). The
``S_`` prefix keeps these from colliding with tuning-knob names and lets the
knob layer skip them in tuning displays.

These features are stamped into ``LoopOp.knobs`` by
``loop/fusion/040_stamp_structural_id`` and carried forward by the engine's
knob-merge, so a fully-lowered kernel's knob dict alone identifies its
structure — and they ARE the structural identity the score cache keys on
(``_score_variant`` folds ``frozenset(merged knobs)``). ``knob.knob_features``
turns the whole knob dict (structural + tuning) into the model's feature vector.

Known limitation: ``TileOp.lazy_score`` reads write-index *coalescing*
(``_coalescing_inner_extent_from_writes``), which a stmt histogram does not
capture — two kernels with identical histograms but different write-index
layouts collide on this identity and share a score. Accepted for now (such
kernels are near-identical and score-interchangeable); the first feature to add
if collisions prove too coarse.
"""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.stmt.body import Body

# Reserved prefix for structural-feature knobs — distinct from any tuning Knob
# name. ``knob.format_tuning_knobs`` / ``knob.knob_features`` key off it.
STRUCT_PREFIX = "S_"


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
