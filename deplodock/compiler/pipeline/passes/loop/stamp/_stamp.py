"""Shared loop-dialect stamp helpers: kernel naming + structural features.

The two ``loop/stamp`` rules are thin rewrite wrappers around the functions
here: ``010_stamp_loop_names`` calls :func:`name_for_loop`, and
``020_stamp_structural_features`` calls :func:`structure_features`. Factoring
the logic out lets callsites that build LoopOps *outside* the pass pipeline —
e.g. the ``lowering/tile/split/005_split_demoted`` fragment assembler — name
and re-stamp their kernels with a plain import instead of reaching into a
leading-digit pass module via ``importlib`` (a pass file's ``NNN_…`` stem
isn't a legal import name).
"""

from __future__ import annotations

from collections import Counter
from math import prod
from typing import TYPE_CHECKING

from deplodock.compiler import provenance
from deplodock.compiler.pipeline.knob import STRUCT_PREFIX

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph, Node
    from deplodock.compiler.ir.loop import LoopOp
    from deplodock.compiler.ir.stmt.body import Body


# ---------------------------------------------------------------------------
# Kernel naming (``010_stamp_loop_names``)
# ---------------------------------------------------------------------------


def name_for_loop(op: LoopOp, node: Node, graph: Graph) -> str:
    """The provenance-derived ``k_…`` kernel label ``010_stamp_loop_names``
    stamps onto ``op``, factored out so fragment builders name their LoopOps
    the same way. Threads the node id + per-node provenance + graph-wide
    coverage totals into :func:`provenance.name_for`."""
    return provenance.name_for(op, node.id, provenance.get(node), provenance.totals(graph))


# ---------------------------------------------------------------------------
# Structural features (``020_stamp_structural_features``)
# ---------------------------------------------------------------------------


def structure_features(body: Body, graph: Graph | None = None) -> dict[str, float]:
    """Flat ``S_``-prefixed structural feature dict for a LoopOp ``body``:
    the extent-free skeleton merged with the ``S_ext_*`` loop extents.

    ``graph`` supplies operand dtypes for the ``S_dtype_*`` multiset; omit it
    (e.g. ad-hoc callers without a surrounding graph) to skip dtype features.
    Values are floats so the dict drops straight into the numeric knob row."""
    return {**_skeleton(body, graph), **_extents(body)}


def restamp_structural_features(op: LoopOp, graph: Graph | None = None) -> None:
    """Strip stale ``S_*`` knobs off ``op`` and re-stamp the structural features
    for its current body, in place. For fragment builders that rewrite a body
    *after* ``020_stamp_structural_features`` already ran (the pass runs once at
    fusion end and never revisits a spliced fragment) — without this the split
    kernels would featurize as the fused kernel for the learned prior."""
    op.knobs = {k: v for k, v in op.knobs.items() if not k.startswith(STRUCT_PREFIX)}
    op.knobs.update(structure_features(op.body, graph))


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
