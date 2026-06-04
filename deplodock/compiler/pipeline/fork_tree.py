"""Hierarchical Fork-tree builder shared by pipeline rules that enumerate a
knob cartesian.

Converts a flat list of variant params into a Fork tree where each ``Level``
groups siblings by a (sub)tuple of knob values, sorts siblings by max-
propagated child score, and collapses levels whose key has a single distinct
value across the group. The LAST level is the leaf level: its grouping
produces one leaf Fork per param (no branch wrapper) — knobs stamped onto
the leaf, ``expand`` thunk yields ``materialize(p)`` once the search engine
resolves that leaf. Branch construction is lazy: only the returned level's
Forks exist up front, and a branch's ``expand`` builds the next level on
demand (see :func:`build_fork_tree`).

Use this when a rule produces ≥2 hierarchical levels of knob bundling with
score-propagated ranking (today: ``passes/lowering/tile/010_partition_loops``).
For flat 2-element forks with no hierarchy (e.g.
``passes/lowering/tile/085_warp_specialize``) a bare ``[Fork(...), Fork(...)]``
list comp is shorter and clearer — don't reach for this builder.

The engine in ``pipeline.py`` consumes ``fork.knobs`` flat (it doesn't walk
ancestors), so the LAST level's knob_names land on each leaf as the
DB-matchable knob delta. Earlier levels' knobs sit on the branch Forks
themselves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline.pipeline import Fork

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.base import Op


@dataclass(frozen=True)
class Level[P]:
    """One grouping level in the Fork tree.

    ``knob_names`` and ``key`` must agree in arity: ``key(p)`` returns a
    tuple of the same length as ``knob_names``, in matching order — or an
    EMPTY tuple when the level doesn't apply to ``p`` (branch levels only;
    the leaf level must apply to every param). Params with an empty key
    skip the level: their next-level subtree splices up as siblings of the
    level's keyed branches, so e.g. scalar tile variants carry no ``MMA``
    branch while warp variants of the same kernel do. Across all Levels the
    ``knob_names`` should partition the knob set the caller wants Forks to
    pin (no duplicates; collapsed-constant knobs may be absent — they're
    pinned by the materialized leaf Op itself).
    """

    knob_names: tuple[str, ...]
    key: Callable[[P], tuple]


def build_fork_tree[P](
    *,
    params: Sequence[P],
    levels: Sequence[Level[P]],
    materialize: Callable[[P], Op | Graph],
    score: Callable[[P], float],
) -> Fork | list[Fork]:
    """Group ``params`` into a Fork tree per ``levels`` (outermost first).

    The LAST level is the leaf level: each param becomes one leaf Fork
    (``is_leaf=True``), knobs stamped from ``level.key(p)``, ``expand``
    thunk yields ``[materialize(p)]``. Earlier levels emit branch Forks
    keyed by ``level.key`` over the enclosing subgroup; siblings sort by
    ``-score`` (max-of-children propagates upward); levels whose key has a
    single distinct value across the group are collapsed (no 1-child branch
    wrapper). Returns a single ``Fork`` when the top level collapses to one
    branch (engine still routes through fork-spawn since
    ``isinstance(option, Fork)``), otherwise the top-level ``list[Fork]``.

    ``score(p)`` is called exactly once per param at builder entry; the
    returned value is reused for both the leaf's own score and any branch
    ``max(child scores)`` propagation.

    **Construction is lazy below the top level.** Only the returned level's
    Fork objects exist up front; a branch's ``expand`` builds the next
    level on demand. A branch's score is ``max`` over the leaf scores of
    its param subgroup — provably equal to the eager
    ``max(child scores)`` propagation (child scores bottom out at the same
    ``leaf_score`` values) without instantiating the subtree. Greedy
    descent therefore builds O(path) Forks instead of one Fork per param
    (~42k for a matmul-class kernel), and MCTS pays one level per pop.
    """
    if not params:
        return []
    if not levels:
        raise ValueError("build_fork_tree: at least one Level required")
    leaf_score = {id(p): score(p) for p in params}
    leaf_level = levels[-1]
    branch_levels = levels[:-1]

    def _sorted(forks: list[Fork]) -> list[Fork]:
        return sorted(forks, key=lambda f: -f.score)

    def _leaves(group: list[P]) -> list[Fork]:
        # Default-arg capture (``p=p``) avoids the late-binding closure
        # trap — without it every thunk would see the final loop var.
        return _sorted(
            [
                Fork(
                    knobs=dict(zip(leaf_level.knob_names, leaf_level.key(p), strict=True)),
                    expand=(lambda p=p: [materialize(p)]),
                    score=leaf_score[id(p)],
                    is_leaf=True,
                )
                for p in group
            ]
        )

    def _build_level(group: list[P], depth: int) -> list[Fork]:
        if depth == len(branch_levels):
            return _leaves(group)
        level = branch_levels[depth]
        keyed: dict[tuple, list[P]] = {}
        # Params whose key is empty skip the level (it doesn't apply to
        # them) — their next-level subtree splices up as siblings of the
        # keyed branches below.
        skipped: list[P] = []
        for p in group:
            key = level.key(p)
            if not key:
                skipped.append(p)
            else:
                keyed.setdefault(key, []).append(p)
        if not keyed:
            # Level applies to nothing in this group — skip it wholesale.
            return _build_level(group, depth + 1)
        # Single-value collapse: the level adds no choice, so skip the
        # 1-child Fork wrapper and recurse straight into the next level.
        if not skipped and len(keyed) == 1:
            return _build_level(next(iter(keyed.values())), depth + 1)
        siblings: list[Fork] = []
        for key, sub in keyed.items():
            # ``expand`` recurses one level on demand (default-arg capture
            # again); the subtree below this branch doesn't exist until the
            # engine pops the branch.
            siblings.append(
                Fork(
                    knobs=dict(zip(level.knob_names, key, strict=True)),
                    expand=(lambda sub=sub, depth=depth: _build_level(sub, depth + 1)),
                    score=max(leaf_score[id(p)] for p in sub),
                    is_leaf=False,
                )
            )
        if skipped:
            siblings.extend(_build_level(skipped, depth + 1))
        return _sorted(siblings)

    top = _build_level(list(params), 0)
    return top[0] if len(top) == 1 else top
