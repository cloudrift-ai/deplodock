"""Hierarchical Fork-tree builder shared by pipeline rules that enumerate a
knob cartesian.

Converts a flat list of variant params into the ROOT ``Fork`` of a lazy
tree: each ``Level`` groups siblings by a (sub)tuple of knob values, sorts
siblings by lazily-computed max-propagated score, and collapses levels
whose key has a single distinct value across the group (params with an
empty key skip the level). The LAST level is the leaf level: its grouping
produces one leaf Fork per param (no branch wrapper) — knobs stamped onto
the leaf, ``expand`` thunk yields ``materialize(p)`` once the search engine
resolves that leaf. Everything is lazy: no Fork below the root exists and
no param is scored until a level expands (see :func:`build_fork_tree`).

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
    """Return the ROOT branch ``Fork`` of a lazy tree grouping ``params``
    per ``levels`` (outermost first).

    Nothing is built or scored at call time — the root carries an
    ``expand`` thunk for level 0 and a lazy score thunk; each branch's
    ``expand`` builds (and sorts) the next level on demand, so greedy
    descent instantiates O(path) Forks instead of one per param (~42k for
    a matmul-class kernel) and MCTS pays one level per pop.

    The LAST level is the leaf level: each param becomes one leaf Fork
    (``is_leaf=True``), knobs stamped from ``level.key(p)``, ``expand``
    thunk yields ``[materialize(p)]``. Earlier levels emit branch Forks
    keyed by ``level.key`` over the enclosing subgroup; siblings sort by
    ``-score()``; levels whose key has a single distinct value across the
    group are collapsed (no 1-child branch wrapper); params whose key is
    empty skip the level (their next-level subtree splices up as siblings
    of the keyed branches).

    Scores are LAZY: ``score(p)`` runs at most once per param (memoized),
    and only when a level containing ``p`` first expands — a branch's
    score thunk is ``max`` over its param subgroup, provably equal to
    eager max-of-children propagation (child scores bottom out at the
    same per-param values) without instantiating the subtree.
    """
    if not params:
        return []
    if not levels:
        raise ValueError("build_fork_tree: at least one Level required")
    all_params = list(params)
    leaf_level = levels[-1]
    branch_levels = levels[:-1]

    _memo: dict[int, float] = {}

    def _score(p: P) -> float:
        sid = id(p)
        v = _memo.get(sid)
        if v is None:
            v = score(p)
            _memo[sid] = v
        return v

    def _group_score(group: list[P]) -> Callable[[], float]:
        return lambda: max(_score(p) for p in group)

    def _sorted(forks: list[Fork]) -> list[Fork]:
        return sorted(forks, key=lambda f: -f.score())

    def _leaves(group: list[P]) -> list[Fork]:
        # Default-arg capture (``p=p``) avoids the late-binding closure
        # trap — without it every thunk would see the final loop var.
        return _sorted(
            [
                Fork(
                    knobs=dict(zip(leaf_level.knob_names, leaf_level.key(p), strict=True)),
                    expand=(lambda p=p: [materialize(p)]),
                    score=(lambda p=p: _score(p)),
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
                    score=_group_score(sub),
                    is_leaf=False,
                )
            )
        if skipped:
            siblings.extend(_build_level(skipped, depth + 1))
        return _sorted(siblings)

    return Fork(knobs={}, expand=(lambda: _build_level(all_params, 0)), score=_group_score(all_params), is_leaf=False)
