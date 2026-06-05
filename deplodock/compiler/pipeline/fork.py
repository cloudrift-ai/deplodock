"""Fork interface + implementations: the deferred fork options the search
engine ranks and resolves, and the hierarchical Fork-tree builder shared by
pipeline rules that enumerate a knob cartesian.

:class:`Fork` is the interface — ``knobs``, ``is_leaf``, ``expand()``,
``score(cache)``. Implementations hold their producer's state as data:
:class:`OptionFork` (a concrete ``Op``/``Graph`` leaf), :class:`ThunkFork`
(generic flat forks), and the tree node classes :class:`_Branch` /
:class:`_Leaf` built by :func:`build_fork_tree`.

The tree builder converts a flat list of variant params into the ROOT
:class:`_Branch` of a lazy tree: each ``Level`` groups siblings by a
(sub)tuple of knob values and collapses levels whose key has a single
distinct value across the group (params with an empty key skip the level).
The LAST level is the leaf level: its grouping produces one :class:`_Leaf`
per param (no branch wrapper) — knobs stamped onto the leaf, ``expand()``
yields ``materialize(p)`` once the search engine resolves that leaf.
Everything is lazy: no Fork below the root exists and no param is scored
until the search reads a fork's score. Siblings are emitted in grouping
order — RANKING IS SEARCH POLICY: each node carries a lazy max-propagated
``score(cache)``, read by the policies via ``Search.score_of`` (which
passes the search-owned value-keyed cache down to the caller's per-param
scorer, so structurally identical kernels across a model share every
score).

Use the builder when a rule produces ≥2 hierarchical levels of knob
bundling with score-propagated ranking (today:
``passes/lowering/tile/010_partition_loops``). For flat 2-element forks
with no hierarchy (e.g. ``passes/lowering/tile/085_warp_specialize``) a
bare ``[ThunkFork(...), ThunkFork(...)]`` list comp is shorter and clearer
— don't reach for the builder.

The engine in ``pipeline.py`` consumes ``fork.knobs`` flat (it doesn't walk
ancestors), so the LAST level's knob_names land on each leaf as the
DB-matchable knob delta. Earlier levels' knobs sit on the branch Forks
themselves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.base import Op


class Fork(ABC):
    """Interface for a deferred fork option in the search tree.

    Two flavors share the interface:

    - **Branch Fork** (``is_leaf=False``) — produced explicitly by a rule's
      ``rewrite()`` to spawn a hierarchical fork point. ``expand()`` returns
      the next level of options (more Forks, concrete leaves, or a mix);
      the search loop drives this via :meth:`LazyCandidate.expand`.
    - **Leaf Fork** (``is_leaf=True``) — wraps one concrete ``Op`` /
      ``Graph`` rewrite. ``expand()`` returns ``[option]`` (one element);
      :meth:`LazyCandidate.resolve` invokes it once at resolve time to
      retrieve the leaf and apply it.

    Sharing one interface lets ``LazyCandidate.pending`` carry just
    ``Fork`` (no tagged union) — the search loop branches on
    ``Fork.is_leaf`` to decide expand-vs-resolve, and ``_best_fork``
    reads ``Fork.knobs`` polymorphically without isinstance.

    ``knobs`` is the knob-delta this Fork pins (used by ``_best_fork``
    to match the lowering DB without expanding). ``score(cache)`` is the
    LAZY planner-prior compute — typically the score of the best-reachable
    leaf under a branch. Ranking is SEARCH policy, not the producer's: the
    engine hands unranked siblings to ``Search.push`` and the policy reads
    ``Search.score_of``, which passes the search-owned ``score_cache``
    dict down to the compute. A scorer with a value identity memoizes its
    unit of work into that dict under its own key — the partition planner
    keys each variant by ``(ctx, merged knobs)``, complete because the
    ``SID`` structural-identity knob rides the dict — so structurally
    identical kernels across a model share every score. Scorers with
    nothing worth caching just ignore ``cache``."""

    knobs: dict
    is_leaf: bool = False

    @abstractmethod
    def expand(self) -> list[Op | Graph | Fork]: ...

    def score(self, cache: dict | None = None) -> float:  # noqa: ARG002 — uniform signature; default prior is neutral
        return 0.0


@dataclass(frozen=True)
class OptionFork(Fork):
    """Leaf Fork around an already-concrete rewrite option. Built by
    :meth:`LazyCandidate.from_op` / :meth:`LazyCandidate.from_graph` so
    every ``LazyCandidate.pending`` carries a uniform Fork shape."""

    option: Op | Graph
    knobs: dict = field(default_factory=dict)
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.option]


@dataclass(frozen=True)
class ThunkFork(Fork):
    """Generic implementation for flat one-off forks (e.g.
    ``tile/085_warp_specialize``'s two-element WS fork): behavior as
    plain functions of the fork's ``knobs`` — ``expand_fn(knobs)`` /
    ``score_fn(knobs)`` — so siblings share ONE function instead of
    per-instance capture lambdas (the knob delta is the only thing that
    varies). Trivial computes — nothing worth the score cache. A
    producer with real per-node state should define a dedicated ``Fork``
    subclass instead (see :class:`_Branch` / :class:`_Leaf` below)."""

    knobs: dict
    expand_fn: Callable[[dict], list[Op | Graph | Fork]]
    score_fn: Callable[[dict], float] | None = None
    is_leaf: bool = False

    def expand(self) -> list[Op | Graph | Fork]:
        return self.expand_fn(self.knobs)

    def score(self, cache: dict | None = None) -> float:  # noqa: ARG002
        return self.score_fn(self.knobs) if self.score_fn is not None else 0.0


# ---------------------------------------------------------------------------
# Hierarchical Fork-tree builder (``Level`` + ``build_fork_tree``).
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class _Tree[P]:
    """One tree's shared builder state — every :class:`_Branch` /
    :class:`_Leaf` node holds a reference back here instead of capturing
    closures. ``memo`` keeps the per-param scorer to at most one call per
    param per tree (a branch's max and its leaves bottom out at the same
    per-param values); cross-tree sharing is the scorer's own job against
    the search-owned ``cache`` it receives."""

    branch_levels: tuple[Level[P], ...]
    leaf_level: Level[P]
    materialize: Callable[[P], Op | Graph]
    score: Callable[[P, dict | None], float]
    memo: dict[int, float] = field(default_factory=dict)

    def param_score(self, p: P, cache: dict | None) -> float:
        sid = id(p)
        v = self.memo.get(sid)
        if v is None:
            v = self.score(p, cache)
            self.memo[sid] = v
        return v

    def build_level(self, group: list[P], depth: int) -> list[Fork]:
        """Build the sibling Forks one level down from a branch at
        ``depth`` (in grouping order — ranking is the search's job)."""
        if depth == len(self.branch_levels):
            return [
                _Leaf(tree=self, param=p, knobs=dict(zip(self.leaf_level.knob_names, self.leaf_level.key(p), strict=True))) for p in group
            ]
        level = self.branch_levels[depth]
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
            return self.build_level(group, depth + 1)
        # Single-value collapse: the level adds no choice, so skip the
        # 1-child Fork wrapper and recurse straight into the next level.
        if not skipped and len(keyed) == 1:
            return self.build_level(next(iter(keyed.values())), depth + 1)
        siblings: list[Fork] = [
            _Branch(tree=self, group=sub, next_depth=depth + 1, knobs=dict(zip(level.knob_names, key, strict=True)))
            for key, sub in keyed.items()
        ]
        if skipped:
            siblings.extend(self.build_level(skipped, depth + 1))
        return siblings


@dataclass(frozen=True)
class _Branch[P](Fork):
    """Branch node: a subgroup of params pinned to ``knobs`` by its level
    key. The subtree below doesn't exist until the engine pops the branch
    and ``expand()`` builds the next level."""

    tree: _Tree[P]
    group: list[P]
    next_depth: int
    knobs: dict

    def expand(self) -> list[Op | Graph | Fork]:
        return self.tree.build_level(self.group, self.next_depth)

    def score(self, cache: dict | None = None) -> float:
        # Max over the param subgroup — provably equal to eager
        # max-of-children propagation without instantiating the subtree.
        return max(self.tree.param_score(p, cache) for p in self.group)


@dataclass(frozen=True)
class _Leaf[P](Fork):
    """Leaf node: one param, ``expand()`` materializes its Op/Graph."""

    tree: _Tree[P]
    param: P
    knobs: dict
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.tree.materialize(self.param)]

    def score(self, cache: dict | None = None) -> float:
        return self.tree.param_score(self.param, cache)


def build_fork_tree[P](
    *,
    params: Sequence[P],
    levels: Sequence[Level[P]],
    materialize: Callable[[P], Op | Graph],
    score: Callable[[P, dict | None], float],
) -> Fork | list[Fork]:
    """Return the ROOT branch ``Fork`` of a lazy tree grouping ``params``
    per ``levels`` (outermost first).

    Nothing is built or scored at call time — the root is a
    :class:`_Branch` over the whole param list; each branch's ``expand()``
    builds the next level on demand, so greedy descent instantiates
    O(path) Forks instead of one per param (~42k for a matmul-class
    kernel) and MCTS pays one level per pop. Scores are equally lazy:
    nothing is scored until the search reads a fork's score, and the
    per-tree memo keeps ``score(p, cache)`` to at most one call per param.

    ``score`` receives the search-owned value-keyed ``cache`` dict
    alongside the param (``None`` when read outside a search) and owns
    its own keying — the partition planner keys each variant by
    ``(ctx, merged knobs)``, so the per-param work transfers across the
    trees of structurally identical kernels.
    """
    if not params:
        return []
    if not levels:
        raise ValueError("build_fork_tree: at least one Level required")
    tree = _Tree(
        branch_levels=tuple(levels[:-1]),
        leaf_level=levels[-1],
        materialize=materialize,
        score=score,
    )
    return _Branch(tree=tree, group=list(params), next_depth=0, knobs={})
