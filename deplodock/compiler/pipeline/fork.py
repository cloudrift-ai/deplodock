"""Fork interface + implementations: the deferred fork options the search
engine ranks and resolves, and the hierarchical Fork-tree builder shared by
pipeline rules that enumerate a knob cartesian.

:class:`Fork` is the interface — ``knobs``, ``is_leaf``, ``expand()``.
Implementations hold their producer's state as data:
:class:`OptionFork` (a concrete ``Op``/``Graph`` leaf), :class:`ThunkFork`
(generic flat forks), and the tree node classes :class:`_Branch` /
:class:`_Leaf` built by :func:`build_fork_tree`.

The tree builder converts a flat list of variant knob rows (plain dicts)
into the ROOT :class:`_Branch` of a lazy tree: each ``Level`` groups
siblings by a (sub)tuple of knob values and collapses levels whose key has
a single distinct value across the group (rows with an empty key skip the
level). Below the last level every row becomes one :class:`_Leaf` carrying
its COMPLETE row as ``knobs`` — the row IS the variant identity (the
``S_*`` structural-feature knobs ride the merged dict), so the perf DB and
the learned prior key leaves and branches by knobs alone, no structural
probing. ``expand()`` yields ``materialize(row)`` once the search engine
resolves a leaf.
Everything is lazy: no Fork below the root exists until the search expands
it. Siblings are emitted in grouping order — RANKING IS SEARCH POLICY: the
policies rank the frontier with the learned prior (Forks carry no score).

Use the builder when a rule produces ≥2 hierarchical levels of knob
bundling (today:
``passes/lowering/tile/010_partition_loops``). For flat 2-element forks
with no hierarchy (e.g. ``passes/lowering/tile/085_warp_specialize``) a
bare ``[ThunkFork(...), ThunkFork(...)]`` list comp is shorter and clearer
— don't reach for the builder.

The engine in ``pipeline.py`` consumes ``fork.knobs`` flat (it doesn't walk
ancestors): branch Forks pin their level's slice of the row, leaves carry
the whole row.
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
    ``Fork.is_leaf`` to decide expand-vs-resolve.

    ``knobs`` is the knob-delta this Fork pins (the variant identity the
    perf DB and the learned prior key on, read without expanding). Ranking
    is SEARCH policy: the engine hands unranked siblings to ``Search.push``
    and the policy ranks them with the learned
    :class:`~deplodock.compiler.pipeline.search.prior.Prior` (greedy
    ``mean_score`` argmin; MCTS PUCT). Forks carry no score of their own —
    the analytic per-fork scorer was removed when the learned prior replaced
    it; siblings are emitted in grouping order and the cold/no-prior fallback
    is that emission order."""

    knobs: dict
    is_leaf: bool = False

    @abstractmethod
    def expand(self) -> list[Op | Graph | Fork]: ...


@dataclass(frozen=True)
class OptionFork(Fork):
    """Leaf Fork around an already-concrete rewrite option. Built by
    :meth:`LazyCandidate.from_option` so every ``LazyCandidate.pending``
    carries a uniform Fork shape."""

    option: Op | Graph
    knobs: dict = field(default_factory=dict)
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.option]


@dataclass(frozen=True)
class ThunkFork(Fork):
    """Generic implementation for flat one-off forks (e.g.
    ``tile/085_warp_specialize``'s two-element WS fork): ``expand_fn(knobs)``
    as a plain function of the fork's ``knobs`` so siblings share ONE function
    instead of per-instance capture lambdas (the knob delta is the only thing
    that varies)."""

    knobs: dict
    expand_fn: Callable[[dict], list[Op | Graph | Fork]]
    is_leaf: bool = False

    def expand(self) -> list[Op | Graph | Fork]:
        return self.expand_fn(self.knobs)


# ---------------------------------------------------------------------------
# Hierarchical Fork-tree builder (``Level`` + ``build_fork_tree``).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Level:
    """One grouping level in the Fork tree.

    ``knob_names`` and ``key`` must agree in arity: ``key(row)`` returns a
    tuple of the same length as ``knob_names``, in matching order — or an
    EMPTY tuple when the level doesn't apply to ``row``. Rows with an
    empty key skip the level: their next-level subtree splices up as
    siblings of the level's keyed branches, so e.g. scalar tile variants
    carry no ``MMA`` branch while warp variants of the same kernel do.
    Across all Levels the ``knob_names`` should partition the knob set the
    caller wants BRANCHES to pin (no duplicates along a path); levels need
    not cover every row knob — leaves carry the complete row regardless.
    """

    knob_names: tuple[str, ...]
    key: Callable[[dict], tuple]


@dataclass(frozen=True)
class _Tree[P]:
    """One tree's shared builder state — every :class:`_Branch` /
    :class:`_Leaf` node holds a reference back here instead of capturing
    closures."""

    levels: tuple[Level, ...]
    materialize: Callable[[dict], Op | Graph]

    def build_level(self, group: list[dict], depth: int) -> list[Fork]:
        """Build the sibling Forks one level down from a branch at
        ``depth`` (in grouping order — ranking is the search's job)."""
        if depth == len(self.levels):
            # One leaf per row, carrying the COMPLETE row as its knobs —
            # the DB-matchable variant identity (levels may not cover
            # every knob, e.g. FK / OVERHANG).
            return [_Leaf(tree=self, knobs=dict(row)) for row in group]
        level = self.levels[depth]
        keyed: dict[tuple, list[dict]] = {}
        # Rows whose key is empty skip the level (it doesn't apply to
        # them) — their next-level subtree splices up as siblings of the
        # keyed branches below.
        skipped: list[dict] = []
        for row in group:
            key = level.key(row)
            if not key:
                skipped.append(row)
            else:
                keyed.setdefault(key, []).append(row)
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
class _Branch(Fork):
    """Branch node: a subgroup of knob rows pinned to ``knobs`` by its
    level key. The subtree below doesn't exist until the engine pops the
    branch and ``expand()`` builds the next level."""

    tree: _Tree
    group: list[dict]
    next_depth: int
    knobs: dict

    def expand(self) -> list[Op | Graph | Fork]:
        return self.tree.build_level(self.group, self.next_depth)


@dataclass(frozen=True)
class _Leaf(Fork):
    """Leaf node: one knob row (= ``knobs``, the complete variant
    identity); ``expand()`` materializes its Op/Graph."""

    tree: _Tree
    knobs: dict
    is_leaf = True

    def expand(self) -> list[Op | Graph | Fork]:
        return [self.tree.materialize(self.knobs)]


def build_fork_tree(
    *,
    params: Sequence[dict],
    levels: Sequence[Level],
    materialize: Callable[[dict], Op | Graph],
) -> Fork:
    """Return the ROOT branch ``Fork`` of a lazy tree grouping the knob
    rows ``params`` per ``levels`` (outermost first); below the last
    level each row becomes one leaf carrying its complete row as
    ``knobs``. ``params`` must be non-empty (a rule with nothing to
    enumerate has no fork point — skip the rule instead) and ``levels``
    non-empty; both raise ``ValueError``.

    Nothing is built at call time — the root is a :class:`_Branch` over
    the whole row list; each branch's ``expand()`` builds the next level
    on demand, so greedy descent instantiates O(path) Forks instead of
    one per row (~42k for a matmul-class kernel) and MCTS pays one level
    per pop. Siblings are emitted in grouping order; ranking is the
    search policy's job (the learned prior), not the tree's.
    """
    if not params:
        raise ValueError("build_fork_tree: params must be non-empty")
    if not levels:
        raise ValueError("build_fork_tree: at least one Level required")
    tree = _Tree(levels=tuple(levels), materialize=materialize)
    return _Branch(tree=tree, group=list(params), next_depth=0, knobs={})
