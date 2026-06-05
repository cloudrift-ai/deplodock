"""Unit tests for ``pipeline/fork.py``'s hierarchical Fork-tree builder.

Covers the generic builder contract using a synthetic param dataclass so
the suite stays decoupled from any specific Tile-IR pass (`partition_loops`
is the canonical caller; its integration tests live in
``tests/compiler/passes/test_partition_planner_forks.py``). The flat
``Fork`` implementations (``OptionFork`` / ``ThunkFork``) are exercised
through the engine in ``tests/compiler/pipeline/search/test_thunk_forks.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from deplodock.compiler.pipeline.fork import Fork, Level, build_fork_tree


@dataclass(frozen=True)
class P:
    """Synthetic variant params — three knob fields, enough for two grouping
    levels + a leaf level."""

    a: int
    b: int
    c: int


def _stub_materialize(p: P) -> str:
    """Stand-in for a real Op materializer; returns a string so leaves'
    ``expand()`` results are trivially comparable."""
    return f"op({p.a},{p.b},{p.c})"


def _identity_score(p: P, cache: dict | None = None) -> float:  # noqa: ARG001 — builder contract passes the search cache
    return float(p.a * 100 + p.b * 10 + p.c)


def _walk_leaves(node: Fork) -> list[Fork]:
    """Collect every leaf Fork in tree order (grouping order first)."""
    out: list[Fork] = []
    stack: list[Fork] = [node]
    while stack:
        cur = stack.pop(0)
        if cur.is_leaf:
            out.append(cur)
        else:
            stack[:0] = list(cur.expand())
    return out


def _walk_branches(node: Fork) -> list[Fork]:
    """Collect every non-leaf branch Fork in the tree."""
    out: list[Fork] = []
    stack: list[Fork] = [node]
    while stack:
        cur = stack.pop()
        if cur.is_leaf:
            continue
        out.append(cur)
        stack.extend(cur.expand())
    return out


# Conventional levels reused across tests: `a` is the outer branch key,
# `b` is the inner branch key, `c` is the leaf level.
_LEVELS = [
    Level(("A",), lambda p: (p.a,)),
    Level(("B",), lambda p: (p.b,)),
    Level(("C",), lambda p: (p.c,)),
]


def test_empty_params_raises():
    """No params ⟹ no fork point — the caller should skip the rule, not
    build a tree. The non-empty invariant keeps the return type a bare
    ``Fork`` (the engine never sees an empty option list)."""
    with pytest.raises(ValueError, match="params must be non-empty"):
        build_fork_tree(params=[], levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)


def test_empty_levels_raises():
    with pytest.raises(ValueError, match="at least one Level"):
        build_fork_tree(params=[P(1, 2, 3)], levels=[], materialize=_stub_materialize, score=_identity_score)


def test_single_param_single_leaf():
    """One param → the root expands straight to one leaf Fork (every level
    above the leaf collapses), knobs from the LAST level."""
    tree = build_fork_tree(params=[P(1, 2, 3)], levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert isinstance(tree, Fork)
    assert not tree.is_leaf and tree.knobs == {}  # lazy root: nothing pinned, nothing built
    (leaf,) = tree.expand()
    assert leaf.is_leaf
    # Leaf carries only the deepest level's pin.
    assert leaf.knobs == {"C": 3}
    assert leaf.expand() == ["op(1,2,3)"]


def test_two_params_identical_branch_key_collapses_branch():
    """Two params sharing every grouping key → both branch levels collapse;
    result is a flat list of leaf Forks at the top level."""
    params = [P(1, 2, 3), P(1, 2, 5)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    # A and B both have one distinct key → collapse. The leaf level (C)
    # sees two distinct values, so the root expands to two sibling leaves.
    top = tree.expand()
    assert all(f.is_leaf for f in top)
    assert {tuple(f.knobs.items()) for f in top} == {(("C", 3),), (("C", 5),)}


def test_two_params_distinct_branch_key_emits_branches():
    """Two params with distinct outer keys → two branch Forks, each with
    one leaf child."""
    params = [P(1, 2, 3), P(2, 2, 3)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    top = tree.expand()
    assert len(top) == 2
    for branch in top:
        assert not branch.is_leaf
        assert set(branch.knobs.keys()) == {"A"}
        children = branch.expand()
        assert len(children) == 1
        assert children[0].is_leaf


def test_branch_score_is_max_of_children():
    """A branch Fork's score equals max(child scores) recursively."""
    params = [P(1, 1, 1), P(1, 1, 9), P(2, 1, 1)]  # distinct A keys → top branches
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    for branch in _walk_branches(tree):
        children = branch.expand()
        assert children
        expected = max(c.score() for c in children)
        assert branch.score() == pytest.approx(expected)


def test_siblings_unranked_in_grouping_order():
    """The builder does NOT sort siblings — ranking is search policy
    (``Search.score_of``). Siblings come out in grouping (= first-
    occurrence) order regardless of score."""
    # identity_score makes A=3 best, but A=1 was enumerated first.
    params = [P(1, 0, 0), P(3, 0, 0), P(2, 0, 0)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert [f.knobs["A"] for f in tree.expand()] == [1, 3, 2]


def test_score_receives_the_search_cache():
    """``Fork.score(cache)`` threads the search-owned dict down to the
    caller's per-param scorer (which owns its own keying), at every
    level — branch max and leaf alike."""
    seen: list[dict | None] = []

    def recording_score(p: P, cache: dict | None = None) -> float:
        seen.append(cache)
        return _identity_score(p)

    params = [P(1, 0, 0), P(2, 0, 0)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=recording_score)
    the_cache: dict = {}
    tree.score(the_cache)
    assert seen and all(c is the_cache for c in seen)


def test_leaf_knobs_keys_equal_last_level_knob_names():
    """Every leaf's ``knobs`` keys match exactly the last level's
    ``knob_names`` (the partition invariant for the leaf layer)."""
    params = [P(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    for leaf in _walk_leaves(tree):
        assert set(leaf.knobs.keys()) == {"C"}


def test_branch_knobs_no_overlap_along_path():
    """Knob multiset along any root→leaf path has no duplicates — each
    knob is pinned exactly once across all Forks on the path."""
    params = [P(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)

    path = [tree]
    node = tree
    while not node.is_leaf:
        node = node.expand()[0]
        path.append(node)
    seen: dict[str, int] = {}
    for f in path:
        for k in f.knobs:
            seen[k] = seen.get(k, 0) + 1
    duplicates = {k: v for k, v in seen.items() if v > 1}
    assert not duplicates, f"knob duplicated along path: {duplicates}"
    # Every pinned knob is one of the level knob_names.
    assert set(seen) <= {"A", "B", "C"}


def test_collapsed_constant_level_omits_branch():
    """A level whose key is constant across all params produces no Fork
    wrapper for that level."""
    # All params share A=1, vary B and C → A level collapses, B emits
    # branches, leaves carry C.
    params = [P(1, 1, 1), P(1, 2, 1)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    for branch in _walk_branches(tree):
        assert "A" not in branch.knobs


def test_materialize_is_lazy():
    """``materialize`` fires 0 times at build, exactly once per leaf
    ``expand()`` resolution."""
    calls: list[P] = []

    def counting_materialize(p: P) -> str:
        calls.append(p)
        return _stub_materialize(p)

    params = [P(1, 2, 3), P(1, 2, 5)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=counting_materialize, score=_identity_score)
    assert calls == [], "materialize fired during build"

    leaves = _walk_leaves(tree)
    leaves[0].expand()
    assert len(calls) == 1
    leaves[1].expand()
    assert len(calls) == 2


def test_score_is_lazy_and_caching_is_the_scorers_job():
    """``score`` fires ZERO times at build AND at expansion (ranking is
    search policy — nothing scores until a score is read). The builder
    adds NO caching of its own: a scorer that memoizes into the ``cache``
    it receives (the production pattern) computes once per param across a
    full tree's reads; repeat reads re-enter the scorer and hit its
    cache."""
    computes: list[P] = []

    def caching_score(p: P, cache: dict | None = None) -> float:
        key = (p.a, p.b, p.c)
        if cache is not None and key in cache:
            return cache[key]
        computes.append(p)
        v = _identity_score(p)
        if cache is not None:
            cache[key] = v
        return v

    params = [P(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]  # 8 unique params
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=caching_score)
    assert computes == [], "score fired during build — it must be lazy"
    leaves = _walk_leaves(tree)
    branches = _walk_branches(tree)
    assert computes == [], "score fired during expansion — ranking is the search's job"
    the_cache: dict = {}
    for fork in (tree, *branches, *leaves):
        fork.score(the_cache)
    tree.score(the_cache)  # repeated reads hit the scorer's cache
    assert len(computes) == len(params)
    assert set(computes) == set(params)


def test_leaf_expand_returns_own_param_materialization():
    """Default-arg lambda capture safety: each leaf's ``expand()`` returns
    that leaf's *own* param's materialization. Without ``lambda p=p:`` the
    closure would late-bind and every leaf would return the last param."""
    params = [P(1, 1, 1), P(1, 1, 2), P(1, 1, 3)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    leaves = _walk_leaves(tree)
    # Map each leaf to its materialized result and check coverage matches
    # the input params, not just the last one repeated.
    results = sorted(leaf.expand()[0] for leaf in leaves)
    assert results == ["op(1,1,1)", "op(1,1,2)", "op(1,1,3)"]
