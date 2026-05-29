"""Unit tests for the hierarchical Fork-tree builder.

Covers the generic builder contract using a synthetic param dataclass so
the suite stays decoupled from any specific Tile-IR pass (`partition_loops`
is the canonical caller; its integration tests live in
``tests/compiler/passes/test_partition_planner_forks.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from deplodock.compiler.pipeline.fork_tree import Level, build_fork_tree
from deplodock.compiler.pipeline.pipeline import Fork


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


def _identity_score(p: P) -> float:
    return float(p.a * 100 + p.b * 10 + p.c)


def _walk_leaves(node: Fork | list[Fork]) -> list[Fork]:
    """Collect every leaf Fork in tree order (option-0 first)."""
    out: list[Fork] = []
    stack: list[Fork] = [node] if isinstance(node, Fork) else list(node)
    while stack:
        cur = stack.pop(0)
        if cur.is_leaf:
            out.append(cur)
        else:
            stack[:0] = list(cur.expand())
    return out


def _walk_branches(node: Fork | list[Fork]) -> list[Fork]:
    """Collect every non-leaf branch Fork in the tree."""
    out: list[Fork] = []
    stack: list[Fork] = [node] if isinstance(node, Fork) else list(node)
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


def test_empty_params_returns_empty_list():
    out = build_fork_tree(params=[], levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert out == []


def test_empty_levels_raises():
    with pytest.raises(ValueError, match="at least one Level"):
        build_fork_tree(params=[P(1, 2, 3)], levels=[], materialize=_stub_materialize, score=_identity_score)


def test_single_param_single_leaf():
    """One param → bare leaf Fork (not list), knobs from the LAST level."""
    tree = build_fork_tree(params=[P(1, 2, 3)], levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert isinstance(tree, Fork)
    assert tree.is_leaf
    # Every level above the leaf collapses (one distinct key each); leaf
    # carries only the deepest level's pin.
    assert tree.knobs == {"C": 3}
    assert tree.expand() == ["op(1,2,3)"]


def test_two_params_identical_branch_key_collapses_branch():
    """Two params sharing every grouping key → both branch levels collapse;
    result is a flat list of leaf Forks at the top level."""
    params = [P(1, 2, 3), P(1, 2, 5)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    # A and B both have one distinct key → collapse. The leaf level (C)
    # sees two distinct values, so it emits two sibling leaves directly.
    assert isinstance(tree, list)
    assert all(f.is_leaf for f in tree)
    assert {tuple(f.knobs.items()) for f in tree} == {(("C", 3),), (("C", 5),)}


def test_two_params_distinct_branch_key_emits_branches():
    """Two params with distinct outer keys → two branch Forks, each with
    one leaf child."""
    params = [P(1, 2, 3), P(2, 2, 3)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert isinstance(tree, list)
    assert len(tree) == 2
    for branch in tree:
        assert not branch.is_leaf
        assert set(branch.knobs.keys()) == {"A"}
        children = branch.expand()
        assert len(children) == 1
        assert children[0].is_leaf


def test_branch_score_is_max_of_children():
    """A branch Fork's score equals max(child scores) recursively."""
    params = [P(1, 1, 1), P(1, 1, 9), P(2, 1, 1)]  # distinct A keys → top branches
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert isinstance(tree, list)
    for branch in _walk_branches(tree):
        children = branch.expand()
        assert children
        expected = max(c.score for c in children)
        assert branch.score == pytest.approx(expected)


def test_siblings_sorted_descending_by_score():
    """Sibling Forks at every level appear in descending score order."""
    # Three distinct A values; identity_score makes A=3 best, A=1 worst.
    params = [P(1, 0, 0), P(3, 0, 0), P(2, 0, 0)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=_identity_score)
    assert isinstance(tree, list)
    scores = [f.score for f in tree]
    assert scores == sorted(scores, reverse=True)


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
    roots = [tree] if isinstance(tree, Fork) else list(tree)

    def _first_path(root: Fork) -> list[Fork]:
        path = [root]
        node = root
        while not node.is_leaf:
            node = node.expand()[0]
            path.append(node)
        return path

    for root in roots:
        path = _first_path(root)
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


def test_score_called_once_per_param():
    """``score`` fires exactly once per param at builder entry (the
    ``leaf_score`` dict) — no rescoring during depth-recursion."""
    calls: list[P] = []

    def counting_score(p: P) -> float:
        calls.append(p)
        return _identity_score(p)

    params = [P(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]  # 8 unique params
    build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize, score=counting_score)
    assert len(calls) == len(params)
    assert set(calls) == set(params)


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
