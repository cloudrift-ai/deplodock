"""Unit tests for ``pipeline/fork.py``'s hierarchical Fork-tree builder.

Covers the generic builder contract using synthetic knob rows so the suite
stays decoupled from any specific Tile-IR pass (`partition_loops` is the
canonical caller; its integration tests live in
``tests/compiler/passes/test_partition_planner_forks.py``). The flat
``Fork`` implementations (``OptionFork`` / ``ThunkFork``) are exercised
through the engine in ``tests/compiler/pipeline/search/test_thunk_forks.py``.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.pipeline.fork import Fork, Level, build_fork_tree


def _row(a: int, b: int, c: int) -> dict:
    """Synthetic variant knob row — two branch-level knobs (A, B) plus a
    knob no level covers (C), mirroring the planner's FK/OVERHANG."""
    return {"A": a, "B": b, "C": c}


def _stub_materialize(row: dict) -> str:
    """Stand-in for a real Op materializer; returns a string so leaves'
    ``expand()`` results are trivially comparable."""
    return f"op({row['A']},{row['B']},{row['C']})"


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


# Conventional levels reused across tests: `A` is the outer branch key,
# `B` the inner one. `C` is deliberately covered by NO level — leaves
# still carry it because a leaf's knobs are its complete row.
_LEVELS = [
    Level(("A",), lambda r: (r["A"],)),
    Level(("B",), lambda r: (r["B"],)),
]


def test_empty_params_raises():
    """No rows ⟹ no fork point — the caller should skip the rule, not
    build a tree. The non-empty invariant keeps the return type a bare
    ``Fork`` (the engine never sees an empty option list)."""
    with pytest.raises(ValueError, match="params must be non-empty"):
        build_fork_tree(params=[], levels=_LEVELS, materialize=_stub_materialize)


def test_empty_levels_raises():
    with pytest.raises(ValueError, match="at least one Level"):
        build_fork_tree(params=[_row(1, 2, 3)], levels=[], materialize=_stub_materialize)


def test_single_param_single_leaf():
    """One row → the root expands straight to one leaf Fork (every level
    collapses), the leaf carrying the COMPLETE row as knobs."""
    tree = build_fork_tree(params=[_row(1, 2, 3)], levels=_LEVELS, materialize=_stub_materialize)
    assert isinstance(tree, Fork)
    assert not tree.is_leaf and tree.knobs == {}  # lazy root: nothing pinned, nothing built
    (leaf,) = tree.expand()
    assert leaf.is_leaf
    assert leaf.knobs == {"A": 1, "B": 2, "C": 3}
    assert leaf.expand() == ["op(1,2,3)"]


def test_two_params_identical_branch_key_collapses_branch():
    """Two rows sharing every grouping key → both branch levels collapse;
    result is a flat list of leaf Forks at the top level."""
    params = [_row(1, 2, 3), _row(1, 2, 5)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)
    # A and B both have one distinct key → collapse. The rows still
    # differ (in C), so the root expands to two sibling leaves.
    top = tree.expand()
    assert all(f.is_leaf for f in top)
    assert [f.knobs["C"] for f in top] == [3, 5]


def test_two_params_distinct_branch_key_emits_branches():
    """Two rows with distinct outer keys → two branch Forks, each with
    one leaf child."""
    params = [_row(1, 2, 3), _row(2, 2, 3)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)
    top = tree.expand()
    assert len(top) == 2
    for branch in top:
        assert not branch.is_leaf
        assert set(branch.knobs.keys()) == {"A"}
        children = branch.expand()
        assert len(children) == 1
        assert children[0].is_leaf


def test_siblings_unranked_in_grouping_order():
    """The builder does NOT sort siblings — ranking is search policy (the
    learned prior). Siblings come out in grouping (= first-occurrence) order
    regardless of any heuristic."""
    params = [_row(1, 0, 0), _row(3, 0, 0), _row(2, 0, 0)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)
    assert [f.knobs["A"] for f in tree.expand()] == [1, 3, 2]


def test_leaf_knobs_are_the_complete_row():
    """Every leaf carries its FULL knob row — including knobs no level
    covers (C) — so the engine's DB replay (``_best_fork``) can match
    leaves by knobs alone, and distinct siblings always differ in a
    recorded knob."""
    params = [_row(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]
    leaves = _walk_leaves(build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize))
    assert sorted(tuple(sorted(leaf.knobs.items())) for leaf in leaves) == sorted(tuple(sorted(p.items())) for p in params)


def test_branch_knobs_partition_and_leaf_row_agrees():
    """BRANCH knobs along any root→leaf path have no duplicates (each
    level pins its slice exactly once); the leaf's complete row agrees
    with every value pinned on the way down."""
    params = [_row(a, b, c) for a in (1, 2) for b in (1, 2) for c in (1, 2)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)

    path = [tree]
    node = tree
    while not node.is_leaf:
        node = node.expand()[0]
        path.append(node)
    leaf, branches = path[-1], path[:-1]
    seen: dict[str, int] = {}
    for f in branches:
        for k, v in f.knobs.items():
            seen[k] = seen.get(k, 0) + 1
            assert leaf.knobs[k] == v, f"leaf row disagrees with pinned branch knob {k}"
    duplicates = {k: n for k, n in seen.items() if n > 1}
    assert not duplicates, f"knob duplicated along branch path: {duplicates}"
    assert set(seen) <= {"A", "B"}


def test_collapsed_constant_level_omits_branch():
    """A level whose key is constant across all rows produces no Fork
    wrapper for that level."""
    # All rows share A=1, vary B and C → A level collapses, B emits
    # branches, leaves carry the full rows.
    params = [_row(1, 1, 1), _row(1, 2, 1)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)
    for branch in _walk_branches(tree):
        assert "A" not in branch.knobs


def test_materialize_is_lazy():
    """``materialize`` fires 0 times at build, exactly once per leaf
    ``expand()`` resolution."""
    calls: list[dict] = []

    def counting_materialize(row: dict) -> str:
        calls.append(row)
        return _stub_materialize(row)

    params = [_row(1, 2, 3), _row(1, 2, 5)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=counting_materialize)
    assert calls == [], "materialize fired during build"

    leaves = _walk_leaves(tree)
    leaves[0].expand()
    assert len(calls) == 1
    leaves[1].expand()
    assert len(calls) == 2


def test_leaf_expand_returns_own_param_materialization():
    """Each leaf's ``expand()`` materializes that leaf's OWN row — the
    leaf holds its row as data (``knobs``), no shared-closure traps."""
    params = [_row(1, 1, 1), _row(1, 1, 2), _row(1, 1, 3)]
    tree = build_fork_tree(params=params, levels=_LEVELS, materialize=_stub_materialize)
    leaves = _walk_leaves(tree)
    results = sorted(leaf.expand()[0] for leaf in leaves)
    assert results == ["op(1,1,1)", "op(1,1,2)", "op(1,1,3)"]
