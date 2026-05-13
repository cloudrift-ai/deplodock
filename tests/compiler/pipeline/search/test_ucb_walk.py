"""Tests for ``TuningSearch._ucb_walk`` — verify exploration order over a
hand-built :class:`SearchTree`.

The walk is the MCTS selection step: starting at the tree root, descend
via UCB-best child at each level, stopping at the first node with an
unvisited child (returning that child) or a node with no children
(return itself). This file constructs deterministic trees with synthetic
measurements and asserts the walk picks the expected frontier.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search import TuningSearch
from deplodock.compiler.pipeline.search.policy.mcts import SearchTree


def _make_search(tree: SearchTree, context_key: str = "ctx") -> TuningSearch:
    return TuningSearch(
        tree=tree,
        context_key=context_key,
        budget_s=float("inf"),
        patience=10**9,
        min_coverage=1.1,
    )


def _set_node(tree: SearchTree, ctx: str, key: str, *, visits: int, seen: int, reward: float) -> None:
    """Test-only direct mutation of a tree node's MCTS counters."""
    node = tree.node(ctx, key)
    assert node is not None
    node.visits = visits
    node.seen_terminals = seen
    node.total_reward = reward


def test_walk_returns_unvisited_child_at_root() -> None:
    tree = SearchTree()
    tree.ensure_root("ctx", "root")
    tree.expand("ctx", "root", ["A", "B", "C"])
    target = _make_search(tree)._ucb_walk("ctx", "root")
    assert target in {"A", "B", "C"}


def test_walk_prefers_unvisited_over_visited_with_low_mean() -> None:
    tree = SearchTree()
    tree.ensure_root("ctx", "root")
    tree.expand("ctx", "root", ["A", "B", "C"])
    tree.record_terminal("ctx", "A", reward=1.0 / 76.0, status="ok")
    target = _make_search(tree)._ucb_walk("ctx", "root")
    assert target in {"B", "C"}, f"expected unvisited child, got {target}"


def test_walk_descends_into_only_visited_child_when_others_drained() -> None:
    tree = SearchTree()
    tree.ensure_root("ctx", "root")
    tree.expand("ctx", "root", ["A", "B", "C"])
    tree.record_terminal("ctx", "A", reward=1.0 / 50.0, status="ok")
    tree.record_terminal("ctx", "B", reward=1.0 / 200.0, status="ok")
    tree.record_terminal("ctx", "C", reward=1.0 / 200.0, status="ok")
    target = _make_search(tree)._ucb_walk("ctx", "root")
    # A has no expanded children, so the walk stops at A (frontier).
    assert target == "A", f"expected best leaf 'A', got {target}"


def test_walk_descends_then_returns_unvisited_grandchild() -> None:
    """Two-level tree where A is clearly the winner at root."""
    tree = SearchTree()
    tree.ensure_root("ctx", "root")
    tree.expand("ctx", "root", ["A", "B"])
    _set_node(tree, "ctx", "A", visits=50, seen=50, reward=1.0)
    _set_node(tree, "ctx", "B", visits=50, seen=50, reward=0.25)
    _set_node(tree, "ctx", "root", visits=100, seen=100, reward=1.25)
    tree.expand("ctx", "A", ["A1", "A2"])
    tree.record_terminal("ctx", "A1", reward=1.0 / 60.0, status="ok")
    target = _make_search(tree)._ucb_walk("ctx", "root")
    assert target == "A2"


def test_walk_revisits_underexplored_sibling_when_exploration_dominates() -> None:
    """A measured many times with low mean; B measured once. The
    exploration term should let B win on the next walk."""
    tree = SearchTree()
    tree.ensure_root("ctx", "root")
    tree.expand("ctx", "root", ["A", "B"])
    # 25 synthetic ok measurements of A at ~76us each.
    _set_node(tree, "ctx", "A", visits=25, seen=25, reward=25 * (1.0 / 76.0))
    _set_node(tree, "ctx", "root", visits=26, seen=26, reward=25 * (1.0 / 76.0) + 1.0 / 83.0)
    # B measured once at 83us → high exploration bonus.
    tree.record_terminal("ctx", "B", reward=1.0 / 83.0, status="ok")
    target = _make_search(tree)._ucb_walk("ctx", "root")
    # B's exploration term ≈ √(log(26)/1) ≈ 1.8 dominates A's 0.0005 + 0.36.
    assert target == "B", f"under-explored arm should win, got {target}"
