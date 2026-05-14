"""Tests for :class:`SearchTree` — the in-memory MCTS state.

No DB, no I/O — just pure-Python expand / propagation invariants.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.policy.mcts import SearchTree


def test_root_starts_unseen() -> None:
    tree = SearchTree()
    tree.ensure_root("root")
    node = tree.node("root")
    assert node.seen_terminals == 0 and node.visits == 0


def test_first_expansion_links_children_to_parent() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b", "c"])
    root = tree.node("root")
    assert [c.key for c in root.children] == ["a", "b", "c"]
    for ck in ("a", "b", "c"):
        child = tree.node(ck)
        assert child.parent is root


def test_expansion_is_idempotent() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b"])
    tree.expand("root", ["a", "b"])
    assert [c.key for c in tree.node("root").children] == ["a", "b"]


def test_record_terminal_propagates_seen_upward() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b"])
    tree.record_terminal("a", reward=0.1, status="ok")
    assert tree.node("a").seen_terminals == 1
    assert tree.node("root").seen_terminals == 1
    tree.record_terminal("b", reward=0.05, status="ok")
    assert tree.node("root").seen_terminals == 2


def test_record_terminal_bumps_visits_and_reward() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    tree.record_terminal("a", reward=0.2, status="ok")
    a = tree.node("a")
    root = tree.node("root")
    assert a.visits == 1 and a.total_reward == 0.2
    # Canonical MCTS: only terminals count as visits — expansion alone
    # does not bump visits. Root has one terminal under it.
    assert root.visits == 1
    assert root.total_reward == 0.2


def test_record_terminal_idempotent_for_same_leaf() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    assert tree.record_terminal("a", reward=0.1, status="ok") is True
    assert tree.record_terminal("a", reward=0.2, status="ok") is False
    assert tree.node("a").seen_terminals == 1
    assert tree.node("root").seen_terminals == 1


def test_record_terminal_failed_counts_separately() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    tree.record_terminal("a", reward=0.0, status="bench_fail")
    assert tree.node("a").failed_terminals == 1
    assert tree.node("root").failed_terminals == 1
    assert tree.node("a").seen_terminals == 1


def test_find_root_returns_first_inserted() -> None:
    tree = SearchTree()
    tree.ensure_root("first")
    tree.expand("first", ["c"])
    # tree.root is the super-root; "first" is its only subtree root.
    assert tree.subtree_roots[0].key == "first"


def test_min_subtree_seen_balances_across_kernel_roots() -> None:
    tree = SearchTree()
    tree.expand("k1", ["a"])
    tree.expand("k2", ["b"])
    tree.record_terminal("a", reward=0.1, status="ok")
    # k1 has 1 seen terminal under it, k2 has 0 — min is 0.
    assert tree.min_subtree_seen() == 0
    tree.record_terminal("b", reward=0.1, status="ok")
    assert tree.min_subtree_seen() == 1


def test_children_preserves_insertion_order() -> None:
    tree = SearchTree()
    tree.expand("root", ["c", "a", "b"])
    assert tree.children("root") == ["c", "a", "b"]
