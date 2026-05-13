"""Tests for :class:`SearchTree` — the in-memory MCTS state.

No DB, no I/O — just pure-Python expand / propagation invariants.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search.policy.mcts import SearchTree


def test_root_placeholder_has_expected_one() -> None:
    tree = SearchTree()
    tree.ensure_root("root")
    node = tree.node("root")
    assert node.expected_terminals == 1 and node.seen_terminals == 0


def test_first_expansion_propagates_delta() -> None:
    """First expansion of a parent consumes its placeholder ``1``, so
    the delta to ancestors is ``n_new - 1``."""
    tree = SearchTree()
    tree.expand("root", ["a", "b", "c"])
    root = tree.node("root")
    assert root.expected_terminals == 3
    for ck in ("a", "b", "c"):
        child = tree.node(ck)
        assert child.expected_terminals == 1
        assert child.parent_key == "root"


def test_second_expansion_is_pure_addition() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    assert tree.node("root").expected_terminals == 1
    tree.expand("root", ["b", "c"])
    assert tree.node("root").expected_terminals == 3


def test_expansion_is_idempotent() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b"])
    tree.expand("root", ["a", "b"])
    assert tree.node("root").expected_terminals == 2


def test_deep_expansion_propagates_to_root() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    tree.expand("a", ["a1", "a2", "a3"])
    assert tree.node("root").expected_terminals == 3
    assert tree.node("a").expected_terminals == 3


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


def test_is_fully_explored_tracks_seen_vs_expected() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b"])
    assert not tree.is_fully_explored("root")
    tree.record_terminal("a", reward=0.1, status="ok")
    assert not tree.is_fully_explored("root")
    tree.record_terminal("b", reward=0.1, status="ok")
    assert tree.is_fully_explored("root")


def test_expansion_grows_denominator_mid_run() -> None:
    tree = SearchTree()
    tree.expand("root", ["a"])
    tree.record_terminal("a", reward=0.1, status="ok")
    assert tree.is_fully_explored("root")
    tree.expand("a", ["a1", "a2"])
    assert not tree.is_fully_explored("root")


def test_find_root_returns_first_inserted() -> None:
    tree = SearchTree()
    tree.ensure_root("first")
    tree.expand("first", ["c"])
    assert tree.root == "first"


def test_root_coverage_sums_over_roots() -> None:
    tree = SearchTree()
    tree.expand("root", ["a", "b"])
    tree.record_terminal("a", reward=0.1, status="ok")
    seen, expected = tree.root_coverage()
    assert seen == 1 and expected == 2


def test_children_preserves_insertion_order() -> None:
    tree = SearchTree()
    tree.expand("root", ["c", "a", "b"])
    assert tree.children("root") == ["c", "a", "b"]
