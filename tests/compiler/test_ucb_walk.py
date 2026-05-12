"""Tests for ``TuningSearch._ucb_walk`` — verify exploration order over a
hand-built cache tree.

The walk is the MCTS selection step: starting at the cache root, descend
via UCB-best child at each level, stopping at the first node with an
unvisited child (returning that child) or a node with no children (return
itself). This file constructs deterministic trees with synthetic
measurements and asserts the walk picks the expected frontier.
"""

from __future__ import annotations

from deplodock.compiler.pipeline.search import TuningSearch
from deplodock.compiler.pipeline.search.cache import TuningCache


def _make_search(cache: TuningCache, context_key: str = "ctx") -> TuningSearch:
    s = TuningSearch(cache=cache, context_key=context_key, budget_s=float("inf"), patience=10**9, min_coverage=1.1)
    return s


def test_walk_returns_unvisited_child_at_root() -> None:
    """Bootstrap phase: root expanded into 3 children, none measured.
    The walk should return one of them (any — they're all priority ∞)."""
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    cache.expand("ctx", "root", ["A", "B", "C"])
    s = _make_search(cache)
    target = s._ucb_walk("ctx", "root")
    assert target in {"A", "B", "C"}


def test_walk_prefers_unvisited_over_visited_with_low_mean() -> None:
    """After A is measured (one rollout, mean ≈ 0.013), the walk should
    still pick an unvisited sibling because unvisited get priority ∞."""
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    cache.expand("ctx", "root", ["A", "B", "C"])
    cache.record_cuda_perf("ctx", "A", latency_us=76.0)  # reward = 0.0132
    s = _make_search(cache)
    target = s._ucb_walk("ctx", "root")
    assert target in {"B", "C"}, f"expected unvisited child, got {target}"


def test_walk_descends_into_only_visited_child_when_others_drained() -> None:
    """All three root-children measured. The walk picks the UCB-best at
    root and descends into it. A is way better (1/50 = 0.020) vs B and
    C at 1/200 = 0.005, so the walk should choose A."""
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    cache.expand("ctx", "root", ["A", "B", "C"])
    cache.record_cuda_perf("ctx", "A", latency_us=50.0)
    cache.record_cuda_perf("ctx", "B", latency_us=200.0)
    cache.record_cuda_perf("ctx", "C", latency_us=200.0)
    s = _make_search(cache)
    target = s._ucb_walk("ctx", "root")
    # A has no children expanded, so the walk stops AT A (frontier).
    assert target == "A", f"expected best leaf 'A', got {target}"


def test_walk_descends_then_returns_unvisited_grandchild() -> None:
    """Two-level tree where A is clearly the winner at root (both A and
    B well-sampled, A has high mean) and A has one expanded grandchild
    plus an un-touched one. The walk should descend A → A2 (unvisited).

    We pre-populate counters via SQL so A and B both have many visits,
    which shrinks their exploration terms and lets the mean dominate."""
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    cache.expand("ctx", "root", ["A", "B"])
    # Synthetic: A has 50 visits with high mean (~0.02), B has 50 with low mean (~0.005).
    cache._conn.execute(
        "UPDATE nodes SET visits = 50, seen_terminals = 50, total_reward = 1.0 WHERE context_key='ctx' AND node_key='A'"
    )
    cache._conn.execute(
        "UPDATE nodes SET visits = 50, seen_terminals = 50, total_reward = 0.25 WHERE context_key='ctx' AND node_key='B'"
    )
    # Root accumulates: visits=100, total_reward=1.25 (just stats, mean unused at root).
    cache._conn.execute(
        "UPDATE nodes SET visits = 100, seen_terminals = 100, total_reward = 1.25 WHERE context_key='ctx' AND node_key='root'"
    )
    # Expand A → A1 (measured), A2 (untouched).
    cache.expand("ctx", "A", ["A1", "A2"])
    cache.record_cuda_perf("ctx", "A1", latency_us=60.0)
    s = _make_search(cache)
    target = s._ucb_walk("ctx", "root")
    assert target == "A2", f"expected to descend A then pick unvisited A2, got {target}"


def test_walk_revisits_underexplored_sibling_when_exploration_term_dominates() -> None:
    """A measured many times with mean=0.013; B measured once with mean=0.012.
    Exploration term for B is √(log(parent.visits)) which dominates when
    A has 25 visits and B has 1. The walk should re-sample B at root."""
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    cache.expand("ctx", "root", ["A", "B"])
    # Push A's stats by recording many "synthetic" measurements through
    # propagation: simulate by directly bumping the node's counters.
    # 25 ok measurements of A at ~76us each.
    for _ in range(25):
        cache._conn.execute("UPDATE nodes SET visits = visits + 1, total_reward = total_reward + 0.0132 WHERE context_key='ctx' AND node_key='A'")
    cache._conn.execute("UPDATE nodes SET seen_terminals = 25 WHERE context_key='ctx' AND node_key='A'")
    # Propagate to root too.
    cache._conn.execute("UPDATE nodes SET visits = visits + 25, total_reward = total_reward + 0.33 WHERE context_key='ctx' AND node_key='root'")
    # B measured once at 83us → mean ~0.012.
    cache.record_cuda_perf("ctx", "B", latency_us=83.0)
    s = _make_search(cache)
    target = s._ucb_walk("ctx", "root")
    # B's exploration term: c * sqrt(log(root.visits=26)/1) ≈ 1.81
    # B's UCB ≈ 0.012 + 1.81 = 1.82
    # A's UCB ≈ 0.0132/25 + c*sqrt(log(26)/25) ≈ 0.0005 + 0.36 = 0.36
    # B wins.
    # But A has no children (frontier) — the walk would still return A
    # if it picked A. Since UCB picks B, walk returns B (B has no children
    # in cache, so frontier).
    assert target == "B", f"under-explored arm should win on exploration bonus, got {target}"
