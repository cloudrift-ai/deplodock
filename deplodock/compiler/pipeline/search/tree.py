"""In-memory MCTS search tree used by the autotune driver.

Rebuilt fresh each process: the engine re-fires every rule on warm
starts, which re-creates the same edges via :meth:`SearchTree.expand`.
The cached ``perf`` rows on disk ensure no re-bench, but the UCB
counters (visits / reward / coverage) start from zero each run.

Three counters are maintained online via upward propagation along
``parent_key``:

- ``expected_terminals`` — each *new* expansion of a parent that had no
  children adds ``n_new - 1`` to every ancestor (the parent's
  placeholder "1" is consumed by the first child); subsequent
  expansions of the same parent add ``n_new``.
- ``seen_terminals`` / ``failed_terminals`` — each measured terminal
  bumps every ancestor by ``+1`` (and ``+1`` for ``failed`` on failure).
- ``visits`` / ``total_reward`` — UCB1 numerator + denominator. Bumped
  both on expansion (treats a rule firing as a soft visit so unvisited
  siblings stay distinguishable in UCB selection) and on terminal
  measurement (``reward = 1/median_us`` for ok, ``0`` for fail).

A node is fully explored when ``seen_terminals == expected_terminals``.
The value can move *down* mid-run when expansion grows the denominator
faster than the numerator — that's the correct semantics ("we just
discovered there's more to explore").
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeRow:
    """One tree node — autotune-state for an op.

    ``seen_terminals`` counts every measured terminal under this node
    (ok + fail; used by coverage queries). ``failed_terminals`` tracks
    fails only (diagnostic). ``visits`` is the MCTS denominator — it
    increments both on each :meth:`SearchTree.expand` (expansion =
    exploration) and on each terminal measurement (measurement =
    visit). ``total_reward`` accumulates the MCTS reward
    (``1/latency_us`` for ok, ``0`` for fail). UCB exploitation uses
    ``total_reward / visits``.
    """

    parent_key: str | None
    expected_terminals: int = 1
    seen_terminals: int = 0
    failed_terminals: int = 0
    visits: int = 0
    total_reward: float = 0.0


class SearchTree:
    """Pure-Python search tree. No I/O. Constructed fresh each process."""

    def __init__(self) -> None:
        # Keyed by (context_key, node_key) so multiple compile contexts
        # can share one tree without cross-pollution. Today the autotune
        # loop only ever uses one context per process, but the keying
        # mirrors the previous SQLite schema for minimal churn at call
        # sites.
        self._nodes: dict[tuple[str, str], NodeRow] = {}
        # Ordered list (dedup on insert) so iteration order is
        # deterministic for tests and tie-breaks in the policy.
        self._expansions: dict[tuple[str, str], list[str]] = {}

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def node(self, context_key: str, node_key: str) -> NodeRow | None:
        return self._nodes.get((context_key, node_key))

    def children(self, context_key: str, parent_key: str) -> list[str]:
        return list(self._expansions.get((context_key, parent_key), ()))

    def find_root(self, context_key: str) -> str | None:
        """Return the first ``parent_key is None`` node inserted for
        this context (insertion order is preserved by ``dict``)."""
        for (ctx, key), row in self._nodes.items():
            if ctx == context_key and row.parent_key is None:
                return key
        return None

    def root_coverage(self, context_key: str) -> tuple[int, int]:
        """``(sum_seen, sum_expected)`` over every root for this context."""
        seen = 0
        expected = 0
        for (ctx, _), row in self._nodes.items():
            if ctx != context_key or row.parent_key is not None:
                continue
            seen += row.seen_terminals
            expected += row.expected_terminals
        return seen, expected

    def is_fully_explored(self, context_key: str, node_key: str) -> bool:
        n = self.node(context_key, node_key)
        return n is not None and n.seen_terminals >= n.expected_terminals

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def ensure_root(self, context_key: str, node_key: str) -> None:
        """Insert a root node (``parent_key = None``) if not present."""
        key = (context_key, node_key)
        if key not in self._nodes:
            self._nodes[key] = NodeRow(parent_key=None)

    def expand(self, context_key: str, parent_key: str, child_keys: list[str]) -> None:
        """Record ``parent_key → child_keys`` edges and maintain
        ``expected_terminals`` on every ancestor.

        Idempotent — re-firing a rule with the same children adds no
        new edges. When the rule's option set has grown, only the
        genuinely-new edges propagate.
        """
        if not child_keys:
            return
        self.ensure_root(context_key, parent_key)

        existing = self._expansions.setdefault((context_key, parent_key), [])
        pre = len(existing)
        n_new = 0
        for ck in child_keys:
            if ck in existing:
                continue
            existing.append(ck)
            n_new += 1

        if n_new == 0:
            return

        # Insert child node rows (placeholders) for the new edges.
        for ck in child_keys:
            self._nodes.setdefault((context_key, ck), NodeRow(parent_key=parent_key))

        # First-ever expansion of this parent consumes its placeholder "1",
        # so the delta is one less than n_new. Later expansions are pure
        # additions (the parent was already accounting for its children).
        delta = n_new - 1 if pre == 0 else n_new
        if delta != 0:
            self._propagate_expected(context_key, parent_key, delta)

        # Treat the expansion event itself as one "visit" of the parent
        # for MCTS-style UCB selection — without this, every sibling at
        # a fork has ``visits = 0`` and indistinguishable ``-inf`` UCB
        # priorities, so the selector can't differentiate.
        self._propagate_mcts_visit(context_key, parent_key, visits_delta=1, reward_delta=0.0)

    def record_terminal(
        self,
        context_key: str,
        leaf_key: str,
        *,
        reward: float,
        status: str,
    ) -> bool:
        """Bump ``seen_terminals = 1`` on the leaf (if not already
        seen), propagate ``+1`` to every ancestor, and add the reward.

        Returns ``True`` iff this was a newly-measured terminal (so the
        caller knows propagation ran)."""
        # Make sure the node exists — rooted single-op graphs may have
        # skipped the expand chain.
        key = (context_key, leaf_key)
        node = self._nodes.setdefault(key, NodeRow(parent_key=None))
        if node.seen_terminals >= 1:
            return False
        failed_delta = 0 if status == "ok" else 1
        node.seen_terminals = 1
        node.failed_terminals = failed_delta
        node.visits += 1
        node.total_reward += reward
        if node.parent_key is not None:
            self._propagate_visit(
                context_key,
                node.parent_key,
                seen_delta=1,
                failed_delta=failed_delta,
                reward_delta=reward,
            )
        return True

    # ------------------------------------------------------------------
    # Internal propagation walks
    # ------------------------------------------------------------------

    def _propagate_expected(self, context_key: str, node_key: str, delta: int) -> None:
        cur: str | None = node_key
        while cur is not None:
            row = self._nodes.get((context_key, cur))
            if row is None:
                return
            row.expected_terminals += delta
            cur = row.parent_key

    def _propagate_visit(
        self,
        context_key: str,
        node_key: str,
        *,
        seen_delta: int,
        failed_delta: int,
        reward_delta: float,
    ) -> None:
        cur: str | None = node_key
        while cur is not None:
            row = self._nodes.get((context_key, cur))
            if row is None:
                return
            row.seen_terminals += seen_delta
            row.failed_terminals += failed_delta
            row.visits += seen_delta
            row.total_reward += reward_delta
            cur = row.parent_key

    def _propagate_mcts_visit(
        self,
        context_key: str,
        node_key: str,
        *,
        visits_delta: int,
        reward_delta: float,
    ) -> None:
        cur: str | None = node_key
        while cur is not None:
            row = self._nodes.get((context_key, cur))
            if row is None:
                return
            row.visits += visits_delta
            row.total_reward += reward_delta
            cur = row.parent_key
