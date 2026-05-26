"""Single-player MCTS for ``deplodock tune`` with max-reward propagation
and normalized UCB1 selection (Schadd et al. 2008, SP-MCTS).

    select   — descend from root, picking at each level
               ``argmax_c [ Q_norm(c) + c · √(ln N_parent / N_c) ]``
               where ``Q_norm = best_reward / global_best``. Unvisited
               children get ``+∞``, breaking ties on the candidate's
               ``score()`` prior. Live-count filtering skips subtrees
               whose frontier has been fully drained.
    expand   — :meth:`TuningSearch.push` adds the engine's spawned
               candidates as children of the last popped node;
    simulate — the engine runs the popped candidate and benches it;
    backprop — :meth:`SearchTree.record_terminal` walks ``parent``
               links bumping ``visits`` and updating
               ``best_reward = max(best_reward, leaf_reward)``.

Reward is normalized to [0,1] against the global best so the UCB
exploration constant is unit-free. The prior never enters arithmetic
with the reward — it only breaks ties among unvisited siblings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import cached_property

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats
from deplodock.compiler.pipeline.search.policy.base import Search


@dataclass
class SearchNode:
    candidate: LazyCandidate | None  # None for the root sentinel
    parent: SearchNode | None = field(default=None, repr=False)
    children: list[SearchNode] = field(default_factory=list, repr=False)
    visits: int = 0
    best_reward: float = 0.0  # max reward over this subtree's measured leaves
    live: int = 0  # count of un-popped frontier leaves in this subtree

    @cached_property
    def score(self) -> float:
        """Cached :meth:`LazyCandidate.score` — UCB descent reads this on
        every unvisited sibling at every level per pop. Without the
        cache, ``Candidate.score`` (which iterates every graph node and
        calls ``op.score(ctx)``) re-runs ~O(siblings) per pop, with
        leaf-Fork siblings additionally materializing their TileOps via
        ``fork.expand()`` (full ``_build_split_body`` + body
        normalization). The cache is sound because ``LazyCandidate.score``
        is a pure function of ``inner.graph`` + ``pending``, both
        frozen once the SearchNode is attached.
        """
        return self.candidate.score() if self.candidate is not None else float("-inf")


class SearchTree:
    def __init__(self) -> None:
        self.root = SearchNode(candidate=None)
        self.last_popped = self.root

    @property
    def best_reward(self) -> float:
        return self.root.best_reward

    def attach(self, candidates: list[LazyCandidate], parent: SearchNode) -> list[SearchNode]:
        nodes = [SearchNode(candidate=c, parent=parent, live=1) for c in candidates]
        parent.children.extend(nodes)
        # Each new child is a fresh frontier — bump live count on every ancestor.
        cur: SearchNode | None = parent
        while cur is not None:
            cur.live += len(nodes)
            cur = cur.parent
        return nodes

    def record_terminal(self, reward: float) -> None:
        """Max-propagate ``reward`` from the most recently popped node
        up to the root, bumping ``visits`` along the way."""
        cur: SearchNode | None = self.last_popped
        while cur is not None:
            cur.visits += 1
            if reward > cur.best_reward:
                cur.best_reward = reward
            cur = cur.parent


class TuningSearch(Search):
    """SP-MCTS: max-Q normalized UCB1 with a rank-only prior."""

    DEFAULT_UCB_C = math.sqrt(2)

    def __init__(
        self,
        tree: SearchTree | None = None,
        *,
        patience: int = 20,
        ucb_c: float = DEFAULT_UCB_C,
    ) -> None:
        self.tree = tree if tree is not None else SearchTree()
        self._ucb_c = ucb_c
        self._patience = patience
        self._best_reward = 0.0
        self._visits_at_best = 0
        self.stop_reason: str | None = None

    def stop(self, reason: str) -> None:
        self.stop_reason = reason

    def observe(self, stats: PerfStats, status: str) -> None:
        reward = (1.0 / stats.median) if status == "ok" and stats.median > 0 else 0.0
        self.tree.record_terminal(reward)

    def push(self, primary: LazyCandidate, *forks: LazyCandidate, best: LazyCandidate | None = None) -> None:
        del best  # tuning explores every fork — DB hint is for greedy
        self.tree.attach([primary, *forks], parent=self.tree.last_popped)

    def pop(self) -> LazyCandidate | None:
        if self._should_stop():
            return None
        node = self.tree.root
        if node.live == 0:
            return None
        while node.children:
            descendable = [c for c in node.children if c.live > 0]
            if not descendable:
                return None
            node = max(descendable, key=lambda c: self._ucb_key(c, node))
        # Frontier just got handed off — drop it from the live count
        # on every ancestor. The engine may push children of this node
        # before the next pop; those pushes re-grow the count.
        cur: SearchNode | None = node
        while cur is not None:
            cur.live -= 1
            cur = cur.parent
        self.tree.last_popped = node
        return node.candidate

    def _ucb_key(self, child: SearchNode, parent: SearchNode) -> tuple[float, float]:
        """Selection score = (UCB1 value, score-as-tiebreak). Returned as
        a tuple so unvisited siblings (UCB1 = +∞) are ranked by their
        prior. Reward is normalized against the global best so the
        ``c`` constant is unit-free."""
        if child.visits == 0:
            return float("inf"), child.score
        global_best = self.tree.best_reward
        q_norm = (child.best_reward / global_best) if global_best > 0 else 0.0
        bonus = self._ucb_c * math.sqrt(math.log(max(parent.visits, 1)) / child.visits)
        return q_norm + bonus, child.score

    def _should_stop(self) -> bool:
        if self.stop_reason is not None:
            return True
        visits = self.tree.root.visits
        if visits == 0:
            return False
        if self.tree.best_reward > self._best_reward:
            self._best_reward = self.tree.best_reward
            self._visits_at_best = visits
        stagnant = visits - self._visits_at_best
        if stagnant >= self._patience:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"patience ({stagnant} stagnant, best {best_us:.2f} us)"
            return True
        return False
