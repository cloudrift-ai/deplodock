"""Single-player MCTS for ``deplodock tune`` with max-reward propagation
and normalized UCB1 selection (Schadd et al. 2008, SP-MCTS).

    select   — descend from root, picking at each level
               ``argmax_c [ Q_norm(c) + c · √(ln N_parent / N_c) ]``
               where ``Q_norm = best_reward / global_best``. Unvisited
               children get ``+∞``, breaking ties on the fork prior
               (``Search.score_of`` — value-keyed, cached on the
               search). Live-count filtering skips subtrees whose
               frontier has been fully drained.
    expand   — :meth:`TuningSearch.push` adds the engine's spawned
               candidates as children of the ``parent`` token (the
               ``SearchNode`` their spawning candidate was popped with);
    simulate — the engine runs the popped candidate and benches it;
    backprop — :meth:`SearchTree.record_terminal` walks ``parent``
               links from the observed terminal's token, bumping
               ``visits`` and updating
               ``best_reward = max(best_reward, leaf_reward)``.

Reward is normalized to [0,1] against the global best so the UCB
exploration constant is unit-free. The prior never enters arithmetic
with the reward — it only breaks ties among unvisited siblings.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field

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
    # Memoized ``Search.score_of`` read (filled by ``_ucb_key``) — UCB
    # descent reads the prior on every unvisited sibling at every level
    # per pop, and a branch fork's prior is a max over its param
    # subgroup, so the per-node memo keeps repeated descents O(1). Sound
    # because the fork's score is frozen once the SearchNode is attached.
    prior: float | None = field(default=None, repr=False)


class SearchTree:
    def __init__(self) -> None:
        self.root = SearchNode(candidate=None)

    @property
    def best_reward(self) -> float:
        return self.root.best_reward

    def attach(self, candidates: list[LazyCandidate], parent: SearchNode) -> None:
        nodes = [SearchNode(candidate=c, parent=parent, live=1) for c in candidates]
        parent.children.extend(nodes)
        # Each new child is a fresh frontier — bump live count on every ancestor.
        cur: SearchNode | None = parent
        while cur is not None:
            cur.live += len(nodes)
            cur = cur.parent

    def record_terminal(self, node: SearchNode, reward: float) -> None:
        """Max-propagate ``reward`` from ``node`` (the terminal's own
        SearchNode — the token it was popped with) up to the root,
        bumping ``visits`` along the way."""
        cur: SearchNode | None = node
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
        max_visits: int | None = None,
        score_cache: dict[Hashable, float] | None = None,
    ) -> None:
        super().__init__(score_cache=score_cache)
        self.tree = tree if tree is not None else SearchTree()
        self._ucb_c = ucb_c
        self._patience = patience
        self._max_visits = max_visits
        self._best_reward = 0.0
        self._visits_at_best = 0
        # Why the search stopped (set by ``_should_stop``): a patience /
        # max_visits message, or ``None`` when the queue drained — the
        # exhaustion signal ``two_level.inner_reward`` records as ``inf``
        # effort.
        self.stop_reason: str | None = None
        # Last benched variant's measurement — read by the tune progress bar
        # after each yielded terminal (the engine calls ``observe`` right before
        # yielding). Carries no role in the search itself.
        self.last_stats: PerfStats | None = None
        self.last_status: str | None = None

    def observe(self, token: object | None, stats: PerfStats, status: str) -> None:
        self.last_stats = stats
        self.last_status = status
        assert isinstance(token, SearchNode), f"TuningSearch.observe needs the terminal's pop token, got {type(token).__name__}"
        reward = (1.0 / stats.median) if status == "ok" and stats.median > 0 else 0.0
        self.tree.record_terminal(token, reward)

    def push(self, *cands: LazyCandidate, parent: object | None = None, best: LazyCandidate | None = None) -> None:
        del best  # tuning explores every fork — DB hint is for greedy
        # ``parent`` is the token the spawning candidate was popped with;
        # ``None`` seeds the run under the root sentinel.
        assert parent is None or isinstance(parent, SearchNode), f"TuningSearch.push needs a SearchNode token, got {type(parent).__name__}"
        self.tree.attach(list(cands), parent=parent if parent is not None else self.tree.root)

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
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
        return node, node.candidate

    def _ucb_key(self, child: SearchNode, parent: SearchNode) -> tuple[float, float]:
        """Selection score = (UCB1 value, score-as-tiebreak). Returned as
        a tuple so unvisited siblings (UCB1 = +∞) are ranked by their
        prior (``Search.score_of``, memoized per node — see
        ``SearchNode.prior``). Reward is normalized against the global
        best so the ``c`` constant is unit-free."""
        prior = child.prior
        if prior is None:
            prior = self.score_of(child.candidate.fork) if child.candidate is not None else float("-inf")
            child.prior = prior
        if child.visits == 0:
            return float("inf"), prior
        global_best = self.tree.best_reward
        q_norm = (child.best_reward / global_best) if global_best > 0 else 0.0
        bonus = self._ucb_c * math.sqrt(math.log(max(parent.visits, 1)) / child.visits)
        return q_norm + bonus, prior

    def _should_stop(self) -> bool:
        if self.stop_reason is not None:
            return True
        visits = self.tree.root.visits
        if visits == 0:
            return False
        if self._max_visits is not None and visits >= self._max_visits:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"max_visits ({visits} reached, best {best_us:.2f} us)"
            return True
        if self.tree.best_reward > self._best_reward:
            self._best_reward = self.tree.best_reward
            self._visits_at_best = visits
        stagnant = visits - self._visits_at_best
        if stagnant >= self._patience:
            best_us = 1.0 / self._best_reward if self._best_reward > 0 else float("inf")
            self.stop_reason = f"patience ({stagnant} stagnant, best {best_us:.2f} us)"
            return True
        return False
