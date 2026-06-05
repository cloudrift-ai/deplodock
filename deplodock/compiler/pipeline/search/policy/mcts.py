"""Single-player MCTS for ``deplodock tune`` with max-reward propagation and
**PUCT** selection — the learned
:class:`~deplodock.compiler.pipeline.search.prior.Prior` is the *only* selection
signal (greedy and the ``+∞``-unvisited UCB rule are gone).

    select   — descend from root, picking at each level
               ``argmax_c [ Q_norm(c) + c · P(c) · √(N_parent+1) / (1+N_c) ]``
               where ``Q_norm = best_reward / global_best`` and ``P`` is the
               softmax over the prior's Thompson-sampled scores of the sibling
               set. No ``+∞``-unvisited rule → no forced breadth: a
               confidently-bad sibling gets a small ``P`` and is skipped. A cold
               or absent prior gives a uniform ``P`` (PUCT still explores via the
               exploration term; a single-shot compile with no prior descends
               emission-order). Live-count filtering skips drained subtrees.
    expand   — :meth:`TuningSearch.push` adds the engine's spawned
               candidates as children of the ``parent`` token (the
               ``SearchNode`` their spawning candidate was popped with);
    simulate — the engine runs the popped candidate and benches it;
    backprop — :meth:`SearchTree.record_terminal` walks ``parent``
               links from the observed terminal's token, bumping
               ``visits`` and updating
               ``best_reward = max(best_reward, leaf_reward)``.

Reward is normalized to [0,1] against the global best so the exploration
constant ``c`` is unit-free.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats
from deplodock.compiler.pipeline.search.policy.base import Search

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.search.prior import Prior


def _softmax(xs: list[float]) -> list[float]:
    """Numerically-stable softmax (shift-invariant, so any constant offset in
    the prior scores — e.g. its ``y_mean`` — cancels)."""
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


@dataclass
class SearchNode:
    candidate: LazyCandidate | None  # None for the root sentinel
    parent: SearchNode | None = field(default=None, repr=False)
    children: list[SearchNode] = field(default_factory=list, repr=False)
    visits: int = 0
    best_reward: float = 0.0  # max reward over this subtree's measured leaves
    live: int = 0  # count of un-popped frontier leaves in this subtree


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
    """SP-MCTS with PUCT selection — the learned prior is the sole signal."""

    DEFAULT_UCB_C = math.sqrt(2)

    def __init__(
        self,
        tree: SearchTree | None = None,
        *,
        patience: int = 20,
        ucb_c: float = DEFAULT_UCB_C,
        max_visits: int | None = None,
        score_cache: dict[Hashable, float] | None = None,
        prior_model: Prior | None = None,
        base_knobs: dict | None = None,
    ) -> None:
        super().__init__(score_cache=score_cache)
        self.tree = tree if tree is not None else SearchTree()
        self._ucb_c = ucb_c
        self._patience = patience
        self._max_visits = max_visits
        # Learned prior driving PUCT selection. Refit from the live tree every
        # ``refit_every`` benches; one Thompson draw per descent. ``None`` for a
        # single-shot compile (no benching) → uniform PUCT → emission-order pick.
        self.prior_model = prior_model
        # The kernel's identity knobs (the ``S_*`` structural features stamped on
        # the LoopOp) — merged under every node's accumulated fork deltas so the
        # GLOBAL prior sees op-structure and can tell kernels apart.
        self._base_knobs = dict(base_knobs) if base_knobs else {}
        self._benches_since_fit = 0
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
        if self.prior_model is not None:
            # Record the leaf for the end-of-run stats; flag a refit due.
            self.prior_model.record_bench(self._node_knobs(token), stats.median, status)
            self._benches_since_fit += 1

    def push(self, *cands: LazyCandidate, parent: object | None = None) -> None:
        # ``parent`` is the token the spawning candidate was popped with;
        # ``None`` seeds the run under the root sentinel.
        assert parent is None or isinstance(parent, SearchNode), f"TuningSearch.push needs a SearchNode token, got {type(parent).__name__}"
        self.tree.attach(list(cands), parent=parent if parent is not None else self.tree.root)

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        if self._should_stop():
            return None
        if self.prior_model is not None:
            # Refit from a tree snapshot when enough new benches have landed
            # (labels are non-stationary — re-read ``best_reward`` each time),
            # then draw one Thompson sample for this whole descent.
            if self._benches_since_fit >= self.prior_model.refit_every:
                rows = self._collect_rows()
                if len(rows) >= self.prior_model.min_rows:
                    self.prior_model.fit(rows)
                    self._benches_since_fit = 0
            self.prior_model.resample()
        node = self.tree.root
        if node.live == 0:
            return None
        while node.children:
            descendable = [c for c in node.children if c.live > 0]
            if not descendable:
                return None
            node = self._select(descendable, node)
        # Frontier just got handed off — drop it from the live count
        # on every ancestor. The engine may push children of this node
        # before the next pop; those pushes re-grow the count.
        cur: SearchNode | None = node
        while cur is not None:
            cur.live -= 1
            cur = cur.parent
        return node, node.candidate

    def _prior_score(self, child: SearchNode) -> float:
        """The prior's Thompson-sampled score for a child (``0`` when no model
        is attached, the model is unfit, or the child is the root sentinel — in
        which case the softmax over siblings is uniform)."""
        if self.prior_model is None or child.candidate is None:
            return 0.0
        return self.prior_model.score(self._node_knobs(child))

    def _select(self, children: list[SearchNode], parent: SearchNode) -> SearchNode:
        """PUCT is the *only* selection rule — the prior is the sole signal.

            score(c) = Q(c) + c_ucb · P(c) · √(N_parent + 1) / (1 + N_c)

        where ``Q = best_reward / global_best`` (``0`` for an unvisited child)
        and ``P`` is the softmax over the prior's Thompson-sampled scores of the
        sibling set. A confidently-bad sibling gets a small ``P`` → tiny
        exploration term → it is deprioritized rather than force-visited. There
        is no ``+∞``-unvisited rule (no forced breadth) and no greedy/UCB
        fallback: a cold or absent prior just yields a uniform ``P``, so PUCT
        still explores via the exploration term (and a single-shot compile with
        no prior descends emission-order). ``c_ucb`` is ``--ucb-c``."""
        global_best = self.tree.best_reward or 1.0
        probs = _softmax([self._prior_score(c) for c in children])
        sqrt_parent = math.sqrt(parent.visits + 1)
        best, best_v = children[0], float("-inf")
        for c, p in zip(children, probs, strict=True):
            q = (c.best_reward / global_best) if c.visits > 0 else 0.0
            v = q + self._ucb_c * p * sqrt_parent / (1 + c.visits)
            if v > best_v:
                best_v, best = v, c
        return best

    def _node_knobs(self, node: SearchNode) -> dict:
        """Accumulated knob dict for a node — the kernel's ``base_knobs`` (its
        ``S_*`` structural identity) merged with every ``fork.knobs`` delta from
        the root down to ``node``. A branch pins its level slice; a leaf carries
        the complete row, so deeper nodes hold a superset. This is the
        (partial-or-full) feature input the prior featurizes."""
        chain: list[dict] = []
        cur: SearchNode | None = node
        while cur is not None and cur.candidate is not None:
            fork = cur.candidate.fork
            if fork is not None:
                chain.append(fork.knobs)
            cur = cur.parent
        merged: dict = dict(self._base_knobs)
        for knobs in reversed(chain):
            merged.update(knobs)
        return merged

    def _collect_rows(self) -> list[tuple[dict, float]]:
        """Value-of-position training rows from the live tree: every node with
        a benched descendant (``visits > 0`` and ``best_reward > 0``) — leaves
        *and* branches — labeled ``log(best_reward)`` (= −log median latency,
        the max over its subtree). Re-read each refit since labels only rise."""
        rows: list[tuple[dict, float]] = []
        stack = list(self.tree.root.children)
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.candidate is None or node.visits == 0 or node.best_reward <= 0:
                continue
            rows.append((self._node_knobs(node), math.log(node.best_reward)))
        return rows

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
