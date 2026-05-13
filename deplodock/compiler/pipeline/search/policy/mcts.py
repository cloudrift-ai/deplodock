"""MCTS-style exhaustive priority search using UCB1 selection.

Used by ``deplodock compile --tune``. Each candidate sits at some
"tip" node in the search tree (the ``op_cache_key`` of the most
recently rewritten kernel-bearing op); pop ordering is driven by a
UCB1 walk over the in-memory :class:`SearchTree` rooted at the first
root inserted, with the score-based prior breaking ties among
unvisited siblings.

The :class:`SearchTree` and its :class:`NodeRow` data class live in
this module because MCTS is the only search policy that reads or
writes them. Engine + recorder feed the tree (via ``expand`` and
``record_terminal``) and the driver pulls it off the search via
``getattr(search, "tree", None)``; nobody outside this file consumes
the counters.

Stopping policy: wall-clock budget, or ``patience`` measured terminals
without a new best latency once coverage clears ``min_coverage``."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key

# ---------------------------------------------------------------------------
# In-memory MCTS tree
# ---------------------------------------------------------------------------


@dataclass(repr=False)
class SearchNode:
    """One tree node — autotune-state for an op.

    ``seen_terminals`` counts every measured terminal under this node
    (ok + fail; used by coverage queries). ``failed_terminals`` tracks
    fails only (diagnostic). ``visits`` is the canonical-MCTS
    denominator — it counts measured terminals under this node and
    nothing else (expansion alone never bumps it). ``total_reward``
    accumulates the MCTS reward (``1/latency_us`` for ok, ``0`` for
    fail). UCB1 exploitation uses ``total_reward / visits``.

    The node carries its own ``key`` plus direct ``parent`` / ``children``
    references, so tree walks are plain attribute access rather than
    ``dict[key]`` round-trips on the parent :class:`SearchTree`.
    """

    key: str
    parent: SearchNode | None = field(default=None, repr=False)
    children: list[SearchNode] = field(default_factory=list, repr=False)
    expected_terminals: int = 1
    seen_terminals: int = 0
    failed_terminals: int = 0
    visits: int = 0
    total_reward: float = 0.0
    # Prior heuristic ``op.score(ctx)`` of the op this node represents.
    # Stashed by :meth:`SearchTree.expand` so the search policy can read
    # it without scanning queued candidates' graphs after the candidate
    # has moved on. ``-inf`` for the root and for nodes inserted without
    # a score (e.g. test fixtures).
    score: float = float("-inf")

    def is_expanded(self) -> bool:
        return bool(self.children)

    def is_measured(self) -> bool:
        return self.visits > 0

    def is_frontier(self) -> bool:
        """Unexpanded and unmeasured — a pop-able rollout target."""
        return not self.children and self.visits == 0

    def is_fully_explored(self) -> bool:
        return self.seen_terminals >= self.expected_terminals

    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0

    def ucb(self, c: float) -> float:
        """Canonical UCB1; ``+inf`` when unmeasured (so unvisited
        children get descent priority over any measured sibling)."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent and self.parent.visits > 0 else 1
        return self.mean_reward() + c * math.sqrt(math.log(max(parent_visits, 1)) / self.visits)

    def __repr__(self) -> str:  # avoid parent/children recursion
        return f"NodeRow(key={self.key!r}, visits={self.visits}, reward={self.total_reward:.4f}, score={self.score})"


class SearchTree:
    """Pure-Python search tree used by :class:`TuningSearch`.

    Rebuilt fresh each process: the engine re-fires every rule on warm
    starts, which re-creates the same edges via :meth:`expand`. The
    cached ``perf`` rows on disk ensure no re-bench, but the UCB
    counters (visits / reward / coverage) start from zero each run.

    Three counters are maintained online via upward propagation along
    ``parent_key``:

    - ``expected_terminals`` — each *new* expansion of a parent that
      had no children adds ``n_new - 1`` to every ancestor (the
      parent's placeholder "1" is consumed by the first child);
      subsequent expansions of the same parent add ``n_new``.
    - ``seen_terminals`` / ``failed_terminals`` — each measured
      terminal bumps every ancestor by ``+1`` (and ``+1`` for
      ``failed`` on failure).
    - ``visits`` / ``total_reward`` — UCB1 numerator + denominator.
      Bumped only on terminal measurement (``reward = 1/median_us``
      for ok, ``0`` for fail). Canonical MCTS: an expansion alone is
      not a visit — unvisited siblings are handled by the "pick any
      unvisited child first" branch in :meth:`_ucb_walk`.

    A node is fully explored when ``seen_terminals == expected_terminals``.
    The value can move *down* mid-run when expansion grows the
    denominator faster than the numerator — that's the correct
    semantics ("we just discovered there's more to explore").
    """

    def __init__(self) -> None:
        # The in-memory tree is single-context per process — the on-disk
        # DB carries context_key for cross-machine partitioning, but the
        # search policy here only ever operates within one context.
        # ``_nodes`` is the by-key index used for O(1) lookup from
        # ``record_terminal`` / ``expand`` callers that hold an
        # ``op_cache_key``. Parent/child structure lives on the
        # :class:`NodeRow` objects themselves.
        self._nodes: dict[str, SearchNode] = {}
        # First root inserted; latched on first ``ensure_root`` so the
        # UCB walk can start there without scanning the dict.
        self._root: SearchNode | None = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def node(self, node_key: str) -> SearchNode | None:
        return self._nodes.get(node_key)

    def children(self, parent_key: str) -> list[str]:
        parent = self._nodes.get(parent_key)
        return [c.key for c in parent.children] if parent else []

    @property
    def root(self) -> SearchNode | None:
        """First root :class:`NodeRow` inserted, or ``None`` if nothing
        has been expanded yet."""
        return self._root

    def root_coverage(self) -> tuple[int, int]:
        """``(seen, expected)`` for the latched root. ``(0, 0)`` if no
        root has been inserted."""
        if self._root is None:
            return 0, 0
        return self._root.seen_terminals, self._root.expected_terminals

    def is_fully_explored(self, node_key: str) -> bool:
        n = self.node(node_key)
        return n is not None and n.is_fully_explored()

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def ensure_root(self, node_key: str) -> SearchNode:
        """Insert a root node if not present and return it."""
        row = self._nodes.get(node_key)
        if row is None:
            row = SearchNode(key=node_key)
            self._nodes[node_key] = row
        if self._root is None:
            self._root = row
        return row

    def expand(self, parent_key: str, child_keys: list[str], *, child_scores: list[float] | None = None) -> None:
        """Record ``parent_key → child_keys`` edges and maintain
        ``expected_terminals`` on every ancestor.

        ``child_scores`` (optional, aligned with ``child_keys``) stashes
        each child's ``op.score(ctx)`` on its :class:`NodeRow` so the
        search policy can read the prior in O(1) without rescanning
        queued candidates' graphs. Re-expansions don't overwrite an
        already-set score.

        Idempotent — re-firing a rule with the same children adds no
        new edges. When the rule's option set has grown, only the
        genuinely-new edges propagate.
        """
        if not child_keys:
            return
        parent = self.ensure_root(parent_key)
        pre = len(parent.children)
        existing_keys = {c.key for c in parent.children}
        scores = child_scores if child_scores is not None else [float("-inf")] * len(child_keys)
        n_new = 0
        for ck, s in zip(child_keys, scores, strict=True):
            if ck in existing_keys:
                # Backfill score on a re-expansion if it wasn't set originally.
                row = self._nodes[ck]
                if row.score == float("-inf") and s != float("-inf"):
                    row.score = s
                continue
            row = self._nodes.get(ck)
            if row is None:
                row = SearchNode(key=ck, parent=parent, score=s)
                self._nodes[ck] = row
            else:
                # Pre-existing orphan (e.g. from a record_terminal that
                # ran before its parent was known) — adopt it.
                row.parent = parent
                if row.score == float("-inf") and s != float("-inf"):
                    row.score = s
            parent.children.append(row)
            existing_keys.add(ck)
            n_new += 1

        if n_new == 0:
            return

        # First-ever expansion of this parent consumes its placeholder "1",
        # so the delta is one less than n_new. Later expansions are pure
        # additions (the parent was already accounting for its children).
        delta = n_new - 1 if pre == 0 else n_new
        if delta != 0:
            self._propagate_expected(parent, delta)

    def record_terminal(self, leaf_key: str, *, reward: float, status: str) -> bool:
        """Bump ``seen_terminals = 1`` on the leaf (if not already
        seen), propagate ``+1`` to every ancestor, and add the reward.

        Returns ``True`` iff this was a newly-measured terminal (so the
        caller knows propagation ran)."""
        node = self._nodes.get(leaf_key)
        if node is None:
            node = SearchNode(key=leaf_key)
            self._nodes[leaf_key] = node
        if node.seen_terminals >= 1:
            return False
        failed_delta = 0 if status == "ok" else 1
        node.seen_terminals = 1
        node.failed_terminals = failed_delta
        node.visits += 1
        node.total_reward += reward
        if node.parent is not None:
            self._propagate_visit(
                node.parent,
                seen_delta=1,
                failed_delta=failed_delta,
                reward_delta=reward,
            )
        return True

    # ------------------------------------------------------------------
    # Internal propagation walks
    # ------------------------------------------------------------------

    def _propagate_expected(self, node: SearchNode, delta: int) -> None:
        cur: SearchNode | None = node
        while cur is not None:
            cur.expected_terminals += delta
            cur = cur.parent

    def _propagate_visit(
        self,
        node: SearchNode,
        *,
        seen_delta: int,
        failed_delta: int,
        reward_delta: float,
    ) -> None:
        cur: SearchNode | None = node
        while cur is not None:
            cur.seen_terminals += seen_delta
            cur.failed_terminals += failed_delta
            cur.visits += seen_delta
            cur.total_reward += reward_delta
            cur = cur.parent


# ---------------------------------------------------------------------------
# MCTS search policy
# ---------------------------------------------------------------------------


class TuningSearch:
    """MCTS-style exhaustive priority search using UCB1 selection.

    Each candidate sits at some "tip" node in the search tree (the
    op_cache_key of the most recently rewritten kernel-bearing op).
    Priority for pop ordering is ``-UCB1(tip)`` where

    ``UCB1 = mean_reward + c * sqrt(log(parent.visits) / tip.visits)``

    Reward per measured terminal is ``1 / latency_us_median`` for an
    ``ok`` bench, ``0`` for ``bench_fail``. Failures stay in the visit
    count, so a subtree with all bench_fails has ``mean_reward = 0``
    and falls in the rankings even as its exploration term decays.

    Tips not yet in the tree (fresh frontier) get ``priority = -∞`` so
    they're always popped first — the algorithm explores once before
    exploiting. Used by ``deplodock compile --tune`` so the sweep
    drifts toward promising subtrees while still covering the space.

    Two-phase selection. The first ``N_BOOTSTRAP`` rollouts skip UCB
    and drill straight down to terminal: pick the deepest unexpanded
    frontier in the tree, tie-breaking on ``Op.score``. This seeds the
    UCB statistics with a handful of measured terminals before
    canonical UCB1 kicks in (which without measurements would just
    re-pick the same level-1 unvisited child forever). After bootstrap,
    selection uses :meth:`_ucb_walk`.

    Stopping policy. ``pop()`` returns ``None`` when:

    - ``patience`` measured terminals in a row without a new best
      latency, *and* coverage ``seen / expected ≥ min_coverage`` so the
      patience clock doesn't fire on a slow start, or
    - the tree is fully drained (no candidate queued, fallback empty).

    There is no wall-clock budget — under parallel workers a wall
    budget chops off promising branches mid-exploration; rely on
    patience + tree exhaustion instead. For manual runs, send Ctrl-C
    and the caller catches ``KeyboardInterrupt`` to print stats."""

    UCB_C = math.sqrt(2)  # canonical UCB1 exploration constant.
    # Measure this many terminals via greedy depth-first descent before
    # switching to UCB. UCB needs some measured rewards to be meaningful;
    # before that it's just "pick any unvisited" which doesn't drill
    # down through expanded subtrees.
    N_BOOTSTRAP = 10
    # Score-gap below which an unvisited sibling is dropped from the
    # expansion frontier (see ``_ucb_walk``). 1.0 spans roughly one of the
    # graduated penalties in ``TileOp.score`` — e.g. CTA-count or
    # thread-distance — so a sibling with one extra failure mode beyond the
    # current best is dropped rather than benched.
    SCORE_CUTOFF = 1.0

    def __init__(
        self,
        tree: SearchTree | None = None,
        *,
        db: SearchDB | None = None,
        patience: int = 20,
        min_coverage: float = 0.3,
    ) -> None:
        self._db = db if db is not None else SearchDB()
        self._tree = tree if tree is not None else SearchTree()
        # Latched lazily on the first ``push`` — the engine never spawns
        # a tuning sweep across mixed contexts so one key per process is fine.
        self._context_key: str | None = None
        self._patience = patience
        self._min_coverage = min_coverage
        self._last_seen = 0
        self._best_latency = float("inf")
        self._stagnant = 0
        self._stop_reason: str | None = None
        # Candidates grouped by their tip op_cache_key. Every push lands
        # here (or in ``_fallback`` if no tip); every pop runs a UCB walk
        # from the root and pulls one from the selected tip's bucket.
        self._by_tip: dict[str, list[Candidate]] = {}
        # FIFO fallback queue for candidates whose tip op_cache_key is
        # missing (no kernel-bearing op in the graph yet) and for the
        # final drain when ``_by_tip`` is empty.
        self._fallback: deque[Candidate] = deque()

    @property
    def db(self) -> SearchDB:
        return self._db

    @property
    def tree(self) -> SearchTree:
        """Exposed for ``driver.py`` / ``engine.py`` / ``recorder.py``
        which all reach this via ``getattr(search, "tree", None)``."""
        return self._tree

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def push(self, c: Candidate, *forks: Candidate) -> None:
        self._push_one(c)
        for fork in forks:
            self._push_one(fork)

    def _push_one(self, c: Candidate) -> None:
        if self._context_key is None:
            self._context_key = c.ctx.structural_key()
        tip_key = self._tip_key(c.graph)
        if tip_key is None:
            # No tip — keep on the FIFO fallback queue (rare).
            self._fallback.append(c)
            return
        self._by_tip.setdefault(tip_key, []).append(c)

    def pop(self) -> Candidate | None:
        if self._should_stop():
            return None
        root = self._tree.root
        if root is None:
            return self._fallback_pop()
        seen, _ = self._tree.root_coverage()
        if seen < self.N_BOOTSTRAP:
            return self._bootstrap_pop()
        target = self._ucb_walk(root)
        if target is None or target.key not in self._by_tip:
            return self._fallback_pop()
        cands = self._by_tip[target.key]
        c = cands.pop(0)
        if not cands:
            del self._by_tip[target.key]
        return c

    def _fallback_pop(self) -> Candidate | None:
        """When the UCB walk has nothing to match (root unknown, all
        frontiers exhausted, etc.), drain the by-tip dict in arbitrary
        order, then the FIFO fallback queue. Keeps the search complete
        even when the tree-walk selector can't find a target."""
        for tip, cands in list(self._by_tip.items()):
            if cands:
                c = cands.pop(0)
                if not cands:
                    del self._by_tip[tip]
                return c
        if self._fallback:
            return self._fallback.popleft()
        return None

    def stop(self, reason: str) -> None:
        """Externally signal early termination (e.g. ``KeyboardInterrupt``
        from the CLI). The next ``pop`` will return ``None`` and the
        caller can read ``stop_reason``."""
        self._stop_reason = reason

    def _should_stop(self) -> bool:
        if self._stop_reason is not None:
            return True
        if self._context_key is None:
            return False
        # Poll the tree for newly-measured terminals since the last
        # pop check. Reset stagnant on any improvement; otherwise
        # count fresh measurements toward patience.
        seen, expected = self._tree.root_coverage()
        new_measurements = seen - self._last_seen
        self._last_seen = seen
        if new_measurements > 0:
            cur_best = self._db.min_latency_for_context(self._context_key, backend="cuda")
            if cur_best is None:
                cur_best = float("inf")
            if cur_best < self._best_latency:
                self._best_latency = cur_best
                self._stagnant = 0
            else:
                self._stagnant += new_measurements
        coverage = (seen / expected) if expected else 0.0
        if coverage >= self._min_coverage and self._stagnant >= self._patience:
            self._stop_reason = f"patience ({self._stagnant} stagnant @ {100 * coverage:.0f}% coverage, best {self._best_latency:.2f} us)"
            return True
        return False

    # -- UCB plumbing --------------------------------------------------

    def _tip_key(self, graph) -> str | None:
        """The candidate's tip is its deepest kernel-bearing op in the
        search tree (most rule applications fired). For single-kernel
        graphs that's just the one body-bearing op's key."""
        keys = [op_cache_key(n.op) for n in graph.nodes.values() if op_cache_key(n.op) is not None]
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return max(keys, key=self._tree_depth)

    def _tree_depth(self, node_key: str) -> int:
        """Hops from ``node_key`` up to the root via ``NodeRow.parent``.
        Capped at 64 as a defensive measure against a hypothetical cycle."""
        node = self._tree.node(node_key)
        d = 0
        while node is not None and node.parent is not None and d < 64:
            node = node.parent
            d += 1
        return d

    def _bootstrap_pop(self) -> Candidate | None:
        """Bootstrap selection: pop the candidate furthest along in the
        pipeline (highest ``cursor.pass_idx`` then ``rule_idx``), tied
        on ``Op.score``. Tree-depth doesn't track pipeline progress —
        Graph-rewrite passes (decomposition / fusion) advance the cursor
        without adding tree edges, so a tree-shallow candidate can be
        closer to terminal than a tree-deep one. Cursor-progress is the
        right "drill toward terminal" signal."""
        best_tip: str | None = None
        best_idx = 0
        best_key: tuple[int, int, float] = (-1, -1, float("-inf"))
        for tip_key, cands in self._by_tip.items():
            node = self._tree.node(tip_key)
            score = node.score if node is not None else float("-inf")
            for i, c in enumerate(cands):
                key = (c.cursor.pass_idx, c.cursor.rule_idx, score)
                if key > best_key:
                    best_key = key
                    best_tip = tip_key
                    best_idx = i
        if best_tip is None:
            # Fall through to fallback (drains by_tip arbitrarily + _fallback).
            return self._fallback_pop()
        c = self._by_tip[best_tip].pop(best_idx)
        if not self._by_tip[best_tip]:
            del self._by_tip[best_tip]
        return c

    def _ucb_walk(self, root: SearchNode) -> SearchNode | None:
        """Walk the search tree via UCB selection. Stops at the first
        :class:`NodeRow` with an unexpanded + unmeasured child (the true
        frontier) and returns it. Expanded-but-unmeasured nodes are
        descended through — :meth:`NodeRow.ucb` returns ``+inf`` when
        ``visits == 0`` so canonical UCB1 picks them ahead of measured
        siblings automatically; ``score`` breaks ``+inf`` ties.

        Measured leaves (terminal, no children, ``visits > 0``) are
        skipped during descent — they contribute their visits to the
        parent's UCB but have no candidate left to pop.

        Among unexpanded siblings, ``score`` ranks priors and a
        ``SCORE_CUTOFF`` purge drops priors too far below the local
        best, freeing the budget for promising variants."""
        cur = root
        # Cap to defend against a hypothetical cycle. Real trees here
        # are tens of levels deep at most.
        for _ in range(256):
            if not cur.children:
                return cur  # frontier: leaf of the search tree

            # True frontier: children that are both unexpanded and unmeasured.
            frontier = [c for c in cur.children if c.is_frontier()]
            if frontier:
                best_score = max(c.score for c in frontier)
                if best_score != float("-inf"):
                    cutoff = best_score - self.SCORE_CUTOFF
                    # Purge cutoff-dropped candidates from the queue so the
                    # ``_fallback_pop`` drain (used when the walk lands on a
                    # node that isn't queued) can't resurrect them later.
                    for c in frontier:
                        if c.score < cutoff:
                            self._by_tip.pop(c.key, None)
                    frontier = [c for c in frontier if c.score >= cutoff]
                if frontier:
                    return max(frontier, key=lambda n: n.score)

            # Descend into expanded children only. ``visits == 0`` gets
            # +inf via ``NodeRow.ucb``; score breaks ties.
            expanded = [c for c in cur.children if c.is_expanded()]
            if not expanded:
                return cur
            cur = max(expanded, key=lambda c: (c.ucb(self.UCB_C), c.score))
        return cur
