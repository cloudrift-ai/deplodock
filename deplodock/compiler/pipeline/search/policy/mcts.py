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
from dataclasses import dataclass

from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key

# ---------------------------------------------------------------------------
# In-memory MCTS tree
# ---------------------------------------------------------------------------


@dataclass
class NodeRow:
    """One tree node — autotune-state for an op.

    ``seen_terminals`` counts every measured terminal under this node
    (ok + fail; used by coverage queries). ``failed_terminals`` tracks
    fails only (diagnostic). ``visits`` is the canonical-MCTS
    denominator — it counts measured terminals under this node and
    nothing else (expansion alone never bumps it). ``total_reward``
    accumulates the MCTS reward (``1/latency_us`` for ok, ``0`` for
    fail). UCB1 exploitation uses ``total_reward / visits``.
    """

    parent_key: str | None
    expected_terminals: int = 1
    seen_terminals: int = 0
    failed_terminals: int = 0
    visits: int = 0
    total_reward: float = 0.0


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
        self._nodes: dict[str, NodeRow] = {}
        # Ordered list (dedup on insert) so iteration order is
        # deterministic for tests and tie-breaks in the policy.
        self._expansions: dict[str, list[str]] = {}
        # First root inserted; latched on first ``ensure_root`` so the
        # UCB walk can start there without scanning the dict.
        self._root_key: str | None = None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def node(self, node_key: str) -> NodeRow | None:
        return self._nodes.get(node_key)

    def children(self, parent_key: str) -> list[str]:
        return list(self._expansions.get(parent_key, ()))

    @property
    def root(self) -> str | None:
        """The first ``parent_key is None`` node inserted, or ``None``
        if nothing has been expanded yet."""
        return self._root_key

    def root_coverage(self) -> tuple[int, int]:
        """``(sum_seen, sum_expected)`` over every root in the tree."""
        seen = 0
        expected = 0
        for row in self._nodes.values():
            if row.parent_key is not None:
                continue
            seen += row.seen_terminals
            expected += row.expected_terminals
        return seen, expected

    def is_fully_explored(self, node_key: str) -> bool:
        n = self.node(node_key)
        return n is not None and n.seen_terminals >= n.expected_terminals

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def ensure_root(self, node_key: str) -> None:
        """Insert a root node (``parent_key = None``) if not present."""
        if node_key not in self._nodes:
            self._nodes[node_key] = NodeRow(parent_key=None)
        if self._root_key is None:
            self._root_key = node_key

    def expand(self, parent_key: str, child_keys: list[str]) -> None:
        """Record ``parent_key → child_keys`` edges and maintain
        ``expected_terminals`` on every ancestor.

        Idempotent — re-firing a rule with the same children adds no
        new edges. When the rule's option set has grown, only the
        genuinely-new edges propagate.
        """
        if not child_keys:
            return
        self.ensure_root(parent_key)

        existing = self._expansions.setdefault(parent_key, [])
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
            self._nodes.setdefault(ck, NodeRow(parent_key=parent_key))

        # First-ever expansion of this parent consumes its placeholder "1",
        # so the delta is one less than n_new. Later expansions are pure
        # additions (the parent was already accounting for its children).
        delta = n_new - 1 if pre == 0 else n_new
        if delta != 0:
            self._propagate_expected(parent_key, delta)

    def record_terminal(self, leaf_key: str, *, reward: float, status: str) -> bool:
        """Bump ``seen_terminals = 1`` on the leaf (if not already
        seen), propagate ``+1`` to every ancestor, and add the reward.

        Returns ``True`` iff this was a newly-measured terminal (so the
        caller knows propagation ran)."""
        node = self._nodes.setdefault(leaf_key, NodeRow(parent_key=None))
        if node.seen_terminals >= 1:
            return False
        failed_delta = 0 if status == "ok" else 1
        node.seen_terminals = 1
        node.failed_terminals = failed_delta
        node.visits += 1
        node.total_reward += reward
        if node.parent_key is not None:
            self._propagate_visit(
                node.parent_key,
                seen_delta=1,
                failed_delta=failed_delta,
                reward_delta=reward,
            )
        return True

    # ------------------------------------------------------------------
    # Internal propagation walks
    # ------------------------------------------------------------------

    def _propagate_expected(self, node_key: str, delta: int) -> None:
        cur: str | None = node_key
        while cur is not None:
            row = self._nodes.get(cur)
            if row is None:
                return
            row.expected_terminals += delta
            cur = row.parent_key

    def _propagate_visit(
        self,
        node_key: str,
        *,
        seen_delta: int,
        failed_delta: int,
        reward_delta: float,
    ) -> None:
        cur: str | None = node_key
        while cur is not None:
            row = self._nodes.get(cur)
            if row is None:
                return
            row.seen_terminals += seen_delta
            row.failed_terminals += failed_delta
            row.visits += seen_delta
            row.total_reward += reward_delta
            cur = row.parent_key


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
        root_key = self._tree.root
        if root_key is None:
            return self._fallback_pop()
        target = self._ucb_walk(root_key)
        if target is None or target not in self._by_tip:
            return self._fallback_pop()
        cands = self._by_tip[target]
        c = cands.pop(0)
        if not cands:
            del self._by_tip[target]
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
        return max(keys, key=self._depth)

    def _ucb_walk(self, root_key: str) -> str | None:
        """Walk the search tree from ``root_key`` via UCB selection at
        each node. Stops at the first node with at least one unvisited
        child (returning that child) or with no children at all
        (returning that node itself). Among unvisited siblings, the
        ``Op.score`` heuristic breaks the otherwise-arbitrary order so
        the MCTS bootstrap visits "well-shaped" candidates first."""
        cur = root_key
        # Cap to defend against a hypothetical cycle in the parent_key
        # graph. Real trees here are tens of levels deep at most.
        for _ in range(256):
            children = self._tree.children(cur)
            if not children:
                return cur  # frontier: leaf of the search tree
            cur_node = self._tree.node(cur)
            parent_visits = cur_node.visits if cur_node else 1
            # Find unvisited children; if any, pick the one whose
            # candidate has the highest ``score`` (heuristic prior).
            # Score cutoff: suppress unvisited candidates whose prior is
            # far below the best score seen at this level. When every
            # unvisited sibling is below the cutoff we *don't* fall
            # through to the worst-of-the-worst — instead we descend via
            # UCB so the budget keeps re-exploring known-good subtrees
            # rather than burning a wall-budget bench on a hopeless
            # variant.
            unvisited: list[tuple[str, float]] = []
            best_score = float("-inf")
            for ck in children:
                child = self._tree.node(ck)
                s = self._candidate_score(ck)
                if s > best_score:
                    best_score = s
                if child is None or child.visits == 0:
                    unvisited.append((ck, s))
            if unvisited and best_score != float("-inf"):
                cutoff = best_score - self.SCORE_CUTOFF
                kept, dropped = [], []
                for ck, s in unvisited:
                    (kept if s >= cutoff else dropped).append((ck, s))
                # Purge cutoff-dropped candidates from the queue so the
                # ``_fallback_pop`` drain (used when the walk lands on a
                # key that isn't queued) can't resurrect them later.
                for ck, _ in dropped:
                    self._by_tip.pop(ck, None)
                unvisited = kept
            if unvisited:
                return max(unvisited, key=lambda kv: kv[1])[0]
            # All worth-exploring children visited (the unvisited got
            # cut off by the score filter, or every child has a
            # measurement). Descend into the UCB-best of the visited
            # ones; skip zero-visit children so the score-cut variants
            # don't cause a divide-by-zero here.
            best_key = None
            best_ucb = float("-inf")
            for ck in children:
                child = self._tree.node(ck)
                if child is None or child.visits == 0:
                    continue
                mean = child.total_reward / child.visits
                exploration = self.UCB_C * math.sqrt(math.log(max(parent_visits, 1)) / child.visits)
                ucb = mean + exploration
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_key = ck
            if best_key is None:
                return cur
            cur = best_key
        return cur

    def _candidate_score(self, tip_key: str) -> float:
        """Highest ``Op.score`` among queued candidates whose tip matches
        ``tip_key``. Falls back to ``-inf`` if no candidate is in
        ``_by_tip`` for this node — the prior cutoff filter compares
        unvisited siblings against the best score among them, so a
        defaulted ``0.0`` would silently outrank legitimately-priored
        negative scores (e.g. an `atomic-fanin -5.0` matmul variant
        would survive the cutoff because a popped-from-queue sibling
        appears as "0.0" and raises the cutoff floor).
        """
        best = float("-inf")
        cands = self._by_tip.get(tip_key, [])
        for c in cands:
            for n in c.graph.nodes.values():
                if op_cache_key(n.op) == tip_key:
                    s = n.op.score(c.ctx)
                    if s > best:
                        best = s
                    break
        return best

    def _depth(self, node_key: str) -> int:
        d = 0
        cur: str | None = node_key
        # Hard cap on chain walk so a degenerate cycle (shouldn't happen, defensive) doesn't loop forever.
        while cur is not None and d < 64:
            row = self._tree.node(cur)
            if row is None or row.parent_key is None:
                break
            cur = row.parent_key
            d += 1
        return d
