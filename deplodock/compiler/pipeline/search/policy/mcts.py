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

import heapq
import time
from dataclasses import dataclass

from deplodock.compiler.ir.base import Op
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.policy.base import _PriorityHeap

# ---------------------------------------------------------------------------
# In-memory MCTS tree
# ---------------------------------------------------------------------------


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
      Bumped both on expansion (treats a rule firing as a soft visit
      so unvisited siblings stay distinguishable in UCB selection) and
      on terminal measurement (``reward = 1/median_us`` for ok, ``0``
      for fail).

    A node is fully explored when ``seen_terminals == expected_terminals``.
    The value can move *down* mid-run when expansion grows the
    denominator faster than the numerator — that's the correct
    semantics ("we just discovered there's more to explore").
    """

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


# ---------------------------------------------------------------------------
# MCTS search policy
# ---------------------------------------------------------------------------


class TuningSearch(_PriorityHeap):
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

    Stopping policy. ``pop()`` returns ``None`` when any of:

    - wall-clock budget ``budget_s`` (from first push) elapsed,
    - ``patience`` measured terminals in a row without a new best
      latency, *and* coverage ``seen / expected ≥ min_coverage`` so the
      patience clock doesn't fire on a slow start.

    Disable a knob by passing ``float("inf")`` (budget / coverage) or
    a very large int (patience)."""

    UCB_C = 1.0  # exploration constant; lower than √2 because we already count expansions as visits.
    # Score-gap below which an unvisited sibling is dropped from the
    # expansion frontier (see ``_ucb_walk``). 1.0 spans roughly one of the
    # graduated penalties in ``TileOp.score`` — e.g. CTA-count or
    # thread-distance — so a sibling with one extra failure mode beyond the
    # current best is dropped rather than benched.
    SCORE_CUTOFF = 1.0

    def __init__(
        self,
        tree: SearchTree | None = None,
        context_key: str | None = None,
        *,
        db: SearchDB | None = None,
        budget_s: float = 60.0,
        patience: int = 20,
        min_coverage: float = 0.3,
    ) -> None:
        super().__init__(context_key, db=db)
        self._tree = tree if tree is not None else SearchTree()
        self._budget_s = budget_s
        self._patience = patience
        self._min_coverage = min_coverage
        self._t_start = time.monotonic()
        self._last_seen = 0
        self._best_latency = float("inf")
        self._stagnant = 0
        self._stop_reason: str | None = None
        # MCTS rollout state: candidates grouped by their tip op_cache_key.
        # ``_current`` is the candidate being drilled to terminal right
        # now — pop returns it on every call until the engine yields it
        # (i.e. doesn't push it back).
        self._current: Candidate | None = None
        self._just_popped: Candidate | None = None
        self._by_tip: dict[str, list[Candidate]] = {}
        # Root of the search tree, latched on first push so the per-pop
        # UCB walk starts at the right place.
        self._root_key: str | None = None

    @property
    def tree(self) -> SearchTree:
        """Exposed for ``driver.py`` / ``engine.py`` / ``recorder.py``
        which all reach this via ``getattr(search, "tree", None)``."""
        return self._tree

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def push(self, c: Candidate, *forks: Candidate, parent: Op | None = None) -> None:
        # Tuning registers every candidate. Order matches the previous
        # driver behavior — forks first, primary last — so the
        # ``_just_popped`` check below still fires on the right one.
        # ``parent`` is unused here (MCTS tracks parents via the
        # ``SearchTree``); accepted for protocol compatibility.
        del parent
        for fork in forks:
            self._push_one(fork)
        self._push_one(c)

    def _push_one(self, c: Candidate) -> None:
        if self._context_key is None:
            self._context_key = c.ctx.structural_key()
        ckey = self._ckey(c)
        # If this is the candidate we just popped being pushed back
        # mid-rollout (engine advanced it by one rule), keep it as the
        # outstanding rollout. Otherwise it's a fork sibling — file it
        # under its tip for later UCB-walk selection.
        if c is self._just_popped:
            self._current = c
            return
        tip_key = self._tip_key(c.graph, ckey)
        if tip_key is None:
            # No tip — keep on the legacy heap as a fallback (rare).
            self._seq += 1
            heapq.heappush(self._heap, (0.0, -self._seq, c))
            return
        self._by_tip.setdefault(tip_key, []).append(c)
        # Latch the search tree's root the first time we see one.
        if self._root_key is None:
            self._root_key = self._tree.find_root(ckey)

    def pop(self) -> Candidate | None:
        if self._should_stop():
            return None
        # Mid-rollout: keep returning the same candidate.
        if self._current is not None:
            c = self._current
            self._current = None
            self._just_popped = c
            return c
        # Previous rollout terminated (engine yielded without push-back).
        # Start a new iteration: walk the search tree from root via UCB
        # and pick a candidate whose tip matches the selected frontier.
        ckey = self._context_key
        if ckey is None:
            return self._fallback_pop()
        if self._root_key is None:
            self._root_key = self._tree.find_root(ckey)
        if self._root_key is None:
            return self._fallback_pop()
        target = self._ucb_walk(ckey, self._root_key)
        if target is None or target not in self._by_tip:
            return self._fallback_pop()
        cands = self._by_tip[target]
        c = cands.pop(0)
        if not cands:
            del self._by_tip[target]
        self._just_popped = c
        return c

    def _fallback_pop(self) -> Candidate | None:
        """When the UCB walk has nothing to match (root unknown, all
        frontiers exhausted, etc.), drain the by-tip dict in arbitrary
        order, then the legacy heap. Keeps the search complete even when
        the tree-walk selector can't find a target."""
        for tip, cands in list(self._by_tip.items()):
            if cands:
                c = cands.pop(0)
                if not cands:
                    del self._by_tip[tip]
                self._just_popped = c
                return c
        if self._heap:
            c = heapq.heappop(self._heap)[2]
            self._just_popped = c
            return c
        return None

    def _should_stop(self) -> bool:
        elapsed = time.monotonic() - self._t_start
        if elapsed >= self._budget_s:
            self._stop_reason = f"wall budget ({self._budget_s:.1f}s, elapsed {elapsed:.1f}s)"
            return True
        if self._context_key is None:
            return False
        # Poll the tree for newly-measured terminals since the last
        # pop check. Reset stagnant on any improvement; otherwise
        # count fresh measurements toward patience.
        seen, expected = self._tree.root_coverage(self._context_key)
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

    def _ucb_priority(self, context_key: str, tip_key: str | None) -> float:
        """Heap-min priority key. Lower = pop first; ``-UCB1`` so higher
        UCB pops earlier. Fresh frontier (no tree row, or row with zero
        ``visits``) gets ``-inf`` — always pop first.

        ``visits`` is the MCTS denominator (expansions + measurements),
        not just measured leaves. That lets the search differentiate
        "rule fired here, accumulating exploration" vs "untouched
        sibling, still totally fresh"."""
        if tip_key is None:
            return 0.0
        node = self._tree.node(context_key, tip_key)
        if node is None or node.visits == 0:
            return float("-inf")
        mean = node.total_reward / node.visits
        parent = self._tree.node(context_key, node.parent_key) if node.parent_key else None
        parent_visits = parent.visits if parent and parent.visits > 0 else node.visits
        import math  # noqa: PLC0415

        exploration = self.UCB_C * math.sqrt(math.log(max(parent_visits, 1)) / max(node.visits, 1))
        return -(mean + exploration)

    def _tip_key(self, graph, context_key: str) -> str | None:
        """The candidate's tip is its deepest kernel-bearing op in the
        search tree (most rule applications fired). For single-kernel
        graphs that's just the one body-bearing op's key."""
        keys = [op_cache_key(n.op) for n in graph.nodes.values() if op_cache_key(n.op) is not None]
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return max(keys, key=lambda k: self._depth(context_key, k))

    def _ucb_walk(self, context_key: str, root_key: str) -> str | None:
        """Walk the search tree from ``root_key`` via UCB selection at
        each node. Stops at the first node with at least one unvisited
        child (returning that child) or with no children at all
        (returning that node itself). Among unvisited siblings, the
        ``Op.score`` heuristic breaks the otherwise-arbitrary order so
        the MCTS bootstrap visits "well-shaped" candidates first."""
        import math  # noqa: PLC0415

        cur = root_key
        # Cap to defend against a hypothetical cycle in the parent_key
        # graph. Real trees here are tens of levels deep at most.
        for _ in range(256):
            children = self._tree.children(context_key, cur)
            if not children:
                return cur  # frontier: leaf of the search tree
            cur_node = self._tree.node(context_key, cur)
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
                child = self._tree.node(context_key, ck)
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
                child = self._tree.node(context_key, ck)
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

    def _depth(self, context_key: str, node_key: str) -> int:
        d = 0
        cur: str | None = node_key
        # Hard cap on chain walk so a degenerate cycle (shouldn't happen, defensive) doesn't loop forever.
        while cur is not None and d < 64:
            row = self._tree.node(context_key, cur)
            if row is None or row.parent_key is None:
                break
            cur = row.parent_key
            d += 1
        return d
