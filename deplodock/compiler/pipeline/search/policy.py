"""Search-policy implementations: the ``Search`` protocol plus the two
concrete strategies used by the autotune driver.

- :class:`GreedySearch` — stops at the first terminal candidate, used by
  ``run_pipeline`` for single-shot compiles.
- :class:`TuningSearch` — MCTS-style exhaustive priority search with
  UCB1 selection, used by ``deplodock compile --tune``.

Both share a min-heap fallback (:class:`_PriorityHeap`) keyed by
``count_unmeasured_ops`` at push time."""

from __future__ import annotations

import heapq
import time
from typing import Protocol

from deplodock.compiler.pipeline.search.cache import TuningCache, count_unmeasured_ops, op_cache_key
from deplodock.compiler.pipeline.search.candidate import Candidate


class Search(Protocol):
    """Search-strategy hook. The engine pushes spawned candidates and
    pops the next one to expand. ``pop`` returning ``None`` ends the
    search. Implementations choose both the ordering (DFS / BFS /
    priority / MCTS / whatever) and the termination condition (greedy
    stops at first terminal; exhaustive runs the queue dry).

    The engine doesn't tell the search when a candidate is terminal —
    instead a terminal candidate is the one the engine yielded without
    pushing it back. Searches that need to detect this can track the
    last-popped candidate and check whether it returned via ``push``."""

    def push(self, c: Candidate) -> None: ...
    def pop(self) -> Candidate | None: ...  # None when exhausted


class _PriorityHeap:
    """Shared push/pop for the two concrete search policies. Priority
    is ``count_unmeasured_ops`` at push time; LIFO tiebreak via
    decreasing ``_seq`` so on a fresh in-memory cache the order is the
    same as a DFS stack."""

    def __init__(self, cache: TuningCache | None = None, context_key: str | None = None) -> None:
        if cache is None:
            cache = TuningCache()
        self._cache = cache
        self._context_key = context_key
        self._heap: list[tuple[int, int, Candidate]] = []
        self._seq = 0

    def _ckey(self, c: Candidate) -> str:
        return self._context_key if self._context_key is not None else c.ctx.structural_key()

    def _push(self, c: Candidate) -> None:
        n = count_unmeasured_ops(c.graph, self._cache, self._ckey(c))
        self._seq += 1
        heapq.heappush(self._heap, (n, -self._seq, c))

    def _pop(self) -> Candidate | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    @property
    def cache(self) -> TuningCache:
        return self._cache


class GreedySearch(_PriorityHeap):
    """Stop at the first terminal candidate.

    The engine yields a terminal candidate without pushing it back. We
    detect that by tracking the last-popped candidate: if the next
    ``pop`` sees that nothing has been ``push``-ed since (the candidate
    didn't return for another rule application), the previous candidate
    must have been terminal — return ``None`` to end the search even if
    the heap still holds unexplored forks.

    Used by ``run_pipeline`` for single-shot compiles. Autotune forks
    beyond option 0 stay in the heap unmeasured."""

    def __init__(self, cache: TuningCache | None = None, context_key: str | None = None) -> None:
        super().__init__(cache, context_key)
        self._outstanding: Candidate | None = None

    def push(self, c: Candidate) -> None:
        if c is self._outstanding:
            self._outstanding = None
        self._push(c)

    def pop(self) -> Candidate | None:
        if self._outstanding is not None:
            # Last popped never came back via ``push`` → it was terminal.
            return None
        c = self._pop()
        self._outstanding = c
        return c


class TuningSearch(_PriorityHeap):
    """MCTS-style exhaustive priority search using UCB1 selection.

    Each candidate sits at some "tip" node in the cache tree (the
    op_cache_key of the most recently rewritten kernel-bearing op).
    Priority for pop ordering is ``-UCB1(tip)`` where

    ``UCB1 = mean_reward + c * sqrt(log(parent.visits) / tip.visits)``

    Reward per measured terminal is ``1 / latency_us`` for an ``ok``
    bench, ``0`` for ``bench_fail``. Failures stay in the visit count,
    so a subtree with all bench_fails has ``mean_reward = 0`` and falls
    in the rankings even as its exploration term decays.

    Tips not yet in the cache (fresh frontier) get ``priority = -∞`` so
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
        cache: TuningCache | None = None,
        context_key: str | None = None,
        *,
        budget_s: float = 60.0,
        patience: int = 20,
        min_coverage: float = 0.3,
    ) -> None:
        super().__init__(cache, context_key)
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
        # Root of the cache tree, latched on first push so the per-pop
        # UCB walk starts at the right place.
        self._root_key: str | None = None

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def push(self, c: Candidate) -> None:
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
        # Latch the cache tree's root the first time we see one.
        if self._root_key is None:
            self._root_key = self._find_root(ckey)

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
        # Start a new iteration: walk the cache tree from root via UCB
        # and pick a candidate whose tip matches the selected frontier.
        ckey = self._context_key
        if ckey is None:
            return self._fallback_pop()
        if self._root_key is None:
            self._root_key = self._find_root(ckey)
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
        # Poll the cache for newly-measured terminals since the last
        # pop check. Reset stagnant on any improvement; otherwise
        # count fresh measurements toward patience.
        seen, expected = self._cache.root_coverage(self._context_key)
        new_measurements = seen - self._last_seen
        self._last_seen = seen
        if new_measurements > 0:
            row = self._cache._conn.execute(  # noqa: SLF001
                "SELECT MIN(latency_us) FROM cuda_perf WHERE context_key = ? AND status = 'ok'",
                (self._context_key,),
            ).fetchone()
            cur_best = row[0] if row and row[0] is not None else float("inf")
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
        UCB pops earlier. Fresh frontier (no cache row, or row with zero
        ``visits``) gets ``-inf`` — always pop first.

        ``visits`` is the MCTS denominator (expansions + measurements),
        not just measured leaves. That lets the search differentiate
        "rule fired here, accumulating exploration" vs "untouched
        sibling, still totally fresh"."""
        if tip_key is None:
            return 0.0
        node = self._cache.node(context_key, tip_key)
        if node is None or node.visits == 0:
            return float("-inf")
        mean = node.total_reward / node.visits
        parent = self._cache.node(context_key, node.parent_key) if node.parent_key else None
        parent_visits = parent.visits if parent and parent.visits > 0 else node.visits
        import math  # noqa: PLC0415

        exploration = self.UCB_C * math.sqrt(math.log(max(parent_visits, 1)) / max(node.visits, 1))
        return -(mean + exploration)

    def _tip_key(self, graph, context_key: str) -> str | None:
        """The candidate's tip is its deepest kernel-bearing op in the
        cache tree (most rule applications fired). For single-kernel
        graphs that's just the one body-bearing op's key."""
        keys = [op_cache_key(n.op) for n in graph.nodes.values() if op_cache_key(n.op) is not None]
        if not keys:
            return None
        if len(keys) == 1:
            return keys[0]
        return max(keys, key=lambda k: self._depth(context_key, k))

    def _find_root(self, context_key: str) -> str | None:
        """Return the node_key of the cache tree's root for this context
        — the first ``parent_key IS NULL`` row inserted (typically the
        post-fusion LoopOp). Latched on first call; subsequent calls
        return the same key even if more roots get inserted later."""
        row = self._cache._conn.execute(  # noqa: SLF001
            "SELECT node_key FROM nodes WHERE context_key = ? AND parent_key IS NULL ORDER BY rowid LIMIT 1",
            (context_key,),
        ).fetchone()
        return row[0] if row else None

    def _ucb_walk(self, context_key: str, root_key: str) -> str | None:
        """Walk the cache tree from ``root_key`` via UCB selection at
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
            children = self._cache.children(context_key, cur)
            if not children:
                return cur  # frontier: leaf of the cache tree
            cur_node = self._cache.node(context_key, cur)
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
                child = self._cache.node(context_key, ck)
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
                child = self._cache.node(context_key, ck)
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
            row = self._cache.node(context_key, cur)
            if row is None or row.parent_key is None:
                break
            cur = row.parent_key
            d += 1
        return d
