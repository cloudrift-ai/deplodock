"""Pattern-based rewrite engine and compile-pipeline entry point.

Public surface:

- ``Pattern`` / ``Match`` / ``match_pattern`` — chain matcher: each
  ``Pattern`` matches one node by ``op_type`` + field constraints;
  ``match_pattern(graph, pattern)`` walks forward from every
  topo-ordered seed along fan-out-1 consumer edges.
- ``run_rule`` / ``run_pass`` — apply one rule module / every rule
  module in a directory to fixed point. Rule modules declare
  ``PATTERN = [Pattern(...), ...]`` and a ``rewrite(...)`` function
  whose return type discriminates the rewrite flavor:
  * ``Graph`` — functional fragment, spliced in place of the match.
  * ``Op`` — in-place rebind of ``root.op`` (id, inputs, hints kept).
  * ``list[Graph | Op]`` — autotuning fork: engine applies option 0
    inline and pushes one ``Candidate`` per remaining option onto the
    search queue.
  Raise ``RuleSkipped`` to decline a match.
- ``Candidate`` / ``Search`` / ``run_pipeline`` — the autotune driver.
  ``run_pipeline`` yields ``Candidate``s; for deterministic rules
  (no list returns) it yields exactly one.

Rule contract: rules MUST be idempotent on their own output. The engine
re-runs the full pipeline on every popped candidate, relying on each
rule's "already applied" guard (often implicit via op-type change) to
skip work that's already done."""

from __future__ import annotations

import copy
import heapq
import importlib.util
import inspect
import logging
import re
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from deplodock.compiler.cache import TuningCache, count_unmeasured_ops, op_cache_key, record_terminal
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node, Tensor, _fmt_op
from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.pipeline.dump import _inline_scalar_loads, _scalar_constant_inputs
from deplodock.compiler.pipeline.rule_diff import display_name, emit, format_skipped, render_rule_diff

if TYPE_CHECKING:
    from deplodock.compiler.pipeline.dump import CompilerDump

_PASSES_DIR = Path(__file__).parent / "passes"
_RULE_PREFIX_RE = re.compile(r"^\d+[a-z]?_")


def _strip_rule_prefix(name: str) -> str:
    """Drop the numeric ordering prefix from a rule file stem
    (``004_cooperative_reduce`` → ``cooperative_reduce``)."""
    return _RULE_PREFIX_RE.sub("", name)


logger = logging.getLogger(__name__)


class RuleSkipped(Exception):
    """Raised by a rule's ``rewrite()`` to signal that the match was
    considered but skipped, with a human-readable reason for why no
    rewrite was applied. The engine catches it, logs the reason at
    DEBUG (visible at ``compile -vv``), and treats the result the same
    as ``return None`` with no in-place mutation. Use this in place of
    a bare ``return None`` whenever the skip reason would help debug
    why a rule didn't fire on a given match."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Chain matcher
# ---------------------------------------------------------------------------


@dataclass
class Pattern:
    """One node in a chain-match pattern.

    ``constraints`` is a dict of ``field_name → expected_value`` checks
    applied to ``node.op`` (e.g. ``{"fn": "softmax"}``).
    """

    name: str
    op_type: type
    constraints: dict = field(default_factory=dict)


@dataclass
class Match:
    """Result of matching a pattern against a graph.

    ``graph`` is the graph this match was built against (rules access
    it via ``match.graph`` for ad-hoc lookups). ``nodes`` maps each
    pattern entry's name to the matched node id. ``consumed`` and
    ``output`` may be overwritten by the rewrite function to control
    which nodes the rewriter removes and which node its edges get
    redirected to. ``output`` defaults to ``root_node_id`` when left
    as ``None``.

    Use the helpers (``root``, ``node()``, ``input()``, ``is_alive()``)
    to resolve ids to ``Node`` objects through ``graph`` — they're the
    intended access pattern for rules that need graph-wide lookups.
    """

    graph: Graph
    root_node_id: str
    nodes: dict[str, str] = field(default_factory=dict)
    consumed: set[str] = field(default_factory=set)
    output: str | None = None
    # Snapshot of id(Node) at match time for every consumed node. The
    # ``is_alive`` check uses this to detect the case where an earlier
    # match in the same batch removed a consumed node and a different
    # node was added at the same id (e.g. splicer auto-rename hitting
    # a recently-freed name). Pure id-existence wouldn't catch that.
    _identities: dict[str, int] = field(default_factory=dict, repr=False)

    @property
    def root(self) -> Node:
        """The root ``Node`` (matched by the first ``Pattern`` entry)."""
        return self.graph.nodes[self.root_node_id]

    def node(self, name_or_id: str) -> Node:
        """Resolve a pattern name (e.g. ``"producer"``) OR a raw node id
        to the current ``Node`` in ``graph``. Raises ``KeyError`` if the
        node has been removed."""
        nid = self.nodes.get(name_or_id, name_or_id)
        return self.graph.nodes[nid]

    def input(self, i: int) -> Node | None:
        """Root's ``i``-th input as a ``Node``, or ``None`` when ``i``
        exceeds the input count or the input node was removed."""
        root = self.root
        if i >= len(root.inputs):
            return None
        return self.graph.nodes.get(root.inputs[i])

    def is_alive(self) -> bool:
        """``True`` when every consumed node still resolves to the same
        ``Node`` object captured at match time. Catches both removal and
        the "removed-then-re-added under same id" case."""
        for nid in self.consumed:
            n = self.graph.nodes.get(nid)
            if n is None or id(n) != self._identities.get(nid):
                return False
        return True


# ---------------------------------------------------------------------------
# Autotune surface: Candidate, TraceEntry, Search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceEntry:
    """One rule application in a candidate's history. ``choice_idx`` is
    the option picked at fork points (0 for deterministic single-option
    rules)."""

    rule_name: str
    choice_idx: int = 0


@dataclass
class Cursor:
    """Pipeline resume state for a ``Candidate``.

    * ``pass_idx`` — index of the pass to apply next.
    * ``rule_idx`` — index of the rule within the current pass to try
      next.
    * ``n_applied`` — number of functional rewrites in the current
      pass scan. When ``rule_idx`` wraps past the last rule with this
      counter ``> 0``, the engine restarts the scan (changes happened);
      with the counter ``== 0``, the engine advances to the next pass."""

    pass_idx: int = 0
    rule_idx: int = 0
    n_applied: int = 0

    def advance(
        self,
        result: RuleResult,
        n_rules: int,
        on_pass_finish: Callable[[int], None] | None = None,
    ) -> None:
        """Drive the cursor forward by one rule attempt.

        First, update ``n_applied`` (functional fires drive end-of-scan
        restart logic) and ``rule_idx`` (only advanced when no
        functional fire happened — in-place rebinds and zero-fire
        batches stay on the same graph state, so re-scanning the rule
        would loop or no-op). Then, when ``rule_idx`` reaches
        ``n_rules``, transition: restart from rule 0 if functional
        rewrites accumulated this scan, otherwise invoke ``on_pass_end``
        with the just-finished ``pass_idx`` and advance to the next
        pass."""
        self.n_applied += result.n_functional
        if result.n_functional == 0:
            self.rule_idx += 1
        if self.rule_idx < n_rules:
            return
        finished = self.n_applied == 0
        self.rule_idx = 0
        self.n_applied = 0
        if finished:
            if on_pass_finish is not None:
                on_pass_finish(self.pass_idx)
            self.pass_idx += 1

    def fork(self, n_applied_delta: int) -> Cursor:
        """A copy at the same ``(pass_idx, rule_idx)`` with ``n_applied``
        shifted by ``n_applied_delta`` — used when spawning autotune
        alternatives mid-batch."""
        return Cursor(self.pass_idx, self.rule_idx, self.n_applied + n_applied_delta)


@dataclass
class RuleResult:
    """Outcome of one ``_try_one_rule`` call.

    * ``forks`` — alternative candidates spawned at autotune fork points
      (empty for deterministic rules).
    * ``n_functional`` — count of ``Graph`` (functional) rewrites applied
      to the candidate's own graph in this batch.
    * ``n_inplace`` — count of ``Op`` (in-place rebind) rewrites applied
      to the candidate's own graph in this batch."""

    forks: list[Candidate] = field(default_factory=list)
    n_functional: int = 0
    n_inplace: int = 0

    @property
    def fired(self) -> bool:
        return (self.n_functional + self.n_inplace) > 0


@dataclass
class Candidate:
    """A single point in the search space. The engine pops a candidate,
    advances it by one rule application attempt, pushes the resulting
    successor(s) back onto the search queue, and yields the candidate
    when ``cursor.pass_idx`` reaches the end of the pipeline.

    ``graph`` is owned by this candidate (deep-copied on multi-option
    forks; mutated in place on single-option steps). ``ctx`` is shared
    by reference. ``trace`` is the immutable history of rule
    applications on this branch. ``cursor`` is the pipeline cursor."""

    graph: Graph
    ctx: Context
    trace: tuple[TraceEntry, ...] = ()
    cursor: Cursor = field(default_factory=Cursor)


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
        ``tip_key``. Falls back to ``0.0`` if no candidate is in
        ``_by_tip`` for this node (e.g. it's a non-frontier interior
        node that doesn't have a pending candidate sitting at it)."""
        cands = self._by_tip.get(tip_key, [])
        if not cands:
            return 0.0
        best = float("-inf")
        for c in cands:
            for n in c.graph.nodes.values():
                if op_cache_key(n.op) == tip_key:
                    s = n.op.score(c.ctx)
                    if s > best:
                        best = s
                    break
        return best if best != float("-inf") else 0.0

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


def match_pattern(graph: Graph, pattern: list[Pattern]) -> list[Match]:
    """Return every pattern match rooted at a topo-ordered node.

    Matches may overlap — e.g. both ``{A, B}`` and ``{B, C}`` for a
    two-node pattern. The rewriter breaks after the first successful
    ``rewrite`` per pass iteration, so overlap is only a candidate-
    enumeration concern.
    """
    results: list[Match] = []
    for nid in graph.topological_order():
        m = _match_at(graph, nid, pattern)
        if m is not None:
            results.append(m)
    return results


def _match_at(graph: Graph, start: str, pattern: list[Pattern]) -> Match | None:
    cursor: str | None = start
    nodes: dict[str, str] = {}
    consumed: set[str] = set()
    identities: dict[str, int] = {}
    for prod in pattern:
        if cursor is None:
            return None
        node = graph.nodes.get(cursor)
        if node is None or not isinstance(node.op, prod.op_type):
            return None
        if not _check_constraints(node, prod):
            return None
        nodes[prod.name] = cursor
        consumed.add(cursor)
        identities[cursor] = id(node)
        cursor = _sole_consumer(graph, cursor)
    return Match(graph=graph, root_node_id=start, nodes=nodes, consumed=consumed, _identities=identities)


def _check_constraints(node, prod: Pattern) -> bool:
    return all(str(getattr(node.op, k, None)) == str(v) for k, v in prod.constraints.items())


def _sole_consumer(graph: Graph, nid: str) -> str | None:
    consumers = graph.consumers(nid)
    return consumers[0] if len(consumers) == 1 else None


# ---------------------------------------------------------------------------
# Rule loading
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    """Loaded rule module — pattern + rewrite plus the rewrite's param list.

    ``param_names`` is captured at load time so the dispatcher can bind
    each rewrite param via signature inspection. The binding rules:

    - ``graph`` — the current ``Graph``
    - ``match`` — the full ``Match`` (escape hatch for advanced rewrites)
    - ``root`` — ``graph.nodes[match.root_node_id]`` (the matched ``Node``)
    - ``out`` — ``root.output`` (the produced ``Tensor``)
    - any ``Pattern.name`` declared in ``PATTERN`` — that pattern entry's
      matched ``Node``
    - anything else — bound positionally to the input ``Node`` at slot
      ``i`` (i.e. ``graph.nodes[root.inputs[i]]``) where ``i`` is the
      param's position among non-reserved / non-pattern params; ``None``
      when ``i ≥ len(root.inputs)`` or the source node was deleted.

    The "anything else" rule lets rewrites read input slots straight off
    the signature::

        def rewrite(inp_x, inp_w, inp_b, out):
            # inp_x = graph.nodes[root.inputs[0]]            (Node)
            # inp_w = graph.nodes[root.inputs[1]]            (Node)
            # inp_b = graph.nodes[root.inputs[2]] or None    (Node | None)
            # out   = root.output                            (Tensor)

    Rules that need ad-hoc graph-wide lookups take ``match`` and use
    ``match.graph`` / ``match.node(id)`` — there's no ``graph`` reserved
    kwarg.
    """

    name: str
    pattern: list[Pattern]
    rewrite: Callable[..., Graph | Op | None]
    param_names: tuple[str, ...]


def _load_rules(pass_dir: Path) -> list[_Rule]:
    rule_files = sorted(f for f in pass_dir.glob("*.py") if f.name != "__init__.py" and not f.name.startswith("_"))
    return [_load_rule(f) for f in rule_files]


def _load_rule(path: Path) -> _Rule:
    import sys

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load rule from {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any ``@dataclass`` defined in the rule
    # module can resolve its own module via ``sys.modules`` —
    # ``dataclasses._is_type`` looks up ``cls.__module__`` there to
    # check for ``KW_ONLY`` and raises ``AttributeError`` on a missing
    # entry.
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    pattern = getattr(module, "PATTERN", None)
    rewrite_fn = getattr(module, "rewrite", None)
    if pattern is None:
        raise ValueError(f"Rule {path} missing PATTERN")
    if rewrite_fn is None:
        raise ValueError(f"Rule {path} missing rewrite() function")
    param_names = tuple(inspect.signature(rewrite_fn).parameters.keys())
    return _Rule(name=path.stem, pattern=pattern, rewrite=rewrite_fn, param_names=param_names)


def _build_rewrite_kwargs(rule: _Rule, match: Match, ctx: Context | None) -> dict:
    """Bind each ``rewrite`` param to its source.

    Reserved-name params (``match`` / ``root`` / ``out`` / ``ctx``) and
    ``PATTERN``-name params bind by name; every remaining param binds
    positionally to ``root.inputs[i]`` (in declaration order, ``None``
    when the position exceeds the available inputs)."""
    pattern_names = {p.name for p in rule.pattern}
    root_node = match.root
    graph = match.graph
    kwargs: dict = {}
    input_slot = 0
    for pname in rule.param_names:
        if pname == "match":
            kwargs[pname] = match
        elif pname == "root":
            kwargs[pname] = root_node
        elif pname == "out":
            kwargs[pname] = root_node.output
        elif pname == "ctx":
            kwargs[pname] = ctx
        elif pname in pattern_names:
            kwargs[pname] = match.node(pname)
        else:
            if input_slot < len(root_node.inputs):
                kwargs[pname] = graph.nodes.get(root_node.inputs[input_slot])
            else:
                kwargs[pname] = None
            input_slot += 1
    return kwargs


def _try_rewrite(
    rule: _Rule,
    match: Match,
    ctx: Context | None,
    *,
    debug_on: bool,
    pass_name: str | None,
) -> list | None:
    """Run ``rule.rewrite`` against ``match`` and return its options.

    Returns ``None`` (caller should ``continue``) when the match is
    stale, the rule raises ``RuleSkipped``, or it returns no options.
    Otherwise returns a non-empty list of ``Op``/``Graph`` options."""
    if not match.is_alive():
        return None
    kwargs = _build_rewrite_kwargs(rule, match, ctx)
    try:
        result = rule.rewrite(**kwargs)
    except RuleSkipped as exc:
        if debug_on:
            emit(format_skipped(display_name(pass_name, rule.name), match.root_node_id, exc.reason))
        return None
    options = list(result) if isinstance(result, (list, tuple)) else [result]
    return options or None


# ---------------------------------------------------------------------------
# Rewrite loop
# ---------------------------------------------------------------------------


def run_pass(
    graph: Graph,
    pass_dir: Path,
    dump: CompilerDump | None = None,
    pass_idx: int | None = None,
    pass_name: str | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
) -> Graph:
    """Load all rule modules in ``pass_dir`` and apply them to fixed
    point. ``select``, if given, restricts the run to rules whose name
    (with or without the numeric ordering prefix, e.g. ``tileify`` or
    ``001_tileify``) appears in the iterable — useful for isolating a
    single rule's behavior in tests.

    Single-graph helper: use ``run_pipeline`` for autotuning. Discards
    any fork ``Candidate``s a multi-option rule might want to spawn."""
    if ctx is None:
        ctx = Context.probe()
    rules = _filter_rules(_load_rules(pass_dir), set(select) if select is not None else None)
    search = GreedySearch()
    search.push(Candidate(graph=graph, ctx=ctx))
    return next(_search_loop(search, [rules], [pass_name or ""], ctx, dump)).graph


def run_rule(graph: Graph, rule_path: Path, ctx: Context | None = None) -> Graph:
    """Load a single rule module and apply it to fixed point. Discards
    fork siblings — for autotuning use the full ``run_pipeline`` driver."""
    if ctx is None:
        ctx = Context.probe()
    search = GreedySearch()
    search.push(Candidate(graph=graph, ctx=ctx))
    return next(_search_loop(search, [[_load_rule(rule_path)]], [""], ctx, None)).graph


def _search_loop(
    search: Search,
    rules_per_pass: list[list[_Rule]],
    pass_names: list[str],
    ctx: Context | None,
    dump: CompilerDump | None,
) -> Iterator[Candidate]:
    """The unified search-driven driver. Each iteration: pop a
    candidate, try one rule application (or end-of-pass bookkeeping),
    push successor(s). Yields when a candidate reaches the end of the
    pipeline (``cursor.pass_idx >= len(pass_names)``).

    Used by every engine entry point — ``run_autotune`` (full pipeline),
    ``run_pass`` (one pass), ``run_rule`` (one rule). They differ only
    in the rules-per-pass list and the ``Search`` instance supplied."""
    cache: TuningCache | None = getattr(search, "cache", None)
    while (cand := search.pop()) is not None:
        cur = cand.cursor
        if cur.pass_idx >= len(pass_names):
            yield cand
            continue
        rules = rules_per_pass[cur.pass_idx]
        # Empty pass (e.g. all rules filtered out): nothing to do, skip.
        if not rules:
            cur.pass_idx += 1
            search.push(cand)
            continue
        rule = rules[cur.rule_idx]
        pass_idx_arg = cur.pass_idx + 1 if pass_names[cur.pass_idx] else None
        pass_name_arg = pass_names[cur.pass_idx] or None
        result = _try_one_rule(cand, rule, ctx, dump, pass_idx_arg, pass_name_arg, cache=cache)

        def _on_pass_finish(idx: int) -> None:
            name = pass_names[idx]
            if name:
                logger.info("compile: %-18s done (%d nodes)", name, len(cand.graph.nodes))
            if dump is not None and name:
                dump.on_pass(idx + 1, name, cand.graph)

        cur.advance(result, n_rules=len(rules), on_pass_finish=_on_pass_finish)
        # Forks first, then ``cand`` last — LIFO ``Search`` pops ``cand``
        # next, driving the inline branch deep before backtracking.
        for fork in result.forks:
            search.push(fork)
        search.push(cand)


def _apply_one(graph: Graph, match: Match, result: Graph | Op, *, rule_name: str) -> Graph:
    """Apply one rewrite outcome to ``graph``. ``Op`` rebinds
    ``root.op`` in place (id, inputs, hints kept); ``Graph`` is a
    fragment spliced via ``Graph.splice``. Returns the (possibly
    same, possibly new) graph.

    On the 1:1 ``Op`` path the engine stamps ``result.source`` with the
    op being replaced (unless the rule already set it). This threads
    the rewrite chain through every in-place rebind for free — lowering
    rules don't need to repeat ``source=root.op`` in every constructor
    call. From a fully lowered ``CudaOp``, ``cuda.source.source.source``
    walks back to the originating ``LoopOp``.
    """
    if isinstance(result, Op):
        old_op = graph.nodes[match.root_node_id].op
        if result is not old_op and result.source is None:
            result.source = old_op
            # Merge predecessor knobs forward; rule-set knobs win on key
            # collision. Build a fresh dict so we don't accidentally
            # mutate the predecessor's metadata.
            result.knobs = {**old_op.knobs, **result.knobs}
        graph.nodes[match.root_node_id].op = result
        return graph
    assert isinstance(result, Graph), f"rule {rule_name} returned {type(result).__name__}; expected Graph, Op, list, or RuleSkipped"
    graph.splice(result, consumed=match.consumed, output=match.output or match.root_node_id)
    return graph


def _try_one_rule(
    cand: Candidate,
    rule: _Rule,
    ctx: Context | None,
    dump: CompilerDump | None,
    pass_idx: int | None,
    pass_name: str | None,
    *,
    cache: TuningCache | None = None,
) -> RuleResult:
    """One iteration: enumerate ``rule``'s matches once and apply each
    live match (with non-skipped rewrite) in batch. Match enumeration
    happens ONCE per call — staged matches that get invalidated by an
    earlier application in the batch are filtered via ``is_alive()``
    rather than re-walking the graph. Per-rule batch semantics are what
    downstream rules (lift / fusion / staging) depend on for
    deterministic structure.

    Mutates ``cand.graph`` and ``cand.trace`` in place with each match's
    option-0 application. Does NOT touch ``cand.cursor`` — the caller
    feeds the returned ``RuleResult`` to ``Cursor.advance`` to drive
    the cursor transition. Each fork carries a deep-copied graph with
    its alt applied at the fork point and a cursor at the same
    ``(pass_idx, rule_idx)`` as ``cand``."""
    debug_on = logger.isEnabledFor(logging.DEBUG)
    dump_on = dump is not None and pass_idx is not None and pass_name is not None
    need_text = debug_on or dump_on

    matches = match_pattern(cand.graph, rule.pattern)
    result = RuleResult()
    context_key = cand.ctx.structural_key() if cache is not None else None
    for match in matches:
        options = _try_rewrite(rule, match, ctx, debug_on=debug_on, pass_name=pass_name)
        if options is None:
            continue
        # Drop options that fail their own validity check (e.g. TileOp
        # variants whose post-register-tile launch would exceed 1024
        # threads). Saves the engine from deep-copying and pushing a
        # candidate that the backend will only fail on. Non-Op options
        # (Graph fragments) skip the check; their structure is opaque
        # at this layer.
        options = [o for o in options if not isinstance(o, Op) or o.validate(ctx)]
        if not options:
            continue
        chosen = options[0]
        # Record the (parent_key → child_keys) expansion in the autotune
        # tree before applying anything. Only fires when every option is
        # an ``Op`` with a derivable ``op_cache_key`` — Graph-returning
        # rewrites (decomposition / fusion) don't have a single post-op
        # to key on, so they don't contribute tree edges. Single-option
        # rules still record an edge (n_new=1, delta=0); the chain in
        # the cache mirrors the source chain on the resulting kernels.
        if cache is not None and context_key is not None:
            parent_key = op_cache_key(cand.graph.nodes[match.root_node_id].op)
            if parent_key is not None and all(isinstance(o, Op) for o in options):
                child_keys = [op_cache_key(o) for o in options]
                if all(k is not None for k in child_keys):
                    cache.expand(context_key, parent_key, child_keys)
        fragment = _wrap_op_as_fragment(cand.graph, match.root_node_id, chosen) if isinstance(chosen, Op) else chosen
        text = _format_rule_application(rule.name, cand.graph, match, fragment, pass_name=pass_name) if need_text else None
        if debug_on:
            emit(text)
        if dump_on:
            record = _record_rule_application(cand.graph, match, fragment)
            dump.on_rule(pass_idx, pass_name, rule.name, record, text)
        # Fork branches: each alt gets a deep-copy of cand.graph at
        # this point in the batch (after prior matches' option-0
        # applications, before this match's). Fork cursors carry the
        # batch's running ``n_functional`` so far plus this alt's
        # contribution — when popped, the fork re-enters the same rule
        # batch from a fresh match enumeration on its alt graph.
        if len(options) > 1:
            snapshot = copy.deepcopy(cand.graph)
            for alt_idx, alt in enumerate(options[1:], start=1):
                fg = copy.deepcopy(snapshot)
                fm = _remap_match_to(fg, match)
                fg = _apply_one(fg, fm, alt, rule_name=rule.name)
                alt_delta = result.n_functional + (0 if isinstance(alt, Op) else 1)
                result.forks.append(
                    Candidate(
                        graph=fg,
                        ctx=cand.ctx,
                        trace=(*cand.trace, TraceEntry(rule.name, alt_idx)),
                        cursor=cand.cursor.fork(alt_delta),
                    )
                )
        cand.graph = _apply_one(cand.graph, match, chosen, rule_name=rule.name)
        cand.trace = (*cand.trace, TraceEntry(rule.name, 0))
        # Functional (Graph) fires drive the end-of-scan restart; in-place
        # (Op) rebinds don't, since the same pattern won't re-match the
        # mutated op (or the rule's idempotence guard handles it).
        if isinstance(chosen, Op):
            result.n_inplace += 1
        else:
            result.n_functional += 1
    return result


def _remap_match_to(forked_graph: Graph, match: Match) -> Match:
    """Build a fresh ``Match`` against ``forked_graph`` (a deep copy)
    that mirrors ``match``'s ids. Re-snapshot ``_identities`` against
    the forked nodes so the new match's ``is_alive`` check works."""
    identities = {nid: id(forked_graph.nodes[nid]) for nid in match.consumed if nid in forked_graph.nodes}
    return Match(
        graph=forked_graph,
        root_node_id=match.root_node_id,
        nodes=dict(match.nodes),
        consumed=set(match.consumed),
        output=match.output,
        _identities=identities,
    )


# ---------------------------------------------------------------------------
# Per-rule snapshot formatting (used at DEBUG, i.e. ``compile -vv``)
# ---------------------------------------------------------------------------


def _format_rule_application(name: str, graph: Graph, match: Match, fragment: Graph, *, pass_name: str | None = None) -> str:
    """Render a one-rule-application snapshot as a unified diff bracketed
    by ``>>> name`` / ``<<< name`` markers (see ``rule_diff``). Kernel
    ops (LoopOp/TileOp/KernelOp/CudaOp) are pretty-printed via their
    dedicated printers rather than dumped as a body repr."""
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    matched_nodes = [graph.nodes[nid] for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes]
    before = _format_nodes(matched_nodes, graph)
    frag_nodes = [fragment.nodes[nid] for nid in fragment.topological_order()]
    after = _format_nodes(frag_nodes, fragment)
    return render_rule_diff(display_name(pass_name, name), before, after, header=f"matched at {match.root_node_id}")


def _wrap_op_as_fragment(graph: Graph, root_id: str, new_op: Op) -> Graph:
    """Build a single-node fragment that mirrors ``graph.nodes[root_id]``
    with ``new_op`` substituted. Lets the engine render an in-place op
    rebind through the same diff/dump path as a functional fragment splice
    (the engine then assigns ``root.op = new_op`` directly, bypassing the
    splicer — node id, inputs list, hints, and output Tensor are kept)."""
    root = graph.nodes[root_id]
    frag = Graph()
    for inp_id in root.inputs:
        if inp_id in frag.nodes:
            continue
        inp = graph.nodes.get(inp_id)
        shape = inp.output.shape if inp is not None else ()
        dtype = inp.output.dtype if inp is not None else "f32"
        frag.add_node(InputOp(), [], Tensor(inp_id, shape, dtype), node_id=inp_id)
    out_id = frag.add_node(new_op, list(root.inputs), root.output, node_id=root.id)
    frag.outputs = [out_id]
    return frag


def _record_rule_application(graph: Graph, match: Match, fragment: Graph) -> dict:
    """Structured analog of ``_format_rule_application`` for JSON dumps.

    Captures the matched-subgraph nodes and the fragment's nodes as plain
    dicts so post-hoc scripts (and the article-side analysis) can iterate
    rule applications without re-parsing the text snapshot.
    """
    matched_ids: set[str] = set(match.consumed) | set(match.nodes.values())
    matched_ids.add(match.root_node_id)
    return {
        "root": match.root_node_id,
        "matched_pattern_nodes": dict(match.nodes),
        "before": [_node_to_dict(graph.nodes[nid]) for nid in graph.topological_order() if nid in matched_ids and nid in graph.nodes],
        "after": [_node_to_dict(fragment.nodes[nid]) for nid in fragment.topological_order()],
    }


def _node_to_dict(node) -> dict:
    return {
        "id": node.output.name,
        "op_class": type(node.op).__name__,
        "inputs": list(node.inputs),
        "output_shape": list(node.output.shape),
        "output_dtype": node.output.dtype,
    }


def _format_nodes(nodes: list, graph: Graph) -> str:
    """Render a list of nodes as readable text. Kernel-IR ops use their
    own ``pretty_body``; everything else falls back to a ``name: ClsName(args)``
    one-liner. Scalar ``ConstantOp`` inputs are inlined as literals (same
    treatment as ``format_kernels`` — see ``_inline_scalar_loads``).

    The leading ``kernel <name>  inputs: ...  outputs: ...`` header that
    ``TileOp.pretty_body`` prepends is stripped here: this path already
    emits ``<output> = TileOp(<inputs>)`` one line above, so the kernel
    header would just duplicate the same info and shift the body's
    indent by 4 spaces, ballooning the diff."""
    lines: list[str] = []
    for node in nodes:
        op = node.op
        if isinstance(op, (InputOp, ConstantOp)):
            continue
        body = op.pretty_body()
        if body is None:
            lines.append(f"{node.output.name} = {_fmt_op(node, graph)}")
            continue
        arg_names = [graph.nodes[inp].output.name for inp in node.inputs if inp in graph.nodes]
        lines.append(f"{node.output.name} = {type(op).__name__}({', '.join(arg_names)})")
        scalar_inputs = _scalar_constant_inputs(graph, node, ConstantOp)
        if scalar_inputs:
            body = _inline_scalar_loads(body, scalar_inputs)
        body_lines = body.splitlines()
        if body_lines and body_lines[0].lstrip().startswith("kernel ") and " inputs: " in body_lines[0] and " outputs: " in body_lines[0]:
            body_lines = [_dedent(ln, 4) for ln in body_lines[1:]]
        lines.extend(f"  {line}" for line in body_lines)
    return "\n".join(lines)


def _dedent(line: str, n: int) -> str:
    """Strip up to ``n`` leading spaces from ``line``."""
    i = 0
    while i < n and i < len(line) and line[i] == " ":
        i += 1
    return line[i:]


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    graph: Graph,
    passes: list[str],
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
    backend=None,
    cache: TuningCache | None = None,
) -> Graph:
    """Run each named pass directory in order; dispatch ``dump.on_pass``
    after each. Single-candidate convenience wrapper around
    :func:`run_autotune` using :class:`GreedySearch` — stops at the
    first terminal so autotune forks beyond option 0 are never explored.

    ``ctx`` is built once (probing the live device if not provided)
    and passed to every rule that takes a ``ctx`` parameter.

    ``backend`` (typically :class:`CudaBackend`) opts the run into real
    GPU measurement: every terminal graph's per-kernel latency is
    recorded to ``cache`` and attributed to every ancestor along the
    ``Op.source`` chain. ``cache`` defaults to a fresh in-memory store;
    pass an explicit :class:`TuningCache` to persist measurements
    across runs.

    For exhaustive autotuning, call :func:`run_autotune` directly with
    :class:`TuningSearch` and iterate every yielded candidate."""
    search = GreedySearch(cache=cache)
    return next(run_autotune(graph, passes, search=search, dump=dump, select=select, ctx=ctx, backend=backend)).graph


def run_autotune(
    graph: Graph,
    passes: list[str],
    *,
    search: Search,
    dump: CompilerDump | None = None,
    select: Iterable[str] | None = None,
    ctx: Context | None = None,
    backend=None,
) -> Iterator[Candidate]:
    """Drive the autotune search. Yields one terminal ``Candidate`` per
    fully-explored branch. With deterministic rules (no list-returning
    rewrites) the search yields exactly one — same shape as
    ``run_pipeline``.

    The loop is fully search-driven: pop a candidate, advance it by one
    rule application via :func:`_try_step`, push successor(s) back to
    ``search``. When no rule fires in the current pass, advance the
    candidate's ``cursor.pass_idx`` and push it back. When ``cursor.pass_idx``
    reaches the end of ``passes``, the candidate is terminal and gets
    yielded.

    ``search`` chooses both the order and the stopping condition:
    :class:`GreedySearch` for single-shot compiles (stops at the first
    terminal); :class:`TuningSearch` for ``--tune`` (runs the queue
    dry, exploring every fork).

    When ``search`` exposes a ``cache: TuningCache`` (both built-in
    searches do), each yielded terminal candidate has its ``CudaOp``
    nodes recorded to the cache via :func:`record_terminal` before being
    yielded — so subsequent candidates see the updated priority signal.
    Pass a ``Backend`` (typically :class:`CudaBackend`) via ``backend=``
    to record real GPU-event latencies; omit it to record the stub
    ``latency_us=1.0``.

    ``ctx`` is built once (probing the live device if not provided)
    and shared by every candidate."""
    if ctx is None:
        ctx = Context.probe()
    select_set = set(select) if select is not None else None
    rules_per_pass = [_filter_rules(_load_rules(_PASSES_DIR / name), select_set) for name in passes]
    t_start = time.monotonic()

    search.push(Candidate(graph=graph, ctx=ctx))

    cache: TuningCache | None = getattr(search, "cache", None)
    n_terminals = 0
    for cand in _search_loop(search, rules_per_pass, passes, ctx, dump):
        n_terminals += 1
        if backend is not None:
            # Collect knobs from every terminal kernel in the graph so
            # the log line reflects the *actual* autotune choices that
            # produced this variant, not just the rule:choice indices.
            knob_strs: list[str] = []
            for nid in cand.graph.topological_order():
                op = cand.graph.nodes[nid].op
                k = getattr(op, "knobs", None) or {}
                if k:
                    knob_strs.append(", ".join(f"{kk}={vv}" for kk, vv in sorted(k.items())))
            label = " | ".join(knob_strs) if knob_strs else "option-0"
            logger.info("[tune] variant #%d  [%s]", n_terminals, label)
        if cache is not None:
            record_terminal(cand.graph, cache, cand.ctx.structural_key(), backend=backend)
        yield cand
    logger.info("compile: total %.2fs (%d terminal(s))", time.monotonic() - t_start, n_terminals)


def _filter_rules(rules: list[_Rule], select_set: set[str] | None) -> list[_Rule]:
    if select_set is None:
        return rules
    return [r for r in rules if r.name in select_set or _strip_rule_prefix(r.name) in select_set]
