"""``Search`` protocol + shared min-heap base for the concrete policies.

Both :class:`GreedySearch` and :class:`TuningSearch` share a
``count_unmeasured_ops``-keyed priority heap as a fallback / tiebreak
queue; ``_PriorityHeap`` factors that out so the policy modules only
add their selection logic on top."""

from __future__ import annotations

import heapq
from typing import Protocol

from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.recorder import count_unmeasured_ops
from deplodock.compiler.pipeline.search.tree import SearchTree


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
    decreasing ``_seq`` so on a fresh in-memory store the order is the
    same as a DFS stack.

    ``tree`` is the in-memory MCTS state (rebuilt per process); ``db``
    is the on-disk inventory + perf store (cross-run cache hits).
    Both default to fresh instances so callers can construct a search
    without wiring persistence."""

    def __init__(
        self,
        tree: SearchTree | None = None,
        context_key: str | None = None,
        *,
        db: SearchDB | None = None,
    ) -> None:
        if tree is None:
            tree = SearchTree()
        if db is None:
            db = SearchDB()
        self._tree = tree
        self._db = db
        self._context_key = context_key
        self._heap: list[tuple[int, int, Candidate]] = []
        self._seq = 0

    def _ckey(self, c: Candidate) -> str:
        return self._context_key if self._context_key is not None else c.ctx.structural_key()

    def _push(self, c: Candidate) -> None:
        n = count_unmeasured_ops(c.graph, self._db, self._ckey(c))
        self._seq += 1
        heapq.heappush(self._heap, (n, -self._seq, c))

    def _pop(self) -> Candidate | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    @property
    def tree(self) -> SearchTree:
        return self._tree

    @property
    def db(self) -> SearchDB:
        return self._db
