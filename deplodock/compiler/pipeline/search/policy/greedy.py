"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.policy.base import Search


class GreedySearch(Search):
    """Stop at the first terminal candidate.

    ``push`` always enqueues exactly one candidate (option-0 from the
    rule's heuristic ordering) and drops the rest, so only a single
    pending slot is ever needed. When the engine yields a terminal
    without pushing it back, the next ``pop`` finds the slot empty and
    returns ``None``, ending the search.

    Used by ``run_pipeline`` for single-shot compiles. At fork points
    the engine looks up the lowering table and passes the DB-best fork
    via ``push(..., best=...)``; greedy prefers it over the rule's
    default option-0. When no DB entry exists (untuned site), greedy
    falls back to ``primary``."""

    def __init__(self, *, db: SearchDB | None = None) -> None:
        self._db = db if db is not None else SearchDB()
        self._pending: LazyCandidate | None = None

    @property
    def db(self) -> SearchDB:
        return self._db

    def push(self, primary: LazyCandidate, *forks: LazyCandidate, best: LazyCandidate | None = None) -> None:
        del forks  # greedy keeps a single candidate per fork point
        self._pending = best if best is not None else primary

    def pop(self) -> LazyCandidate | None:
        c, self._pending = self._pending, None
        return c
