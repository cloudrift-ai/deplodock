"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.policy.base import Search


class GreedySearch(Search):
    """Stop at the first terminal candidate.

    ``push`` keeps exactly one candidate per fork point and drops the
    rest, so only a single pending slot is ever needed. When the engine
    yields a terminal without pushing it back, the next ``pop`` finds
    the slot empty and returns ``None``, ending the search.

    Used by ``run_pipeline`` for single-shot compiles. At fork points
    the engine looks up the lowering table and passes the DB-best fork
    via ``push(..., best=...)``; greedy takes it WITHOUT scoring
    anything — a fully tuned compile does near-zero ranking work. When
    no DB entry exists (untuned site), greedy keeps the max-prior
    sibling via :meth:`Search.score_of` (ties keep emission order, so
    rules whose options carry no scores still get their option-0
    default)."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None

    def push(self, *cands: LazyCandidate, parent: object | None = None, best: LazyCandidate | None = None) -> None:
        del parent  # greedy keeps no lineage — one pending slot, no tree
        self._pending = best if best is not None else max(cands, key=lambda c: self.score_of(c.fork))

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None
