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

    Used by ``run_pipeline`` for single-shot compiles. Selection by the
    DB-best fork (``_best_fork``) and the static ``score_of`` prior were
    both nuked, so greedy now keeps the **first** emitted sibling (the
    rule's option-0 default). Reconnecting greedy to the learned prior is
    deferred until the prior proves out in search."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: LazyCandidate | None = None

    def push(self, *cands: LazyCandidate, parent: object | None = None) -> None:
        del parent  # greedy keeps no lineage — one pending slot, no tree
        self._pending = cands[0] if cands else None

    def pop(self) -> tuple[object | None, LazyCandidate] | None:
        c, self._pending = self._pending, None
        return (None, c) if c is not None else None
