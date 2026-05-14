"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.policy.base import Search


class GreedySearch(Search):
    """Stop at the first terminal candidate.

    ``push`` always enqueues exactly one candidate (option-0 from the
    rule's heuristic ordering) and drops the rest, so only a single
    pending slot is ever needed. When the engine yields a terminal
    without pushing it back, the next ``pop`` finds the slot empty and
    returns ``None``, ending the search.

    Used by ``run_pipeline`` for single-shot compiles. The DB is held
    only so callers (and ``record_terminal``) have a place to write
    measurements through; greedy itself never reads it — every fork
    point picks option 0 unconditionally. Hop-by-hop DB replay can be
    layered on later as a dedicated policy if/when the autotuner needs
    deterministic resume from prior runs.

    No ``tree`` attribute — greedy never participates in the MCTS
    accounting, so :func:`record_terminal` skips the tree bump when it
    sees ``getattr(search, "tree", None) is None``."""

    def __init__(self, *, db: SearchDB | None = None) -> None:
        self._db = db if db is not None else SearchDB()
        self._pending: Candidate | None = None

    @property
    def db(self) -> SearchDB:
        return self._db

    def push(self, c: Candidate, *forks: Candidate) -> None:
        del forks  # greedy always picks option-0 (``c``)
        self._pending = c

    def pop(self) -> Candidate | None:
        c, self._pending = self._pending, None
        return c
