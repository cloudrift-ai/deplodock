"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.ir.base import Op
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.policy.base import _PriorityHeap


class GreedySearch(_PriorityHeap):
    """Stop at the first terminal candidate.

    The engine yields a terminal candidate without pushing it back. We
    detect that by tracking the last-popped candidate: if the next
    ``pop`` sees that nothing has been ``push``-ed since (the candidate
    didn't return for another rule application), the previous candidate
    must have been terminal — return ``None`` to end the search even if
    the heap still holds unexplored forks.

    Used by ``run_pipeline`` for single-shot compiles. At every
    fork point we consult the search DB's ``lowering`` table: if a
    previously-tuned winner exists for the parent op's structural key,
    we push that fork instead of option-0. Decisions are memoized in
    ``_choices`` so the same parent key encountered repeatedly in one
    compile doesn't hit the DB again. Without a DB hit we fall back to
    option-0 (the rule's heuristic ordering) — same behavior as before.

    No ``tree`` attribute — greedy never participates in the MCTS
    accounting, so :func:`record_terminal` skips the tree bump when it
    sees ``getattr(search, "tree", None) is None``."""

    def __init__(self, context_key: str | None = None, *, db: SearchDB | None = None) -> None:
        super().__init__(context_key, db=db)
        self._outstanding: Candidate | None = None
        # parent_key → (child_key, knobs) for this compile session.
        self._choices: dict[str, tuple[str, dict]] = {}

    def push(self, c: Candidate, *forks: Candidate, parent: Op | None = None) -> None:
        if c is self._outstanding:
            self._outstanding = None
        if not forks or parent is None:
            self._push(c)
            return
        winner = self._select(c, forks, parent)
        self._push(winner)

    def _select(self, c: Candidate, forks: tuple[Candidate, ...], parent: Op) -> Candidate:
        parent_key = op_cache_key(parent)
        if parent_key is None:
            return c
        if parent_key not in self._choices:
            row = self._db.lookup_lowering(parent_key)
            if row is None:
                return c
            perf = self._db.lookup_perf_any(row.child_key)
            self._choices[parent_key] = (row.child_key, perf.knobs if perf else {})
        child_key, _knobs = self._choices[parent_key]
        # Identify the fork whose newly-introduced op points back to the
        # rewritten parent (by ``op_cache_key`` since forks deep-copy the
        # graph, so a Python ``is`` against ``parent`` only matches the
        # primary candidate) and matches the DB-preferred child key.
        # ``_apply_one`` stamps ``op.source`` on every rebind so the
        # structural lookup is reliable.
        for cand in (c, *forks):
            for node in cand.graph.nodes.values():
                src = node.op.source
                if src is None:
                    continue
                if op_cache_key(src) == parent_key and op_cache_key(node.op) == child_key:
                    return cand
        # DB winner not present in this fork group (structural-key
        # collision, stale DB after a rule change, etc.) — fall back to
        # option-0 rather than failing.
        return c

    def pop(self) -> Candidate | None:
        if self._outstanding is not None:
            # Last popped never came back via ``push`` → it was terminal.
            return None
        c = self._pop()
        self._outstanding = c
        return c
