"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.ir.base import Op
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key
from deplodock.compiler.pipeline.search.policy.base import _PriorityHeap


class GreedySearch(_PriorityHeap):
    """Stop at the first terminal candidate.

    ``push`` always enqueues exactly one candidate (either the primary
    ``c`` or the DB-preferred fork) and drops the rest, so the heap
    holds at most one item at any time. When the engine yields a
    terminal without pushing it back, the next ``pop`` finds the heap
    empty and returns ``None``, ending the search — no explicit
    "last-popped" tracking needed.

    Used by ``run_pipeline`` for single-shot compiles. At every fork
    point we consult the search DB's ``lowering`` table: if a
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
        # parent_key → (child_key, knobs) for this compile session.
        self._choices: dict[str, tuple[str, dict]] = {}

    def push(self, c: Candidate, *forks: Candidate, parent: Op | None = None) -> None:
        if not forks or parent is None:
            self._push(c)
            return
        self._push(self._select(c, forks, parent))

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
        return self._pop()
