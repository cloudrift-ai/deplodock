"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

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

    The rewrite site rides on ``Candidate.last_rewritten`` (set by the
    engine after every rule application), so the policy reads the new
    child op and its source parent directly without scanning the graph.

    No ``tree`` attribute — greedy never participates in the MCTS
    accounting, so :func:`record_terminal` skips the tree bump when it
    sees ``getattr(search, "tree", None) is None``."""

    def __init__(self, context_key: str | None = None, *, db: SearchDB | None = None) -> None:
        super().__init__(context_key, db=db)
        # parent_key → (child_key, knobs) for this compile session.
        self._choices: dict[str, tuple[str, dict]] = {}

    def push(self, c: Candidate, *forks: Candidate) -> None:
        if not forks:
            self._push(c)
            return
        self._push(self._select(c, forks))

    def _select(self, c: Candidate, forks: tuple[Candidate, ...]) -> Candidate:
        # Forks deep-copy the graph but preserve node IDs, so every
        # candidate in the group exposes the rule's new op at the same
        # ``last_rewritten`` node. Read it off ``c`` to recover the
        # parent (via ``op.source``) for the DB lookup.
        if not c.last_rewritten:
            return c
        nid = c.last_rewritten[0]
        new_op = c.graph.nodes[nid].op
        parent = new_op.source
        if parent is None:
            return c
        parent_key = op_cache_key(parent)
        if parent_key is None:
            return c
        if parent_key not in self._choices:
            row = self._db.lookup_lowering(parent_key)
            if row is None:
                return c
            # Look up the winning child's measured perf row under *this*
            # compile's context + backend. The DB partitions perf by
            # both, so a cross-target row (e.g. measured on sm_90 while
            # we compile for sm_80) won't return knobs here — we still
            # use the lowering edge for selection but skip the knob
            # carry-over since it wouldn't apply. Backend identity rides
            # on ``ctx.backend_name`` (stamped by ``run_autotune``).
            perf = self._db.lookup_perf(self._ckey(c), row.child_key, backend=c.ctx.backend_name)
            self._choices[parent_key] = (row.child_key, perf.knobs if perf else {})
        child_key, _knobs = self._choices[parent_key]
        for cand in (c, *forks):
            if op_cache_key(cand.graph.nodes[nid].op) == child_key:
                return cand
        # DB winner not present in this fork group (structural-key
        # collision, stale DB after a rule change, etc.) — fall back to
        # option-0 rather than failing.
        return c

    def pop(self) -> Candidate | None:
        return self._pop()
