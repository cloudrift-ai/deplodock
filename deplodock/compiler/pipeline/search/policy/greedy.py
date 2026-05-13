"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.db import SearchDB
from deplodock.compiler.pipeline.search.keys import op_cache_key


class GreedySearch:
    """Stop at the first terminal candidate.

    ``push`` always enqueues exactly one candidate (either the primary
    ``c`` or the DB-preferred fork) and drops the rest, so only a
    single pending slot is ever needed. When the engine yields a
    terminal without pushing it back, the next ``pop`` finds the slot
    empty and returns ``None``, ending the search.

    Used by ``run_pipeline`` for single-shot compiles. At every fork
    point we consult the search DB's ``lowering`` table: the row
    carries the knob delta the winning rule application stamped on its
    child. Greedy picks the fork whose newly-stamped knobs agree with
    that delta. Without a DB hit we fall back to option-0 (the rule's
    heuristic ordering). Decisions are memoized in ``_choices`` so
    repeated parent keys in one compile don't hit the DB twice.

    Replay is hop-by-hop: each fork point reads its own row. Because
    structural keys compose (a post-blockify TileOp's key already
    encodes the BN/BM picked upstream), the lookup at register_tile is
    naturally scoped to "this particular BN/BM branch", and so on
    down the chain.

    No ``tree`` attribute — greedy never participates in the MCTS
    accounting, so :func:`record_terminal` skips the tree bump when it
    sees ``getattr(search, "tree", None) is None``."""

    def __init__(self, *, db: SearchDB | None = None) -> None:
        self._db = db if db is not None else SearchDB()
        # parent_key → winning knob delta for this compile session.
        self._choices: dict[str, dict] = {}
        self._pending: Candidate | None = None

    @property
    def db(self) -> SearchDB:
        return self._db

    def push(self, c: Candidate, *forks: Candidate) -> None:
        self._pending = self._select(c, forks) if forks else c

    def _select(self, c: Candidate, forks: tuple[Candidate, ...]) -> Candidate:
        # Forks deep-copy the graph but preserve node IDs, so every
        # candidate in the group exposes the rule's new op at the same
        # ``last_rewritten`` node. Read it off ``c`` to recover the
        # parent (via ``op.source``) for the DB lookup.
        if not c.last_rewritten:
            return c
        nid = c.last_rewritten[0]
        parent = c.graph.nodes[nid].op.source
        if parent is None:
            return c
        parent_key = op_cache_key(parent)
        if parent_key is None:
            return c
        if parent_key not in self._choices:
            row = self._db.lookup_lowering(parent_key)
            if row is None:
                return c
            self._choices[parent_key] = dict(row.knobs)
        target_knobs = self._choices[parent_key]
        if not target_knobs:
            # Deterministic hop (no knobs stamped) — option-0 is correct.
            return c
        # Pick the fork whose newly-stamped knobs match the recorded
        # delta. The fork's op carries cumulative knobs from upstream
        # hops too; checking ``target ⊆ op.knobs`` is enough since the
        # delta is exactly what this step adds.
        for cand in (c, *forks):
            new_knobs = cand.graph.nodes[nid].op.knobs
            if all(new_knobs.get(k) == v for k, v in target_knobs.items()):
                return cand
        # DB winner not present in this fork group (structural-key
        # collision, stale DB after a rule change, etc.) — fall back to
        # option-0 rather than failing.
        return c

    def pop(self) -> Candidate | None:
        c, self._pending = self._pending, None
        return c
