"""Autotune search infrastructure: candidates, search policies, the
on-disk inventory + perf store, and the in-memory MCTS tree.

- :mod:`.candidate` — :class:`Candidate` / :class:`LazyCandidate` /
  :class:`Cursor` data classes.
- :mod:`.policy` — :class:`Search` ABC (``base``), :class:`TuningSearch`
  (``mcts``, PUCT — the exploration policy), and :func:`greedy_decide`
  (``greedy``, the deterministic ``Run.resolve`` pick for single-shot
  compiles — not a ``Search``).
- :mod:`.db` — :class:`SearchDB` SQLite store (op inventory, lowering
  edges, ``perf`` table).
- :mod:`.policy.mcts` — :class:`TuningSearch` + the in-memory
  :class:`SearchTree` / :class:`SearchNode` (MCTS-only — no other policy
  reads or writes the tree).
- :mod:`.keys` — ``op_cache_key`` / ``dialect_of`` / ``source_chain``.

The bench + DB write orchestration lives in
:func:`deplodock.compiler.pipeline.pipeline._bench_terminal`;
``Pipeline.tune`` calls it per yielded terminal and passes the
aggregate :class:`PerfStats` to :meth:`Search.observe` for the policy
to consume.
"""

from deplodock.compiler.pipeline.search.candidate import Candidate, Cursor, LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfRow, PerfStats, SearchDB
from deplodock.compiler.pipeline.search.keys import dialect_of, op_cache_key, source_chain
from deplodock.compiler.pipeline.search.policy import Search, TuningSearch, greedy_decide
from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree

__all__ = [
    "Candidate",
    "Cursor",
    "LazyCandidate",
    "SearchNode",
    "PerfRow",
    "PerfStats",
    "Search",
    "SearchDB",
    "SearchTree",
    "TuningSearch",
    "dialect_of",
    "greedy_decide",
    "op_cache_key",
    "source_chain",
]
