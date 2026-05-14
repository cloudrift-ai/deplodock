"""Autotune search infrastructure: candidates, search policies, the
on-disk inventory + perf store, and the in-memory MCTS tree.

- :mod:`.candidate` — ``Candidate`` / ``Cursor`` data classes.
- :mod:`.policy` — ``Search`` ABC (``base``) plus :class:`GreedySearch`
  (``greedy``) / :class:`TuningSearch` (``mcts``) concrete strategies.
- :mod:`.db` — :class:`SearchDB` SQLite store (op inventory, lowering
  edges, generic ``perf`` table).
- :mod:`.policy.mcts` — :class:`TuningSearch` + the in-memory
  :class:`SearchTree` / :class:`NodeRow` (MCTS-only — no other policy
  reads or writes the tree).
- :mod:`.recorder` — top-level :func:`record_terminal` orchestrating
  bench → DB writes → optional tree updates.
- :mod:`.keys` — ``op_cache_key`` / ``dialect_of`` / ``source_chain``.

The ``run_pipeline`` / ``run_autotune`` entry points and the
``_search_loop`` driver live in :mod:`pipeline.engine` (re-exported
from :mod:`pipeline`).
"""

from deplodock.compiler.pipeline.search.candidate import Candidate, Cursor
from deplodock.compiler.pipeline.search.db import PerfRow, PerfStats, SearchDB
from deplodock.compiler.pipeline.search.keys import dialect_of, op_cache_key, source_chain
from deplodock.compiler.pipeline.search.policy import GreedySearch, Search, TuningSearch
from deplodock.compiler.pipeline.search.policy.mcts import SearchNode, SearchTree
from deplodock.compiler.pipeline.search.recorder import TuneAborted, count_unmeasured_ops, record_terminal

__all__ = [
    "Candidate",
    "Cursor",
    "GreedySearch",
    "SearchNode",
    "PerfRow",
    "PerfStats",
    "Search",
    "SearchDB",
    "SearchTree",
    "TuneAborted",
    "TuningSearch",
    "count_unmeasured_ops",
    "dialect_of",
    "op_cache_key",
    "record_terminal",
    "source_chain",
]
