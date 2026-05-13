"""Autotune search infrastructure: candidates, search policies, the
driver loop, the on-disk inventory + perf store, and the in-memory
MCTS tree.

- :mod:`.candidate` — ``Candidate`` / ``Cursor`` / ``TraceEntry`` /
  ``RuleResult`` data classes.
- :mod:`.policy` — ``Search`` protocol (``base``) plus :class:`GreedySearch`
  (``greedy``) / :class:`TuningSearch` (``mcts``) concrete strategies.
- :mod:`.driver` — ``run_pipeline`` / ``run_autotune`` entry points and
  the unified ``_search_loop`` driver.
- :mod:`.db` — :class:`SearchDB` SQLite store (op inventory, lowering
  edges, generic ``perf`` table).
- :mod:`.policy.mcts` — :class:`TuningSearch` + the in-memory
  :class:`SearchTree` / :class:`NodeRow` (MCTS-only — no other policy
  reads or writes the tree).
- :mod:`.recorder` — top-level :func:`record_terminal` orchestrating
  bench → DB writes → optional tree updates.
- :mod:`.keys` — ``op_cache_key`` / ``dialect_of`` / ``source_chain``.
"""

from deplodock.compiler.pipeline.search.candidate import Candidate, Cursor, RuleResult, TraceEntry
from deplodock.compiler.pipeline.search.db import PerfRow, PerfStats, SearchDB
from deplodock.compiler.pipeline.search.driver import run_autotune, run_pipeline
from deplodock.compiler.pipeline.search.keys import dialect_of, op_cache_key, source_chain
from deplodock.compiler.pipeline.search.policy import GreedySearch, Search, TuningSearch
from deplodock.compiler.pipeline.search.policy.mcts import NodeRow, SearchTree
from deplodock.compiler.pipeline.search.recorder import TuneAborted, count_unmeasured_ops, record_terminal

__all__ = [
    "Candidate",
    "Cursor",
    "GreedySearch",
    "NodeRow",
    "PerfRow",
    "PerfStats",
    "RuleResult",
    "Search",
    "SearchDB",
    "SearchTree",
    "TraceEntry",
    "TuneAborted",
    "TuningSearch",
    "count_unmeasured_ops",
    "dialect_of",
    "op_cache_key",
    "record_terminal",
    "run_autotune",
    "run_pipeline",
    "source_chain",
]
