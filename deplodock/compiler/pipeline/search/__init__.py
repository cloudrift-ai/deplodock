"""Autotune search infrastructure: candidates, search policies, the
driver loop, and the persistent measurement cache.

- :mod:`.candidate` — ``Candidate`` / ``Cursor`` / ``TraceEntry`` /
  ``RuleResult`` data classes.
- :mod:`.policy` — ``Search`` protocol plus :class:`GreedySearch` /
  :class:`TuningSearch` concrete strategies.
- :mod:`.driver` — ``run_pipeline`` / ``run_autotune`` entry points and
  the unified ``_search_loop`` driver.
- :mod:`.cache` — :class:`TuningCache` SQLite store + ``op_cache_key`` /
  ``record_terminal`` helpers.
"""

from deplodock.compiler.pipeline.search.cache import (
    CudaPerf,
    NodeRow,
    TuneAborted,
    TuningCache,
    count_unmeasured_ops,
    op_cache_key,
    record_terminal,
)
from deplodock.compiler.pipeline.search.candidate import Candidate, Cursor, RuleResult, TraceEntry
from deplodock.compiler.pipeline.search.driver import run_autotune, run_pipeline
from deplodock.compiler.pipeline.search.policy import GreedySearch, Search, TuningSearch

__all__ = [
    "Candidate",
    "CudaPerf",
    "Cursor",
    "GreedySearch",
    "NodeRow",
    "RuleResult",
    "Search",
    "TraceEntry",
    "TuneAborted",
    "TuningCache",
    "TuningSearch",
    "count_unmeasured_ops",
    "op_cache_key",
    "record_terminal",
    "run_autotune",
    "run_pipeline",
]
