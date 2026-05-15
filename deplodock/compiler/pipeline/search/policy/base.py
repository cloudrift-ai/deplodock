"""``Search`` ABC shared by the concrete policies.

- ``push`` enqueues spawned candidates (primary + forks);
- ``pop`` returns the next candidate to explore (``None`` ends the run);
- ``observe`` is called by :meth:`Pipeline.tune` once per yielded
  terminal, with the aggregated ``reward`` (``1/total_us``) and
  ``status`` from the bench. Default is a no-op; :class:`TuningSearch`
  overrides it to backprop on its MCTS tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats


class Search(ABC):
    @abstractmethod
    def push(self, primary: LazyCandidate, *forks: LazyCandidate, best: LazyCandidate | None = None) -> None:
        """Enqueue ``primary`` and the rest of the fork siblings.

        ``best`` (optional) carries the fork that matches the
        DB's best-known lowering for the rewrite site. Greedy uses it
        to prefer the post-tuning winner over the rule's default
        option-0; tuning ignores it (it explores every fork)."""

    @abstractmethod
    def pop(self) -> LazyCandidate | None: ...

    def observe(self, stats: PerfStats, status: str) -> None:  # noqa: B027
        """Hook for the policy to consume the terminal's measurement.
        ``stats`` aggregates the terminal's per-iter latencies in
        microseconds (median / mean / variance / etc.); ``status`` is
        ``"ok"`` or ``"bench_fail"``. Default no-op (greedy doesn't
        use it)."""
