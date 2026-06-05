"""``Search`` ABC shared by the concrete policies.

- ``push`` enqueues spawned candidates (unranked fork siblings);
- ``pop`` returns the next candidate to explore (``None`` ends the run);
- ``score_of`` is the value-keyed scorer the policies rank with — the
  search owns the cache, the ``Fork`` carries the key + compute thunk;
- ``observe`` is called by :meth:`Pipeline.tune` once per yielded
  terminal, with the aggregated ``reward`` (``1/total_us``) and
  ``status`` from the bench. Default is a no-op; :class:`TuningSearch`
  overrides it to backprop on its MCTS tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable

from deplodock.compiler.pipeline.pipeline import Fork
from deplodock.compiler.pipeline.search.candidate import LazyCandidate
from deplodock.compiler.pipeline.search.db import PerfStats


class Search(ABC):
    def __init__(self, *, score_cache: dict[Hashable, float] | None = None) -> None:
        # Value-keyed variant-score cache, handed to every ``Fork.score``
        # compute. Owned by the search because scoring is ranking policy:
        # one cache per search run shares scores across every fork point
        # it ranks. The KEYING is the scorer's own (it knows its value
        # identity — the partition planner keys ``(ctx, merged knobs)``,
        # complete because the ``SID`` structural-identity knob rides the
        # dict), so structurally identical kernels (the same layer
        # repeated through a whole model) hit the same entries. Pass a
        # shared dict to pool entries across several searches (see
        # ``two_level.inner_reward``, which tunes per-op slices with one
        # fresh search each).
        self.score_cache: dict[Hashable, float] = score_cache if score_cache is not None else {}

    def score_of(self, fork: Fork | None) -> float:
        """The fork's planner prior — ``fork.score(self.score_cache)``;
        ``None`` (a no-pending wrapper) scores ``0.0``.

        This prior is the ONLY ranking signal for unresolved candidates:
        siblings at a fork point share their ``inner`` snapshot by
        reference, and policies only ever compare children of one parent,
        so any inner-graph term would be a constant offset within every
        comparison set. We deliberately do NOT fire ``fork.expand()`` to
        recover the option and re-score it: expanding a partition leaf
        runs the full body build + normalization, and the policies read
        this score for every unranked sibling — materializing every leaf
        just to rank it would defeat the lazy-planner design."""
        return fork.score(self.score_cache) if fork is not None else 0.0

    @abstractmethod
    def push(self, *cands: LazyCandidate, best: LazyCandidate | None = None) -> None:
        """Enqueue the spawned candidates — all siblings of one fork
        point, in rule-emission order (NOT ranked; ranking is the
        policy's job via :meth:`score_of`).

        ``best`` (optional) carries the fork that matches the
        DB's best-known lowering for the rewrite site. Greedy uses it
        to skip ranking entirely and take the post-tuning winner;
        tuning ignores it (it explores every fork)."""

    @abstractmethod
    def pop(self) -> LazyCandidate | None: ...

    def observe(self, stats: PerfStats, status: str) -> None:  # noqa: B027
        """Hook for the policy to consume the terminal's measurement.
        ``stats`` aggregates the terminal's per-iter latencies in
        microseconds (median / mean / variance / etc.); ``status`` is
        ``"ok"`` or ``"bench_fail"``. Default no-op (greedy doesn't
        use it)."""
