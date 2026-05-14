"""``Search`` ABC shared by the concrete policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from deplodock.compiler.pipeline.search.candidate import LazyCandidate


class Search(ABC):
    """Search-strategy hook. The engine pushes spawned candidates and
    pops the next one to expand. ``pop`` returning ``None`` ends the
    search. Implementations choose both the ordering (DFS / BFS /
    priority / MCTS / whatever) and the termination condition (greedy
    stops at first terminal; exhaustive runs the queue dry).

    Everything in the search layer is :class:`LazyCandidate` —
    concrete candidates from the engine arrive wrapped in a
    zero-chain ``LazyCandidate`` (``resolve`` returns the inner
    Candidate unchanged) so push/pop stays uniform across "the
    rollout's current cand" and "an autotune fork that hasn't been
    materialized yet".

    ``push(primary, *forks)`` carries the primary candidate plus
    every sibling spawned at the same rewrite point. Exhaustive
    policies register all of them; greedy keeps only ``primary``."""

    @abstractmethod
    def push(self, primary: LazyCandidate, *forks: LazyCandidate) -> None: ...

    @abstractmethod
    def pop(self) -> LazyCandidate | None:  # None when exhausted
        ...
