"""``Search`` ABC shared by the concrete policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from deplodock.compiler.pipeline.search.candidate import Candidate


class Search(ABC):
    """Search-strategy hook. The engine pushes spawned candidates and
    pops the next one to expand. ``pop`` returning ``None`` ends the
    search. Implementations choose both the ordering (DFS / BFS /
    priority / MCTS / whatever) and the termination condition (greedy
    stops at first terminal; exhaustive runs the queue dry).

    The engine doesn't tell the search when a candidate is terminal —
    instead a terminal candidate is the one the engine yielded without
    pushing it back. Searches that need to detect this can track the
    last-popped candidate and check whether it returned via ``push``.

    ``push(c, *forks)`` carries the primary candidate ``c`` plus every
    sibling alternative the engine spawned at the same rewrite point.
    Exhaustive policies register all of them; greedy policies discard
    the forks and keep only ``c`` (option-0)."""

    @abstractmethod
    def push(self, c: Candidate, *forks: Candidate) -> None: ...

    @abstractmethod
    def pop(self) -> Candidate | None:  # None when exhausted
        ...
