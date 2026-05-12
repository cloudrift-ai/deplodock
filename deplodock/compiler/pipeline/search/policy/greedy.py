"""Greedy single-shot search — stops at the first terminal candidate."""

from __future__ import annotations

from deplodock.compiler.pipeline.search.cache import TuningCache
from deplodock.compiler.pipeline.search.candidate import Candidate
from deplodock.compiler.pipeline.search.policy.base import _PriorityHeap


class GreedySearch(_PriorityHeap):
    """Stop at the first terminal candidate.

    The engine yields a terminal candidate without pushing it back. We
    detect that by tracking the last-popped candidate: if the next
    ``pop`` sees that nothing has been ``push``-ed since (the candidate
    didn't return for another rule application), the previous candidate
    must have been terminal — return ``None`` to end the search even if
    the heap still holds unexplored forks.

    Used by ``run_pipeline`` for single-shot compiles. Autotune forks
    beyond option 0 stay in the heap unmeasured."""

    def __init__(self, cache: TuningCache | None = None, context_key: str | None = None) -> None:
        super().__init__(cache, context_key)
        self._outstanding: Candidate | None = None

    def push(self, c: Candidate) -> None:
        if c is self._outstanding:
            self._outstanding = None
        self._push(c)

    def pop(self) -> Candidate | None:
        if self._outstanding is not None:
            # Last popped never came back via ``push`` → it was terminal.
            return None
        c = self._pop()
        self._outstanding = c
        return c
