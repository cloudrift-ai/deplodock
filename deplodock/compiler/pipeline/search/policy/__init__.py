"""Search policies: the ``Search`` protocol plus the two concrete
strategies the autotune driver chooses between.

- :mod:`.base` — ``Search`` protocol.
- :mod:`.greedy` — :class:`GreedySearch` (single-shot compile).
- :mod:`.mcts` — :class:`TuningSearch` (MCTS/UCB1 sweep).
"""

from deplodock.compiler.pipeline.search.policy.base import Search
from deplodock.compiler.pipeline.search.policy.greedy import GreedySearch
from deplodock.compiler.pipeline.search.policy.mcts import TuningSearch

__all__ = ["GreedySearch", "Search", "TuningSearch"]
