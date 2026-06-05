"""Search policies.

- :mod:`.base` — ``Search`` protocol.
- :mod:`.mcts` — :class:`TuningSearch`: PUCT over the learned prior — the only
  *exploration* policy (``tune``).
- :mod:`.greedy` — :class:`GreedySearch`: the O(1)-per-step single-shot
  ``Pipeline.run`` driver for ``compile`` / ``run`` (deterministic option-0;
  not exploration).
"""

from deplodock.compiler.pipeline.search.policy.base import Search
from deplodock.compiler.pipeline.search.policy.greedy import GreedySearch
from deplodock.compiler.pipeline.search.policy.mcts import TuningSearch

__all__ = ["GreedySearch", "Search", "TuningSearch"]
