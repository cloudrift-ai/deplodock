"""Search policies.

- :mod:`.base` — ``Search`` protocol.
- :mod:`.mcts` — :class:`TuningSearch`: PUCT over the learned prior — the only
  *exploration* policy (``tune``).
- :mod:`.greedy` — :func:`greedy_decide`: the deterministic greedy pick for
  ``Pipeline.run`` / the structural pricing probes. Not a ``Search``: it is a
  ``Run.resolve`` decide factory (a deterministic resolution has no frontier
  to rank — see ``plans/resolve-trace-driver.md``).
"""

from emmy.compiler.pipeline.search.policy.base import Search
from emmy.compiler.pipeline.search.policy.greedy import greedy_decide
from emmy.compiler.pipeline.search.policy.mcts import TuningSearch

__all__ = ["Search", "TuningSearch", "greedy_decide"]
