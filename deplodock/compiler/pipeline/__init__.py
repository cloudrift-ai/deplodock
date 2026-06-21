"""Compiler pipeline: rewrite engine + pass directories + dump hooks.

- :mod:`.pipeline` — value types (``Pattern``, ``Match``, ``Rule``,
  ``RuleSkipped``, ``Pass``, ``Pipeline``), the per-run state object
  ``Run`` (ctx / search / db / backend / dump / rejections + the engine
  loop, ``Run.drive``), and the compile entry points
  (``Pipeline.build``, ``.run``, ``.tune``). Pattern matching goes
  through ``pipeline.match(graph, rule)``; tests can use
  ``Pipeline.from_pattern(...)`` for a one-rule shim. The same greedy /
  autotune semantics apply to every caller.
- :mod:`.search` — search policies (``Candidate`` / ``Search`` /
  ``TuningSearch``; ``greedy_decide`` is the deterministic ``Run.resolve``
  pick, not a policy) and persistent measurement
  cache. ``Candidate.apply`` owns the per-rule logging + dump hooks
  (reads ``cand.run.dump``).
- :mod:`.dump` — ``CompilerDump`` artifact collector + ``on_pass``
  dispatch that routes post-pass dumps by pass name.
- :mod:`.passes` — pass directories grouped by IR level:
  ``frontend/{decomposition,optimization}``, ``loop/{lifting,fusion}``,
  ``lowering/{tile,kernel,cuda}``. Each leaf contains ``NNN_<name>.py``
  rule modules picked up by ``Pass.load``.
"""

from deplodock.compiler.pipeline.dump import CompilerDump
from deplodock.compiler.pipeline.pipeline import (
    LoweringError,
    Match,
    Pass,
    Pattern,
    Pipeline,
    Rule,
    RuleSkipped,
    _strip_rule_prefix,
)
from deplodock.compiler.pipeline.search import (
    Candidate,
    Search,
    TuningSearch,
)

# Canonical pass lists, indexed by the --ir stage they produce. Backends
# and tests should reference these rather than re-listing pass names.
TENSOR_PASSES = ["frontend/decomposition", "frontend/optimization"]
LOOP_PASSES = [*TENSOR_PASSES, "loop/lifting", "loop/fusion", "loop/recognize", "loop/stamp"]
TILE_PASSES = [*LOOP_PASSES, "lowering/tile"]
KERNEL_PASSES = [*TILE_PASSES, "lowering/kernel"]
CUDA_PASSES = [*KERNEL_PASSES, "lowering/cuda"]

__all__ = [
    "CUDA_PASSES",
    "Candidate",
    "CompilerDump",
    "KERNEL_PASSES",
    "LOOP_PASSES",
    "LoweringError",
    "Match",
    "Pass",
    "Pattern",
    "Pipeline",
    "Rule",
    "RuleSkipped",
    "Search",
    "TENSOR_PASSES",
    "TILE_PASSES",
    "TuningSearch",
    "_strip_rule_prefix",
]
