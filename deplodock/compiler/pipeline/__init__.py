"""Compiler pipeline: rewrite engine + pass directories + dump hooks.

- :mod:`.engine` — pattern matcher + rule runner. Exports ``Pattern``,
  ``Match``, ``match_pattern``, ``run_pass``. To run a single rule
  module, call ``run_pass(rules_dir, select=[<rule_stem>])``.
- :mod:`.search` — autotune driver (``Candidate`` / ``Search`` /
  ``run_pipeline`` / ``run_autotune``) and persistent measurement cache.
- :mod:`.dump` — ``CompilerDump`` artifact collector + ``on_pass``
  dispatch that routes post-pass dumps by pass name.
- :mod:`.passes` — pass directories grouped by IR level:
  ``frontend/{decomposition,optimization}``, ``loop/{lifting,fusion}``,
  ``lowering/{tile,kernel,cuda}``. Each leaf contains ``NNN_<name>.py``
  rule modules picked up by ``run_pass``.
"""

from deplodock.compiler.pipeline.dump import CompilerDump
from deplodock.compiler.pipeline.engine import (
    Match,
    Pattern,
    RuleSkipped,
    match_pattern,
    run_pass,
)
from deplodock.compiler.pipeline.search import (
    Candidate,
    GreedySearch,
    Search,
    TraceEntry,
    TuningSearch,
    run_autotune,
    run_pipeline,
)

# Canonical pass lists, indexed by the --ir stage they produce. Backends
# and tests should reference these rather than re-listing pass names.
TENSOR_PASSES = ["frontend/decomposition", "frontend/optimization"]
LOOP_PASSES = [*TENSOR_PASSES, "loop/lifting", "loop/fusion"]
TILE_PASSES = [*LOOP_PASSES, "lowering/tile"]
KERNEL_PASSES = [*TILE_PASSES, "lowering/kernel"]
CUDA_PASSES = [*KERNEL_PASSES, "lowering/cuda"]

__all__ = [
    "CUDA_PASSES",
    "Candidate",
    "CompilerDump",
    "GreedySearch",
    "KERNEL_PASSES",
    "LOOP_PASSES",
    "Match",
    "Pattern",
    "RuleSkipped",
    "Search",
    "TENSOR_PASSES",
    "TILE_PASSES",
    "TraceEntry",
    "TuningSearch",
    "match_pattern",
    "run_autotune",
    "run_pass",
    "run_pipeline",
]
