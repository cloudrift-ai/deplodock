"""Compiler pipeline: rewrite engine + pass directories + dump hooks.

- :mod:`.engine` — pattern matcher + rule runner + ``run_pipeline``
  entry point. Exports ``Pattern``, ``Match``, ``match_pattern``,
  ``run_rule``, ``run_pass``, ``run_pipeline``.
- :mod:`.dump` — ``CompilerDump`` artifact collector + ``on_pass``
  dispatch that routes post-pass dumps by pass name.
- :mod:`.passes` — pass directories grouped by IR level:
  ``frontend/{decomposition,optimization}``, ``loop/{lifting,fusion}``,
  ``lowering/{tile,kernel,cuda}``. Each leaf contains ``NNN_<name>.py``
  rule modules picked up by ``run_pass``.
"""

from deplodock.compiler.pipeline.dump import CompilerDump
from deplodock.compiler.pipeline.engine import (
    Candidate,
    Match,
    MeasurementPrioritySearch,
    Pattern,
    RuleSkipped,
    Search,
    TraceEntry,
    match_pattern,
    run_autotune,
    run_pass,
    run_pipeline,
    run_rule,
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
    "KERNEL_PASSES",
    "LOOP_PASSES",
    "Match",
    "MeasurementPrioritySearch",
    "Pattern",
    "RuleSkipped",
    "Search",
    "TENSOR_PASSES",
    "TILE_PASSES",
    "TraceEntry",
    "match_pattern",
    "run_autotune",
    "run_pass",
    "run_pipeline",
    "run_rule",
]
