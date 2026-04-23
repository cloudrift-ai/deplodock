"""Compiler pipeline: rewrite engine + pass directories + dump hooks.

- :mod:`.engine` — pattern matcher + rule runner + ``run_pipeline``
  entry point. Exports ``Pattern``, ``Match``, ``match_pattern``,
  ``run_rule``, ``run_pass``, ``run_pipeline``.
- :mod:`.dump` — ``CompilerDump`` artifact collector + ``on_pass``
  dispatch that routes post-pass dumps by pass name.
- :mod:`.passes` — pass directories (``decomposition/``, ``optimization/``,
  ``fusion/``, ``lowering/{kernel,cuda}/``); each contains ``NNN_<name>.py``
  rule modules picked up by ``run_pass``.
"""

from deplodock.compiler.pipeline.dump import CompilerDump
from deplodock.compiler.pipeline.engine import (
    Match,
    Pattern,
    match_pattern,
    run_pass,
    run_pipeline,
    run_rule,
)

__all__ = [
    "CompilerDump",
    "Match",
    "Pattern",
    "match_pattern",
    "run_pass",
    "run_pipeline",
    "run_rule",
]
