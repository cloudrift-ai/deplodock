"""Compiler pipeline: rule-based rewriter, pass directories, dump hooks.

Layout:

- :mod:`.base` — ``run_pipeline(graph, passes, dump=None)`` entry point.
- :mod:`.rewriter` — pattern-matching rule engine (``run_pass``, ``run_rule``).
- :mod:`.matcher` — chain pattern matcher (``Pattern``, ``Match``, ``match_pattern``).
- :mod:`.dump` — ``CompilerDump`` artifact collector + ``on_pass`` dispatch.
- :mod:`.passes` — pass directories (``decomposition/``, ``optimization/``,
  ``fusion/``, ``lowering/``) whose rule modules (``NNN_<name>.py``) are
  picked up by ``run_pass``.
"""

from deplodock.compiler.pipeline.base import run_pipeline
from deplodock.compiler.pipeline.dump import CompilerDump
from deplodock.compiler.pipeline.matcher import Match, Pattern, match_pattern
from deplodock.compiler.pipeline.rewriter import run_pass, run_rule

__all__ = [
    "CompilerDump",
    "Match",
    "Pattern",
    "match_pattern",
    "run_pass",
    "run_pipeline",
    "run_rule",
]
