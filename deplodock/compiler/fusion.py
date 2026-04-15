"""Fusion shim: delegates to the rule-based fusion pass.

The legacy greedy edge-merger (`auto_fuse` + helpers in this file's prior
revision) is gone. Fusion is now exclusively rule-based — see
`rules/fusion/*.py` for the seed and absorb rules.

`auto_fuse` is preserved as a thin entry point because ~10 test files
import it directly. It loads the fusion pass and applies it to the given
graph; new code should use `Rewriter.from_directory(rules_dir)` directly.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.rewriter import Pass

_RULES_DIR = Path(__file__).parent / "rules" / "fusion"


def auto_fuse(graph: Graph) -> Graph:
    """Run the rule-based fusion pass to fixed point.

    Equivalent to `Pass.from_directory(rules/fusion).apply(graph)` —
    callers in test code keep working unchanged. New code should use
    `Rewriter.from_directory(...)` so decomposition + optimization +
    fusion compose properly.
    """
    return Pass.from_directory(_RULES_DIR).apply(graph)
