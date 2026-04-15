"""Test helper: a thin ``auto_fuse`` that runs the rule-based fusion pass.

Legacy test code imported ``auto_fuse`` from ``deplodock.compiler.fusion``.
Production code no longer needs that shim (fusion is the last pass in
``Rewriter``'s pipeline, so production callers get it via
``Rewriter.from_directory``). This helper preserves the old one-liner
shape for tests that construct a primitive graph and want just the
fusion step applied.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.rewriter import Pass

_RULES_DIR = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion"


def auto_fuse(graph: Graph) -> Graph:
    """Run the fusion pass to fixed point on ``graph``."""
    return Pass.from_directory(_RULES_DIR).apply(graph)
