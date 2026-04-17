"""Compiler pipeline entry points.

Lowers a traced ``Graph`` to a ``LoopProgram`` (the post-fusion program
form) via three rewriter stages:

    1. **Decomposition** — rewrites high-level ops to primitives.
    2. **Optimization** — canonicalizes primitive graph.
    3. **Fusion** — assembles primitives into ``LoopOp`` nodes.

The resulting ``LoopProgram`` is the single input to backend codegen
(``backend/cuda/emit.compile_kernels``).
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.program.loop import LoopProgram
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent / "rules"


def compile_graph(graph: Graph, name: str = "prog") -> LoopProgram:
    """Lower a traced ``Graph`` to a ``LoopProgram``.

    The returned program is authoritative for buffer shapes and launch
    order; downstream codegen reads shapes from it and never recomputes
    them.
    """
    rewriter_pre = Rewriter.from_directory(_RULES_DIR, pass_order=["decomposition", "optimization"])
    graph = rewriter_pre.apply(graph)

    rewriter_fusion = Rewriter.from_directory(_RULES_DIR, pass_order=["fusion"])
    graph = rewriter_fusion.apply(graph)

    return LoopProgram.from_graph(graph, name=name)
