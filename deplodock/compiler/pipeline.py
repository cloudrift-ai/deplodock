"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of structural ``KernelOp``s:

    1. **Decomposition** — rewrites high-level ops to primitives.
    2. **Optimization** — canonicalizes primitive graph (e.g. merge IndexMaps).
    3. **Fusion** — assembles primitives into ``KernelOp`` nodes using the
       chain grammar.
    4. **Extraction** — collects ``KernelOp``s in topo order for backend codegen.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.lower import extract_kernels
from deplodock.compiler.ops import KernelOp
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent / "rules"


def compile_graph(graph: Graph) -> list[KernelOp]:
    """Lower a traced ``Graph`` to a list of ``KernelOp``.

    Runs decomposition → optimization → fusion (via the rewriter), then
    extracts the resulting ``KernelOp`` nodes.
    """
    rewriter = Rewriter.from_directory(_RULES_DIR)
    graph = rewriter.apply(graph)
    return extract_kernels(graph)
