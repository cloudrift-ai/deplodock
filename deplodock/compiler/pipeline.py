"""Compiler pipeline entry points.

Maps a traced ``Graph`` to a list of structural ``KernelOp``s ready for
backend codegen:

    1. **Decomposition** — rewrites high-level traced ops (``LinearOp``,
       ``MatmulOp``, ``SdpaOp``, ``MeanOp``, layout ops) into primitives
       (``ElementwiseOp``, ``ReduceOp``, ``IndexMapOp``).
    2. **Optimization** — canonicalizes the primitive graph (e.g. merge
       adjacent IndexMaps).
    3. **Lowering** — walks the primitive graph and emits one ``KernelOp``
       per compute unit.
"""

from __future__ import annotations

from pathlib import Path

from deplodock.compiler.ir import Graph
from deplodock.compiler.lower import lower
from deplodock.compiler.ops import KernelOp
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent / "rules"


def compile_graph(graph: Graph) -> list[KernelOp]:
    """Lower a traced ``Graph`` to a list of ``KernelOp``.

    Runs decomposition + optimization passes (via the rule-based
    rewriter) to reduce the graph to primitives, then calls
    ``lower()`` to emit structural ``KernelOp``s.
    """
    rewriter = Rewriter.from_directory(_RULES_DIR)
    graph = rewriter.apply(graph)
    return lower(graph)
