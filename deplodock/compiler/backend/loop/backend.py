"""Loop backend: run the fusion pipeline, then interpret the fused graph.

Equivalent to ``NumpyBackend`` except that ``compile`` first runs
decomposition → optimization → fusion so the executed graph contains
``LoopOp`` nodes. Execution goes through the default ``Backend.run``
topo-walk: ``LoopOp.forward`` (defined in
:mod:`deplodock.compiler.ir.loop.ir`) JIT-compiles each kernel to C++
via cppyy / Cling and runs it on numpy buffers.

Used as a correctness-triangulation reference: CUDA vs. loop
disagreement implicates codegen; loop vs. numpy disagreement implicates
fusion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.backend import Backend
from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


class LoopBackend(Backend):
    """Execute a fused ``Graph[LoopOp]`` via the default topo-walk; LoopOps are JIT'd by ``LoopOp.forward``."""

    name = "loop"

    def compile(self, graph: Graph) -> Graph:
        return Pipeline.build(LOOP_PASSES).run(graph)
