"""Loop backend: run the fusion pipeline, then interpret the fused graph.

Equivalent to ``NumpyBackend`` except that ``compile`` first runs
decomposition → optimization → fusion so the executed graph contains
``LoopOp`` nodes. Execution goes through the shared
:func:`interpret_graph` walker — ``LoopOp.forward`` (defined in
:mod:`deplodock.compiler.ir.loop.interpret`) handles the body walk.

Used as a correctness-triangulation reference: CUDA vs. loop
disagreement implicates codegen; loop vs. numpy disagreement implicates
fusion.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, RunResult
from deplodock.compiler.backend.interpret import interpret_graph
from deplodock.compiler.pipeline import LOOP_PASSES, run_pipeline

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


class LoopBackend(Backend):
    """Execute a fused ``Graph[LoopOp]`` via numpy whole-tensor operations."""

    def compile(self, graph: Graph) -> Graph:
        return run_pipeline(graph, LOOP_PASSES)

    def run(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        t0 = time.perf_counter()
        outputs = interpret_graph(compiled, input_data)
        elapsed = (time.perf_counter() - t0) * 1000
        return RunResult(outputs=outputs, time_ms=elapsed)
