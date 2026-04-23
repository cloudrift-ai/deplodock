"""Numpy backend: interpret a Graph IR using numpy arrays.

No GPU required — useful for correctness testing and debugging.
``compile`` is a no-op (the graph is its own compiled artifact);
``run`` delegates to the shared :func:`interpret_graph` walker.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, RunResult
from deplodock.compiler.backend.interpret import interpret_graph

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


class NumpyBackend(Backend):
    """Graph → in-memory interpreter → numpy arrays.

    Inherits the default wall-time ``benchmark`` from ``Backend``.
    """

    def compile(self, graph: Graph) -> Graph:
        return graph

    def run(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        t0 = time.perf_counter()
        outputs = interpret_graph(compiled, input_data)
        elapsed = (time.perf_counter() - t0) * 1000
        return RunResult(outputs=outputs, time_ms=elapsed)
