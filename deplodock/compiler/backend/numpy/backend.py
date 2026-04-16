"""Numpy backend: interpret a Graph IR using numpy arrays.

Walks the graph in topological order, calling ``node.op.forward()`` at each
node. No GPU required — useful for correctness testing and debugging.
"""

from __future__ import annotations

import time

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph


class NumpyBackend(Backend):
    """Numpy backend: Graph → in-memory interpreter → numpy arrays."""

    def compile(self, graph: Graph, *, input_data: dict[str, np.ndarray] | None = None) -> NumpyProgram:
        """Store the graph and optional input data for execution."""
        return NumpyProgram(graph=graph, input_data=input_data or {})

    def run(self, program: NumpyProgram, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        """Execute the graph and return outputs as flat float lists."""
        inputs = {**program.input_data, **(input_data or {})}
        t0 = time.perf_counter()
        arrays = _execute(program.graph, inputs)
        elapsed = (time.perf_counter() - t0) * 1000
        outputs = {nid: arr.flatten().tolist() for nid, arr in arrays.items()}
        return ProgramResult(outputs=outputs, time_ms=elapsed)

    def benchmark(self, program: NumpyProgram, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        """Execute the graph repeatedly and return timing stats."""
        inputs = program.input_data
        for _ in range(warmup):
            _execute(program.graph, inputs)
        times = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            _execute(program.graph, inputs)
            times.append((time.perf_counter() - t0) * 1000)
        return BenchmarkResult(
            time_ms=sum(times) / len(times),
            min_ms=min(times),
            max_ms=max(times),
        )

    def run_arrays(self, program: NumpyProgram, input_data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Execute the graph and return outputs as numpy arrays (no flattening)."""
        inputs = {**program.input_data, **(input_data or {})}
        return _execute(program.graph, inputs)


class NumpyProgram:
    """A compiled numpy program: just the graph + cached input data."""

    def __init__(self, graph: Graph, input_data: dict[str, np.ndarray]) -> None:
        self.graph = graph
        self.input_data = input_data


def _execute(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Execute a graph on numpy arrays.

    ``inputs`` maps graph input node IDs (and optionally weight ConstantOp
    IDs) to numpy arrays. Returns a dict mapping graph output node IDs to
    their computed numpy arrays.
    """
    values: dict[str, np.ndarray] = {}

    for nid in graph.topological_order():
        node = graph.nodes[nid]

        if isinstance(node.op, InputOp):
            if nid not in inputs:
                raise KeyError(f"Missing input for node {nid!r}")
            values[nid] = inputs[nid]
            continue

        if isinstance(node.op, ConstantOp):
            if nid in inputs:
                values[nid] = inputs[nid]
            elif node.op.value is not None:
                values[nid] = np.array([node.op.value], dtype=np.float32)
            else:
                raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in inputs")
            continue

        args = [values[inp] for inp in node.inputs]
        result = node.op.forward(*args)

        target_shape = tuple(int(d) for d in node.output.shape if isinstance(d, int))
        if target_shape and result.shape != target_shape:
            result = np.reshape(result, target_shape)

        values[nid] = result

    return {nid: values[nid] for nid in graph.outputs}
