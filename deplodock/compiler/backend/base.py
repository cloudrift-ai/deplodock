"""Backend abstraction for compiling and running a ``Graph``.

A Backend lowers a ``Graph`` to a backend-specific runnable artifact and
executes it. Each backend (NumPy, Loop, CUDA, ...) implements this
interface; all backends share the same call surface:

    compiled = backend.compile(graph)
    result   = backend.run(compiled, input_data={...})
    outputs  = result.outputs   # dict[name, ndarray]
    elapsed  = result.time_ms

Individual backends may do different internal lowerings: the CUDA backend
calls ``run_pipeline`` through the full chain (decomposition →
optimization → fusion → lowering/tile → lowering/cuda), the Loop
backend stops after fusion, and the numpy backend walks the graph
directly. From the caller's perspective the interface is identical.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from deplodock.compiler.ir.base import ConstantOp, InputOp

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


@dataclass
class RunResult:
    """Result of running a program: outputs as ndarrays + optional wall-time."""

    outputs: dict[str, Any]  # actually dict[str, np.ndarray] at runtime
    time_ms: float | None = None


@dataclass
class LaunchTime:
    """Per-launch GPU-event timing inside a benchmark run."""

    idx: int
    kernel_name: str
    time_ms: float


@dataclass
class BenchmarkResult:
    """Result of benchmarking a program."""

    time_ms: float
    min_ms: float | None = None
    max_ms: float | None = None
    num_launches: int = 0
    per_launch: list[LaunchTime] | None = None


class Backend(ABC):
    """Abstract backend: Graph → compiled artifact → run."""

    @abstractmethod
    def compile(self, graph: Graph) -> Any:
        """Lower a ``Graph`` to a backend-specific runnable artifact."""

    def run(self, compiled: Any, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        """Execute; return ``RunResult`` with outputs as ndarrays + wall-time.

        Default implementation walks ``compiled`` (a ``Graph``) in topological
        order and dispatches every compute node through ``op.forward``. Inputs
        and ``ConstantOp`` overrides come from ``input_data``; scalar
        ``ConstantOp``s without overrides materialize as single-element float32
        arrays. ``LoopOp.forward`` plugs into this same walk, so post-fusion
        graphs interpret identically — used by ``NumpyBackend`` and
        ``LoopBackend``. Backends with native runtimes (e.g. CUDA) override.
        """
        input_data = input_data or {}
        input_set = set(compiled.inputs)
        values: dict[str, np.ndarray] = {}

        t0 = time.perf_counter()
        for nid in compiled.topological_order():
            node = compiled.nodes[nid]
            shape = tuple(int(d) for d in node.output.shape if isinstance(d, int))

            if nid in input_set:
                if nid not in input_data:
                    raise KeyError(f"Missing input for node {nid!r}")
                values[nid] = _coerce(input_data[nid], shape)
                continue

            if isinstance(node.op, ConstantOp):
                if nid in input_data:
                    values[nid] = _coerce(input_data[nid], shape)
                elif node.op.value is not None:
                    values[nid] = np.array([node.op.value], dtype=np.float32)
                else:
                    raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in input_data")
                continue

            if isinstance(node.op, InputOp):
                continue

            args = [values[inp] for inp in node.inputs]
            result = node.op.forward(*args)
            arr = np.asarray(result, dtype=np.float32)
            if shape and arr.shape != shape:
                arr = arr.reshape(shape)
            values[nid] = arr

        elapsed = (time.perf_counter() - t0) * 1000
        outputs = {name: values[name] for name in compiled.outputs}
        return RunResult(outputs=outputs, time_ms=elapsed)

    def benchmark(
        self,
        compiled: Any,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int = 20,
    ) -> BenchmarkResult:
        """Wall-time benchmark over repeated ``run()`` calls.

        Default implementation — good enough for numpy / loop interpreters.
        Backends with better timing (e.g. CUDA using GPU events inside an
        nvcc-compiled subprocess) override for device-precise measurements.
        """
        for _ in range(warmup):
            self.run(compiled, input_data=input_data)
        times: list[float] = []
        for _ in range(num_iters):
            t0 = time.perf_counter()
            self.run(compiled, input_data=input_data)
            times.append((time.perf_counter() - t0) * 1000)
        return BenchmarkResult(
            time_ms=sum(times) / len(times),
            min_ms=min(times),
            max_ms=max(times),
        )


def _coerce(data, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if shape and arr.shape != shape:
        arr = arr.reshape(shape)
    return arr
