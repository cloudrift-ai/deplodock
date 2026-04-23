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
optimization → fusion → lowering/kernel → lowering/cuda), the Loop
backend stops after fusion, and the numpy backend walks the graph
directly. From the caller's perspective the interface is identical.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from deplodock.compiler.pipeline.graph import Graph


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

    @abstractmethod
    def run(self, compiled: Any, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        """Execute; return ``RunResult`` with outputs as ndarrays + wall-time."""

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
