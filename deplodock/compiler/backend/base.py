"""Backend abstraction for compiling and running a ``Graph``.

A Backend lowers a ``Graph`` to a backend-specific runnable artifact and
executes it. Each backend (NumPy, Loop, CUDA, ...) implements this
interface; all backends share the same call surface:

    compiled = backend.compile(graph)
    outputs  = backend.run_arrays(compiled, input_data={...})   # dict[name, ndarray]

Individual backends may do different internal lowerings: the CUDA backend
runs ``compile_graph → compile_kernels`` internally, the Loop backend runs
``compile_graph`` only, the numpy backend walks the graph directly. From
the caller's perspective the interface is identical.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from deplodock.compiler.ir.graph import Graph


@dataclass
class ProgramResult:
    """Result of running a program: outputs as flat lists + optional timing."""

    outputs: dict[str, list[float]]
    time_ms: float | None = None


@dataclass
class BenchmarkResult:
    """Result of benchmarking a program."""

    time_ms: float
    min_ms: float | None = None
    max_ms: float | None = None
    num_launches: int = 0


class Backend(ABC):
    """Abstract backend: Graph → compiled artifact → run."""

    @abstractmethod
    def compile(self, graph: Graph) -> Any:
        """Lower a ``Graph`` to a backend-specific runnable artifact."""

    @abstractmethod
    def run(self, compiled: Any, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        """Execute; return outputs as flat lists (subprocess-friendly)."""

    @abstractmethod
    def run_arrays(self, compiled: Any, *, input_data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Execute; return outputs as numpy arrays with declared shapes."""

    @abstractmethod
    def benchmark(self, compiled: Any, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        """Execute with timing."""
