"""Backend abstraction for compiling and running structural KernelOps.

A Backend converts a list of ``KernelOp`` (the structural compiler IR)
into a runnable artifact and executes it. Each backend (CUDA, ROCm, ...)
implements this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ProgramResult:
    """Result of running a program."""

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
    """Abstract backend for compiling + executing a list of KernelOps."""

    @abstractmethod
    def compile(self, kernels: list) -> Any:
        """Lower structural KernelOps to a backend-specific Program."""

    @abstractmethod
    def run(self, program: Any) -> ProgramResult:
        """Execute a compiled program and return outputs."""

    @abstractmethod
    def benchmark(self, program: Any, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        """Execute a compiled program with timing."""
