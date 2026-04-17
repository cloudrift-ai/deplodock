"""Backend abstraction for compiling and running a ``LoopProgram``.

A Backend lowers a ``LoopProgram`` (the post-fusion program form) to a
backend-specific runnable artifact and executes it. Each backend (CUDA,
ROCm, NumPy, ...) implements this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from deplodock.compiler.program.loop import LoopProgram


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
    """Abstract backend for compiling + executing a ``LoopProgram``."""

    @abstractmethod
    def compile(self, program: LoopProgram) -> Any:
        """Lower a ``LoopProgram`` to a backend-specific runnable program."""

    @abstractmethod
    def run(self, program: Any) -> ProgramResult:
        """Execute a compiled program and return outputs."""

    @abstractmethod
    def benchmark(self, program: Any, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        """Execute a compiled program with timing."""
