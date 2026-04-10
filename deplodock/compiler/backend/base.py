"""Backend abstraction for compiling and running execution plans.

A Backend converts a backend-agnostic ExecutionPlan into a runnable
artifact and executes it.  Each backend (CUDA, ROCm, ...) implements
this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from deplodock.compiler.plan import ExecutionPlan


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
    """Abstract backend for executing an ExecutionPlan."""

    @abstractmethod
    def compile(self, plan: ExecutionPlan) -> Any:
        """Convert a backend-agnostic plan to a backend-specific program."""

    @abstractmethod
    def run(self, program: Any) -> ProgramResult:
        """Execute a compiled program and return outputs."""

    @abstractmethod
    def benchmark(self, program: Any, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        """Execute a compiled program with timing."""
