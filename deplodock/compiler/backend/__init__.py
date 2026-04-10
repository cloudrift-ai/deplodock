"""Backend abstraction for compiling and running execution plans."""

from deplodock.compiler.backend.base import Backend, BenchmarkResult, ProgramResult

__all__ = ["Backend", "BenchmarkResult", "ProgramResult"]
