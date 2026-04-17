"""CUDA backend: converts a ``LoopProgram`` into a runnable ``GpuProgram``.

The recursive-descent code emitter lives in
``deplodock.compiler.backend.cuda.emit``; this module is the stable
interface the runner / commands call.
"""

from __future__ import annotations

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.emit import compile_kernels
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
from deplodock.compiler.program.gpu import GpuProgram
from deplodock.compiler.program.loop import LoopProgram


class CudaBackend(Backend):
    """CUDA backend: ``LoopProgram`` → ``GpuProgram`` → nvcc → GPU."""

    def compile(self, program: LoopProgram) -> GpuProgram:
        """Lower a ``LoopProgram`` to a runnable CUDA ``GpuProgram``."""
        return compile_kernels(program)

    def run(self, program: GpuProgram, input_data: dict[str, list[float]] | None = None) -> ProgramResult:
        result = run_program(program, input_data=input_data)
        return ProgramResult(outputs=result.outputs, time_ms=result.time_ms)

    def benchmark(self, program: GpuProgram, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        result = benchmark_program(program, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(time_ms=result.time_ms, num_launches=result.num_launches)
