"""CUDA backend: converts a list of structural ``KernelOp`` into a runnable Program.

The recursive-descent code emitter lives in
``deplodock.compiler.backend.cuda.emit`` (added in a follow-up commit);
this module is the stable interface the runner / commands call.
"""

from __future__ import annotations

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
from deplodock.compiler.backend.program import Program


class CudaBackend(Backend):
    """CUDA backend: list[KernelOp] → Program → nvcc → GPU."""

    def compile(self, kernels: list) -> Program:
        """Emit CUDA source per ``KernelOp`` and assemble a runnable Program."""
        raise NotImplementedError("CudaBackend.compile: structural codegen lands in feature/structural-compiler c4")

    def run(self, program: Program, input_data: dict[str, list[float]] | None = None) -> ProgramResult:
        result = run_program(program, input_data=input_data)
        return ProgramResult(outputs=result.outputs, time_ms=result.time_ms)

    def benchmark(self, program: Program, warmup: int = 5, num_iters: int = 20) -> BenchmarkResult:
        result = benchmark_program(program, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(time_ms=result.time_ms, num_launches=result.num_launches)
