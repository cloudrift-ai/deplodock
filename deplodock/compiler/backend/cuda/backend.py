"""CUDA backend: ``Graph`` → ``LoopProgram`` → ``GpuProgram`` → nvcc → GPU.

The compiled artifact is the ``GpuProgram`` itself — it carries buffer
shapes and constant values, so no extra wrapper is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.emit import compile_kernels
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.gpu import GpuProgram

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph


class CudaBackend(Backend):
    """CUDA backend."""

    def compile(self, graph: Graph) -> GpuProgram:
        """Lower ``Graph`` → ``LoopProgram`` → ``GpuProgram``."""
        return compile_kernels(compile_graph(graph))

    def run(self, compiled: GpuProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        """Execute the compiled program; returns outputs as ndarrays with declared shapes."""
        flat_input = {k: np.asarray(v, dtype=np.float32).flatten().tolist() for k, v in (input_data or {}).items()}
        result = run_program(compiled, input_data=flat_input)
        outputs: dict[str, np.ndarray] = {}
        for name, vals in result.outputs.items():
            shape = tuple(int(d) for d in compiled.shape(name))
            outputs[name] = np.asarray(vals, dtype=np.float32).reshape(shape)
        return ProgramResult(outputs=outputs, time_ms=result.time_ms)

    def benchmark(
        self,
        compiled: GpuProgram,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int = 20,
    ) -> BenchmarkResult:
        """Override the default wall-time loop with GPU-event timing
        (via the nvcc-compiled subprocess). ``input_data`` is currently
        unused — the benchmark subprocess generates random inputs."""
        del input_data  # subprocess generates its own random inputs
        result = benchmark_program(compiled, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(time_ms=result.time_ms, num_launches=result.num_launches)
