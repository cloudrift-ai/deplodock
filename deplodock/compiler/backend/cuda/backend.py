"""CUDA backend: ``Graph`` → ``LoopProgram`` → ``GpuProgram`` → nvcc → GPU.

The compiled artifact bundles both the ``GpuProgram`` (what actually runs)
and the ``LoopProgram`` (shape metadata for ``run_arrays`` to reshape
flat subprocess outputs back to ndarrays).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.emit import compile_kernels
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.gpu import GpuProgram
from deplodock.compiler.program.loop import LoopProgram

if TYPE_CHECKING:
    from deplodock.compiler.ir.graph import Graph


@dataclass
class CompiledCudaProgram:
    """Compiled-CUDA artifact: the runnable ``GpuProgram`` + ``LoopProgram`` for shape lookups."""

    gpu: GpuProgram
    loop: LoopProgram


class CudaBackend(Backend):
    """CUDA backend."""

    def compile(self, graph: Graph) -> CompiledCudaProgram:
        """Lower ``Graph`` → ``LoopProgram`` → ``GpuProgram``."""
        loop = compile_graph(graph)
        gpu = compile_kernels(loop)
        return CompiledCudaProgram(gpu=gpu, loop=loop)

    def run(self, compiled: CompiledCudaProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        """Execute the compiled program; returns outputs as flat float lists."""
        flat_input = {k: np.asarray(v, dtype=np.float32).flatten().tolist() for k, v in (input_data or {}).items()}
        result = run_program(compiled.gpu, input_data=flat_input)
        return ProgramResult(outputs=result.outputs, time_ms=result.time_ms)

    def run_arrays(self, compiled: CompiledCudaProgram, *, input_data: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """Execute and reshape flat outputs to ndarrays using LoopProgram shapes."""
        result = self.run(compiled, input_data=input_data)
        outputs: dict[str, np.ndarray] = {}
        for name, vals in result.outputs.items():
            shape = tuple(int(d) for d in compiled.loop.shape(name))
            outputs[name] = np.asarray(vals, dtype=np.float32).reshape(shape)
        return outputs

    def benchmark(
        self,
        compiled: CompiledCudaProgram,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int = 20,
    ) -> BenchmarkResult:
        """Override the default wall-time loop with GPU-event timing
        (via the nvcc-compiled subprocess). ``input_data`` is currently
        unused — the benchmark subprocess generates random inputs."""
        del input_data  # subprocess generates its own random inputs
        result = benchmark_program(compiled.gpu, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(time_ms=result.time_ms, num_launches=result.num_launches)
