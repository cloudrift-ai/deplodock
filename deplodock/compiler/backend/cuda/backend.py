"""CUDA backend: ``Graph`` → ``LoopProgram`` → ``GpuProgram`` → nvcc → GPU.

The compiled artifact is the ``GpuProgram`` itself — it carries buffer
shapes and constant values, so no extra wrapper is needed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.emit import compile_kernels
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program, run_program_debug
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.program.gpu import GpuProgram

if TYPE_CHECKING:
    from deplodock.compiler.dump import CompilerDump
    from deplodock.compiler.ir.graph import Graph


_DEBUG_ENV = "DEPLODOCK_DEBUG"


class CudaBackend(Backend):
    """CUDA backend.

    When ``debug`` is True (or the ``DEPLODOCK_DEBUG`` env var is set),
    ``run()`` uses the per-launch debug path that dumps every non-input
    buffer after each kernel launch. ``last_debug_result`` is then
    populated with the per-launch snapshots for the last ``run()`` call.

    When ``dump`` is provided, intermediate compilation artifacts are
    written to the dump directory during ``compile()``.
    """

    def __init__(self, *, debug: bool | None = None, dump: CompilerDump | None = None) -> None:
        if debug is None:
            debug = os.environ.get(_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes")
        self.debug = debug
        self.dump = dump
        self.last_debug_result = None  # set by run() when debug=True

    def compile(self, graph: Graph) -> GpuProgram:
        """Lower ``Graph`` → ``LoopProgram`` → ``GpuProgram``."""
        return compile_kernels(compile_graph(graph, dump=self.dump))

    def run(self, compiled: GpuProgram, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
        """Execute the compiled program; returns outputs as ndarrays with declared shapes."""
        if self.debug:
            debug_result = run_program_debug(compiled, input_data=input_data)
            self.last_debug_result = debug_result
            result_outputs = debug_result.outputs
            time_ms = None
        else:
            self.last_debug_result = None
            result = run_program(compiled, input_data=input_data)
            result_outputs = result.outputs
            time_ms = result.time_ms
        outputs: dict[str, np.ndarray] = {}
        for name, vals in result_outputs.items():
            shape = tuple(int(d) for d in compiled.shape(name))
            outputs[name] = np.asarray(vals, dtype=np.float32).reshape(shape)
        return ProgramResult(outputs=outputs, time_ms=time_ms)

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
