"""CUDA backend: ``Graph`` → lowered ``Graph[CudaOp]`` → nvcc → GPU.

The compiled artifact is the lowered ``Graph`` itself — every compute
node carries a rendered CUDA kernel source plus its launch geometry.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, ProgramResult
from deplodock.compiler.backend.cuda.program import benchmark_program, run_program, run_program_debug
from deplodock.compiler.passes.lowering.cuda import lower as lower_to_cuda
from deplodock.compiler.passes.lowering.kernel import lower as lower_to_kernel
from deplodock.compiler.pipeline import compile_graph

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
    """

    def __init__(self, *, debug: bool | None = None, dump: CompilerDump | None = None) -> None:
        if debug is None:
            debug = os.environ.get(_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes")
        self.debug = debug
        self.dump = dump
        self.last_debug_result = None

    def compile(self, graph: Graph) -> Graph:
        """Lower ``Graph`` → ``Graph[LoopOp]`` → ``Graph[KernelOp]`` → ``Graph[CudaOp]``."""
        graph = compile_graph(graph, dump=self.dump)
        graph = lower_to_kernel(graph)
        if self.dump is not None:
            self.dump.dump_kernel_graph(graph)
        graph = lower_to_cuda(graph)
        if self.dump is not None:
            self.dump.dump_cuda_graph(graph)
        return graph

    def run(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None) -> ProgramResult:
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
            shape = tuple(int(d) for d in compiled.nodes[name].output.shape)
            outputs[name] = np.asarray(vals, dtype=np.float32).reshape(shape)
        return ProgramResult(outputs=outputs, time_ms=time_ms)

    def benchmark(
        self,
        compiled: Graph,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int = 20,
    ) -> BenchmarkResult:
        del input_data
        result = benchmark_program(compiled, warmup=warmup, num_iters=num_iters)
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
            per_launch=result.per_launch,
        )
