"""CUDA backend: ``Graph`` → lowered ``Graph[CudaOp]`` → nvcc → GPU.

The compiled artifact is the lowered ``Graph`` itself — every compute
node carries a rendered CUDA kernel source plus its launch geometry.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, RunResult
from deplodock.compiler.backend.cuda.program import benchmark_program, make_runner, run_program, run_program_debug
from deplodock.compiler.pipeline import CUDA_PASSES, run_pipeline

logger = logging.getLogger(__name__)

# Hard wall-clock cap for ``benchmark()`` calls. Wraps the whole bench
# in a daemon worker thread; if it doesn't return within this budget we
# abandon and raise ``RuntimeError``. Needed because NVRTC compilation
# inside the first kernel launch can take 30+ s on heavily-replicated
# kernels (e.g. autotune variants with F_M*F_N=256 cells), which would
# otherwise stall the whole sweep on one bad variant. Exposed via the
# inherited ``Backend.bench_wall_timeout_s`` attribute for callers
# (autotune cache reads it to set fail-row latencies).



if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline.dump import CompilerDump


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
        if dump is None:
            from deplodock.compiler.pipeline.dump import CompilerDump as _CD

            dump = _CD.from_env()
        self.debug = debug
        self.dump = dump
        self.last_debug_result = None

    def compile(self, graph: Graph) -> Graph:
        """Lower ``Graph`` → ``Graph[LoopOp]`` → ``Graph[TileOp]`` → ``Graph[CudaOp]``."""
        return run_pipeline(graph, CUDA_PASSES, dump=self.dump)

    def run(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        # GPU serialization happens inside ``run_program`` /
        # ``run_program_debug`` / ``benchmark_program`` — they grab
        # ``gpu_lock()`` only around the launches + sync so parallel
        # workers can NVRTC-compile in parallel and only contend for
        # the device when it's time to time.
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
        return RunResult(outputs=outputs, time_ms=time_ms)

    def benchmark(
        self,
        compiled: Graph,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int | str = 20,
        on_iter=None,
    ) -> BenchmarkResult:
        del input_data
        # GPU serialization is held inside ``benchmark_program`` (only
        # around the iter loop, not NVRTC compile + alloc). The compile
        # stage is bounded by ``bench_compile_timeout_s``; the iter loop
        # is bounded both by ``num_iters="auto"``'s 100 ms GPU-time
        # target *and* the cumulative ``bench_run_timeout_s`` budget,
        # which catches pathological variants where every launch fits
        # under the per-launch ``_KERNEL_TIMEOUT_MS`` watchdog but they
        # accumulate over many iters.
        result = benchmark_program(
            compiled,
            warmup=warmup,
            num_iters=num_iters,
            on_iter=on_iter,
            compile_timeout_s=self.bench_compile_timeout_s,
            run_timeout_s=self.bench_run_timeout_s,
        )
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
            per_launch=result.per_launch,
        )

    def make_runner(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None):
        """Return a zero-arg ``run_once()`` callable that issues one full
        kernel-sequence pass on the same pre-allocated buffers. Used for
        interleaved benchmarking against PyTorch.

        Per-call locking is intentionally omitted: callers that need
        process-level serialization should hold ``gpu_lock()`` around
        their entire iter loop instead of acquiring/releasing on every
        kernel launch."""
        return make_runner(compiled, input_data=input_data)
