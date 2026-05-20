"""CUDA backend: ``Graph`` → lowered ``Graph[CudaOp]`` → nvcc → GPU.

The compiled artifact is the lowered ``Graph`` itself — every compute
node carries a rendered CUDA kernel source plus its launch geometry.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from deplodock.compiler.backend import Backend, BenchmarkResult, RunResult
from deplodock.compiler.backend.cuda.program import (
    benchmark_program,
    benchmark_program_isolated,
    make_runner,
    run_program,
    run_program_debug,
)
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

logger = logging.getLogger(__name__)

# Hard wall-clock cap for ``benchmark()`` calls. Wraps the whole bench
# in a daemon worker thread; if it doesn't return within this budget we
# abandon and raise ``RuntimeError``. Needed because NVRTC compilation
# inside the first kernel launch can take 30+ s on heavily-replicated
# kernels (e.g. autotune variants with FM*FN=256 cells), which would
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

    name = "cuda"

    def __init__(
        self,
        *,
        debug: bool | None = None,
        dump: CompilerDump | None = None,
        bench_wall_timeout_s: float | None = None,
        tune_db: Path | str | None = None,
    ) -> None:
        if debug is None:
            debug = os.environ.get(_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes")
        if dump is None:
            from deplodock.compiler.pipeline.dump import CompilerDump as _CD

            dump = _CD.from_env()
        self.debug = debug
        self.dump = dump
        self.last_debug_result = None
        # When set, ``benchmark()`` runs in a subprocess so the parent
        # can SIGKILL a wedged worker. Defaults to ``None`` (in-process,
        # required when ``on_iter`` callbacks are supplied).
        self.bench_wall_timeout_s = bench_wall_timeout_s
        # Persistent autotune cache. When provided, ``compile()`` opens
        # this SearchDB so ``GreedySearch`` can pick tuned forks via
        # ``best=``; otherwise it falls back to rule option-0.
        self.tune_db = Path(tune_db) if tune_db is not None else None

    def compile(self, graph: Graph) -> Graph:
        """Lower ``Graph`` → ``Graph[LoopOp]`` → ``Graph[TileOp]`` → ``Graph[CudaOp]``."""
        db = None
        if self.tune_db is not None:
            from deplodock.compiler.pipeline.search.db import SearchDB

            db = SearchDB(path=self.tune_db)
        return Pipeline.build(CUDA_PASSES, dump=self.dump).run(graph, db=db)

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
            outputs[name] = np.asarray(vals, dtype=compiled.nodes[name].output.dtype.np).reshape(shape)
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
        # ``bench_wall_timeout_s`` selects between two paths:
        # - Set (autotune sweep): run in a subprocess-isolated worker so
        #   a wedged kernel can be SIGKILLed without leaving the
        #   parent's CUDA stream dirty. Wall-clock backstop on top of the
        #   in-worker ``bench_compile_timeout_s`` / ``bench_run_timeout_s``.
        # - ``None`` (interactive ``deplodock run --bench``): run in-process.
        #   Required when ``on_iter`` callbacks are supplied — they can't
        #   cross the subprocess boundary.
        # ``on_iter`` forces the in-process path even when
        # ``bench_wall_timeout_s`` is set — interleaved benches need to
        # share torch state with the parent and can't cross the
        # subprocess boundary. The autotune sweep never passes
        # ``on_iter``, so this fallback only fires for ``--tune --bench``
        # where the post-tune interleaved bench reuses the same backend.
        if self.bench_wall_timeout_s is not None and on_iter is None:
            result = benchmark_program_isolated(
                compiled,
                wall_timeout_s=self.bench_wall_timeout_s,
                warmup=warmup,
                num_iters=num_iters,
                compile_timeout_s=self.bench_compile_timeout_s,
                run_timeout_s=self.bench_run_timeout_s,
            )
        else:
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
