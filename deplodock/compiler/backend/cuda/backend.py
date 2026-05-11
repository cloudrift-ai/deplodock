"""CUDA backend: ``Graph`` → lowered ``Graph[CudaOp]`` → nvcc → GPU.

The compiled artifact is the lowered ``Graph`` itself — every compute
node carries a rendered CUDA kernel source plus its launch geometry.
"""

from __future__ import annotations

import logging
import os
import threading
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


def _benchmark_with_timeout(graph, *, warmup, num_iters, on_iter, wall_timeout_s):
    """Run ``benchmark_program`` in a daemon worker; raise on timeout.

    The worker captures the return value or exception in a shared dict.
    On timeout the worker is abandoned (still daemon, dies with the
    process) — GPU memory it allocated leaks until the eventual NVRTC
    compile/kernel finishes or the process exits. Acceptable for an
    autotune sweep: a handful of timeouts cost some headroom but the
    sweep keeps moving instead of stalling forever."""
    box: dict[str, object] = {}

    def _worker() -> None:
        try:
            box["value"] = benchmark_program(graph, warmup=warmup, num_iters=num_iters, on_iter=on_iter)
        except BaseException as exc:  # noqa: BLE001 — pass everything back through the box
            box["error"] = exc

    t = threading.Thread(target=_worker, daemon=True, name="deplodock-bench")
    t.start()
    t.join(timeout=wall_timeout_s)
    if t.is_alive():
        logger.warning("benchmark exceeded %.1fs wall budget — abandoning worker thread", wall_timeout_s)
        raise RuntimeError(f"benchmark exceeded {wall_timeout_s:.1f}s wall budget — variant marked bench_fail")
    if "error" in box:
        raise box["error"]
    return box["value"]

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
        result = _benchmark_with_timeout(compiled, warmup=warmup, num_iters=num_iters, on_iter=on_iter, wall_timeout_s=self.bench_wall_timeout_s)
        return BenchmarkResult(
            time_ms=result.time_ms,
            num_launches=result.num_launches,
            per_launch=result.per_launch,
        )

    def make_runner(self, compiled: Graph, *, input_data: dict[str, np.ndarray] | None = None):
        """Return a zero-arg ``run_once()`` callable that issues one full
        kernel-sequence pass on the same pre-allocated buffers. Used for
        interleaved benchmarking against PyTorch."""
        return make_runner(compiled, input_data=input_data)
