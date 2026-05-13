"""Backend abstraction for compiling and running a ``Graph``.

A Backend lowers a ``Graph`` to a backend-specific runnable artifact and
executes it. Each backend (NumPy, Loop, CUDA, ...) implements this
interface; all backends share the same call surface:

    compiled = backend.compile(graph)
    result   = backend.run(compiled, input_data={...})
    outputs  = result.outputs   # dict[name, ndarray]
    elapsed  = result.time_ms

Individual backends may do different internal lowerings: the CUDA backend
calls ``run_pipeline`` through the full chain (decomposition →
optimization → fusion → lowering/tile → lowering/cuda), the Loop
backend stops after fusion, and the numpy backend walks the graph
directly. From the caller's perspective the interface is identical.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from deplodock.compiler.ir.base import ConstantOp, InputOp

if TYPE_CHECKING:
    from deplodock.compiler.graph import Graph


@dataclass
class RunResult:
    """Result of running a program: outputs as ndarrays + optional wall-time."""

    outputs: dict[str, Any]  # actually dict[str, np.ndarray] at runtime
    time_ms: float | None = None


@dataclass
class LaunchTime:
    """Per-launch GPU-event timing inside a benchmark run.

    ``time_ms`` is the median over ``samples`` (the canonical selection
    statistic — robust to single-iter outliers from cupy framing
    jitter). ``samples`` carries every measured per-iter latency in
    ms so callers downstream (e.g. the tuning DB recorder) can compute
    min/max/mean/variance without re-running the bench."""

    idx: int
    kernel_name: str
    time_ms: float
    samples: tuple[float, ...] | None = None


@dataclass
class BenchmarkResult:
    """Result of benchmarking a program."""

    time_ms: float
    min_ms: float | None = None
    max_ms: float | None = None
    num_launches: int = 0
    per_launch: list[LaunchTime] | None = None


class Backend(ABC):
    """Abstract backend: Graph → compiled artifact → run."""

    # Short identifier used by the autotune ``perf`` table to partition
    # measurements by which backend produced them (so e.g. the CUDA
    # backend and the loop interpreter can share a DB without clobber).
    # Subclasses override.
    name: str = "stub"

    # Wall-clock cap on the NVRTC-compile stage of a single
    # ``benchmark()`` call. Enforced at the C-call boundary after compile
    # finishes so the worker raises cleanly and no daemon thread is ever
    # left holding the CUDA context. Autotune cache pins ``bench_fail``
    # latency to (this + ``bench_run_timeout_s``).
    bench_compile_timeout_s: float = 2.0
    # Cumulative GPU-time cap on the iter loop. Enforced *after* each
    # iter completes — checked against the running sum of per-launch
    # event-measured ms. Catches the case where every iter is just
    # under the per-launch ``_KERNEL_TIMEOUT_MS`` watchdog (e.g. 999 ms
    # × 20 iters = 20 s of GPU time) which the watchdog by design lets
    # through. Distinct from a wall-clock cap so Python/cupy framing
    # overhead doesn't artificially shrink the budget for tiny ops.
    bench_run_timeout_s: float = 2.0
    # Optional hard wall-clock cap on a single ``benchmark()`` call.
    # When set, the call runs in a subprocess-isolated worker so the
    # parent can SIGKILL the GPU process if a kernel keeps the device
    # busy past the in-process per-launch / per-iter budgets (those
    # budgets rely on ``cupy.cuda.Event.done`` which never trips on
    # some hangs). Set this for autotune sweeps; leave ``None`` for
    # interactive ``deplodock run`` so on-iter callbacks and the
    # parent's torch instance can be shared in-process.
    bench_wall_timeout_s: float | None = None

    @abstractmethod
    def compile(self, graph: Graph) -> Any:
        """Lower a ``Graph`` to a backend-specific runnable artifact."""

    def run(self, compiled: Any, *, input_data: dict[str, np.ndarray] | None = None) -> RunResult:
        """Execute; return ``RunResult`` with outputs as ndarrays + wall-time.

        Default implementation walks ``compiled`` (a ``Graph``) in topological
        order and dispatches every compute node through ``op.forward``. Inputs
        and ``ConstantOp`` overrides come from ``input_data``; scalar
        ``ConstantOp``s without overrides materialize as single-element float32
        arrays. ``LoopOp.forward`` plugs into this same walk, so post-fusion
        graphs interpret identically — used by ``NumpyBackend`` and
        ``LoopBackend``. Backends with native runtimes (e.g. CUDA) override.
        """
        input_data = input_data or {}
        input_set = set(compiled.inputs)
        values: dict[str, np.ndarray] = {}

        t0 = time.perf_counter()
        for nid in compiled.topological_order():
            node = compiled.nodes[nid]
            shape = tuple(int(d) for d in node.output.shape if isinstance(d, int))

            if nid in input_set:
                if nid not in input_data:
                    raise KeyError(f"Missing input for node {nid!r}")
                values[nid] = _coerce(input_data[nid], shape)
                continue

            if isinstance(node.op, ConstantOp):
                if nid in input_data:
                    values[nid] = _coerce(input_data[nid], shape)
                elif node.op.value is not None:
                    values[nid] = np.array([node.op.value], dtype=np.float32)
                else:
                    raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in input_data")
                continue

            if isinstance(node.op, InputOp):
                continue

            args = [values[inp] for inp in node.inputs]
            result = node.op.forward(*args)
            arr = np.asarray(result, dtype=np.float32)
            if shape and arr.shape != shape:
                arr = arr.reshape(shape)
            values[nid] = arr

        elapsed = (time.perf_counter() - t0) * 1000
        outputs = {name: values[name] for name in compiled.outputs}
        return RunResult(outputs=outputs, time_ms=elapsed)

    def benchmark(
        self,
        compiled: Any,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        warmup: int = 5,
        num_iters: int | str = 20,
    ) -> BenchmarkResult:
        """Wall-time benchmark over repeated ``run()`` calls.

        Default implementation — good enough for numpy / loop interpreters.
        Backends with better timing (e.g. CUDA using GPU events inside an
        nvcc-compiled subprocess) override for device-precise measurements.

        Single loop covers warmup + measurement (the first ``warmup`` iters
        are discarded). A per-call wall-time watchdog (``1000 ms``) raises
        ``RuntimeError`` if any single ``run()`` exceeds it — autotune
        sweeps catch this and mark the variant ``bench_fail``.

        ``num_iters="auto"`` adapts the iter count to the per-call cost:
        keep running measured iters until accumulated wall time reaches
        100 ms, capped at ``100_000`` measured iters.
        """
        if isinstance(num_iters, str):
            if num_iters != "auto":
                raise ValueError(f"num_iters must be int or 'auto', got {num_iters!r}")
            target_total_ms = 100.0
            max_measured = 100_000
            auto = True
        else:
            target_total_ms = float("inf")
            max_measured = int(num_iters)
            auto = False

        kernel_timeout_ms = 1000.0
        iters_run = 0
        times: list[float] = []
        cumulative_ms = 0.0
        while True:
            t0 = time.perf_counter()
            self.run(compiled, input_data=input_data)
            dt = (time.perf_counter() - t0) * 1000
            if dt > kernel_timeout_ms:
                raise RuntimeError(f"run() took {dt:.1f} ms — exceeds {kernel_timeout_ms:.0f} ms timeout")
            iters_run += 1
            if iters_run <= warmup:
                continue
            times.append(dt)
            cumulative_ms += dt
            if len(times) >= max_measured:
                break
            if auto and cumulative_ms >= target_total_ms:
                break
        return BenchmarkResult(
            time_ms=sum(times) / len(times),
            min_ms=min(times),
            max_ms=max(times),
        )


def _coerce(data, shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if shape and arr.shape != shape:
        arr = arr.reshape(shape)
    return arr
