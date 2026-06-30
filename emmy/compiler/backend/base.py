"""Backend abstraction for compiling and running a ``Graph``.

A Backend lowers a ``Graph`` to a backend-specific runnable artifact and
executes it. Each backend (NumPy, Loop, CUDA, ...) implements this
interface; all backends share the same call surface:

    compiled = backend.compile(graph)
    result, pre = backend.run(compiled, input_data={...})
    outputs    = result.outputs   # dict[name, ndarray]
    elapsed    = result.time_ms

``run()`` always returns a 2-tuple ``(RunResult, T | None)``. ``T`` is
whatever the optional ``pre_run`` callback returned; with no callback
it is ``None``. ``pre_run`` runs once before the backend executes the
graph and inside whatever serialization the backend uses
(``gpu_lock()`` on the CUDA backend) — accuracy tests use this to
compute a torch eager reference on the same GPU window the deplodock
launches will see, so peer-worker CUDA activity can't interleave
between the eager forward and the deplodock comparison.

Individual backends may do different internal lowerings: the CUDA backend
calls ``run_pipeline`` through the full chain (decomposition →
optimization → fusion → lowering/tile → lowering/cuda), the Loop
backend stops after fusion, and the numpy backend walks the graph
directly. From the caller's perspective the interface is identical.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
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
    ms so callers downstream (e.g. ``Pipeline._bench_terminal``) can
    compute min/max/mean/variance without re-running the bench."""

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
    # Timings came from CUDA-graph-captured launch windows (pure GPU time, no
    # per-launch dispatch gaps). False when capture was disabled or failed and
    # the bench fell back to plain launches — callers that pair this result
    # with peer torch timings use the flag to keep one table single-semantics.
    captured: bool = False
    # Whole-program time per iteration, measured as replays of ONE CUDA graph
    # holding every launch in program order (median / min over measured iters).
    # ``time_ms``/``min_ms`` sum per-launch windows that each replay a single
    # kernel back-to-back — accurate per-kernel, but the sum is not an
    # end-to-end number (no cross-kernel cache effects, no inter-kernel gaps).
    # Populated automatically for multi-launch programs when capture held;
    # ``None`` for single-launch programs (the solo window IS the program
    # time) and when capture was off or fell back.
    e2e_ms: float | None = None
    e2e_min_ms: float | None = None


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
    # latency to (this + ``bench_run_timeout_s``). Default 30 s gives the
    # whole-graph compile/run commands headroom for a large kernel's -O3 nvcc
    # compile; the ``tune`` sweep overrides it down (single kernels, and it
    # compiles at -Xcicc -O1 which is fast) — see ``commands/tune.py``.
    bench_compile_timeout_s: float = 30.0
    # Cumulative GPU-time cap on the iter loop. Enforced *after* each
    # iter completes — checked against the running sum of per-launch
    # event-measured ms. Catches the case where every iter is just
    # under the per-launch ``_KERNEL_TIMEOUT_MS`` watchdog (e.g. 999 ms
    # × 20 iters = 20 s of GPU time) which the watchdog by design lets
    # through. Distinct from a wall-clock cap so Python/cupy framing
    # overhead doesn't artificially shrink the budget for tiny ops.
    bench_run_timeout_s: float = 10.0
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

    def run(
        self,
        compiled: Any,
        *,
        input_data: dict[str, np.ndarray] | None = None,
        pre_run: Callable[[], Any] | None = None,
    ) -> tuple[RunResult, Any]:
        """Execute; return ``(RunResult, pre_run_result)``.

        ``pre_run`` runs once before the backend executes the graph and
        inside whatever serialization the backend uses (the GPU lock on
        :class:`~deplodock.compiler.backend.cuda.backend.CudaBackend`).
        Its return value flows through as the tuple's second element.
        With no callback the second element is ``None``.

        Default implementation walks ``compiled`` (a ``Graph``) in topological
        order and dispatches every compute node through ``op.forward``. Inputs
        and ``ConstantOp`` overrides come from ``input_data``; scalar
        ``ConstantOp``s without overrides materialize as single-element float32
        arrays. ``LoopOp.forward`` plugs into this same walk, so post-fusion
        graphs interpret identically — used by ``NumpyBackend`` and
        ``LoopBackend``. Backends with native runtimes (e.g. CUDA) override.
        """
        pre_result = pre_run() if pre_run is not None else None
        input_data = input_data or {}
        input_set = set(compiled.inputs)
        values: dict[str, np.ndarray] = {}

        # Symbolic dims bind from the supplied input array shapes (mirrors
        # ``CudaBackend.run``): each atomic symbolic input dim records its
        # runtime size; downstream node shapes — possibly composite Dim
        # exprs — resolve via ``expr.eval(sym_env)``.
        from deplodock.compiler.ir.expr import Var  # noqa: PLC0415

        sym_env: dict[str, int] = {}
        for nid in input_set:
            node = compiled.nodes.get(nid)
            if node is None or nid not in input_data:
                continue
            arr_shape = np.asarray(input_data[nid]).shape
            for i, d in enumerate(node.output.shape):
                if not d.is_static and isinstance(d.expr, Var) and i < len(arr_shape):
                    sym_env.setdefault(d.expr.name, int(arr_shape[i]))

        def _shape_of(node) -> tuple[int, ...]:  # noqa: ANN001
            return tuple(d.as_static() if d.is_static else int(d.expr.eval(sym_env)) for d in node.output.shape)

        t0 = time.perf_counter()
        for nid in compiled.topological_order():
            node = compiled.nodes[nid]
            shape = _shape_of(node)

            dtype_np = node.output.dtype.np

            if nid in input_set:
                if nid not in input_data:
                    raise KeyError(f"Missing input for node {nid!r}")
                values[nid] = _coerce(input_data[nid], shape, dtype_np)
                continue

            if isinstance(node.op, ConstantOp):
                if nid in input_data:
                    values[nid] = _coerce(input_data[nid], shape, dtype_np)
                elif node.op.value is not None:
                    values[nid] = np.array([node.op.value], dtype=dtype_np)
                else:
                    raise KeyError(f"ConstantOp {nid!r} has no value and was not supplied in input_data")
                continue

            if isinstance(node.op, InputOp):
                continue

            args = [values[inp] for inp in node.inputs]
            result = node.op.forward(*args)
            arr = np.asarray(result, dtype=dtype_np)
            if shape and arr.shape != shape:
                arr = arr.reshape(shape)
            values[nid] = arr

        elapsed = (time.perf_counter() - t0) * 1000
        outputs = {name: values[name] for name in compiled.outputs}
        return RunResult(outputs=outputs, time_ms=elapsed), pre_result


def _coerce(data, shape: tuple[int, ...], dtype: np.dtype | None = None) -> np.ndarray:
    arr = np.asarray(data, dtype=dtype if dtype is not None else np.float32)
    if shape and arr.shape != shape:
        arr = arr.reshape(shape)
    return arr
