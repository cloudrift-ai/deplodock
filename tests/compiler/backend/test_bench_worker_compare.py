"""The deployable eager / torch.compile / emmy comparison runs in the SIGKILL-able worker.

This is what makes ``tune --bench`` / ``run --bench`` survive a hung generated kernel: the whole
comparison (emmy + the torch peer-bench, rebuilt in the child from a recipe) runs in the worker,
so a non-terminating kernel hangs the *child* and the parent SIGKILLs it on ``wall_timeout_s`` —
freeing the device and leaving the parent clean, instead of the ~109-minute in-process wedge.

- ``test_compare_in_worker_returns_torch_and_emmy`` — the happy path: a real frontend graph is
  rebuilt + benched against eager torch in the child, numbers come back.
- ``test_worker_hang_is_sigkilled_not_wedged`` — a worker that hangs on a real non-terminating GPU
  kernel is SIGKILLed at the wall-timeout and surfaces a ``RuntimeError`` promptly (not a wedge).
"""

from __future__ import annotations

import asyncio
import sys
import textwrap
import time

from ..conftest import requires_cuda


@requires_cuda
def test_compare_in_worker_returns_torch_and_emmy() -> None:
    from emmy.commands.run import _detect_stage, _passes_after_stage
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.program import benchmark_compare_isolated_async
    from emmy.compiler.pipeline import Pipeline

    # A small torch_ref-runnable op; lowered in-parent (as the per-kernel sweep does), then the
    # frontend snapshot + lowered graph go to the worker, which rebuilds the torch ref and benches.
    g, _, _ = graph_from_code("torch.nn.RMSNorm(512)(torch.randn(8, 512))")
    fe = g.copy()
    tail = _passes_after_stage(_detect_stage(g))
    lowered = Pipeline.build(tail).run(g) if tail else g

    results, bench, torch_available, _captured = asyncio.run(
        benchmark_compare_isolated_async(
            lowered=lowered,
            torch_spec=("frontend_graph", fe),
            bench_backends="eager,emmy",
            wall_timeout_s=180.0,
            warmup=2,
            iters=5,
            seed=0,
            nvcc_flags="",
        )
    )
    assert torch_available, "the worker should have rebuilt the torch reference from the frontend graph"
    assert results.get("Emmy", 0) > 0, f"missing emmy number: {results}"
    assert results.get("Eager PyTorch", 0) > 0, f"missing eager torch number: {results}"
    assert bench is not None


class _HangWorker:
    """An ``_AsyncBenchWorker`` whose child's ``_run_job`` launches a non-terminating GPU kernel and
    blocks on it forever — to exercise the parent's wall-timeout SIGKILL on a genuinely hung worker."""

    _CHILD = textwrap.dedent(
        """
        import emmy.compiler.backend.cuda._bench_worker as w
        import cupy
        def _hang(req):
            spin = cupy.RawKernel(r'extern "C" __global__ void spin(volatile int* f){ while(f[0]==0){} }', 'spin')
            flag = cupy.zeros(1, dtype=cupy.int32)   # never set → infinite loop
            spin((1,), (1,), (flag,))
            cupy.cuda.runtime.deviceSynchronize()    # blocks forever on the hung kernel
            return {}
        w._run_job = _hang
        w.main()
        """
    )

    def __init__(self) -> None:
        from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

        self._impl = _AsyncBenchWorker()
        # Override _spawn to launch our hanging child instead of ``-m _bench_worker``.
        self._impl._spawn = self._spawn_hang  # type: ignore[method-assign]

    async def _spawn_hang(self) -> None:
        import asyncio

        self._impl._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            self._CHILD,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    def run_job(self, *a, **k):
        return self._impl.run_job(*a, **k)

    @property
    def proc(self):
        return self._impl._proc


def test_run_job_send_times_out_on_unresponsive_worker() -> None:
    """A worker that is alive but never reads stdin — e.g. wedged in CUDA-context
    teardown behind a hung kernel after a dirty exit — must trip the wall budget
    during the request SEND (``stdin.drain`` blocks once the request exceeds the
    ~64 KB pipe buffer). No CUDA needed: the deaf child is plain Python."""
    import asyncio

    import pytest

    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    worker = _AsyncBenchWorker()

    async def _spawn_deaf() -> None:
        worker._proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    worker._spawn = _spawn_deaf  # type: ignore[method-assign]
    t0 = time.time()
    with pytest.raises(RuntimeError, match="did not accept the request"):
        asyncio.run(worker.run_job({"blob": b"x" * (1 << 20)}, wall_timeout_s=2.0))  # 1 MB >> pipe buffer
    assert time.time() - t0 < 20.0, "send must respect the wall budget, not block on the pipe"
    assert worker._proc is None, "the unresponsive worker must be killed and its handle released"


@requires_cuda
def test_worker_hang_is_sigkilled_not_wedged() -> None:
    import asyncio

    import pytest

    worker = _HangWorker()
    t0 = time.time()
    # The child launches an infinite kernel and never responds; the parent must SIGKILL it at the
    # 3 s wall budget and raise — not block forever (the pre-isolation in-process failure mode).
    with pytest.raises(RuntimeError, match="wall budget"):
        asyncio.run(worker.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=3.0))
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"run_job took {elapsed:.1f}s — the wall-timeout SIGKILL did not fire promptly"
    assert worker.proc is None, "the hung worker must be killed and its handle released"
