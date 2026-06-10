"""The deployable eager / torch.compile / deplodock comparison runs in the SIGKILL-able worker.

This is what makes ``tune --bench`` / ``run --bench`` survive a hung generated kernel: the whole
comparison (deplodock + the torch peer-bench, rebuilt in the child from a recipe) runs in the worker,
so a non-terminating kernel hangs the *child* and the parent SIGKILLs it on ``wall_timeout_s`` —
freeing the device and leaving the parent clean, instead of the ~109-minute in-process wedge.

- ``test_compare_in_worker_returns_torch_and_deplodock`` — the happy path: a real frontend graph is
  rebuilt + benched against eager torch in the child, numbers come back.
- ``test_worker_hang_is_sigkilled_not_wedged`` — a worker that hangs on a real non-terminating GPU
  kernel is SIGKILLed at the wall-timeout and surfaces a ``RuntimeError`` promptly (not a wedge).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time

from ..conftest import requires_cuda


@requires_cuda
def test_compare_in_worker_returns_torch_and_deplodock() -> None:
    from deplodock.commands.run import _detect_stage, _passes_after_stage
    from deplodock.commands.trace import graph_from_code
    from deplodock.compiler.backend.cuda.program import benchmark_compare_isolated
    from deplodock.compiler.pipeline import Pipeline

    # A small torch_ref-runnable op; lowered in-parent (as the per-kernel sweep does), then the
    # frontend snapshot + lowered graph go to the worker, which rebuilds the torch ref and benches.
    g, _, _ = graph_from_code("torch.nn.RMSNorm(512)(torch.randn(8, 512))")
    fe = g.copy()
    tail = _passes_after_stage(_detect_stage(g))
    lowered = Pipeline.build(tail).run(g) if tail else g

    results, bench, torch_available, _captured = benchmark_compare_isolated(
        lowered=lowered,
        torch_spec=("frontend_graph", fe),
        bench_backends="eager,deplodock",
        wall_timeout_s=180.0,
        warmup=2,
        iters=5,
        seed=0,
        nvcc_flags="",
    )
    assert torch_available, "the worker should have rebuilt the torch reference from the frontend graph"
    assert results.get("Deplodock", 0) > 0, f"missing deplodock number: {results}"
    assert results.get("Eager PyTorch", 0) > 0, f"missing eager torch number: {results}"
    assert bench is not None


class _HangWorker:
    """A ``_BenchWorker`` whose child's ``_run_job`` launches a non-terminating GPU kernel and blocks
    on it forever — to exercise the parent's wall-timeout SIGKILL on a genuinely hung worker."""

    _CHILD = textwrap.dedent(
        """
        import deplodock.compiler.backend.cuda._bench_worker as w
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
        from deplodock.compiler.backend.cuda.program import _BenchWorker

        self._impl = _BenchWorker()
        # Override _spawn to launch our hanging child instead of ``-m _bench_worker``.
        self._impl._spawn = self._spawn_hang  # type: ignore[method-assign]

    def _spawn_hang(self) -> None:
        self._impl._proc = subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", self._CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    def run_job(self, *a, **k):
        return self._impl.run_job(*a, **k)

    @property
    def proc(self):
        return self._impl._proc


@requires_cuda
def test_worker_hang_is_sigkilled_not_wedged() -> None:
    import pytest

    worker = _HangWorker()
    t0 = time.time()
    # The child launches an infinite kernel and never responds; the parent must SIGKILL it at the
    # 3 s wall budget and raise — not block forever (the pre-isolation in-process failure mode).
    with pytest.raises(RuntimeError, match="wall budget"):
        worker.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=3.0)
    elapsed = time.time() - t0
    assert elapsed < 30.0, f"run_job took {elapsed:.1f}s — the wall-timeout SIGKILL did not fire promptly"
    assert worker.proc is None, "the hung worker must be killed and its handle released"
