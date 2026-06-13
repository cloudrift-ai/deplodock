"""Unit tests for the async bench worker's device/lock env overlay.

The multi-GPU tune driver pins each ``_AsyncBenchWorker`` to a physical GPU via
the child's spawn env (``CUDA_VISIBLE_DEVICES``) plus a per-device gpu-lock path,
and must NEVER mutate the parent's shared ``os.environ`` (every slot shares one
event-loop thread). These tests pin that contract without spawning a subprocess
or touching CUDA.
"""

from __future__ import annotations

import os

from deplodock import config
from deplodock.compiler.backend.cuda.program import _AsyncBenchWorker

from ..conftest import requires_cuda


def test_child_env_pins_device_without_mutating_os_environ() -> None:
    before = dict(os.environ)
    worker = _AsyncBenchWorker(device_id=3)
    env = worker._child_env()
    assert env["CUDA_VISIBLE_DEVICES"] == "3"
    # The parent env is untouched — the pin rides the child only.
    assert os.environ == before
    assert "CUDA_VISIBLE_DEVICES" not in os.environ or os.environ.get("CUDA_VISIBLE_DEVICES") == before.get("CUDA_VISIBLE_DEVICES")


def test_child_env_unpinned_is_passthrough() -> None:
    worker = _AsyncBenchWorker(device_id=None)
    env = worker._child_env()
    assert env.get("CUDA_VISIBLE_DEVICES") == os.environ.get("CUDA_VISIBLE_DEVICES")
    assert "DEPLODOCK_GPU_LOCK" not in env or env["DEPLODOCK_GPU_LOCK"] == os.environ.get("DEPLODOCK_GPU_LOCK")


def test_child_env_per_device_gpu_lock(monkeypatch) -> None:
    monkeypatch.setenv(config.GPU_LOCK, "/tmp/deplodock-gpu.lock")
    env = _AsyncBenchWorker(device_id=2)._child_env()
    # Distinct devices take distinct lock files so they don't serialise on one FileLock.
    assert env["DEPLODOCK_GPU_LOCK"] == "/tmp/deplodock-gpu.lock-2"
    env0 = _AsyncBenchWorker(device_id=0)._child_env()
    assert env0["DEPLODOCK_GPU_LOCK"] == "/tmp/deplodock-gpu.lock-0"
    assert env["DEPLODOCK_GPU_LOCK"] != env0["DEPLODOCK_GPU_LOCK"]


def test_child_env_no_lock_suffix_when_base_unset(monkeypatch) -> None:
    monkeypatch.delenv(config.GPU_LOCK, raising=False)
    env = _AsyncBenchWorker(device_id=1)._child_env()
    # No base lock → no per-device suffix (one worker per GPU has its own context).
    assert "DEPLODOCK_GPU_LOCK" not in env


@requires_cuda
def test_async_worker_real_roundtrip_single_gpu() -> None:
    """Smoke the real transport: the async inner-reward pool benches a tiny matmul
    end-to-end on GPU 0 through a real ``_AsyncBenchWorker`` (``create_subprocess_exec``
    + framed-pickle protocol + ``asyncio.wait_for`` wall cap), producing a measured
    per-op best. The transport mirrors the proven sync worker; this confirms the
    asyncio I/O round-trips on real hardware. ``prior=None`` keeps it off catboost."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline
    from deplodock.compiler.pipeline.search.db import SearchDB
    from tests.compiler.conftest import run_inner_reward

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (64, 128)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (128, 48)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (64, 48)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    fused = Pipeline.build(LOOP_PASSES).run(g, db=SearchDB())

    backend = CudaBackend(bench_compile_timeout_s=15.0, bench_run_timeout_s=15.0, bench_wall_timeout_s=40.0, device_id=0)
    try:
        reward = run_inner_reward(fused, ctx=Context.probe(), db=SearchDB(), backends=[backend], patience=2, prior=None)
    finally:
        backend.close_async_worker()
    assert reward.ok
    assert any(r.best_us and r.best_us > 0 for r in reward.per_op)
