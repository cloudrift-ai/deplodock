"""``tune --bench`` must not march into a CUDA device that a hung kernel poisoned.

A non-terminating kernel trips the per-launch watchdog (``HungKernelError``), but the
kernel stays *resident* on the device — the in-process deployable bench can't SIGKILL-reset
it the way the isolated tuning worker can. If ``_run_bench`` continued to the per-kernel
sweep after that, the sweep's torch peer-bench would block forever on ``synchronize()``
behind the still-running kernel (observed: a 109-minute wedge). So a ``HungKernelError`` from
the full-model bench must skip the per-kernel sweep, while a *benign* ``RuntimeError`` (e.g. a
slow-compiling kernel, device healthy) must still fall through to it.

These tests drive the real ``_run_bench`` control flow with the GPU-touching collaborators
(``CudaBackend`` / ``bench_full_model_real`` / ``_bench_per_kernel``) stubbed out — no CUDA.
"""

from __future__ import annotations

import types

import deplodock.commands.run as run_mod
import deplodock.commands.tune as tune_mod
import deplodock.compiler.backend.cuda.backend as backend_mod
from deplodock import config
from deplodock.compiler.backend.cuda.program import HungKernelError


class _DummyBackend:
    """Stands in for ``CudaBackend`` — ``tune_db=None`` keeps ``_run_bench`` off the SearchDB path."""

    def __init__(self, *, tune_db=None, **_kw) -> None:  # noqa: ARG002
        self.tune_db = None


def _args() -> types.SimpleNamespace:
    return types.SimpleNamespace(nvcc_flags=None, warmup=1, iters=1, seed=0, bench_backends="deplodock")


def _patch_common(monkeypatch, *, full_model_raises: Exception) -> list[bool]:
    """Stub the GPU collaborators; return a one-element list flipped to True iff per-kernel ran."""
    called = [False]

    def _fake_full_model(*_a, **_k):
        raise full_model_raises

    def _fake_per_kernel(*_a, **_k):
        called[0] = True
        return []

    monkeypatch.setattr(backend_mod, "CudaBackend", _DummyBackend)
    monkeypatch.setattr(run_mod, "bench_full_model_real", _fake_full_model)
    monkeypatch.setattr(tune_mod, "_bench_per_kernel", _fake_per_kernel)
    monkeypatch.setenv(config.NVCC_FLAGS, "")  # registers cleanup of the flag _run_bench sets
    return called


def test_hung_kernel_error_is_runtimeerror() -> None:
    # Subclassing RuntimeError keeps every existing ``except RuntimeError`` (the autotune
    # sweep's bench_fail handling) catching it unchanged.
    assert issubclass(HungKernelError, RuntimeError)


def test_run_bench_skips_per_kernel_on_hung_kernel(monkeypatch) -> None:
    called = _patch_common(monkeypatch, full_model_raises=HungKernelError("kernel 'k_x' did not complete"))
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert called[0] is False, "per-kernel bench must be skipped after a hung kernel poisons the device"


def test_run_bench_runs_per_kernel_on_benign_runtime_error(monkeypatch) -> None:
    called = _patch_common(monkeypatch, full_model_raises=RuntimeError("slow-compiling kernel"))
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert called[0] is True, "a benign full-model failure (healthy device) must still run the per-kernel bench"
