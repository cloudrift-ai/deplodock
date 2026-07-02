"""``tune --bench`` runs its deployable benches in the SIGKILL-able worker, so a hung kernel can't
wedge the run — and, because the parent device stays clean, a failed full-model bench no longer has
to skip the per-kernel sweep (the pre-isolation behavior); it just continues.

A non-terminating kernel trips the worker's per-launch watchdog (``HungKernelError``) and the parent
SIGKILLs the child, surfacing as a ``RuntimeError`` to ``_run_bench``. These tests drive the
``_run_bench`` control flow with the worker call (``benchmark_compare_isolated_async``) and the per-kernel
sweep stubbed — no CUDA. The real-GPU recovery is covered in
``tests/compiler/backend/test_bench_worker_compare.py``.
"""

from __future__ import annotations

import types

import emmy.commands.run as run_mod
import emmy.commands.tune as tune_mod
import emmy.compiler.backend.cuda.backend as backend_mod
import emmy.compiler.backend.cuda.program as program_mod
from emmy import config
from emmy.compiler.backend.cuda.program import HungKernelError


class _DummyBackend:
    """Stands in for ``CudaBackend`` — ``tune_db=None`` keeps ``_run_bench`` off the SearchDB path."""

    def __init__(self, *, tune_db=None, **_kw) -> None:  # noqa: ARG002
        self.tune_db = None


def _args() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        nvcc_flags=None,
        warmup=1,
        iters=1,
        seed=0,
        bench_backends="emmy",
        code=None,
        input="some/model",
        layer=0,
        seq_len=32,
        dynamic=None,
    )


def _patch_common(monkeypatch, *, compare_raises: Exception | None) -> list[bool]:
    """Stub the worker compare call + the per-kernel sweep; return a flag flipped iff per-kernel ran."""
    per_kernel_ran = [False]

    async def _fake_compare(**_kw):
        if compare_raises is not None:
            raise compare_raises
        return {"Emmy": 1.0}, object(), True, False  # (results, bench, torch_available, captured)

    def _fake_per_kernel(*_a, **_k):
        per_kernel_ran[0] = True
        return [], []

    monkeypatch.setattr(backend_mod, "CudaBackend", _DummyBackend)
    monkeypatch.setattr(program_mod, "benchmark_compare_isolated_async", _fake_compare)
    monkeypatch.setattr(tune_mod, "_bench_per_kernel", _fake_per_kernel)
    monkeypatch.setattr(run_mod, "_print_table", lambda *_a, **_k: None)
    monkeypatch.setenv(config.NVCC_FLAGS, "")  # registers cleanup of the flag _run_bench sets
    return per_kernel_ran


def test_hung_kernel_error_is_runtimeerror() -> None:
    # Subclassing RuntimeError keeps every existing ``except RuntimeError`` (the autotune sweep's
    # bench_fail handling, _run_bench's continue) catching it unchanged.
    assert issubclass(HungKernelError, RuntimeError)


def test_run_bench_continues_to_per_kernel_on_full_model_failure(monkeypatch) -> None:
    # The worker SIGKILLs a hung kernel and surfaces a RuntimeError; the parent device is clean
    # (the bench ran in the child), so the per-kernel sweep must still run — no skip.
    ran = _patch_common(monkeypatch, compare_raises=RuntimeError("bench worker exceeded wall budget — SIGKILL'd"))
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert ran[0] is True, "per-kernel bench must still run after an isolated full-model failure"


def test_run_bench_runs_per_kernel_on_success(monkeypatch) -> None:
    ran = _patch_common(monkeypatch, compare_raises=None)
    dump = types.SimpleNamespace(dir="/tmp/does-not-matter")

    tune_mod._run_bench(_args(), ("module", "args", "kwargs"), assembled=None, dump=dump, html_dir=None)

    assert ran[0] is True, "per-kernel bench must run after a successful full-model bench"
