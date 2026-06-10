"""Persistent benchmark worker — runs bench / comparison jobs (:func:`_run_job`) in
an isolated subprocess so the parent can SIGKILL on wall-timeout and
the dirty CUDA stream (or a non-terminating kernel) dies with the child.

Protocol (length-prefixed pickle on stdin/stdout):

- Parent writes ``<8-byte little-endian length><pickled request>``.
- One request shape, handled by :func:`_run_job`: ``{"graph": Graph, "nvcc_flags": str|None,
  "torch_spec": None | (...), ...}``. ``torch_spec`` selects the work — ``None`` is a pure
  deplodock bench (the autotune sweep, ``kwargs`` carries warmup / num_iters / timeouts), otherwise
  the deployable eager / torch.compile / deplodock comparison (``tune --bench`` / ``run --bench``),
  rebuilt and run *here* so a hung kernel hangs this child (SIGKILL-recoverable).
- Response: ``{"ok": True, "result": BenchmarkResult, "results": dict|None, "torch_available": bool}``
  or ``{"ok": False, "error": str, "traceback": str}`` on exception.
- Worker imports cupy / torch lazily on first request, writes ``<8-byte length><pickled response>``.
- EOF on stdin (or parent SIGKILL) terminates the worker.

Errors raised inside ``benchmark_program`` (bench_compile_timeout_s,
bench_run_timeout_s, per-launch ``_KERNEL_TIMEOUT_MS``) propagate back
as ``ok: False`` and the parent surfaces them as ``RuntimeError``.

A failing bench may have left the CUDA context in a sticky-error state: an
illegal / misaligned memory access corrupts the context so that *every*
subsequent CUDA call returns the same error until the context is destroyed.
Reusing the worker would then cascade identical false bench_fails across all
later configs. So after any error we probe the context (:func:`_context_dirty`)
and, if it's poisoned, send the response and exit — the parent's next request
sees a dead process and respawns a clean context. Benign failures (NVRTC
compile errors, OOM that's cleaned up) leave the context healthy and keep the
worker alive, so they don't pay the respawn cost.
"""

from __future__ import annotations

import os
import pickle
import sys
import traceback


def _read_n(fd: int, n: int) -> bytes:
    out = bytearray()
    while len(out) < n:
        chunk = os.read(fd, n - len(out))
        if not chunk:
            return bytes(out)  # short read → caller treats as EOF
        out.extend(chunk)
    return bytes(out)


def _hung(exc: BaseException) -> bool:
    """``True`` iff ``exc`` is the per-launch hung-kernel watchdog. A hung kernel is still
    resident on the device, so :func:`_context_dirty`'s ``deviceSynchronize`` probe would block
    on it forever — treat it as dirty without probing, so the worker exits promptly and process
    teardown kills the kernel."""
    from deplodock.compiler.backend.cuda.program import HungKernelError

    return isinstance(exc, HungKernelError)


def _run_job(req: dict) -> dict:
    """Run one bench job; return ``{"result": BenchmarkResult, "results": dict|None,
    "torch_available": bool, "captured": bool}`` (``captured``: timings came from
    CUDA-graph-captured windows; False = the all-or-nothing uncaptured fallback ran).

    ``req["torch_spec"]`` picks the work — a pure deplodock bench is just the comparison with a
    no-op torch request:

    - ``None`` → the deplodock-only autotune bench (``benchmark_program``), no torch comparison.
    - ``("trace_args", {code/input/layer/seq_len/dynamic})`` → ``load_or_trace`` rebuilds the real
      module here (HF model id or ``--code`` expr) → ``bench_full_model_real``.
    - ``("frontend_graph", Graph | None)`` → ``bench_lowered_vs_torch`` (per-kernel reproducer).

    Rebuilding the torch side **here** (not pickling a live module) means a hung deplodock kernel
    hangs *this* child, which the parent SIGKILLs — recovering the device. ``nvcc_flags`` re-points
    the compile at a given opt level (the cubin cache key folds it in)."""
    from deplodock import config

    with config.nvcc_flags_override(req.get("nvcc_flags")):
        spec = req.get("torch_spec")
        if spec is None:
            from deplodock.compiler.backend.cuda.program import benchmark_program

            result = benchmark_program(req["graph"], **req["kwargs"])
            return {"result": result, "results": None, "torch_available": False, "captured": result.captured}

        from deplodock.compiler.backend.cuda.backend import CudaBackend

        # In-process within this child — the parent's SIGKILL is the wall-timeout backstop.
        backend = CudaBackend(bench_compile_timeout_s=60.0, bench_run_timeout_s=60.0)
        kind, payload = spec
        if kind == "frontend_graph":
            from deplodock.commands.run import bench_lowered_vs_torch

            results, bench, avail, captured = bench_lowered_vs_torch(
                payload,
                req["graph"],
                backend,
                seed=req["seed"],
                do_bench=True,
                warmup=req["warmup"],
                iters=req["iters"],
                bench_backends=req["bench_backends"],
            )
        elif kind == "trace_args":
            import types

            from deplodock.commands.compile import load_or_trace
            from deplodock.commands.run import bench_full_model_real

            _, _, bundle = load_or_trace(types.SimpleNamespace(**payload))
            if bundle is None:
                raise RuntimeError("trace_args produced no runnable module (--ir JSON path has none)")
            module, args_t, kwargs = bundle
            results, bench, captured = bench_full_model_real(
                module,
                args_t,
                kwargs,
                req["graph"],
                backend,
                warmup=req["warmup"],
                iters=req["iters"],
                bench_backends=req["bench_backends"],
            )
            avail = True
        else:
            raise ValueError(f"unknown torch_spec kind: {kind!r}")
        return {"result": bench, "results": results, "torch_available": avail, "captured": captured}


def _context_dirty() -> bool:
    """``True`` iff the live CUDA context is in a sticky-error state.

    A cheap ``deviceSynchronize`` surfaces a context-wide sticky error (e.g.
    ``CUDA_ERROR_MISALIGNED_ADDRESS`` / ``CUDA_ERROR_ILLEGAL_ADDRESS``) left by
    a prior illegal access — those keep returning the same status on every call
    until the context is torn down. Returns ``False`` when cupy was never
    imported / no context exists (a compile-only failure touches no context),
    or when the sync succeeds (context healthy)."""
    cupy = sys.modules.get("cupy")
    if cupy is None:
        return False  # no CUDA context was ever created in this worker
    try:
        cupy.cuda.runtime.deviceSynchronize()
        return False
    except Exception:  # noqa: BLE001 — any CUDA error here means the context is unusable
        return True


def main() -> None:
    in_fd = sys.stdin.buffer.fileno()
    out_fd = sys.stdout.buffer.fileno()
    while True:
        header = _read_n(in_fd, 8)
        if len(header) < 8:
            return
        n = int.from_bytes(header, "little")
        body = _read_n(in_fd, n)
        if len(body) < n:
            return
        dirty = False
        try:
            resp = {"ok": True, **_run_job(pickle.loads(body))}
        except BaseException as exc:  # noqa: BLE001 — surface every failure mode to the parent
            resp = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
            dirty = _hung(exc) or _context_dirty()
        payload = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        os.write(out_fd, len(payload).to_bytes(8, "little"))
        os.write(out_fd, payload)
        if dirty:
            # Corrupted context — don't serve more requests from it. Exit so the
            # parent respawns a fresh context on its next bench (program.py
            # ``_BenchWorker.bench`` re-spawns when ``poll()`` shows us dead).
            return


if __name__ == "__main__":
    main()
