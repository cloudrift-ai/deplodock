"""Persistent benchmark worker — runs ``benchmark_program`` requests in
an isolated subprocess so the parent can SIGKILL on wall-timeout and
the dirty CUDA stream dies with the child.

Protocol (length-prefixed pickle on stdin/stdout):

- Parent writes ``<8-byte little-endian length><pickled request>``.
- Request is ``{"graph": Graph, "kwargs": dict}``.
- Worker imports cupy lazily on first request, calls ``benchmark_program``,
  serializes ``{"ok": True, "result": BenchmarkResult}`` (or
  ``{"ok": False, "error": str, "traceback": str}`` on exception) and
  writes ``<8-byte length><pickled response>`` to stdout.
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
            req = pickle.loads(body)
            from deplodock.compiler.backend.cuda.program import benchmark_program

            result = benchmark_program(req["graph"], **req["kwargs"])
            resp = {"ok": True, "result": result}
        except BaseException as exc:  # noqa: BLE001 — surface every failure mode to the parent
            resp = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
            dirty = _context_dirty()
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
