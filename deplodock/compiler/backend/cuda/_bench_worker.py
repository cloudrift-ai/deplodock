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
        try:
            req = pickle.loads(body)
            from deplodock.compiler.backend.cuda.program import benchmark_program

            result = benchmark_program(req["graph"], **req["kwargs"])
            resp = {"ok": True, "result": result}
        except BaseException as exc:  # noqa: BLE001 — surface every failure mode to the parent
            resp = {"ok": False, "error": repr(exc), "traceback": traceback.format_exc()}
        payload = pickle.dumps(resp, protocol=pickle.HIGHEST_PROTOCOL)
        os.write(out_fd, len(payload).to_bytes(8, "little"))
        os.write(out_fd, payload)


if __name__ == "__main__":
    main()
