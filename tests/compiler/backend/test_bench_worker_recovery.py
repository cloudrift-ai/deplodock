"""The persistent bench worker must not serve from a corrupted CUDA context.

A kernel that does an illegal / misaligned memory access leaves the CUDA
context in a *sticky*-error state: every subsequent CUDA call returns the same
error until the context is destroyed. The bench worker is a long-lived
subprocess reused across every autotune config, so a single such crash used to
cascade identical false ``bench_fail``s across all later configs (and ops).

The worker now probes its context after a failure (``_context_dirty``) and
exits if it's poisoned, so the parent (``program.py`` ``_BenchWorker.bench``)
respawns a clean context on the next request. A benign failure (NVRTC compile
error, etc.) leaves the context healthy and keeps the worker alive.

These tests drive the real ``_bench_worker.main`` loop over the real
length-prefixed pickle protocol, with ``benchmark_program`` monkeypatched in the
child to either corrupt the context or raise without touching CUDA.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import textwrap

from ..conftest import requires_cuda

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _spawn(child_src: str) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(child_src)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=PROJECT_ROOT,
    )


def _send(proc: subprocess.Popen, obj: object) -> None:
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(len(body).to_bytes(8, "little"))
    proc.stdin.write(body)
    proc.stdin.flush()


def _recv(proc: subprocess.Popen) -> dict | None:
    header = proc.stdout.read(8)
    if len(header) < 8:
        return None  # worker exited / closed stdout
    n = int.from_bytes(header, "little")
    return pickle.loads(proc.stdout.read(n))


# The patched ``benchmark_program`` ignores the request graph, so the parent can
# send a dummy request — no real compiled Graph needed.
_DUMMY_REQ = {"graph": None, "kwargs": {}}


@requires_cuda
def test_worker_exits_after_context_corruption() -> None:
    # Child: first bench launches an out-of-bounds write (sticky illegal
    # access), then raises. The worker should detect the dirty context, answer
    # the first request with the error, and exit — so the *second* request gets
    # no response (EOF), proving it won't serve from the poisoned context.
    child = """
        import deplodock.compiler.backend.cuda.program as program
        def _corrupt(graph, **kw):
            import cupy
            k = cupy.RawKernel(r'extern "C" __global__ void oob(float* p){ p[268435456] = 1.0f; }', 'oob')
            buf = cupy.zeros(8, dtype=cupy.float32)
            k((1,), (1,), (buf,))
            cupy.cuda.runtime.deviceSynchronize()  # surfaces the sticky error
            raise RuntimeError('unreached')
        program.benchmark_program = _corrupt
        from deplodock.compiler.backend.cuda._bench_worker import main
        main()
    """
    proc = _spawn(child)
    try:
        _send(proc, _DUMMY_REQ)
        resp1 = _recv(proc)
        assert resp1 is not None and resp1["ok"] is False
        # The worker must not answer a second request — its context is dirty.
        _send(proc, _DUMMY_REQ)
        resp2 = _recv(proc)
        assert resp2 is None, f"worker kept serving from a corrupted context: {resp2}"
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_kill_idempotent_when_no_proc() -> None:
    """``_kill()`` on a worker that was never spawned must be a silent no-op."""
    from deplodock.compiler.backend.cuda.program import _BenchWorker

    w = _BenchWorker()
    assert w._proc is None
    w._kill()
    assert w._proc is None
    w._kill()  # repeated calls stay no-ops
    assert w._proc is None


def test_kill_releases_already_dead_subprocess() -> None:
    """A worker subprocess that exited on its own (e.g. dirty-context path) is
    still attached to ``self._proc``; ``_kill()`` must release it without
    raising even though ``kill()`` on a reaped pid would normally surface a
    ``ProcessLookupError``."""
    from deplodock.compiler.backend.cuda.program import _BenchWorker

    proc = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.wait(timeout=5) == 0

    w = _BenchWorker()
    w._proc = proc
    w._kill()
    assert w._proc is None


def test_bench_retries_after_broken_pipe_on_first_write(monkeypatch) -> None:
    """The dirty-context exit path can race the parent's ``poll()`` check: by
    the time we write the next request, the worker's stdin has been closed
    but ``poll()`` may not yet show the exit. The first ``stdin.write``
    then raises ``BrokenPipeError``. ``bench()`` must respawn and retry the
    send once before surfacing the failure."""
    from deplodock.compiler.backend import BenchmarkResult
    from deplodock.compiler.backend.cuda import program as P

    spawn_count = 0

    class _FakeStdin:
        """A real pipe write end (the send path uses ``fileno()`` + ``os.write``).
        ``broken=True`` closes the read end up front, so the first ``os.write``
        raises ``BrokenPipeError`` organically — the stale-worker race."""

        def __init__(self, *, broken: bool) -> None:
            r, w = os.pipe()
            self._w = w
            if broken:
                os.close(r)
            else:
                self._r = r  # keep the read end open; the small request fits the pipe buffer

        def fileno(self) -> int:
            return self._w

    class _FakeStderr:
        def read(self) -> bytes:
            return b""

    class _FakeProc:
        def __init__(self, *, fail_first_write: bool) -> None:
            self.pid = 1000 + (0 if fail_first_write else 1)
            self.stdin = _FakeStdin(broken=fail_first_write)
            self.stdout = type("S", (), {"fileno": lambda self: -1})()
            self.stderr = _FakeStderr()
            self._alive = True

        def poll(self) -> int | None:
            return None if self._alive else 0

        def kill(self) -> None:
            self._alive = False

        def wait(self, timeout: float | None = None) -> int:
            return 0

    procs = [_FakeProc(fail_first_write=True), _FakeProc(fail_first_write=False)]

    def fake_spawn(self: P._BenchWorker) -> None:
        nonlocal spawn_count
        self._proc = procs[spawn_count]
        spawn_count += 1

    # Construct the wire-format response the second proc "writes" back so the
    # recv loop pickle-loads a real BenchmarkResult.
    response_body = pickle.dumps(
        {"ok": True, "result": BenchmarkResult(time_ms=42.0, num_launches=0)},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    response_wire = len(response_body).to_bytes(8, "little") + response_body
    read_pos = [0]

    def fake_select(rlist, wlist, xlist, timeout):  # noqa: ARG001
        return rlist, wlist, []  # writes (real pipe fds) and reads (faked below) both "ready"

    def fake_read(fd: int, n: int) -> bytes:  # noqa: ARG001
        chunk = response_wire[read_pos[0] : read_pos[0] + n]
        read_pos[0] += len(chunk)
        return chunk

    monkeypatch.setattr(P._BenchWorker, "_spawn", fake_spawn)
    monkeypatch.setattr(P.select, "select", fake_select)
    monkeypatch.setattr(P._os, "read", fake_read)

    w = P._BenchWorker()
    resp = w.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=5.0)

    assert spawn_count == 2, "BrokenPipeError on first write must trigger one respawn"
    assert resp["result"].time_ms == 42.0


def test_worker_survives_benign_error() -> None:
    # A failure that never touches CUDA (e.g. an NVRTC compile error) leaves the
    # context healthy — the worker must stay alive and keep serving so the
    # autotune sweep doesn't pay a respawn per rejected config.
    child = """
        import deplodock.compiler.backend.cuda.program as program
        def _benign(graph, **kw):
            raise ValueError('benign compile-like failure')
        program.benchmark_program = _benign
        from deplodock.compiler.backend.cuda._bench_worker import main
        main()
    """
    proc = _spawn(child)
    try:
        _send(proc, _DUMMY_REQ)
        resp1 = _recv(proc)
        assert resp1 is not None and resp1["ok"] is False
        # Still alive: a second request gets a real response, not EOF.
        _send(proc, _DUMMY_REQ)
        resp2 = _recv(proc)
        assert resp2 is not None and resp2["ok"] is False, "worker exited on a benign error"
    finally:
        proc.stdin.close()
        if proc.poll() is None:
            proc.wait(timeout=5)
        if proc.poll() is None:
            proc.kill()
