"""The persistent bench worker must not serve from a corrupted CUDA context.

A kernel that does an illegal / misaligned memory access leaves the CUDA
context in a *sticky*-error state: every subsequent CUDA call returns the same
error until the context is destroyed. The bench worker is a long-lived
subprocess reused across every autotune config, so a single such crash used to
cascade identical false ``bench_fail``s across all later configs (and ops).

The worker now probes its context after a failure (``_context_dirty``) and
exits if it's poisoned, so the parent (``program.py`` ``_AsyncBenchWorker.run_job``)
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
        import emmy.compiler.backend.cuda.program as program
        def _corrupt(graph, **kw):
            import cupy
            k = cupy.RawKernel(r'extern "C" __global__ void oob(float* p){ p[268435456] = 1.0f; }', 'oob')
            buf = cupy.zeros(8, dtype=cupy.float32)
            k((1,), (1,), (buf,))
            cupy.cuda.runtime.deviceSynchronize()  # surfaces the sticky error
            raise RuntimeError('unreached')
        program.benchmark_program = _corrupt
        from emmy.compiler.backend.cuda._bench_worker import main
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
    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    w = _AsyncBenchWorker()
    assert w._proc is None
    w._kill()
    assert w._proc is None
    w._kill()  # repeated calls stay no-ops
    assert w._proc is None


def test_kill_releases_already_dead_subprocess() -> None:
    """A worker subprocess that exited on its own (e.g. dirty-context path) is
    still attached to ``self._proc``; ``_kill()`` must release it without raising
    (a dead proc has ``returncode`` set, so no SIGKILL is attempted)."""
    import asyncio

    from emmy.compiler.backend.cuda.program import _AsyncBenchWorker

    async def _run() -> None:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import sys; sys.exit(0)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert await proc.wait() == 0
        w = _AsyncBenchWorker()
        w._proc = proc
        w._kill()
        assert w._proc is None

    asyncio.run(_run())


def test_bench_retries_after_broken_pipe_on_first_write(monkeypatch) -> None:
    """The dirty-context exit path can race a respawn: the first ``stdin.drain``
    raises ``BrokenPipeError`` (the worker's read end is gone). ``run_job`` must
    respawn and retry the send once before surfacing the failure."""
    import asyncio

    from emmy.compiler.backend import BenchmarkResult
    from emmy.compiler.backend.cuda import program as P

    response_body = pickle.dumps(
        {"ok": True, "result": BenchmarkResult(time_ms=42.0, num_launches=0)},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    response_wire = len(response_body).to_bytes(8, "little") + response_body

    class _FakeStdin:
        def __init__(self, *, broken: bool) -> None:
            self._broken = broken

        def write(self, _data: bytes) -> None:
            pass

        async def drain(self) -> None:
            if self._broken:
                raise BrokenPipeError("stale worker — read end closed")

    class _FakeStdout:
        def __init__(self, wire: bytes) -> None:
            self._buf = bytearray(wire)

        async def readexactly(self, n: int) -> bytes:
            if len(self._buf) < n:
                raise asyncio.IncompleteReadError(bytes(self._buf), n)
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
            return chunk

    class _FakeStderr:
        async def read(self) -> bytes:
            return b""

    class _FakeProc:
        def __init__(self, *, fail_send: bool) -> None:
            self.pid = 1000 + (0 if fail_send else 1)
            self.returncode = None
            self.stdin = _FakeStdin(broken=fail_send)
            self.stdout = _FakeStdout(b"" if fail_send else response_wire)
            self.stderr = _FakeStderr()

        def kill(self) -> None:
            self.returncode = -9

        async def wait(self) -> int:
            return -9

    procs = [_FakeProc(fail_send=True), _FakeProc(fail_send=False)]
    spawn_count = 0

    async def fake_spawn(self: P._AsyncBenchWorker) -> None:
        nonlocal spawn_count
        self._proc = procs[spawn_count]
        spawn_count += 1

    monkeypatch.setattr(P._AsyncBenchWorker, "_spawn", fake_spawn)

    w = P._AsyncBenchWorker()
    resp = asyncio.run(w.run_job({"graph": None, "torch_spec": None, "kwargs": {}}, wall_timeout_s=5.0))

    assert spawn_count == 2, "BrokenPipeError on first send must trigger one respawn"
    assert resp["result"].time_ms == 42.0


def test_worker_survives_benign_error() -> None:
    # A failure that never touches CUDA (e.g. an NVRTC compile error) leaves the
    # context healthy — the worker must stay alive and keep serving so the
    # autotune sweep doesn't pay a respawn per rejected config.
    child = """
        import emmy.compiler.backend.cuda.program as program
        def _benign(graph, **kw):
            raise ValueError('benign compile-like failure')
        program.benchmark_program = _benign
        from emmy.compiler.backend.cuda._bench_worker import main
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
