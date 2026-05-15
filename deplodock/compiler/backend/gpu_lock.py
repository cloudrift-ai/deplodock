"""Cross-process GPU lock used by the CUDA backend.

Every entry point that issues CUDA work (NVRTC compile, kernel launch,
bench loop) acquires this lock so that concurrent worker processes —
``make bench-kernels``, ``make bench-kernels-tune``, parallel
``deplodock run`` invocations from xdist — never interleave kernels on
the same GPU. Without it, two processes' kernels share clocks / caches /
thermal state and timings turn into noise (we saw 2× variance on tiny
ops like ``rmsnorm`` and ``silu_mul`` at small seqlens).

Activated when ``DEPLODOCK_GPU_LOCK`` is set to a path (the perf
conftest exports ``/tmp/deplodock-gpu.lock``); otherwise the context
manager is a no-op so ad-hoc ``deplodock run`` invocations don't pay
any coordination overhead.

Re-entrant within a single process: the same thread can ``with
gpu_lock():`` nested arbitrarily deep. ``filelock`` already handles this
on a per-instance basis; we share the instance via ``_LOCK_CACHE`` so
nested calls inside the same process don't deadlock against themselves.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

_LOCK_CACHE: dict[str, object] = {}


def _resolve_lock():
    """Return the cached ``FileLock`` instance, or ``None`` for no-op."""
    path = os.environ.get("DEPLODOCK_GPU_LOCK")
    if not path:
        return None
    cached = _LOCK_CACHE.get(path)
    if cached is not None:
        return cached
    from filelock import FileLock  # noqa: PLC0415 — optional dep, deferred

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(path)
    _LOCK_CACHE[path] = lock
    return lock


@contextlib.contextmanager
def gpu_lock() -> Iterator[None]:
    """Hold the cross-process GPU lock for the duration of the block.

    No-op when ``DEPLODOCK_GPU_LOCK`` is unset. Otherwise wraps the
    shared ``FileLock`` so any code inside is guaranteed sole access
    to the device across processes."""
    lock = _resolve_lock()
    if lock is None:
        yield
        return
    with lock:
        yield
