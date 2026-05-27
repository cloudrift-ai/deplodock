"""Offline kernel compilation via the ``nvcc`` binary (ptxas), as a faster
drop-in for cupy's in-process NVRTC path.

cupy's cold compile goes NVRTC → PTX → **driver JIT** (PTX→SASS), which is both
slow and globally serialized across processes. ``nvcc --cubin`` runs **offline
ptxas** instead: ~3x faster on the complex tile-search kernels that dominate
autotune, GPU-free for the compile step, and the cubin loads with no driver
JIT (~25ms). Same SASS quality → identical kernel output + latency (validated).

The two halves are split on purpose:

- :func:`compile_to_cubin` — ``nvcc --cubin`` into a content-addressed disk
  cache. GPU-free and independent per kernel, so a compile **pool** can warm
  the cache off the GPU (the planned next step).
- :func:`load_function` — ensure the cubin exists, then ``RawModule`` load it
  on the GPU. This is all the bench worker needs once the cache is warm.

Falls back to ``cp.RawKernel`` (NVRTC) when ``nvcc`` is absent or a compile
fails, so correctness never depends on the toolkit being installed. Set
``DEPLODOCK_NO_NVCC=1`` to force the cupy path.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from deplodock import config

logger = logging.getLogger(__name__)

# Base nvcc flags deplodock always compiles with (matches ``program._nvrtc_options``).
_BASE_FLAGS = ["--use_fast_math"]


def effective_flags() -> list[str]:
    """The full nvcc flag list: the base flags plus any extra flags from the
    ``DEPLODOCK_NVCC_FLAGS`` env var (space-separated), which the CLI commands
    set — ``tune`` defaults to ``-Xcicc -O1`` (fast compile; see the O1 note in
    ``compile_to_cubin``), ``compile`` / ``run`` to nvcc's default cicc -O3.
    Read fresh each call so a per-invocation override / the bench-worker
    subprocess (which inherits the env) both see the same value, and so the
    flags fold into the cache key."""
    return [*_BASE_FLAGS, *config.nvcc_flags().split()]


def cubin_cache_dir() -> Path:
    """Directory holding the content-addressed cubin cache (``DEPLODOCK_CUBIN_CACHE``)."""
    return config.cubin_cache_dir()


def clear_cubin_cache() -> None:
    """Delete the entire cubin cache (used by ``deplodock tune --clean``)."""
    shutil.rmtree(cubin_cache_dir(), ignore_errors=True)


@functools.cache
def nvcc_path() -> str | None:
    """Resolve the ``nvcc`` binary (PATH, then ``$CUDA_HOME``/``$CUDA_PATH``),
    or ``None`` when unavailable. Cached — looked up once per process."""
    if config.nvcc_disabled():
        return None
    found = shutil.which("nvcc")
    if found:
        return found
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root and (cand := Path(root) / "bin" / "nvcc").exists():
            return str(cand)
    return None


def device_arch(uses_tma: bool) -> str:
    """``sm_<cc>`` for the live device, plus the ``a`` (arch-accelerated)
    suffix for TMA kernels — matching ``program._nvrtc_options``' arch."""
    import cupy as cp  # noqa: PLC0415

    cap = str(cp.cuda.Device().compute_capability)  # e.g. "120"
    return f"sm_{cap}" + ("a" if uses_tma else "")


@functools.cache
def _toolkit_tag() -> str:
    """Short digest of the ``nvcc`` toolchain (``nvcc --version``), folded into
    the cache key so a CUDA upgrade never reuses a cubin compiled by an older
    ptxas (which could emit different / worse SASS for the same source). Run
    once per process."""
    nvcc = nvcc_path()
    try:
        ver = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001 — fall back to the path; never block a compile on version probing
        ver = nvcc or "?"
    return hashlib.sha1(ver.encode()).hexdigest()[:12]


def _cache_key(source: str, name: str, arch: str) -> str:
    # Content-addressed: identical (source, name, arch, toolkit, flags) → same
    # cubin, so the persistent cache is safe to share across (even concurrent)
    # runs. Toolkit + flags are in the key so an nvcc / opt-level (e.g. tune's
    # -Xcicc -O1 vs compile's -O3) / flags change recompiles rather than
    # serving a stale or wrong-opt cubin.
    h = hashlib.sha1()
    for part in (source, name, arch, _toolkit_tag(), "\x1f".join(effective_flags())):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def compile_to_cubin(source: str, name: str, *, arch: str) -> Path:
    """Compile ``source`` to a cubin with ``nvcc --cubin``, content-addressed in
    the on-disk cache. Idempotent + atomic (compile to a temp file, then
    ``os.replace``) so concurrent compilers / the bench loader never observe a
    half-written cubin. GPU-free — safe to call from a worker pool. Raises
    ``RuntimeError`` if ``nvcc`` is unavailable; ``CalledProcessError`` on a
    compile error (caller decides whether to fall back).

    ⚠️  The opt level comes from :func:`effective_flags` (``DEPLODOCK_NVCC_FLAGS``).
    ``deplodock tune`` defaults to ``-Xcicc -O1`` to dodge a cicc/LLVM blowup on
    big unrolled register-tile kernels (up to ~200× faster compile), but **-O1
    is NOT runtime-optimal** — reduction/attention kernels can run ~1.5–3×
    slower than -O3, so tune-measured latencies are a *ranking* signal, not the
    deployed speed. ``compile`` / ``run`` use -O3; re-bench there for real
    numbers."""
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError("nvcc unavailable")
    cache = cubin_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_cache_key(source, name, arch)}.cubin"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(dir=cache) as td:
        cu = Path(td) / "k.cu"
        cu.write_text(source)
        tmp_cubin = Path(td) / "k.cubin"
        subprocess.run(
            [nvcc, "--cubin", f"-arch={arch}", *effective_flags(), "-o", str(tmp_cubin), str(cu)],
            check=True,
            capture_output=True,
        )
        os.replace(tmp_cubin, out)  # atomic publish
    return out


def load_function(source: str, name: str, options, *, uses_tma: bool):
    """Compile (via nvcc, cached) + ``RawModule``-load ``name``, returning a
    cupy ``Function`` usable exactly like a ``RawKernel`` at launch (callable,
    and ``max_dynamic_shared_size_bytes`` is settable for the >48KB smem path).

    Falls back to ``cp.RawKernel`` (lazy NVRTC) when nvcc is unavailable or the
    compile fails — so this is always safe to drop in for ``cp.RawKernel``."""
    import cupy as cp  # noqa: PLC0415

    if nvcc_path() is None:
        return cp.RawKernel(source, name, options=tuple(options))
    try:
        cubin = compile_to_cubin(source, name, arch=device_arch(uses_tma))
        return cp.RawModule(path=str(cubin)).get_function(name)
    except Exception as exc:  # noqa: BLE001 — any nvcc/load failure → safe NVRTC fallback
        detail = exc.stderr.decode(errors="replace")[-400:] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else exc
        logger.warning("nvcc compile failed for kernel %r — falling back to NVRTC (%s)", name, detail)
        return cp.RawKernel(source, name, options=tuple(options))
