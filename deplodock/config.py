"""Single source of truth for ``DEPLODOCK_*`` environment-variable handling.

Every read or write of a ``DEPLODOCK_*`` config var goes through this module.
It is intentionally stdlib-only (no ``deplodock`` imports) so that ``knob.py``
— imported transitively by every pipeline pass — can depend on it without a
cycle.

Design contract:

- ``os.environ`` stays the backing store. Bench-worker subprocesses
  (``backend/cuda/program.py``) and the ncu child (``commands/run.py``) spawn
  with ``env=dict(os.environ)``, so anything written here propagates to them;
  tests monkeypatch ``os.environ`` directly, so getters must read it **live**.
- Getters read ``os.environ`` on every call (never cache at import).
- The one setter, :func:`set_nvcc_flags`, centralizes the ``--flag > env >
  default`` override that used to live in the CLI layer, so every callsite
  (CLI, programmatic, tests) shares it.

Out of scope: provider/secret vars (``HF_TOKEN``, ``CLOUDRIFT_*``, ``GCP_*``,
``NO_COLOR``) — those stay at their use sites, and ``deplodock/redact.py`` owns
secret redaction. The dynamic ``DEPLODOCK_<KNOB>`` namespace is owned by
``compiler/pipeline/knob.py``; it borrows :data:`PREFIX` / :func:`knob_var` and
the parse primitives here but keeps its own descriptor logic.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Var-name constants (the single source of truth for spellings) ---------

PREFIX = "DEPLODOCK_"
TUNE_DB = "DEPLODOCK_TUNE_DB"
NVCC_FLAGS = "DEPLODOCK_NVCC_FLAGS"
DEBUG = "DEPLODOCK_DEBUG"
DUMP_DIR = "DEPLODOCK_DUMP_DIR"
KNOBS = "DEPLODOCK_KNOBS"
TUNE_PATIENCE = "DEPLODOCK_TUNE_PATIENCE"
BENCH_BACKENDS = "DEPLODOCK_BENCH_BACKENDS"
CUBIN_CACHE = "DEPLODOCK_CUBIN_CACHE"
NO_NVCC = "DEPLODOCK_NO_NVCC"
GPU_LOCK = "DEPLODOCK_GPU_LOCK"
NCU_CHILD = "DEPLODOCK_NCU_CHILD"

_CACHE_ROOT = Path.home() / ".cache" / "deplodock"


def knob_var(name: str) -> str:
    """The ``DEPLODOCK_<NAME>`` env-var key for a knob named ``name``.

    Sole place the knob-name → env-var join lives. Used by
    :class:`~deplodock.compiler.pipeline.knob.Knob` (via ``Knob.env``), the
    ``DEPLODOCK_KNOBS`` splat, and ``compiler/tuning.py``'s literal knob reads."""
    return f"{PREFIX}{name.upper()}"


def knob_raw(name: str) -> str | None:
    """Raw string value of the knob env var ``DEPLODOCK_<NAME>``, or ``None`` if
    unset. The per-type decode (INT / BOOL / BINMASK) stays in the ``Knob``
    descriptor (``compiler/pipeline/knob.py``); this is just the env read."""
    return os.environ.get(knob_var(name))


def knobs_aggregate() -> str:
    """Raw ``DEPLODOCK_KNOBS`` aggregate string (``""`` if unset)."""
    return _str(KNOBS)


def set_knob(name: str, value: str, *, overwrite: bool = True) -> bool:
    """Write ``DEPLODOCK_<NAME>=value`` into ``os.environ`` (so pipeline passes
    and bench subprocesses see it). With ``overwrite=False`` only writes when the
    key is absent. Returns ``True`` iff a write happened."""
    key = knob_var(name)
    if not overwrite and key in os.environ:
        return False
    os.environ[key] = value
    return True


# --- Shared parse primitives -----------------------------------------------

_TRUTHY = {"1", "true", "yes", "on"}


def _bool(name: str, default: bool = False) -> bool:
    """Truthy env read. ``{"1","true","yes","on"}`` (case-insensitive) → True;
    unset → ``default``; anything else → False."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


def int_env(name: str, default: int) -> int:
    """Int env read. Empty / unset / unparseable → ``default``."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


# --- Typed getters (read os.environ live) ----------------------------------


def tune_db_path() -> Path:
    """Autotune SQLite cache path: ``DEPLODOCK_TUNE_DB`` → ``~/.cache/deplodock/autotune.db``.

    The shared resolution for ``compile`` / ``run`` / ``tune`` / ``knobs`` and
    for :class:`CudaBackend` constructed with ``tune_db="auto"``. The path is
    advisory — the engine only opens it when the file exists."""
    override = os.environ.get(TUNE_DB)
    return Path(override) if override else _CACHE_ROOT / "autotune.db"


def nvcc_flags() -> str:
    """Extra nvcc flags for this compile (``DEPLODOCK_NVCC_FLAGS``, ``""`` if unset).

    Read fresh each call so a per-invocation override (set via
    :func:`set_nvcc_flags`) and the bench-worker subprocess (which inherits the
    env) see the same value, and so the flags fold into cache keys."""
    return _str(NVCC_FLAGS)


def debug_enabled() -> bool:
    """``DEPLODOCK_DEBUG`` — per-launch debug dump path in the CUDA backend."""
    return _bool(DEBUG)


def dump_dir() -> Path | None:
    """``DEPLODOCK_DUMP_DIR`` as an expanded ``Path``, or ``None`` when unset."""
    raw = os.environ.get(DUMP_DIR)
    return Path(raw).expanduser() if raw else None


def tune_patience(default: int = 50) -> int:
    """``DEPLODOCK_TUNE_PATIENCE`` — inner-MCTS patience fallback for ``tune``."""
    return int_env(TUNE_PATIENCE, default)


def bench_backends_raw(cli_value: str | None) -> str:
    """Raw comma-separated bench-backend selection. Precedence: ``cli_value`` >
    ``DEPLODOCK_BENCH_BACKENDS`` > ``"eager,deplodock"``. Backend-key
    normalization stays at the call site (``run.py:_resolve_backends``)."""
    return cli_value or os.environ.get(BENCH_BACKENDS) or "eager,deplodock"


def cubin_cache_dir() -> Path:
    """Content-addressed cubin cache dir: ``DEPLODOCK_CUBIN_CACHE`` → ``~/.cache/deplodock/cubin``."""
    override = os.environ.get(CUBIN_CACHE)
    return Path(override) if override else _CACHE_ROOT / "cubin"


def nvcc_disabled() -> bool:
    """``DEPLODOCK_NO_NVCC`` — force the cupy/NVRTC path instead of offline nvcc."""
    return _bool(NO_NVCC)


def gpu_lock_path() -> str | None:
    """``DEPLODOCK_GPU_LOCK`` path, or ``None`` for the no-op (unset) case."""
    return os.environ.get(GPU_LOCK)


def ncu_child() -> bool:
    """``DEPLODOCK_NCU_CHILD`` — set in the ncu-profiled child to prevent
    recursive re-spawning of ncu."""
    return _bool(NCU_CHILD)


def tma_swizzle_enabled() -> bool:
    """``DEPLODOCK_TMA_SWIZZLE`` — opt in to TMA hardware-swizzle modes."""
    return _bool(knob_var("tma_swizzle"))


# Allowed values for the CTA-swizzle group size. ``1`` is the no-op (row-major
# decode); ``8`` is the Triton/CUTLASS default the swizzle pass stamps.
_GROUP_M_ALLOWED = (1, 2, 4, 8, 16)
_GROUP_M_DEFAULT = 8


def group_m() -> int:
    """``DEPLODOCK_GROUP_M`` — CTA-swizzle row-group size used by
    ``compiler/pipeline/passes/lowering/tile/025_swizzle_blocks.py``.

    ``1`` disables the swizzle (renderer falls back to row-major decode).
    Unset → ``8`` (Triton/CUTLASS default). Garbage and out-of-set values
    raise ``ValueError`` so a typo doesn't silently degrade matmul perf.
    Reads :func:`knob_var` so it shares the ``DEPLODOCK_KNOBS`` splat path
    with the rest of the planner knobs."""
    raw = knob_raw("GROUP_M")
    if raw is None or raw == "":
        return _GROUP_M_DEFAULT
    try:
        v = int(raw)
    except ValueError as e:
        raise ValueError(f"DEPLODOCK_GROUP_M must be one of {_GROUP_M_ALLOWED}, got {raw!r}") from e
    if v not in _GROUP_M_ALLOWED:
        raise ValueError(f"DEPLODOCK_GROUP_M must be one of {_GROUP_M_ALLOWED}, got {v}")
    return v


# --- Setters (write os.environ so subprocesses inherit) --------------------


def set_nvcc_flags(cli_value: str | None, default: str) -> str:
    """Resolve and publish the effective extra nvcc flags via
    ``DEPLODOCK_NVCC_FLAGS`` (the carrier the cubin compiler, the bench-worker
    subprocess, and ``Context.structural_key`` all read).

    Precedence: ``cli_value`` (a ``--nvcc-flags`` override, when not ``None``) >
    a pre-set env var > ``default`` (per-command policy: ``""`` for compile/run,
    ``"-Xcicc -O1"`` for tune). Must run before any compile/bench. Returns the
    effective string."""
    if cli_value is not None:
        os.environ[NVCC_FLAGS] = cli_value
    elif NVCC_FLAGS not in os.environ:
        os.environ[NVCC_FLAGS] = default
    return os.environ.get(NVCC_FLAGS, "")
