"""Compile-time hardware target selection.

The tile-IR passes that gate on compute capability (``050_use_tma`` for
TMA, ``060_use_async_copy`` for cp.async) read the target via
:func:`compute_capability`. By default that probes the live CUDA device
through cupy. Callers can override the target via :func:`set_target` so
the compiler emits code for a different architecture than the host —
useful for ``deplodock compile`` on a CPU box that wants to see the
sm_120 codegen path, or for cross-compiling for a benchmark target.

CLI commands attach the ``--target`` flag with :func:`add_target_arg`
and resolve it via :func:`apply_target_arg`. Both forms accept the
canonical NVIDIA spelling (``sm_80``, ``sm_90``, ``sm_120``); the
optional architecture suffix on Hopper (``sm_90a``) is stripped.
"""

from __future__ import annotations

import functools
import logging
import re

_logger = logging.getLogger(__name__)

_OVERRIDE: tuple[int, int] | None = None


def parse_sm(spec: str) -> tuple[int, int]:
    """Parse ``sm_NN`` / ``sm_NNN`` (with optional arch suffix) into ``(major, minor)``.

    Examples: ``sm_80`` → (8, 0), ``sm_86`` → (8, 6), ``sm_90`` → (9, 0),
    ``sm_90a`` → (9, 0), ``sm_120`` → (12, 0).
    """
    m = re.fullmatch(r"sm_(\d+)[a-z]?", spec.strip().lower())
    if not m:
        raise ValueError(f"invalid SM target {spec!r} — expected e.g. sm_80, sm_90, sm_120")
    digits = m.group(1)
    return (int(digits[:-1]), int(digits[-1]))


def set_target(cap: tuple[int, int] | None) -> None:
    """Set (or clear) the compile-time compute-capability override.

    Pass ``None`` to revert to live device probing. Clears the
    :func:`compute_capability` cache so the next caller sees the change.
    """
    global _OVERRIDE
    _OVERRIDE = cap
    compute_capability.cache_clear()


def add_target_arg(parser, *, dest: str = "target") -> None:
    """Add a ``--target sm_NN`` argument to ``parser``.

    Commands that produce or run code (``compile``, ``run``, ``bench``)
    use this so the same flag means the same thing everywhere. The
    parsed value is a string; pass it to :func:`apply_target_arg` after
    parsing to install the override.
    """
    parser.add_argument(
        "--target",
        dest=dest,
        default=None,
        metavar="sm_NN",
        help=(
            "Compile-time target compute capability (e.g. sm_80, sm_90, sm_120). "
            "Overrides the live device's capability, so passes that gate on "
            "hardware features (TMA, cp.async) take the same path they would on "
            "the target GPU. Default: probe the active CUDA device."
        ),
    )


def apply_target_arg(args, *, dest: str = "target") -> None:
    """Install the target from a parsed-args namespace, if set."""
    spec = getattr(args, dest, None)
    if spec is None:
        return
    cap = parse_sm(spec)
    set_target(cap)
    _logger.info("compile target set to sm_%d%d (override)", cap[0], cap[1])


@functools.cache
def compute_capability() -> tuple[int, int]:
    """Active compute capability as ``(major, minor)``.

    Returns the override set via :func:`set_target` if any; otherwise
    delegates to :func:`live_compute_capability`. Cached so repeated
    rule firings don't re-query the driver; :func:`set_target` clears
    the cache.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    return live_compute_capability()


@functools.cache
def live_compute_capability() -> tuple[int, int]:
    """The live CUDA device's compute capability, ignoring any
    :func:`set_target` override.

    Returns ``(0, 0)`` when cupy is unavailable. Used by ``Context.probe``
    to size ``max_dynamic_smem`` to what the actual hardware can honor
    even when the target-derived gate cap is higher.
    """
    try:
        import cupy as cp

        dev = cp.cuda.Device()
        # cupy returns the capability as a string ``"MMm"``: ``"86"`` for
        # sm_86, ``"120"`` for sm_12.0. Minor is always the last digit.
        cap = str(dev.compute_capability)
        return (int(cap[:-1]), int(cap[-1]))
    except Exception as e:  # pragma: no cover
        _logger.debug("live_compute_capability query failed (%s)", e)
        return (0, 0)
