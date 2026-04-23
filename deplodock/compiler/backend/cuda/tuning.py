"""GPU-specific default hint profiles for the matmul kernel.

Per-GPU tunings for the SGEMM bench. Profiles are indexed by size (max square
dimension) and produce a ``MatmulHints`` dict that downstream matmul codegen
reads. Dispatch is by GPU name because sm_120 cards (5090, RTX PRO 6000)
share the same compute capability but tune to different values. Unknown GPUs
fall back to the 5090 profile.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

# Type alias for a matmul hint dict (keys are cuda.matmul.* field names
# without the prefix, e.g. {"block_k": 32, "thread_m": 8}).
MatmulHints = dict[str, Any]


def _hints(bk: int = 32, tm: int = 8, ks: int = 1) -> MatmulHints:
    return {
        "block_k": bk,
        "threads_y": 8,
        "threads_x": 32,
        "thread_m": tm,
        "k_splits": ks,
    }


# Per-GPU strategy maps. Each entry is `(max_size, hints)` — pick the first
# entry whose threshold is >= the matrix size. Tuned for square FP32 matmul.
_PROFILE_5090: list[tuple[int, MatmulHints]] = [
    (256, _hints(bk=32, tm=8, ks=4)),
    (512, _hints(bk=32, tm=8, ks=4)),
    (1024, _hints(bk=32, tm=8, ks=1)),
    (2048, _hints(bk=32, tm=26, ks=1)),
    (4096, _hints(bk=32, tm=20, ks=1)),
    (8192, _hints(bk=32, tm=28, ks=1)),
    (16384, _hints(bk=32, tm=28, ks=1)),
]

_PROFILE_H200: list[tuple[int, MatmulHints]] = [
    (256, _hints(bk=32, tm=8, ks=4)),
    (512, _hints(bk=32, tm=8, ks=4)),
    (1024, _hints(bk=32, tm=8, ks=1)),
    (2048, _hints(bk=32, tm=8, ks=1)),
    (4096, _hints(bk=32, tm=8, ks=1)),
    (8192, _hints(bk=32, tm=8, ks=1)),
    (16384, _hints(bk=32, tm=8, ks=1)),
]

_PROFILE_PRO6000: list[tuple[int, MatmulHints]] = [
    (256, _hints(bk=32, tm=8, ks=4)),
    (512, _hints(bk=32, tm=8, ks=4)),
    (1024, _hints(bk=32, tm=8, ks=4)),
    (2048, _hints(bk=32, tm=26, ks=1)),
    (4096, _hints(bk=32, tm=26, ks=1)),
    (8192, _hints(bk=32, tm=26, ks=1)),
    (16384, _hints(bk=32, tm=26, ks=1)),
]

# Match against the GPU name reported by `nvidia-smi --query-gpu=name`.
# First matching pattern wins.
_PROFILES: list[tuple[re.Pattern[str], list[tuple[int, MatmulHints]], str]] = [
    (re.compile(r"RTX\s*PRO\s*6000", re.IGNORECASE), _PROFILE_PRO6000, "rtx_pro_6000"),
    (re.compile(r"RTX\s*5090", re.IGNORECASE), _PROFILE_5090, "rtx_5090"),
    (re.compile(r"\bH200\b", re.IGNORECASE), _PROFILE_H200, "h200"),
]

_DEFAULT_PROFILE = _PROFILE_5090
_DEFAULT_PROFILE_NAME = "rtx_5090 (fallback)"


def detect_gpu_name() -> str | None:
    """Return the first GPU's name from nvidia-smi, or None if unavailable."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    name = result.stdout.strip().split("\n")[0].strip()
    return name or None


def default_matmul_strategy_map(
    gpu_name: str | None = None,
) -> tuple[list[tuple[int, MatmulHints]], str]:
    """Return the best-known hint map for the given GPU.

    If ``gpu_name`` is None, auto-detects via ``nvidia-smi``. Falls back to
    the 5090 profile when the GPU is unknown or detection fails.

    Returns ``(strategy_map, profile_name)`` so callers can log which profile
    was selected. Each entry in the map is ``(max_size, hints_dict)`` where
    ``hints_dict`` contains matmul hint keys (without the ``cuda.matmul.``
    prefix).
    """
    if gpu_name is None:
        gpu_name = detect_gpu_name()
    if gpu_name is not None:
        for pattern, profile, profile_name in _PROFILES:
            if pattern.search(gpu_name):
                return profile, profile_name
        logger.info("No tuned matmul profile for GPU %r — using %s", gpu_name, _DEFAULT_PROFILE_NAME)
    return _DEFAULT_PROFILE, _DEFAULT_PROFILE_NAME
