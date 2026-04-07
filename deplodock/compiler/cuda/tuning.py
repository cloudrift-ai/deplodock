"""GPU-specific default configs for the matmul kernel.

The best `MatmulConfig` for `tma_db` depends on the GPU's SM count, register
file, and clocks — Blackwell sm_120 cards (5090, RTX PRO 6000) share the same
compute capability but tune to different `thread_m` values at large sizes.

Dispatch is by GPU name (from `nvidia-smi --query-gpu=name`) because both cards
report `sm_120` and identical per-SM smem, so compute capability cannot
distinguish them. Unknown GPUs fall back to the 5090 profile, which is the most
thoroughly tuned and has been validated to also work reasonably on the Pro 6000
(within ~5-10% of the per-card optimum).

To add a new GPU: measure the best `(thread_m, block_k, k_splits)` per size with
`scripts/bench_matmul.py --strategy tma_db --thread-m N --sizes S` and append a
new entry to `_PROFILES`.
"""

from __future__ import annotations

import logging
import re
import subprocess

from deplodock.compiler.cuda.lower import MatmulConfig

logger = logging.getLogger(__name__)


def _tma(bk: int = 32, tm: int = 8, ks: int = 1) -> MatmulConfig:
    return MatmulConfig(strategy="tma_db", block_k=bk, thread_m=tm, k_splits=ks)


# Per-GPU strategy maps. Each entry is `(max_size, config)` — pick the first
# entry whose threshold is >= the matrix size. Tuned for square FP32 matmul.
#
# 5090 numbers come from the blog post (CUDA 13.2, sm_120, RTX 5090).
# Pro 6000 numbers come from a TM sweep on driver 595 / CUDA 13.2 (April 2026).
_PROFILE_5090: list[tuple[int, MatmulConfig]] = [
    (256, _tma(bk=32, tm=8, ks=4)),
    (512, _tma(bk=32, tm=8, ks=4)),
    (1024, _tma(bk=32, tm=8, ks=1)),
    (2048, _tma(bk=32, tm=26, ks=1)),
    (4096, _tma(bk=32, tm=20, ks=1)),
    (8192, _tma(bk=32, tm=28, ks=1)),
    (99999, _tma(bk=32, tm=28, ks=1)),
]

_PROFILE_PRO6000: list[tuple[int, MatmulConfig]] = [
    (256, _tma(bk=32, tm=8, ks=4)),
    (512, _tma(bk=32, tm=8, ks=4)),
    (1024, _tma(bk=32, tm=8, ks=1)),
    (2048, _tma(bk=32, tm=24, ks=1)),
    (4096, _tma(bk=32, tm=24, ks=1)),
    (8192, _tma(bk=32, tm=28, ks=1)),
    (99999, _tma(bk=32, tm=28, ks=1)),
]

# Match against the GPU name reported by `nvidia-smi --query-gpu=name`.
# First matching pattern wins.
_PROFILES: list[tuple[re.Pattern[str], list[tuple[int, MatmulConfig]], str]] = [
    (re.compile(r"RTX\s*PRO\s*6000", re.IGNORECASE), _PROFILE_PRO6000, "rtx_pro_6000"),
    (re.compile(r"RTX\s*5090", re.IGNORECASE), _PROFILE_5090, "rtx_5090"),
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
) -> tuple[list[tuple[int, MatmulConfig]], str]:
    """Return the best-known `tma_db` strategy map for the given GPU.

    If `gpu_name` is None, auto-detects via `nvidia-smi`. Falls back to the
    5090 profile when the GPU is unknown or detection fails.

    Returns `(strategy_map, profile_name)` so callers can log which profile
    was selected.
    """
    if gpu_name is None:
        gpu_name = detect_gpu_name()
    if gpu_name is not None:
        for pattern, profile, profile_name in _PROFILES:
            if pattern.search(gpu_name):
                return profile, profile_name
        logger.info("No tuned matmul profile for GPU %r — using %s", gpu_name, _DEFAULT_PROFILE_NAME)
    return _DEFAULT_PROFILE, _DEFAULT_PROFILE_NAME
