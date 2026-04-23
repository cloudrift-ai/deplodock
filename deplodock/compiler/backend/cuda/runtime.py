"""CUDA runtime availability predicates (used by pytest skip markers)."""

from __future__ import annotations

import shutil


def has_nvcc() -> bool:
    """Check if nvcc is available on PATH (legacy — kernel dispatch uses NVRTC via cupy)."""
    return shutil.which("nvcc") is not None


def has_cuda_gpu() -> bool:
    """Check if cupy is importable and sees at least one CUDA device."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
