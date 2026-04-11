"""Kernel generators for the CUDA backend.

matmul.py — hand-optimised SGEMM strategies (naive, TMA double-buffer, TF32).
fused.py  — automatic pointwise / reduction kernels from FusedRegionOp.
"""

from deplodock.compiler.backend.cuda.generators.fused import generate_kernel
from deplodock.compiler.backend.cuda.generators.matmul import lower_matmul

__all__ = [
    "generate_kernel",
    "lower_matmul",
]
