"""Kernel generators for the CUDA backend.

tiled.py    — unified generator: TileAnalysis → KernelDef for all patterns
              (pointwise, row_reduce, reduce_broadcast, contraction).
              Supports naive and tma_db strategies for contractions.
analysis.py — TileAnalysis: classify FusedRegionOp patterns.
"""

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, analyze
from deplodock.compiler.backend.cuda.generators.tiled import generate_kernel, lower_tiled

__all__ = [
    "TileAnalysis",
    "analyze",
    "generate_kernel",
    "lower_tiled",
]
