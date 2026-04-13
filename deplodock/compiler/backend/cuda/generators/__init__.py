"""Kernel generators for the CUDA backend.

Pipeline: FusedRegionOp → analyze() → TileAnalysis → lower_to_loop_ir() → LoopIR → loop_ir_to_kernel() → KernelDef

analysis.py    — TileAnalysis: classify FusedRegionOp patterns.
loop_lower.py  — Lower TileAnalysis → LoopIR (loop-nest IR).
loop_codegen.py — Lower LoopIR → KernelDef (imperative C AST).
tiled.py       — Public API: generate_kernel(), lower_tiled() (routes through LoopIR).
"""

from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis, analyze
from deplodock.compiler.backend.cuda.generators.loop_codegen import loop_ir_to_kernel
from deplodock.compiler.backend.cuda.generators.loop_lower import lower_to_loop_ir
from deplodock.compiler.backend.cuda.generators.tiled import generate_kernel, lower_tiled

__all__ = [
    "TileAnalysis",
    "analyze",
    "generate_kernel",
    "loop_ir_to_kernel",
    "lower_tiled",
    "lower_to_loop_ir",
]
