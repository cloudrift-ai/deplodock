"""Kernel generators for the CUDA backend.

Pipeline: KernelOp → build_schedule() → Schedule → lower_generic()
          → LoopIR → loop_ir_to_kernel() → KernelDef

schedule.py     — Schedule: all kernel structure decisions as a dataclass.
loop_lower.py   — lower_generic(): Schedule-driven LoopIR emission.
loop_codegen.py — loop_ir_to_kernel(): LoopIR → KernelDef (imperative C AST).
tiled.py        — Public API: generate_kernel(), lower_tiled().
"""

from deplodock.compiler.backend.cuda.generators.loop_codegen import loop_ir_to_kernel
from deplodock.compiler.backend.cuda.generators.loop_lower import lower_generic, lower_to_loop_ir
from deplodock.compiler.backend.cuda.generators.tiled import generate_kernel, lower_tiled
from deplodock.compiler.backend.cuda.schedule import Schedule, build_schedule

__all__ = [
    "Schedule",
    "build_schedule",
    "generate_kernel",
    "loop_ir_to_kernel",
    "lower_generic",
    "lower_tiled",
    "lower_to_loop_ir",
]
