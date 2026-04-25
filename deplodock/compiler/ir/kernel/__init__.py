"""Kernel IR — fully-scheduled kernel form, lowered directly to CUDA source.

- :mod:`.ir` — dataclass definitions: ``KernelOp`` wrapper, ``Enclosure`` /
  ``Tile`` / ``Smem`` / ``Sync`` / ``TreeHalve`` / ``StridedLoop`` hardware
  primitives, plus re-exports of Loop-IR leaves.
- :mod:`.render` — ``render_kernelop`` emitting CUDA source.
- :mod:`.pretty` — structural pretty-printer.
"""

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.kernel.ir import (
    Accum,
    Assign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Cond,
    ElementwiseImpl,
    Enclosure,
    Expr,
    FuncCallExpr,
    KernelOp,
    Literal,
    Load,
    Loop,
    Select,
    SelectBranch,
    Smem,
    Stmt,
    StridedLoop,
    Sync,
    TernaryExpr,
    TreeHalve,
    Var,
    Write,
)

__all__ = [
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    "Enclosure",
    "Smem",
    "Sync",
    "TreeHalve",
    "StridedLoop",
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "Stmt",
    "KernelOp",
    "Axis",
    "ElementwiseImpl",
]
