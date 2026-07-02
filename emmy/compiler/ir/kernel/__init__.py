"""Kernel IR — fully-scheduled kernel form, lowered directly to CUDA source.

- :mod:`.ir` — dataclass definitions: ``KernelOp`` wrapper plus
  ``Smem`` / ``Sync`` / ``TreeHalve`` hardware primitives. Shared
  structural types (``Loop``, ``StridedLoop``) come from ``ir.stmt``;
  the typed tile flavors (``GridTile``, ``ThreadTile``, etc.) come from
  ``ir.tile.ir``.
- :mod:`.render` — ``render_kernelop`` emitting CUDA source.
"""

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import (
    Accum,
    Assign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Cond,
    ElementwiseImpl,
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
    Tile,
    TreeHalve,
    Var,
    WarpShuffle,
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
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "WarpShuffle",
    "StridedLoop",
    "Stmt",
    "KernelOp",
    "Axis",
    "ElementwiseImpl",
]
