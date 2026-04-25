"""Tile IR — schedule decisions as structural Stmts, pre-materialization.

- :mod:`.ir` — dataclass definitions: ``Block`` / ``BoundLoop`` /
  ``Combine`` + binding constants, plus re-exports of Loop-IR leaves
  and shared expressions.
- :mod:`.lower` — ``lower_naive`` translating Loop-IR ``LoopOp`` to Tile-IR
  ``TileOp`` with a logical ``Block``.
- :mod:`.pretty` — structural pretty-printer for ``TileOp``.

Materialization (Tile IR → Kernel IR) lives under
``passes/lowering/kernel``; rendering of Kernel IR to CUDA source lives
under ``ir.kernel``.
"""

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.tile.ir import (
    COMBINE_REGISTER,
    COMBINE_SMEM_TREE_HALVE,
    WALK_SERIAL,
    WALK_STRIDED,
    Accum,
    Assign,
    BinaryExpr,
    Block,
    BoundLoop,
    Builtin,
    CastExpr,
    Combine,
    Cond,
    ElementwiseImpl,
    Expr,
    FuncCallExpr,
    Literal,
    Load,
    Loop,
    Select,
    SelectBranch,
    Stmt,
    TernaryExpr,
    TileOp,
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
    "Block",
    "BoundLoop",
    "Combine",
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "WALK_SERIAL",
    "WALK_STRIDED",
    "COMBINE_REGISTER",
    "COMBINE_SMEM_TREE_HALVE",
    "Stmt",
    "TileOp",
    "Axis",
    "ElementwiseImpl",
]
