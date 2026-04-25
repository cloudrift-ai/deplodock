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

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.tile.ir import (
    COMBINE_BLOCK_REDUCE,
    COMBINE_THREAD_LOCAL,
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
    "BIND_BLOCK_STRIDED",
    "BIND_SERIAL",
    "COMBINE_THREAD_LOCAL",
    "COMBINE_BLOCK_REDUCE",
    "Stmt",
    "TileOp",
    "Axis",
    "ElementwiseImpl",
]
