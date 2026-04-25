"""Tile IR — schedule decisions as structural Stmts, pre-materialization.

- :mod:`.ir` — dataclass definitions: ``Tile`` / ``BoundLoop`` /
  ``Combine`` + binding constants, plus re-exports of Loop-IR leaves
  and shared expressions.
- :mod:`.pretty` — structural pretty-printer for ``TileOp``.

Loop-IR → Tile-IR lowering (``lower_naive``) lives next to its rule at
``passes/lowering/tile/_lower.py`` — the convention is "rules own their
logic; shared helpers underscore-prefixed in the rules dir".

Materialization (Tile IR → Kernel IR) lives under
``passes/lowering/kernel``; rendering of Kernel IR to CUDA source lives
under ``ir.kernel``.
"""

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.tile.ir import (
    Accum,
    Assign,
    BinaryExpr,
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
    Stage,
    Stmt,
    TernaryExpr,
    Tile,
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
    "Tile",
    "BoundLoop",
    "Combine",
    "Stage",
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "BIND_BLOCK_STRIDED",
    "BIND_SERIAL",
    "Stmt",
    "TileOp",
    "Axis",
    "ElementwiseImpl",
]
