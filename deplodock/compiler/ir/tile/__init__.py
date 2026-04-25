"""Tile IR — schedule + leaf compute, lowered directly to CUDA source.

- :mod:`.ir` — the dataclass definitions: ``Enclosure`` schedule wrapper,
  ``TileOp`` wrapper, plus re-exports of Loop IR's leaf stmts + control
  flow (``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum`` /
  ``Cond`` / ``Loop``) and the shared expression types from :mod:`ir.expr`.

Subsequent siblings (``render.py``, lowering passes, schedule strategies)
land alongside; this module holds only the type definitions.
"""

from deplodock.compiler.ir.tile.ir import (
    Accum,
    Assign,
    Axis,
    BinaryExpr,
    Builtin,
    CastExpr,
    Cond,
    ElementwiseImpl,
    Enclosure,
    Expr,
    FuncCallExpr,
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
    TileOp,
    TreeHalve,
    Var,
    Write,
)

__all__ = [
    # Shared expressions
    "Var",
    "Literal",
    "BinaryExpr",
    "Builtin",
    "FuncCallExpr",
    "TernaryExpr",
    "CastExpr",
    "Expr",
    # Loop-IR leaves + control flow
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    "Cond",
    "Loop",
    # Tile-IR statements
    "Enclosure",
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "StridedLoop",
    "Stmt",
    # Top-level
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
