"""Tile IR — schedule + leaf compute, lowered directly to CUDA source.

- :mod:`.ir` — the dataclass definitions: Tile-IR-specific schedule
  wrappers (``Tile`` / ``Coop`` / ``Sync`` / ``Enclosure``), ``TileOp``
  wrapper, ``SmemBuf``, plus re-exports of Loop IR's leaf stmts + control
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
    Coop,
    ElementwiseImpl,
    Enclosure,
    Expr,
    FuncCallExpr,
    Literal,
    Load,
    Loop,
    Select,
    SelectBranch,
    SmemBuf,
    Stmt,
    Sync,
    TernaryExpr,
    Tile,
    TileOp,
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
    "Sync",
    "Tile",
    "Coop",
    "Enclosure",
    "Stmt",
    # Top-level
    "SmemBuf",
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
