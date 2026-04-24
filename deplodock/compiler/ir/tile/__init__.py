"""Tile IR — schedule + leaf compute, lowered directly to CUDA source.

- :mod:`.ir` — the dataclass definitions: schedule wrappers (``FreeLoop``
  / ``Reduce`` / ``Tile`` / ``Coop`` / ``Cond`` / ``Sync``), ``Kernel``
  wrapper, ``Param`` / ``SmemBuf``, plus re-exports of Loop IR's leaf
  stmts (``Load`` / ``Assign`` / ``Select`` / ``Write`` / ``Accum``) and
  the shared expression types from :mod:`ir.expr`.

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
    Expr,
    FreeLoop,
    FuncCallExpr,
    Kernel,
    Literal,
    Load,
    Param,
    Reduce,
    Select,
    SelectBranch,
    SmemBuf,
    Stmt,
    Sync,
    TernaryExpr,
    Tile,
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
    # Loop-IR leaves
    "Load",
    "Assign",
    "Select",
    "SelectBranch",
    "Write",
    "Accum",
    # Tile-IR statements
    "Sync",
    "Cond",
    "FreeLoop",
    "Reduce",
    "Tile",
    "Coop",
    "Stmt",
    # Top-level
    "Param",
    "SmemBuf",
    "Kernel",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
