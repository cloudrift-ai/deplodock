"""Tile IR — schedule + leaf compute, lowered directly to CUDA source.

- :mod:`.ir` — the dataclass definitions: ``Kernel``, statement types
  (``Let`` / ``Store`` / ``AccumFold`` / ``Sync`` / ``Cond`` / ``FreeLoop`` /
  ``Reduce`` / ``Tile`` / ``Coop``), Tile-IR-specific ``Index`` expression,
  and re-exports of the shared expression types from :mod:`ir.expr`.

Subsequent siblings (``render.py``, lowering passes, schedule strategies)
land alongside; this module holds only the type definitions.
"""

from deplodock.compiler.ir.tile.ir import (
    Acc,
    AccumFold,
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
    Index,
    Kernel,
    Let,
    Literal,
    Param,
    Reduce,
    SmemBuf,
    Stmt,
    Store,
    Sync,
    TernaryExpr,
    Tile,
    Var,
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
    # Tile-IR expressions
    "Index",
    # Statements
    "Let",
    "Store",
    "AccumFold",
    "Sync",
    "Cond",
    "FreeLoop",
    "Reduce",
    "Tile",
    "Coop",
    "Stmt",
    # Top-level
    "Acc",
    "Param",
    "SmemBuf",
    "Kernel",
    # Re-exported from ir.loop / ir.elementwise
    "Axis",
    "ElementwiseImpl",
]
