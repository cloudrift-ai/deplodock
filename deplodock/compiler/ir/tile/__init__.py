"""Tile IR — schedule + leaf compute, lowered directly to CUDA source.

- :mod:`.ir` — dataclass definitions: high-level ``Block`` /
  ``BoundLoop`` / ``Combine`` (pre-materialization), low-level
  ``Enclosure`` / ``Tile`` / ``Smem`` / ``Sync`` / ``StridedLoop`` /
  ``TreeHalve`` (post-materialization), plus the ``TileOp`` wrapper.
  Loop-IR leaves (``Load`` / ``Assign`` / ``Select`` / ``Write`` /
  ``Accum`` / ``Cond`` / ``Loop``) and shared expressions from
  :mod:`ir.expr` are re-exported here.

Subsequent siblings (``render.py``, lowering passes, schedule strategies)
land alongside; this module holds only the type definitions.
"""

from deplodock.compiler.ir.tile.ir import (
    BIND_BLOCK,
    BIND_SERIAL,
    BIND_STRIDED,
    BIND_THREAD,
    COMBINE_REGISTER,
    COMBINE_SMEM_TREE_HALVE,
    Accum,
    Assign,
    Axis,
    BinaryExpr,
    Block,
    BoundLoop,
    Builtin,
    CastExpr,
    Combine,
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
    # Tile-IR statements — low-level (post-materialization)
    "Enclosure",
    "Tile",
    "Smem",
    "Sync",
    "TreeHalve",
    "StridedLoop",
    # Tile-IR statements — high-level (pre-materialization)
    "Block",
    "BoundLoop",
    "Combine",
    # Binding + Combine kind constants
    "BIND_SERIAL",
    "BIND_STRIDED",
    "BIND_THREAD",
    "BIND_BLOCK",
    "COMBINE_REGISTER",
    "COMBINE_SMEM_TREE_HALVE",
    "Stmt",
    # Top-level
    "TileOp",
    # Re-exports
    "Axis",
    "ElementwiseImpl",
]
