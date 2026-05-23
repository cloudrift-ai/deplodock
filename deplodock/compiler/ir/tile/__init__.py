"""Tile IR — schedule decisions as structural Stmts, pre-materialization.

- :mod:`.ir` — typed tile flavor dataclasses (``GridTile``,
  ``ThreadTile``, ``RegisterTile``, ``SerialTile``, ``StridedTile``) +
  ``Stage`` family, plus re-exports of Loop-IR leaves and shared
  expressions.

Loop-IR → Tile-IR lowering is owned by ``passes/lowering/tile/000_partition_planner``
(constructs tile flavors directly via ``_wrap_tower``). Materialization
(Tile IR → Kernel IR) lives under ``passes/lowering/kernel``; rendering
of Kernel IR to CUDA source lives under ``ir.kernel``.
"""

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.tile.ir import (
    Accum,
    Assign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Cond,
    ElementwiseImpl,
    Expr,
    FuncCallExpr,
    GridTile,
    Literal,
    Load,
    Loop,
    ParallelTile,
    RegisterTile,
    Select,
    SelectBranch,
    SerialKind,
    SerialTile,
    SerialTileBase,
    Stage,
    Stmt,
    StridedLoop,
    StridedTile,
    TernaryExpr,
    ThreadTile,
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
    "StridedLoop",
    # Typed tile flavor hierarchy
    "ParallelTile",
    "GridTile",
    "ThreadTile",
    "RegisterTile",
    "SerialTileBase",
    "SerialTile",
    "StridedTile",
    "SerialKind",
    "Stage",
    "Stmt",
    "TileOp",
    "Axis",
    "ElementwiseImpl",
]
