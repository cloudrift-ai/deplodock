"""Tile IR — schedule decisions as structural Stmts, pre-materialization.

- :mod:`.ir` — dataclass definitions: ``Tile`` / ``Combine`` /
  ``Stage`` + binding constants, plus re-exports of Loop-IR leaves
  and shared expressions.
- :mod:`.pretty` — structural pretty-printer for ``TileOp``.

Loop-IR → Tile-IR lowering (``launch_geometry``) lives next to its
rule at ``passes/lowering/tile/001_launch_geometry.py`` — the
convention is "rules own their logic".

Materialization (Tile IR → Kernel IR) lives under
``passes/lowering/kernel``; rendering of Kernel IR to CUDA source lives
under ``ir.kernel``.
"""

from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.tile.ir import (
    Accum,
    Assign,
    BinaryExpr,
    Builtin,
    CastExpr,
    Combine,
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
    "StridedLoop",
    # Legacy Tile (Loop-IR shared)
    "Tile",
    # New tile flavor hierarchy
    "ParallelTile",
    "GridTile",
    "ThreadTile",
    "RegisterTile",
    "SerialTileBase",
    "SerialTile",
    "StridedTile",
    "SerialKind",
    "Combine",
    "Stage",
    "BoundAxis",
    "BIND_THREAD",
    "BIND_BLOCK",
    "Stmt",
    "TileOp",
    "Axis",
    "ElementwiseImpl",
]
