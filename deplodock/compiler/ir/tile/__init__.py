"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` carries a :class:`~.schedule.Kernel` (op-tree node + a kind-free
:class:`~.schedule.TileSchedule`) so the *schedule* (free axes, reduce partition, grid
binding) stays separate from the *combine* (the op tree), and one ``TileOp`` covers
MAP / MONOID / SEMIRING with no per-kind schedule type (dispatch reads ``ops.axis_role``).
"""

from deplodock.compiler.ir.tile.atom import AtomKind
from deplodock.compiler.ir.tile.binding import AtomBinding, Operand
from deplodock.compiler.ir.tile.ir import Schedule, TileOp
from deplodock.compiler.ir.tile.role import RoleKind, role_for
from deplodock.compiler.ir.tile.schedule import (
    Fold,
    Kernel,
    Level,
    Placement,
    ReducePlan,
    ReduceStage,
    RoleAlloc,
    Stage,
    TilePlan,
    TileSchedule,
    WarpSpec,
    kernel_for,
)
from deplodock.compiler.ir.tile.structural import Contraction

__all__ = [
    "AtomBinding",
    "AtomKind",
    "Contraction",
    "Fold",
    "Kernel",
    "Level",
    "Operand",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "RoleAlloc",
    "RoleKind",
    "Schedule",
    "Stage",
    "TileOp",
    "TilePlan",
    "TileSchedule",
    "WarpSpec",
    "kernel_for",
    "role_for",
]
