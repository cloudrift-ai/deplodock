"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` carries a typed :class:`~.schedule.Kernel` (op-tree node + matching
``*Schedule``) so the *schedule* (free axes, reduce partition, grid binding) stays
separate from the *combine* (the op tree), and one ``TileOp`` covers MAP / MONOID /
SEMIRING.
"""

from deplodock.compiler.ir.tile.atom import AtomKind
from deplodock.compiler.ir.tile.binding import AtomBinding, Operand
from deplodock.compiler.ir.tile.ir import Schedule, TileOp
from deplodock.compiler.ir.tile.role import RoleKind, role_for
from deplodock.compiler.ir.tile.schedule import (
    Fold,
    Kernel,
    Level,
    MapKernel,
    MapSchedule,
    MonoidKernel,
    MonoidSchedule,
    Placement,
    ReducePlan,
    ReduceStage,
    RoleAlloc,
    SemiringKernel,
    SemiringSchedule,
    Stage,
    TilePlan,
    WarpSpec,
    kernel_for,
)

__all__ = [
    "AtomBinding",
    "AtomKind",
    "Fold",
    "Kernel",
    "Level",
    "MapKernel",
    "MapSchedule",
    "MonoidKernel",
    "MonoidSchedule",
    "Operand",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "RoleAlloc",
    "RoleKind",
    "Schedule",
    "SemiringKernel",
    "SemiringSchedule",
    "Stage",
    "TileOp",
    "TilePlan",
    "WarpSpec",
    "kernel_for",
    "role_for",
]
