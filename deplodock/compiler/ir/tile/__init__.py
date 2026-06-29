"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` carries a typed :class:`~.schedule.Kernel` (op-tree node + matching
``*Schedule``) so the *schedule* (free axes, reduce partition, grid binding) stays
separate from the *combine* (the op tree), and one ``TileOp`` covers MAP / MONOID /
SEMIRING.
"""

from deplodock.compiler.ir.tile.binding import AtomBinding, Operand
from deplodock.compiler.ir.tile.ir import Schedule, TileOp
from deplodock.compiler.ir.tile.schedule import (
    Channel,
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
    SemiringKernel,
    SemiringSchedule,
    Stage,
    TilePlan,
    WarpRole,
    WarpSpec,
    WarpTile,
    kernel_for,
)

__all__ = [
    "AtomBinding",
    "Channel",
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
    "Schedule",
    "SemiringKernel",
    "SemiringSchedule",
    "Stage",
    "TileOp",
    "TilePlan",
    "WarpRole",
    "WarpSpec",
    "WarpTile",
    "kernel_for",
]
