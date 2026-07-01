"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (the *combine* — a :class:`~.structural.Map` /
:class:`~.structural.Reduction` / :class:`~.structural.Contraction`) directly, plus thin schedule
fields (``place`` / ``workers`` / the residual reduce/tier/stage) so the *schedule* (free axes,
reduce partition, grid binding) stays separate from the *combine*, and one ``TileOp`` covers
MAP / MONOID / SEMIRING with no per-kind schedule type (dispatch reads ``ops.axis_role``).
"""

from deplodock.compiler.ir.tile.atom import AtomKind
from deplodock.compiler.ir.tile.binding import AtomBinding, Operand
from deplodock.compiler.ir.tile.ir import Schedule, TileOp
from deplodock.compiler.ir.tile.role import RoleKind, role_for
from deplodock.compiler.ir.tile.schedule import (
    Fold,
    Level,
    Placement,
    ReducePlan,
    ReduceStage,
    RoleAlloc,
    Stage,
    TilePlan,
    WarpSpec,
)
from deplodock.compiler.ir.tile.structural import Contraction, Map, Reduction

__all__ = [
    "AtomBinding",
    "AtomKind",
    "Contraction",
    "Fold",
    "Level",
    "Map",
    "Operand",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "Reduction",
    "RoleAlloc",
    "RoleKind",
    "Schedule",
    "Stage",
    "TileOp",
    "TilePlan",
    "WarpSpec",
    "role_for",
]
