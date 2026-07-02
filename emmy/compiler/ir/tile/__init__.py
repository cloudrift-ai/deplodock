"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir` and :mod:`.schedule`. The layer between Loop IR and Kernel IR: a
:class:`TileOp` holds the structural-IR root ``op`` (the *combine* — a :class:`~.ir.Map` /
:class:`~.ir.Reduction` / :class:`~.ir.Contraction`) directly, plus thin schedule
fields (``place`` / ``workers`` / the residual reduce/tier/stage) so the *schedule* (free axes,
reduce partition, grid binding) stays separate from the *combine*, and one ``TileOp`` covers
MAP / MONOID / SEMIRING with no per-kind schedule type (dispatch reads ``ops.axis_role``).
"""

from emmy.compiler.ir.atom import AtomKind
from emmy.compiler.ir.schedule import (
    Fold,
    Level,
    Placement,
    ReducePlan,
    ReduceStage,
    RoleAlloc,
    RoleKind,
    Stage,
    TilePlan,
    WarpSpec,
    role_for,
)
from emmy.compiler.ir.tile.ir import Contraction, Map, Reduction, TileOp

__all__ = [
    "AtomKind",
    "Contraction",
    "Fold",
    "Level",
    "Map",
    "Placement",
    "ReducePlan",
    "ReduceStage",
    "Reduction",
    "RoleAlloc",
    "RoleKind",
    "Stage",
    "TileOp",
    "TilePlan",
    "WarpSpec",
    "role_for",
]
