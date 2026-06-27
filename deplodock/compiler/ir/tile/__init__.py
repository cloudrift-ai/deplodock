"""Tile IR — a map/reduce kernel with its schedule made explicit.

See :mod:`.ir`. The layer between Loop IR and Kernel IR: a :class:`TileOp`
records the *schedule* (which axes tile onto the thread grid) while the
*combine* stays in the body, so one op covers MAP / MONOID / SEMIRING.
"""

from deplodock.compiler.ir.tile.ir import TileOp

__all__ = ["TileOp"]
