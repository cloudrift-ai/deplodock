"""Body visitor helpers used by the normalization passes.

Tree-walking primitives (``iter``, ``map``) live as methods on
:class:`Body`. This module keeps only the small helpers that the
normalization passes share — recursion through non-Loop block stmts
plus the trivial identity renamers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.stmt.blocks import Cond, StridedLoop, Tile


def _identity_rename(n: str) -> str:
    return n


def _make_axis_renamer(old: str, new: Axis) -> Callable[[Axis], Axis]:
    return lambda a: new if a.name == old else a


def _recurse_through_block(s: Stmt, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Stmt | None:
    """If ``s`` is a non-Loop block stmt, return a copy with its body / bodies
    re-walked through ``fn`` via ``Body.map``. Else return ``None``."""
    if isinstance(s, StridedLoop):
        return StridedLoop(axis=s.axis, start=s.start, step=s.step, body=s.body.map(fn))
    if isinstance(s, Tile):
        return Tile(axes=s.axes, body=s.body.map(fn))
    if isinstance(s, Cond):
        return Cond(cond=s.cond, body=s.body.map(fn), else_body=s.else_body.map(fn))
    return None
