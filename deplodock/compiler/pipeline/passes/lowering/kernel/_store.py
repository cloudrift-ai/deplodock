"""Shared output-store glue for the kernel materializer + the contraction constructor.

The two tiny helpers that decide whether a lowered body already writes its output and, if
not, append the grid-cell ``Write``. Used by both ``005_contract`` (the warp/mma contraction
node) and ``010_materialize`` (the scalar / reduce / register-tile tiers), so they live here
rather than in either rule module. Leading ``_`` so the pass loader (globs ``*.py``, skips
``_``-prefixed) skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Write
from deplodock.compiler.ir.stmt.base import Stmt


def has_write(stmts: list[Stmt]) -> bool:
    """Any ``Write`` reachable in ``stmts`` (deep — a projection's output sweep nests its
    ``Write`` inside a per-cell ``Loop``)."""
    for s in stmts:
        if isinstance(s, Write):
            return True
        if any(has_write(list(b)) for b in s.nested()):
            return True
    return False


def with_store(stmts: list[Stmt], output: str, grid, op) -> list[Stmt]:
    """Append the output-store glue when the body has none — a bare reduction (``op`` a
    ``Monoid`` / ``Semiring``) produces its finalized value as an SSA name (``op.out``) that
    must be written to the output buffer at the grid cell. A body that already carries a
    ``Write`` needs no glue (and ``op.out`` is left unread)."""
    if has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=op.out)]


__all__ = ["has_write", "with_store"]
