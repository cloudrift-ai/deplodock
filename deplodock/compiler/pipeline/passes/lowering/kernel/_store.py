"""Shared output-store glue for the kernel materializer.

The two tiny helpers that decide whether a lowered body already writes its output and, if
not, append the grid-cell ``Write``. Used by ``_factor.factorize`` for the tiled ``Contraction``
node's bare grid-``Write`` (synthesized since it needs ``root.output``) and the scalar / reduce
tiers, so they live here rather than in the rule module. Leading ``_`` so the pass loader (globs
``*.py``, skips ``_``-prefixed) skips this module."""

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


def with_store(stmts: list[Stmt], output: str, grid, value: str) -> list[Stmt]:
    """Append the output-store glue when the body has none — a bare reduction / contraction produces
    its finalized value as the SSA name ``value`` (the carrier state / accumulator, or a projection's
    last def) that must be written to the output buffer at the grid cell. A body that already carries
    a ``Write`` needs no glue (``value`` is left unread). The caller resolves ``value`` off the node
    (``Contraction.out`` / the recursion's produced ``Handle``) so this helper stays node-agnostic."""
    if has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=value)]


__all__ = ["has_write", "with_store"]
