"""LoopBuilder — incremental body construction for loop IR.

Accumulates a ``LoopOp`` body by inserting stmts one at a time at a given
enclosing ``Scope``. Callers drive two concerns:

- **Fresh SSA names**: ``fresh(hint)`` returns an unused name derived from
  ``hint``, reserving it in the builder's name pool.
- **Scope-aware insertion**: ``insert(stmt, scope)`` descends the body
  tree along the enclosing axis path, creates ``Loop`` nodes as needed,
  and prepends the stmt at the leaf.

Insertions are prepend-at-leaf. Callers that want defined-before-use
ordering should insert in reverse-topological order (consumers first,
producers after) — that's what the fusion splicer does.
"""

from __future__ import annotations

from deplodock.compiler.ir.loop.ir import Axis, Loop, Scope, Stmt


class LoopBuilder:
    """Mutable accumulator for a ``LoopOp`` body."""

    def __init__(self, used_names: set[str]) -> None:
        self._body: tuple[Stmt, ...] = ()
        self._used: set[str] = set(used_names)

    def fresh(self, hint: str) -> str:
        """Return an unused name derived from ``hint`` and reserve it."""
        if hint not in self._used:
            self._used.add(hint)
            return hint
        i = 1
        while f"{hint}_s{i}" in self._used:
            i += 1
        name = f"{hint}_s{i}"
        self._used.add(name)
        return name

    def insert(self, stmt: Stmt, enclosure: Scope) -> None:
        """Prepend ``stmt`` at the leaf of the path denoted by ``enclosure``."""
        self._body = _prepend_at(self._body, enclosure.enclosing, stmt)

    def finish(self) -> tuple[Stmt, ...]:
        """Return the accumulated body."""
        return self._body


def _prepend_at(body: tuple[Stmt, ...], path: tuple[Axis, ...], stmt: Stmt) -> tuple[Stmt, ...]:
    """Descend ``body`` following ``path``; create missing ``Loop`` nodes;
    prepend ``stmt`` at the leaf."""
    if not path:
        return (stmt,) + tuple(body)
    head, rest = path[0], path[1:]
    for i, s in enumerate(body):
        if isinstance(s, Loop) and s.axis == head:
            new_inner = _prepend_at(s.body, rest, stmt)
            return tuple(body[:i]) + (Loop(axis=head, body=new_inner),) + tuple(body[i + 1 :])
    return (Loop(axis=head, body=_prepend_at((), rest, stmt)),) + tuple(body)
