"""``Body`` — an immutable container for a sequence of body Stmts.

Used as the storage type for ``LoopOp.body`` and ``TileOp.body``,
replacing the bare ``tuple[Stmt, ...]`` so future analysis methods
(def-use queries, type-filtered lookups, region transforms) have a
natural home. Iteration, indexing, length, and bool-truthiness all
delegate to the wrapped tuple — most existing tuple-shaped call sites
work unchanged.

Phase 1 keeps Body to its minimal surface: storage + the four protocol
methods that make it tuple-compatible at call sites. Analysis methods
(``def_table``, ``external_reads``, ``backward_slice_names``, etc.)
will be added in a follow-up once the type is wired through the Ops.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from deplodock.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class Body:
    """An immutable sequence of body Stmts.

    Constructed from any iterable of ``Stmt`` (tuple, list, generator);
    stored as ``tuple[Stmt, ...]`` internally. Frozen + hashable so
    instances can serve as dict keys / cache keys.

    Sliced indexing (``body[i:j]``) returns a plain ``tuple`` because
    that's what ``self.stmts[i:j]`` produces — tuple-style usage in
    rules (``body[:idx] + (new,) + body[idx + 1:]``) keeps working
    without an extra wrap. Wrap explicitly with ``Body(...)`` when the
    Body type is needed back.
    """

    stmts: tuple[Stmt, ...] = ()

    def __post_init__(self) -> None:
        # Accept any iterable; normalize to tuple. Frozen dataclasses
        # need ``object.__setattr__`` to mutate in __post_init__.
        if not isinstance(self.stmts, tuple):
            object.__setattr__(self, "stmts", tuple(self.stmts))

    def __iter__(self) -> Iterator[Stmt]:
        return iter(self.stmts)

    def __len__(self) -> int:
        return len(self.stmts)

    def __getitem__(self, key):
        return self.stmts[key]

    def __bool__(self) -> bool:
        return bool(self.stmts)

    @staticmethod
    def coerce(value: Body | Iterable[Stmt]) -> Body:
        """Wrap a tuple / iterable as a Body if it isn't already one.
        ``LoopOp`` / ``TileOp`` use this in ``__post_init__`` so the
        common ``Op(body=tuple_value, ...)`` construction shape keeps
        working without forcing every caller to wrap explicitly."""
        return value if isinstance(value, Body) else Body(value)
