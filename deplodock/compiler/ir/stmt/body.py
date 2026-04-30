"""``Body`` — an immutable sequence of body Stmts with built-in
def-use / iteration / transform queries.

Implemented as a ``tuple`` subclass so it interoperates transparently
everywhere a ``tuple[Stmt, ...]`` was previously accepted: iteration,
indexing, length, slicing, equality, hashing, and ``isinstance(body,
tuple)`` all work without thinking. The methods on Body are the
recommended way to phrase common analyses (def-use, iteration,
type-filtered lookups) so they can be added incrementally without
rippling through call sites.

Phase 1 surface (this file): the protocol that lets every
``tuple[Stmt, ...]`` site accept Body, plus :meth:`iter` / :meth:`map`
as method-shaped wrappers around the existing free functions.

Phase 2 (follow-up): def-use queries (``def_table``,
``external_reads``, ``backward_slice_names``), type-filtered lookups
(``loops``, ``loads``, ``stages``), region transforms
(``replace_at``, ``partition_at``). Add as needed; the storage shape
is already in place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from deplodock.compiler.ir.stmt.base import Stmt


class Body(tuple[Stmt, ...]):
    """Immutable Stmt sequence. Tuple-subclass so existing tuple-shaped
    APIs accept Body for free; preserves its own type through
    :meth:`__getitem__` slicing and :meth:`__add__` concatenation so
    callers don't keep falling back to plain tuples.

    Constructed from any iterable: ``Body(some_tuple)``,
    ``Body([s1, s2])``, ``Body(s for s in ... if ...)``.

    No ``__slots__`` — instances retain ``__dict__`` so
    ``functools.cached_property`` works for the analysis methods we
    add incrementally (``def_table``, ``external_reads``, etc.). The
    per-instance dict adds a small memory overhead vs a bare tuple,
    but Body counts are bounded by the number of kernel bodies in a
    pipeline run (tens to hundreds), so it's not a concern.
    """

    def __new__(cls, stmts: Iterable[Stmt] = ()) -> Body:
        return super().__new__(cls, tuple(stmts))

    def __getitem__(self, key):
        r = super().__getitem__(key)
        return Body(r) if isinstance(key, slice) else r

    def __add__(self, other: Iterable[Stmt]) -> Body:
        if isinstance(other, tuple):
            return Body(tuple.__add__(self, other))
        return Body(tuple.__add__(self, tuple(other)))

    def __radd__(self, other: Iterable[Stmt]) -> Body:
        if isinstance(other, tuple):
            return Body(tuple.__add__(other, self))
        return Body(tuple.__add__(tuple(other), self))

    def __repr__(self) -> str:
        return f"Body({tuple.__repr__(self)})"

    @staticmethod
    def coerce(value: Body | Iterable[Stmt]) -> Body:
        """Wrap if not already a Body. Used by ``LoopOp`` /
        ``TileOp`` ``__post_init__`` so the legacy
        ``Op(body=tuple_value)`` construction shape keeps working."""
        return value if isinstance(value, Body) else Body(value)

    # -- iteration -------------------------------------------------------

    def iter(self) -> Iterator[Stmt]:
        """Pre-order iteration over this body and every nested body
        (``Loop`` / ``Tile`` / ``Cond`` / ``StridedLoop`` recurse via
        ``Stmt.nested()``). Method-shaped wrapper around the free
        function :func:`iter_body`; new code should prefer the method
        for discoverability."""
        from deplodock.compiler.ir.stmt.visit import iter_body

        return iter_body(self)

    # -- transformation --------------------------------------------------

    def map(self, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Body:
        """Flat 1:N body transformer. Returns a new Body with each stmt
        replaced by ``fn(stmt)`` (or dropped if ``None`` / inlined if
        an iterable). Method-shaped wrapper around :func:`map_body`."""
        from deplodock.compiler.ir.stmt.visit import map_body

        return Body(map_body(self, fn))
