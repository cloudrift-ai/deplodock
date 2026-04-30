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

    # No custom ``__repr__`` — inherit ``tuple.__repr__`` so a Body
    # round-trips through ``repr(...)`` / ``eval(...)`` as a tuple.
    # Loop / Tile / Cond / StridedLoop / LoopOp / TileOp /
    # KernelOp ``__post_init__`` coerce on construction so the ingest
    # path ends up with Body either way.

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
        ``Stmt.nested()``)."""
        for s in self:
            yield s
            for child_body in s.nested():
                yield from child_body.iter()

    # -- transformation --------------------------------------------------

    def map(self, fn: Callable[[Stmt], Stmt | None | Iterable[Stmt]]) -> Body:
        """Flat 1:N body transformer. Returns a new Body with each stmt
        replaced by ``fn(stmt)``:

        - a single ``Stmt`` (kept in place of the input),
        - ``None`` (drop the input), or
        - an iterable of ``Stmt`` (inline all of them).

        ``fn`` is called on *every* stmt including ``Loop`` / ``Tile`` /
        etc. wrappers; recursion into a wrapper's body is the caller's
        responsibility (typically by writing a self-recursive ``fn``
        that returns ``Loop(axis=..., body=s.body.map(fn))`` for Loop
        cases). Lets callers pick their own policy for axis renames,
        Loop skipping, or selective recursion.
        """
        out: list[Stmt] = []
        for s in self:
            r = fn(s)
            if r is None:
                continue
            if isinstance(r, Stmt):
                out.append(r)
            else:
                out.extend(r)
        return Body(out)
