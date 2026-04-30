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
from functools import cached_property

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
        """Recursive 1:N body transformer. Post-order: each block stmt's
        nested body is mapped first, then ``fn`` is applied to the
        children-rewritten wrapper. Returns a new Body with each stmt
        replaced by ``fn(stmt)``:

        - a single ``Stmt`` (kept in place of the input),
        - ``None`` (drop the input), or
        - an iterable of ``Stmt`` (inline all of them).

        ``fn`` is called on *every* stmt including ``Loop`` / ``Tile`` /
        ``Cond`` / ``StridedLoop`` wrappers — but with their bodies already
        recursively mapped, so ``fn`` only needs to handle the leaf cases
        it cares about (callers no longer need a self-recursive
        ``Loop(..., body=s.body.map(fn))`` branch). Mirrors :meth:`iter`'s
        full-tree traversal.

        Iterable returns *replace* the wrapper: the returned stmts are
        spliced into the body as-is (their interiors have already been
        recursed when the wrapper was visited), so a caller that
        ``return tuple(c for c in s.body)`` to inline a Loop's body sees
        already-rewritten children.
        """

        def descend(s: Stmt) -> Stmt:
            nested = s.nested()
            if not nested:
                return s
            return s.with_bodies(tuple(b.map(fn) for b in nested))

        out: list[Stmt] = []
        for s in self:
            r = fn(descend(s))
            if r is None:
                continue
            if isinstance(r, Stmt):
                out.append(r)
            else:
                out.extend(r)
        return Body(out)

    # -- generic backward dataflow --------------------------------------

    def fold[T](
        self,
        fn: Callable[[Stmt, tuple[T | None, ...], frozenset[str]], T],
    ) -> dict[int, T]:
        """Generic backward dataflow over this body's def-use DAG.

        Walks every stmt in source order (= SSA topo order). At each stmt,
        calls ``fn(stmt, child_T, bound)``:

        - ``child_T`` — one entry per name in ``stmt.deps()``, pulled from
          the running memo via ``self.definitions``. ``None`` when the dep
          is read but not defined locally (Tile-input buffer reference,
          constant, or an SSA from an enclosing scope — i.e. an external
          read). Position-preserving: same order as ``stmt.deps()``.
        - ``bound`` — set of axis names introduced by enclosing
          ``Loop`` / ``StridedLoop`` / ``Tile`` wrappers via
          ``Stmt.binds_axes()``. ``Cond`` doesn't bind axes. Callbacks
          that don't care about scope can ignore this.

        Returns the per-stmt memo keyed by ``id(stmt)`` — ``Tile`` is a
        non-frozen dataclass and not hashable, so id-keying is the
        lowest-friction choice. Callers that want a name-keyed view do
        ``{n: memo[id(s)] for s in body.iter() for n in s.defines()}``.

        Recursion order: nested bodies are processed *before* the wrapper
        stmt. So when ``fn`` is called on a wrapper that doesn't define a
        name itself (Loop / Tile / StridedLoop), the memo entries for any
        Accums inside its body already exist — downstream consumers at the
        wrapper's scope can read them through ``deps()``.

        Caveat: when multiple stmts define the same SSA name (matmul-shape
        bodies with several ``Accum`` stmts sharing one accumulator),
        ``self.definitions`` resolves to the last definer. ``child_T`` will
        carry that last definer's ``T`` only. Callers needing a multi-defs
        union (e.g. unioning axes across all Accums for ``acc``) iterate
        ``body.accums`` themselves at the call site.
        """
        memo: dict[int, T] = {}
        defs = self.definitions

        def walk(body: Body, bound: frozenset[str]) -> None:
            for s in body:
                child_bound = bound | s.binds_axes()
                for child_body in s.nested():
                    walk(child_body, child_bound)
                child_T: tuple[T | None, ...] = tuple(memo.get(id(defs[d])) if d in defs else None for d in s.deps())
                memo[id(s)] = fn(s, child_T, bound)

        walk(self, frozenset())
        return memo

    # -- def-use analysis ------------------------------------------------

    @cached_property
    def definitions(self) -> dict[str, Stmt]:
        """Map every SSA name produced anywhere inside this body
        (recursive) to its defining ``Stmt``.

        Built once per Body via :meth:`Stmt.defines` over :meth:`iter`;
        cached on the instance, so repeated queries (``def_of`` from
        many call sites in a single rule) are O(1) after the first
        access. Body is immutable, so the cache stays valid for its
        lifetime.

        Names not present in the dict are either Tile-input buffer
        references, constants, or SSA names defined in an enclosing
        scope outside this body — i.e. external reads.
        """
        return {n: s for s in self.iter() for n in s.defines()}

    def deps_of(self, stmt: Stmt) -> tuple[Stmt | None, ...]:
        """Defining stmts inside this body for each of ``stmt``'s SSA
        reads, in the same order as ``stmt.deps()``. Position-preserving:
        each entry is the ``Stmt`` that produces the corresponding dep,
        or ``None`` if the dep is read but not defined locally (Tile-
        input buffer reference, constant, or an SSA from an enclosing
        scope — i.e. an external read).

        Replaces the ``[body.def_of(d) for d in stmt.deps()]`` pattern
        rules used to write inline. Use ``isinstance(s, T)`` predicates
        to filter results — ``None`` won't match any concrete stmt
        type, so external reads drop out automatically; check
        ``s is None`` explicitly when the gate cares about externals."""
        defs = self.definitions
        return tuple(defs.get(d) for d in stmt.deps())

    # -- type-filtered lookups -------------------------------------------

    def of_type(self, *types: type) -> tuple[Stmt, ...]:
        """Top-level stmts in this body matching any of the given
        types. The named helpers below (:meth:`loads`, :meth:`writes`,
        :meth:`accums`, :meth:`loops`, :meth:`stages`) walk the entire
        body recursively; use ``of_type`` when you need only the
        top-level slice (e.g. "Loops directly in this Tile body, not
        inside nested wrappers")."""
        return tuple(s for s in self if isinstance(s, types))

    def iter_of_type(self, *types: type) -> tuple[Stmt, ...]:
        """All stmts (recursive — via :meth:`iter`) matching any of the
        given types. The base primitive the named helpers
        (:meth:`loads`, :meth:`writes`, ...) wrap."""
        return tuple(s for s in self.iter() if isinstance(s, types))

    @cached_property
    def loads(self) -> tuple[Stmt, ...]:
        """All ``Load`` stmts in the body (recursive). Replaces the
        per-Op ``loads`` properties on ``LoopOp`` / ``TileOp`` /
        ``KernelOp``. Cached on the instance — Body is immutable."""
        from deplodock.compiler.ir.stmt.leaves import Load  # noqa: PLC0415

        return self.iter_of_type(Load)

    @cached_property
    def writes(self) -> tuple[Stmt, ...]:
        """All ``Write`` stmts in the body (recursive)."""
        from deplodock.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        return self.iter_of_type(Write)

    @cached_property
    def accums(self) -> tuple[Stmt, ...]:
        """All ``Accum`` stmts in the body (recursive). May contain
        multiple Accums sharing a single accumulator name (matmul-shape
        K-inner reduces, 008's per-cell replicated accumulator chains).
        Validation enforces op-consistency across same-name Accums in
        ``LoopOp.__post_init__``; callers that want a one-per-name view
        can dedup at the call site (``{a.name: a for a in body.accums}``)."""
        from deplodock.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        return self.iter_of_type(Accum)

    @cached_property
    def loops(self) -> tuple[Stmt, ...]:
        """All ``Loop`` stmts in the body (recursive)."""
        from deplodock.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

        return self.iter_of_type(Loop)

    @cached_property
    def stages(self) -> tuple[Stmt, ...]:
        """All ``Stage`` stmts in the body (recursive — Tile-IR only)."""
        from deplodock.compiler.ir.tile.ir import Stage  # noqa: PLC0415

        return self.iter_of_type(Stage)
