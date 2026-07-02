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

Phase 2 surface: def-use queries (``definitions``, ``deps_closure``,
``depends_on`` / ``independent``, ``deps_of``), type-filtered lookups
(``loads``, ``writes``, ``accums``, …), and dependence cones
(:class:`Cone`, :meth:`Body.backward_cone` / :meth:`Body.forward_cone`
/ :meth:`Body.defs_die_at`) — the shared substrate behind the rules
that slice computed-operand cones. Region transforms (``replace_at``,
``partition_at``) remain follow-ups; add as needed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property, lru_cache

from emmy.compiler.ir.stmt.base import Stmt


@dataclass(frozen=True)
class Cone:
    """A dependence cone over ONE scope level: the subset of a Body's
    immediate stmts closed under SSA dependence (in body order), plus every
    name the cone reads from outside itself — sibling/enclosing scopes and
    axis vars alike. Built by :meth:`Body.backward_cone` /
    :meth:`Body.forward_cone`.

    Construction never fails and applies no eligibility judgment: an
    unresolved name is data (``external_reads``), not an error. Which
    external reads are acceptable, which member kinds are cuttable, and
    whether the cone's values escape (:meth:`Body.defs_die_at`) are the
    calling rule's conditions — bail decisions stay in rules, the dataflow
    walk lives here (``passes/ARCHITECTURE.md``: phrase conditions over cone
    properties instead of re-walking shapes).

    A member is a whole top-level stmt: a wrapper (Loop / Cond / Tile) joins
    as a unit, exposing names per :func:`_exposed_defines` and reading per
    :func:`_member_reads` (subtree rolled up, internally-bound axes
    excluded). Axis vars from enclosing scopes survive into
    ``external_reads`` — intersect with an axis-name set to get the cone's
    axis usage, subtract it to get the SSA names that must resolve
    elsewhere."""

    members: tuple[Stmt, ...]
    external_reads: frozenset[str]

    @property
    def loads(self) -> tuple[Stmt, ...]:
        """Every ``Load`` in the members, nested included, body order —
        the cone's leaf operands (dtype checks, graph resolution)."""
        return Body(self.members).loads


def _exposed_defines(s: Stmt) -> set[str]:
    """SSA names ``s`` makes visible at its own scope level — own defines
    plus every nested define. A deliberate over-approximation: nested non-Accum
    names don't truly cross a Loop boundary, but well-formed SSA never reads them
    from outside, so resolving through them is harmless and cheap."""
    out = set(s.defines())
    for body in s.nested():
        for c in body.iter():
            out.update(c.defines())
    return out


def _member_reads(s: Stmt) -> frozenset[str]:
    """Names ``s`` (incl. a whole wrapper subtree) reads from its enclosing
    scope: SSA deps + Expr free vars (Load/Write indices, Select predicates,
    Cond conditions), recursive, minus internally-defined names and axes
    bound inside the subtree."""
    reads: set[str] = set()
    defs: set[str] = set()

    def walk(st: Stmt, bound: frozenset[str]) -> None:
        reads.update(set(st.deps()) - bound)
        for e in st.exprs():
            reads.update(e.free_vars() - bound)
        defs.update(st.defines())
        inner_bound = bound | st.binds_axes()
        for body in st.nested():
            for c in body:
                walk(c, inner_bound)

    walk(s, frozenset())
    return frozenset(reads - defs)


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

    @cached_property
    def axis_names(self) -> frozenset[str]:
        """Every axis name bound by any wrapper anywhere in this body
        (``Loop`` / ``StridedLoop`` / ``Tile.axes``). Axes from
        enclosing scopes above this body are not included."""
        return frozenset(ax for s in self.iter() for ax in s.binds_axes())

    @cached_property
    def deps_closure(self) -> dict[str, frozenset[str]]:
        """For every SSA name defined in this body (recursive), the
        set of names it transitively reads. Values include both SSA
        names (defined elsewhere in the body or externally) and axis
        names (free vars from Load/Write indices, Select predicates,
        Cond conditions, etc.).

        ``Accum`` is recorded with the *outside-the-loop* form: the
        immediately-enclosing reduce-Loop's axis is subtracted from
        the value's closure, because the reduced result no longer
        varies with that axis. Reads of an Accum's *running* value
        from inside its own loop body get the wrong answer here —
        passes that gate on those (the in-loop online-softmax merge
        pattern) keep the explicit ``deps_of(c)`` check that returns
        the Accum's defining stmt directly.

        This is the substrate behind :meth:`depends_on` and
        :meth:`independent`. Most call sites prefer those phrased
        helpers over poking the closure directly.
        """
        closure: dict[str, frozenset[str]] = {}

        def _immediate(s: Stmt) -> set[str]:
            reads: set[str] = set(s.deps())
            for e in s.exprs():
                reads.update(e.free_vars())
            return reads

        def _transitive(reads: set[str]) -> frozenset[str]:
            out: set[str] = set(reads)
            for r in reads:
                out |= closure.get(r, frozenset())
            return frozenset(out)

        def walk(body: Body) -> None:
            for s in body:
                # Recurse first (post-order) so inner Accums / leaves are
                # in ``closure`` before we record this stmt or close the
                # wrapper.
                for child in s.nested():
                    walk(child)
                # After a Loop / StridedLoop closes, its body's Accums
                # become visible at the outer scope with the loop axis
                # subtracted (Loop) or kept (StridedLoop — partial value
                # carries the strided axis). Mirrors hoist_loop_invariants.
                from emmy.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
                from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

                if isinstance(s, Loop):
                    for c in s.body:
                        if isinstance(c, Accum):
                            closure[c.name] = closure.get(c.value, frozenset()) - {s.axis.name}
                    continue
                if isinstance(s, StridedLoop):
                    for c in s.body:
                        if isinstance(c, Accum):
                            closure[c.name] = closure.get(c.value, frozenset())
                    continue
                # Leaves and non-Loop wrappers (Tile, Cond): record
                # closure for each name this stmt defines.
                for name in s.defines():
                    closure[name] = _transitive(_immediate(s))

        walk(self)
        return closure

    def _stmt_reads(self, s: Stmt) -> frozenset[str]:
        """Names ``s`` transitively reads (axes + SSA), with axes bound
        by ``s`` subtracted. For leaf stmts this matches
        ``deps_closure[s.defines()[0]]``; for compound stmts (Loop /
        StridedLoop / Tile / Cond) it rolls up every nested stmt's
        reads and removes the wrapper's own bound axes — so e.g.
        ``Loop(b, ...)._stmt_reads()`` does not contain ``b``."""
        closure = self.deps_closure
        seeds: set[str] = set(s.deps())
        for e in s.exprs():
            seeds.update(e.free_vars())
        for sub in s.nested():
            for c in sub.iter():
                seeds.update(c.defines())
                seeds.update(c.deps())
                for e in c.exprs():
                    seeds.update(e.free_vars())
        out: set[str] = set(seeds)
        for n in seeds:
            out |= closure.get(n, frozenset())
        return frozenset(out) - s.binds_axes()

    def depends_on(self, a: Stmt | str | Iterable[Stmt | str], b: str | Iterable[str]) -> bool:
        """True iff anything in ``a`` transitively reads any name in
        ``b``. Directional — does not check whether ``b`` reads ``a``;
        callers can swap arg order to flip direction.

        ``a`` may be a name, a ``Stmt``, or an iterable of either.
        Passing a ``Stmt`` expands it to its read set: for a leaf this
        is the closure of its defined name; for a compound stmt
        (``Loop`` / ``StridedLoop`` / ``Tile`` / ``Cond``) it's the
        rolled-up reads of every nested stmt with the wrapper's own
        bound axes subtracted. ``b`` is always names — SSA or axis.
        Names not in :attr:`deps_closure` (external references — Tile-
        input buffers, ConstantOps, names from enclosing scopes) are
        treated as having empty closure.
        """
        b_set = {b} if isinstance(b, str) else set(b)
        if not b_set:
            return False
        closure = self.deps_closure
        a_iter: Iterable[Stmt | str] = [a] if isinstance(a, (str, Stmt)) else a
        for x in a_iter:
            if isinstance(x, Stmt):
                if not self._stmt_reads(x).isdisjoint(b_set):
                    return True
            else:
                if x in b_set or not closure.get(x, frozenset()).isdisjoint(b_set):
                    return True
        return False

    def independent(self, a: Stmt | str | Iterable[Stmt | str], b: Stmt | str | Iterable[Stmt | str]) -> bool:
        """True iff ``a`` and ``b`` share no dataflow path — neither
        ``a`` transitively reads any name in ``b`` nor vice versa.
        Symmetric counterpart to :meth:`depends_on`. Use this when
        asking "are these two things related at all?" (fusion safety,
        motion legality); use :meth:`depends_on` when direction
        matters (invariance, hoist gates).

        For symmetric usage, ``Stmt`` arguments on either side expand
        to their *defined* names (what the stmt produces) when used as
        a read target — same swap-and-call behavior as
        :meth:`depends_on`."""

        def _as_target(x: Stmt | str | Iterable[Stmt | str]) -> set[str]:
            items = [x] if isinstance(x, (str, Stmt)) else list(x)
            out: set[str] = set()
            for it in items:
                if isinstance(it, Stmt):
                    for sub in it.nested():
                        for c in sub.iter():
                            out.update(c.defines())
                    out.update(it.defines())
                else:
                    out.add(it)
            return out

        return not (self.depends_on(a, _as_target(b)) or self.depends_on(b, _as_target(a)))

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

    # -- dependence cones --------------------------------------------------

    def backward_cone(self, roots: Iterable[str]) -> Cone:
        """The backward dependence :class:`Cone` of ``roots`` over THIS
        body's immediate stmts: every top-level member whose exposed names
        transitively feed a root, in body order. Names resolving to no
        member here (axis vars, enclosing/sibling scopes, buffer constants)
        surface in ``external_reads`` — chain another scope level by seeding
        its ``backward_cone`` with them. A root not defined at this level is
        itself an external read (members come out empty)."""
        by_name: dict[str, Stmt] = {}
        for s in self:
            for n in _exposed_defines(s):
                by_name[n] = s
        member_ids: set[int] = set()
        external: set[str] = set()
        pending = list(roots)
        seen: set[str] = set()
        while pending:
            n = pending.pop()
            if n in seen:
                continue
            seen.add(n)
            s = by_name.get(n)
            if s is None:
                external.add(n)
                continue
            if id(s) in member_ids:
                continue
            member_ids.add(id(s))
            pending.extend(_member_reads(s))
        return Cone(members=tuple(s for s in self if id(s) in member_ids), external_reads=frozenset(external))

    def forward_cone(self, seeds: Iterable[Stmt]) -> Cone:
        """The forward (taint) :class:`Cone` of the ``seeds`` — top-level
        stmts of this body — over THIS body's immediate stmts: the seeds
        plus every member transitively reading a name they expose, to
        fixpoint, in body order. ``external_reads`` are the member reads not
        produced inside the cone (reads of earlier non-member siblings
        included)."""
        member_ids = {id(s) for s in seeds}
        names: set[str] = set()
        for s in seeds:
            names.update(_exposed_defines(s))
        reads = {id(s): _member_reads(s) for s in self}
        changed = True
        while changed:
            changed = False
            for s in self:
                if id(s) in member_ids:
                    continue
                if reads[id(s)] & names:
                    member_ids.add(id(s))
                    names.update(_exposed_defines(s))
                    changed = True
        members = tuple(s for s in self if id(s) in member_ids)
        external = set().union(*(reads.get(id(s), _member_reads(s)) for s in members)) if members else set()
        return Cone(members=members, external_reads=frozenset(external - names))

    def defs_die_at(self, members: Iterable[Stmt], *, roots: Iterable[str], allowed: Iterable[Stmt]) -> bool:
        """True iff no stmt in this body outside ``members`` reads a name
        they expose — except the ``allowed`` stmts, which may read names in
        ``roots`` (and nothing else from the cone). The escape check for
        cutting a cone out: its values must die at the designated consumers
        (e.g. the matmul multiplies) or the cut would break a reader left
        behind. ``members`` may span several scope levels of this body
        (a cell cone plus its prologue deps); each member's whole subtree
        counts as inside."""
        members = tuple(members)  # may arrive as a generator; iterated twice
        member_ids = {id(s) for s in members}
        moved_defs: set[str] = set()
        for s in members:
            moved_defs |= _exposed_defines(s)
        root_set = frozenset(roots)
        allowed_ids = {id(s) for s in allowed}

        def walk(stmts: Iterable[Stmt]) -> bool:
            for s in stmts:
                if id(s) in member_ids:
                    continue  # the whole subtree moves; internal uses are fine
                reads = set(s.deps()) & moved_defs
                if reads and not (id(s) in allowed_ids and reads <= root_set):
                    return False
                for sub in s.nested():
                    if not walk(sub):
                        return False
            return True

        return walk(self)

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
        from emmy.compiler.ir.stmt.leaves import Load  # noqa: PLC0415

        return self.iter_of_type(Load)

    @cached_property
    def writes(self) -> tuple[Stmt, ...]:
        """All ``Write`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.leaves import Write  # noqa: PLC0415

        return self.iter_of_type(Write)

    @cached_property
    def accums(self) -> tuple[Stmt, ...]:
        """All ``Accum`` stmts in the body (recursive). May contain
        multiple Accums sharing a single accumulator name (matmul-shape
        K-inner reduces, 008's per-cell replicated accumulator chains).
        Validation enforces op-consistency across same-name Accums in
        ``LoopOp.__post_init__``; callers that want a one-per-name view
        can dedup at the call site (``{a.name: a for a in body.accums}``)."""
        from emmy.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

        return self.iter_of_type(Accum)

    @cached_property
    def loops(self) -> tuple[Stmt, ...]:
        """All ``Loop`` stmts in the body (recursive)."""
        from emmy.compiler.ir.stmt.blocks import Loop  # noqa: PLC0415

        return self.iter_of_type(Loop)

    # -- structural identity --------------------------------------------

    def structural_key(self) -> str:
        """Implements :class:`emmy.compiler.structural.Structural`.

        Canonical text rendering used for structural-equivalence
        queries. Two bodies that differ only by SSA / axis names,
        commutative-arg order, external-buffer names, or the specific
        op within a compute-unit cluster (``add`` vs ``sub``, ``div``
        vs ``mod``) produce the same key.

        Built by re-running :func:`normalize_body` with ``hoist=False``
        (safe for both Loop-IR and Tile-IR bodies — hoisting can move
        Loads above Stage decls in Tile bodies),
        ``canonical_buffers=True`` (renames ``Load.input`` /
        ``Write.output`` to ``b0, b1, ...``), and ``cluster_ops=True``
        (collapses each op to its compute-unit cluster representative
        — see :func:`emmy.compiler.ir.elementwise.cluster_representative`),
        then joining :func:`pretty_body`'s line list. Cached on the
        instance — Body is immutable."""
        return self._cached_structural_key

    @cached_property
    def _cached_structural_key(self) -> str:
        # Delegates to a module-level lru_cache keyed by Body content
        # (Body is ``tuple[Stmt, ...]`` and every Stmt subclass is a
        # frozen dataclass, so the cache key is structural). Two
        # different Body instances with identical stmts now share the
        # one ``normalize_body`` call — matters in tune mode where
        # ``_record_op_inventory`` walks the source chain of every
        # CudaOp in every terminal and hammers ``op_cache_key`` ->
        # ``Body.structural_key()`` on bodies that frequently recur
        # structurally across variants. The cache is safe because
        # ``structural_key`` always normalizes with the same fixed flag
        # combination (``hoist=False, canonical_buffers=True,
        # cluster_ops=True``); generic ``normalize_body`` callers with
        # other flags do not share this cache.
        return _shared_structural_key(self)


@lru_cache(maxsize=4096)
def _shared_structural_key(body: Body) -> str:
    """Module-level memoization for :meth:`Body.structural_key`.

    The structural-key formula is fixed: ``normalize_body(body,
    hoist=False, canonical_buffers=True, cluster_ops=True)`` joined as
    pretty-printed text. With every concrete ``Stmt`` subclass a frozen
    dataclass and ``Body`` a ``tuple[Stmt, ...]`` subclass, equal-content
    bodies hash equal — so two structurally identical Body instances
    share one normalize+pretty walk through this cache. Tune mode hits
    this hard from ``_record_op_inventory`` (one ``op_cache_key`` call
    per ancestor in every CudaOp's source chain, per terminal candidate).

    The cache is sound because the flag combination is hard-coded here.
    Generic :func:`normalize_body` callers with other flags don't share
    this cache — ``cluster_ops=True`` collapses semantically distinct ops
    to a single cluster representative (``add``↔``sub``, ``div``↔``mod``,
    …), which is the right canonicalization for structural-equivalence
    queries but would be a *correctness bug* for any callsite running
    the normalized body.
    """
    from emmy.compiler.ir.stmt.base import pretty_body  # noqa: PLC0415
    from emmy.compiler.ir.stmt.normalize import normalize_body  # noqa: PLC0415

    normalized = normalize_body(body, hoist=False, canonical_buffers=True, cluster_ops=True)
    return "\n".join(pretty_body(normalized))
