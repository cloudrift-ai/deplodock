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
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING

from deplodock.compiler.ir.stmt.base import Stmt

if TYPE_CHECKING:
    from deplodock.compiler.ir.stmt.leaves import Write


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
                from deplodock.compiler.ir.stmt.blocks import Loop, StridedLoop  # noqa: PLC0415
                from deplodock.compiler.ir.stmt.leaves import Accum  # noqa: PLC0415

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

    # -- structural identity --------------------------------------------

    def structural_key(self) -> str:
        """Implements :class:`deplodock.compiler.structural.Structural`.

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
        — see :func:`deplodock.compiler.ir.elementwise.cluster_representative`),
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

    @cached_property
    def coordination(self) -> Coordination:
        """Per-Write atomic / broadcast-guard classifications and
        per-Accum cooperative-axis sets, derived from one walk of this
        body. Materializer + Kernel-IR render consume this to pick
        ``atomicAdd`` vs plain store, ``Cond(t == 0)`` guard wrapping,
        and warp-shuffle / smem tree-halve emission points. Cached on
        the instance.

        Derivation (one body.iter() pass):

        - block axes = union of every ``GridTile.axes``
        - thread axes = union of every ``ThreadTile.axes``
        - staging buffers = ``Smem.name`` + ``StageBundle.sources.name``
        - per-Accum cooperative axes = ``Accum.axes ∩ thread axes``
        - per-Write atomic axes = ``block axes − Write.index free vars``
          (skipped for staging-buffer Writes)
        - per-Write broadcast axes = ``cooperative thread axes −
          Write.index free vars`` (skipped for staging-buffer Writes)

        A ``TileOp`` body has at most one outer ``GridTile`` and one
        outer ``ThreadTile`` (enforced by ``TileOp.__post_init__``),
        and axis names are unique within the body after
        ``normalize_body``, so the axis sets are global — no per-stmt
        scope walk needed. Staging-buffer Writes (smem stores from the
        cooperative-load nest / warp-shuffle emission) are excluded
        because those are per-thread slab slots, not racing global
        stores."""
        # Lazy imports avoid the ir/stmt → ir/tile cycle (ir/tile/ir.py
        # imports Body from this module). Smem / StageBundle staging buffers
        # are picked up generically via ``Stmt.local_decls`` so no kernel-IR
        # import is needed.
        from deplodock.compiler.ir.stmt.leaves import Accum, Write  # noqa: PLC0415
        from deplodock.compiler.ir.tile.ir import GridTile, ThreadTile  # noqa: PLC0415

        block_axes: set[str] = set()
        thread_axes: set[str] = set()
        staging_buffers: set[str] = set()
        accums: list[Accum] = []
        writes: list[Write] = []

        for s in self.iter():
            staging_buffers.update(s.local_decls())
            if isinstance(s, GridTile):
                block_axes.update(ax.name for ax in s.axes)
            elif isinstance(s, ThreadTile):
                thread_axes.update(ax.name for ax in s.axes)
            elif isinstance(s, Accum):
                accums.append(s)
            elif isinstance(s, Write):
                writes.append(s)

        block_axes_fz = frozenset(block_axes)
        thread_axes_fz = frozenset(thread_axes)
        staging_buffers_fz = frozenset(staging_buffers)

        accum_cooperative = {acc.name: frozenset(acc.axes) & thread_axes_fz for acc in accums}
        cooperative_thread_axes = frozenset().union(*accum_cooperative.values()) if accum_cooperative else frozenset()

        atomic_by_id: dict[int, frozenset[str]] = {}
        broadcast_by_id: dict[int, frozenset[str]] = {}
        for w in writes:
            if w.output in staging_buffers_fz:
                atomic_by_id[id(w)] = frozenset()
                broadcast_by_id[id(w)] = frozenset()
                continue
            idx_vars: set[str] = set()
            for e in w.index:
                idx_vars |= e.free_vars()
            atomic_by_id[id(w)] = block_axes_fz - idx_vars
            broadcast_by_id[id(w)] = cooperative_thread_axes - idx_vars

        return Coordination(
            cooperative_thread_axes=cooperative_thread_axes,
            accum_cooperative_axes=accum_cooperative,
            writes=tuple(writes),
            _write_atomic_axes=atomic_by_id,
            _write_broadcast_axes=broadcast_by_id,
        )


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
    from deplodock.compiler.ir.stmt.base import pretty_body  # noqa: PLC0415
    from deplodock.compiler.ir.stmt.normalize import normalize_body  # noqa: PLC0415

    normalized = normalize_body(body, hoist=False, canonical_buffers=True, cluster_ops=True)
    return "\n".join(pretty_body(normalized))


@dataclass(frozen=True)
class Coordination:
    """Result of :attr:`Body.coordination`. Per-Write atomic / broadcast
    classifications and per-Accum cooperative-axis sets.

    ``Write`` lookups use ``id(...)`` internally because ``Write.index``
    may hold ``BinaryExpr`` nodes that aren't hashable — use the
    :meth:`atomic_axes` / :meth:`broadcast_axes` accessors. ``writes``
    is the analyzed Writes in body-walk order so callers can iterate
    deterministically.
    """

    cooperative_thread_axes: frozenset[str] = frozenset()
    accum_cooperative_axes: dict[str, frozenset[str]] = field(default_factory=dict)
    writes: tuple[Write, ...] = field(default_factory=tuple)
    _write_atomic_axes: dict[int, frozenset[str]] = field(default_factory=dict)
    _write_broadcast_axes: dict[int, frozenset[str]] = field(default_factory=dict)

    def atomic_axes(self, w: Write) -> frozenset[str]:
        """Block axes NOT in ``w.index`` — non-empty ⇒ ``atomicAdd``."""
        return self._write_atomic_axes.get(id(w), frozenset())

    def broadcast_axes(self, w: Write) -> frozenset[str]:
        """Cooperative thread axes NOT in ``w.index`` — non-empty ⇒
        ``Cond(axis == 0)`` guard around the Write."""
        return self._write_broadcast_axes.get(id(w), frozenset())
