"""Planner-driven register tile — runs *after* ``020_stage_inputs``.

When ``010_partition_loops`` pre-splits a matmul's output Loops and
tags the inner halves ``Role.REGISTER``, this pass unwraps those
REGISTER Loops and replicates their bodies per-cell. By the time this
pass runs, ``020_stage_inputs`` has already emitted Stages with
REGISTER axes (M_r / N_r) as part of their cache axes — the slab spans
the full ``BM·FM × BK`` (and similar) with Affine addressing. Stages
are treated as opaque here: their internal gmem-load body has its own
cache-axis iteration that shadows the outer REGISTER Loops, so the
replicator passes them through unchanged. Consumer body Loads
(reading from the Stages) carry outer-REGISTER ``Var``s in their
smem indices, and those replicas multiply along the unwrapping axis
in the usual way.

When no REGISTER tags are present (non-matmul kernels), the pass
skips. Stamps ``FM`` / ``FN`` so the planner-stamped values persist
and the rule is idempotent on a second visit.

For MMA kernels, the sibling
``005_lower_atom_tile`` pass has already replaced the AtomTile-wrapped
matmul body with an Mma* fragment chain by the time this pass runs.
This pass then sees the same ``RegisterTile`` wrapper as in the scalar
path and replicates the Mma* chain per (M_r, N_r) cell — the
``Mma*.rewrite(...)`` registrations in ``ir/kernel/ir.py`` thread
``rename`` through fragment SSA names so each cell gets its own
``c_frag_<i>_<j>`` etc.
"""

from __future__ import annotations

from collections.abc import Callable

from emmy.compiler.graph import Graph, Node
from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.kernel.ir import Reassign
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Stmt
from emmy.compiler.ir.tile.ir import RegisterTile, SerialTile, StageBundle, TileOp
from emmy.compiler.pipeline import Pattern, RuleSkipped
from emmy.compiler.pipeline.passes.lowering.kernel._helpers import parallel_tile_of, replace_parallel_tile_body, single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, outer = single_tile(body)
    tt = parallel_tile_of(outer)

    new_body, factors, saw_register, folds = _replicate_register_tiles(tt.body)
    if not saw_register:
        # No RegisterTile in body — non-matmul kernel with no FK strip-mine, or
        # this rule has already run and consumed them. Either way, nothing to do.
        raise RuleSkipped("no RegisterTile in body")
    # Every fold is placed at its enclosing non-K-serial scope inside the walk;
    # the top-level call is itself a non-K-serial scope, so none should escape.
    assert not folds, "reduce-accumulator fold escaped the register-tile walk"

    # FM/FN are stamped by the planner; preserve them rather than overwriting.
    # ``factors`` carries only the output-cell (reduce=False) tile extents in
    # outermost-first order — the reduce-axis (K_f) tiles are excluded so the
    # FM/FN stamping stays role-driven, not positional, even when an FK tile
    # coexists with the FM/FN output tile.
    knobs = dict(root.op.knobs)
    if len(factors) >= 1 and "FM" not in knobs:
        knobs["FM"] = factors[0]
    if len(factors) >= 2 and "FN" not in knobs:
        knobs["FN"] = factors[1]
    rebuilt = replace_parallel_tile_body(outer, new_body)
    return TileOp(body=body[:idx] + (rebuilt,) + body[idx + 1 :], name=root.op.name, knobs=knobs)


def _replicate_register_tiles(body: Body, *, in_k_serial: bool = False) -> tuple[Body, list[int], bool, list[Stmt]]:
    """Inside-out unwrap of ``RegisterTile`` flavors. For each tile, recurse
    into nested RegisterTiles first, then replicate this layer's body by
    ``axis.extent`` with ``σ: axis → literal(i)`` for each of the tile's
    axes (outermost first). Returns ``(new_body, factors, saw_register,
    folds)`` — ``factors`` lists only the output-cell (``reduce=False``)
    tile extents in outermost-first order (FM/FN); ``saw_register`` is True
    iff any RegisterTile (output or reduce) was consumed; ``folds`` are the
    pending cross-accumulator folds bubbling up to their enclosing scope.

    Walks into non-RegisterTile block stmts (e.g. ``SerialTile`` wrapping
    a softmax prologue + ``RegisterTile``-tiled matmul body for SDPA P@V)
    so deeply-nested RegisterTiles get replicated too.

    **FK reduce tiles** (``RegisterTile.reduce=True``, the K_f strip-mine):
    after replicating the wrapped ``Accum``s into ``acc_0 .. acc_{FK-1}``,
    a cross-accumulator tree-fold ``acc = op(acc_0, …, acc_{FK-1})`` is
    emitted *after the enclosing K serial loops close* — i.e. at the first
    enclosing non-K-serial scope (the ThreadTile body, or the FM/FN output
    RegisterTile body when one wraps the reduce). ``in_k_serial`` tracks
    whether the body being walked is the body of a K serial loop
    (``SerialTile`` of kind ``serial_outer`` / ``stage_inner``); folds
    bubble up through those and land at the first scope where it's False.
    """
    out: list[Stmt] = []
    factors: list[int] = []
    saw_register = False
    bubble_folds: list[Stmt] = []
    for s in body:
        if isinstance(s, RegisterTile):
            saw_register = True
            inner_unwrapped, inner_factors, inner_saw, inner_folds = _replicate_register_tiles(s.body, in_k_serial=False)
            current = inner_unwrapped
            saw_register = saw_register or inner_saw
            local_factors: list[int] = []
            # Replicate from innermost axis outward.
            for ax in reversed(s.axes):
                factor = ax.extent.as_static()
                current = _replicate_along_axis(current, ax.name, factor, _sigma_to_literal(ax.name))
                local_factors.append(factor)
            local_factors.reverse()
            out.extend(current)
            # inner_folds were generated with in_k_serial=False, so they've
            # already been placed inside ``current``; none escape here.
            my_folds = list(inner_folds)
            if s.reduce:
                my_folds.extend(_build_accum_fold(s))
            else:
                factors.extend(local_factors)
            if in_k_serial:
                bubble_folds.extend(my_folds)
            else:
                out.extend(my_folds)
        elif s.nested():
            # Descend into block stmts (SerialTile / StridedTile / Cond / etc.)
            # to find nested RegisterTiles. Each nested body is rewritten
            # independently and re-attached via ``with_bodies``. A K serial
            # loop body propagates ``in_k_serial=True`` so any reduce fold from
            # below keeps bubbling until the loop closes.
            #
            # A ``StageBundle`` is **transparent** to the K-serial nesting: it
            # wraps the staged K_i loop *inside* K_o (gmem→smem cooperative load +
            # the inner reduce loop), so a reduce fold generated under it must keep
            # bubbling past K_o, not stop at the bundle scope. Propagate the
            # current ``in_k_serial`` through it rather than resetting to False
            # (without this, a multi-stage fp16 matmul drops the FK cross-accumulator
            # fold inside K_o — its master accumulators are then out of scope at the
            # post-K_o store).
            if isinstance(s, StageBundle):
                child_in_k = in_k_serial
            else:
                child_in_k = isinstance(s, SerialTile) and s.kind in ("serial_outer", "stage_inner")
            new_bodies: list[Body] = []
            child_folds: list[Stmt] = []
            for sub in s.nested():
                new_sub, sub_factors, sub_saw, sub_folds = _replicate_register_tiles(sub, in_k_serial=child_in_k)
                new_bodies.append(new_sub)
                factors.extend(sub_factors)
                saw_register = saw_register or sub_saw
                child_folds.extend(sub_folds)
            out.append(s.with_bodies(tuple(new_bodies)))
            # The K serial loop ``s`` has now closed at this scope. If this scope
            # is itself a K serial loop body, keep bubbling; otherwise the
            # accumulators are final — drop the fold right after ``s``.
            if in_k_serial:
                bubble_folds.extend(child_folds)
            else:
                out.extend(child_folds)
        else:
            out.append(s)
    return Body(out), factors, saw_register, bubble_folds


def _build_accum_fold(tile: RegisterTile) -> list[Stmt]:
    """Cross-accumulator tree-fold for an FK reduce tile. The tile's single
    K_f axis has extent FK; ``010``'s replication has just turned each wrapped
    ``Accum(acc, …)`` into siblings ``acc_0 .. acc_{FK-1}``. Emit
    ``acc = op(acc_0, …, acc_{FK-1})`` as a balanced binary tree of ``Assign``s
    (same shape as ``_combine.TreeHalve``) so downstream reads of ``acc`` see
    the combined partial — the materializer / cross-thread combine then run on
    a single ``acc`` exactly as in the FK=1 path. Returns ``[]`` when the tile
    wraps no ``Accum`` (a post-pointwise K_f tile — FK-unrolled writes only)."""
    (k_f,) = tile.axes
    fk = k_f.extent.as_static()
    folds: list[Stmt] = []
    for accum in _iter_accums(tile.body):
        # The FK strip-mine reorders the per-thread reduction (independent partial
        # sums folded at the end), so the combine must be associative — the same
        # fp-reassociation the BR cross-thread warp-shuffle combine relies on.
        if not accum.op.associative:
            raise RuleSkipped(f"FK fold requires an associative reduce op; got {accum.op.name!r} on accumulator {accum.name!r}")
        folds.extend(_fold_tree(accum.name, fk, accum.op))
    return folds


def _repl_defs(s: Stmt) -> tuple[str, ...]:
    """The SSA names ``s`` binds **for replication dataflow**. A :class:`Reassign` rebinds an
    already-declared carried name (its global ``Stmt.defines()`` stays ``()`` to keep SSA-uniqueness
    analyses happy), but for register-axis replication it must count as defining that name — otherwise
    a carried state rebound by a ``Reassign`` and dependent on the replicated axis (the FA-2 ``O[d]``
    accumulator a ``ScalarCombiner`` rebinds) never propagates ``keep`` and stays a single scalar."""
    if isinstance(s, Reassign):
        return (s.name,)
    return s.defines()


def _iter_accums(body: Body) -> list[Accum]:
    """All ``Accum``s reachable in ``body`` (recursing through nested block
    stmts). For an FK reduce tile the body is the flat innermost reduce
    ``[Load, Assign…, Accum]``, but recurse defensively."""
    found: list[Accum] = []
    for s in body:
        if isinstance(s, Accum):
            found.append(s)
        for sub in s.nested():
            found.extend(_iter_accums(sub))
    return found


def _fold_tree(acc: str, fk: int, op) -> list[Stmt]:
    """Balanced binary fold of ``acc_0 .. acc_{fk-1}`` into ``acc`` via ``op``.
    Intermediate sums get unique ``<acc>__fk<n>`` names; the final ``Assign``
    defines ``acc`` itself, so the original accumulator name is rebound to the
    combined value (matching ``acc = (acc0+acc1)+(acc2+acc3)`` in the plan)."""
    level = [f"{acc}_{i}" for i in range(fk)]
    out: list[Stmt] = []
    ctr = 0
    while len(level) > 1:
        nxt: list[str] = []
        i = 0
        while i < len(level):
            if i + 1 < len(level):
                final = len(level) == 2  # the last surviving pair names the result
                name = acc if final else f"{acc}__fk{ctr}"
                ctr += 1
                out.append(Assign(name=name, op=op, args=(level[i], level[i + 1])))
                nxt.append(name)
                i += 2
            else:
                nxt.append(level[i])
                i += 1
        level = nxt
    return out


def _sigma_to_literal(axis: str) -> Callable[[int], Sigma]:
    """σ-factory: ``axis → Literal(i)``."""

    def _f(i: int) -> Sigma:
        return Sigma({axis: Literal(i, "int")})

    return _f


def _replicate_along_axis(
    body: Body,
    axis: str,
    factor: int,
    sigma_for: Callable[[int], Sigma],
) -> Body:
    """F× replicate every stmt whose value transitively depends on
    ``axis``. Each such stmt is emitted ``factor`` times with σ given
    by ``sigma_for(i)`` and SSA names suffixed ``_<i>``. Stmts that
    don't depend on ``axis`` pass through. Block stmts recurse into
    their bodies and rebuild via :meth:`Stmt.with_bodies`; a wrapper
    that shadows ``axis`` isn't itself replicated (the fold's bound-
    axis filter keeps shadowed references local).

    Dependency analysis is one :meth:`Body.fold` over the def-use DAG
    with bound-axis filtering. ``keep[name]`` records which SSA names
    must carry the suffix vs. pass through unchanged (Tile-input
    buffers, constants, axis-free producers)."""

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        # StageBundle cache-axis Vars are smem-local — they don't vary
        # per replica. Mark them bound here so the staging IR isn't tagged for
        # replication; only the consumer Loads (which σ-rewrite cache-axis
        # Vars) multiply.
        if isinstance(s, StageBundle):
            local_bound = bound | frozenset(ax.name for src in s.sources for ax in src.cache_axes)
        else:
            local_bound = bound
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in local_bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in _repl_defs(s)}

    # SSA def-use propagation: if any SSA name a stmt reads has keep=True,
    # then everything it defines must also be marked keep. The fold above
    # tracks free-var presence per Expr but doesn't chase the SSA chain —
    # e.g. ``in1 = load w[(int)in0, a3]`` has free vars ``{in0, a3}`` (no
    # ``a2``), so its keep stays False even though ``in0`` is replicated.
    # Without this propagation the dependent Load survives as one copy and
    # all replicas read the lane-0 idx (the embedding-lookup bug).
    defined_names = set(keep)
    changed = True
    while changed:
        changed = False
        for s in body.iter():
            reads: set[str] = set(s.deps())
            for e in s.exprs():
                reads.update(e.free_vars())
            reads &= defined_names
            if any(keep.get(r, False) for r in reads):
                for n in _repl_defs(s):
                    if not keep.get(n, False):
                        keep[n] = True
                        changed = True

    def rename_for(i: int):
        def _rename(name: str) -> str:
            return f"{name}_{i}" if keep.get(name, False) else name

        return _rename

    def go(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            nested = s.nested()
            # Block stmts whose OWN exprs (predicate, etc.) reference the
            # replicated axis need full replication — descending into the
            # nested bodies only leaves the wrapper's expression unsubstituted.
            # Example: a masked-tile ``Cond(<post-σ N expr> < real_extent)``
            # wrapping a per-cell Write — each replica must get its own σ-folded
            # predicate, otherwise the wrapper references a no-longer-defined
            # register-axis Var (or worse, collides with a later loop axis
            # named the same). StageBundle hides its cache axes from this
            # check — those Vars are smem-local and shouldn't drive
            # replication of the wrapper.
            if isinstance(s, StageBundle):
                wrapper_bound = frozenset(ax.name for src in s.sources for ax in src.cache_axes)
            else:
                wrapper_bound = frozenset()
            own_refs_axis = axis not in wrapper_bound and any(axis in e.free_vars() for e in s.exprs())
            if isinstance(s, StageBundle) and nested and not own_refs_axis:
                # A StageBundle's COMPUTE phase is a cooperative slab fill (the
                # SMEM fused edge's producer — it writes the WHOLE slab, every
                # register cell; ``emit_compute_phase`` re-iterates its cache
                # axes), so it must NOT be register-replicated. Only the consumer
                # BODY descends — its Loads read the cell's own slab position and
                # so replicate across the REGISTER axis. (Source-side state stays
                # untouched: the cache-axis Vars are smem-local, masked above.)
                out.append(s.with_bodies((s.compute if s.compute is not None else Body(()), go(s.body))))
            elif nested and not own_refs_axis:
                # Wrap-body Stage's consumer body must be descended so the
                # consumer Loads inside the staged scope replicate across the
                # REGISTER axis. Stage's source-side state (cache_axes, origin)
                # stays untouched — the per-source Vars are smem-local and
                # the cache-axis 'bound' mask in the fold guard above prevents
                # 010_split_register_axes from tagging them for replication.
                out.append(s.with_bodies(tuple(go(child) for child in nested)))
            elif _needs_replication(s, axis, deps, keep):
                for i in range(factor):
                    out.append(s.rewrite(rename_for(i), sigma_for(i)))
            else:
                out.append(s)
        return Body(out)

    return go(body)


def _needs_replication(s, axis: str, deps: dict, keep: dict[str, bool]) -> bool:
    """A stmt needs replication along ``axis`` iff (a) the per-id deps for
    this exact stmt include axis, OR (b) it defines an SSA name marked
    ``keep[name] = True`` (the cross-scope-tolerant version of the same
    check). The keep-fallback exists because ``body.fold``'s per-id deps
    use ``body.definitions[name]`` (name → last-defining stmt) to look up
    child memos: for two sibling scopes that re-use a name, the FIRST
    sibling's stmts see ``memo[id(other_sibling_def)] = None`` and end up
    with empty deps even though they transitively read the axis. Using
    keep[name] (which is last-wins across the body and correctly reflects
    *some* definer reads the axis) covers the missed case."""
    if axis in deps.get(id(s), frozenset()):
        return True
    return any(keep.get(n, False) for n in _repl_defs(s))
