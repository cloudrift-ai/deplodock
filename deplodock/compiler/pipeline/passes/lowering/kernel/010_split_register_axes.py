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
"""

from __future__ import annotations

from collections.abc import Callable

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import RegisterTile, Stage, StageBundle, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import replace_thread_tile_body, single_tile, thread_tile_of

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, outer = single_tile(body)
    tt = thread_tile_of(outer)

    # Body-global keep set per register axis. Computed once over the entire
    # ThreadTile body so sibling RegisterTile(axis) towers (the blocked-GEMM
    # nest emits three: Init / K-reduce-Accum / Write) agree on which SSA
    # names carry the per-cell ``_<i>`` suffix. Without this, each tower's
    # local keep-analysis would run independently — the Init tower doesn't
    # reference N_r, so ``acc`` would stay one name there, while the Accum
    # tower produces ``acc_0..N-1`` — a naming mismatch the consumer Write
    # would inherit.
    global_keep = _collect_body_global_keep(tt.body)
    new_body, factors = _replicate_register_tiles(tt.body, global_keep=global_keep)
    if not factors:
        # No RegisterTile in body — non-matmul kernel (planner only emits
        # RegisterTile for matmul) or this rule has already run and
        # consumed them. Either way, nothing to do.
        raise RuleSkipped("no RegisterTile in body")

    # FM/FN are stamped by the planner; preserve them rather than
    # overwriting (factors carry the same values).
    knobs = dict(root.op.knobs)
    if len(factors) >= 1 and "FM" not in knobs:
        knobs["FM"] = factors[0]
    if len(factors) >= 2 and "FN" not in knobs:
        knobs["FN"] = factors[1]
    rebuilt = replace_thread_tile_body(outer, new_body)
    return TileOp(body=body[:idx] + (rebuilt,) + body[idx + 1 :], name=root.op.name, knobs=knobs)


def _replicate_register_tiles(body: Body, *, global_keep: dict[str, dict[str, bool]] | None = None) -> tuple[Body, list[int]]:
    """Inside-out unwrap of ``RegisterTile`` flavors. For each tile, recurse
    into nested RegisterTiles first, then replicate this layer's body by
    ``axis.extent`` with ``σ: axis → literal(i)`` for each of the tile's
    axes (outermost first). Returns ``(new_body, factors)`` with factors
    in outermost-first order. Caller stamps factors[0] → FM, factors[1] → FN.

    Walks into non-RegisterTile block stmts (e.g. ``SerialTile`` wrapping
    a softmax prologue + ``RegisterTile``-tiled matmul body for SDPA P@V)
    so deeply-nested RegisterTiles get replicated too.

    ``global_keep`` is the body-global keep map (axis → name → True/False)
    computed at the TileOp body level. It's threaded into every per-axis
    ``_replicate_along_axis`` call so sibling RegisterTile(axis) towers
    agree on which names get the per-cell ``_<i>`` suffix — the local
    fold inside one tower can't see that a name is register-tiled by
    another sibling tower's body.
    """
    out: list[Stmt] = []
    factors: list[int] = []
    for s in body:
        if isinstance(s, RegisterTile):
            inner_unwrapped, inner_factors = _replicate_register_tiles(s.body, global_keep=global_keep)
            current = inner_unwrapped
            local_factors: list[int] = []
            # Replicate from innermost axis outward.
            for ax in reversed(s.axes):
                factor = ax.extent.as_static()
                extra_keep = global_keep.get(ax.name) if global_keep is not None else None
                current = _replicate_along_axis(current, ax.name, factor, _sigma_to_literal(ax.name), extra_keep=extra_keep)
                local_factors.append(factor)
            local_factors.reverse()
            out.extend(current)
            factors.extend(local_factors)
            factors.extend(inner_factors)
        elif s.nested():
            # Descend into block stmts (SerialTile / StridedTile / Cond / etc.)
            # to find nested RegisterTiles. Each nested body is rewritten
            # independently and re-attached via ``with_bodies``.
            new_bodies: list[Body] = []
            for sub in s.nested():
                new_sub, sub_factors = _replicate_register_tiles(sub, global_keep=global_keep)
                new_bodies.append(new_sub)
                factors.extend(sub_factors)
            out.append(s.with_bodies(tuple(new_bodies)))
        else:
            out.append(s)
    return Body(out), factors


def _collect_body_global_keep(body: Body) -> dict[str, dict[str, bool]]:
    """For every register axis appearing in any ``RegisterTile`` within
    ``body``, compute the body-global keep set — which SSA names defined
    anywhere in ``body`` transitively reference that axis. Used to align
    replicator suffixing across sibling/cousin RegisterTile(axis) towers
    (the blocked-GEMM nest's three towers, SDPA-prologue+matmul, etc.).
    """
    axes: set[str] = set()
    for s in body.iter():
        if isinstance(s, RegisterTile):
            for ax in s.axes:
                axes.add(ax.name)
    return {ax: _global_keep_for_axis(body, ax) for ax in axes}


def _global_keep_for_axis(body: Body, axis: str) -> dict[str, bool]:
    """Body-global keep set for ``axis``: which SSA names defined anywhere
    in ``body`` transitively reference ``axis`` (and therefore must carry
    the per-cell ``_<i>`` suffix at replicate time, regardless of which
    sibling RegisterTile(axis) tower defines them).

    Mirrors the per-body fold in :func:`_replicate_along_axis`, but
    exempts ``axis`` from ``bound`` so a RegisterTile(axis) wrapper
    doesn't mask references inside it — the whole point of this walk
    is to see those references. Stage / StageBundle cache axes stay
    masked: they're smem-local and don't vary per replica. Other
    enclosing-wrapper-bound axes pass through unaffected (they aren't
    ``axis`` by construction).

    Multiple defs of the same name (e.g. an Accum's name re-defined in
    every sibling tower) OR together — keep[name]=True iff ANY definer
    reads ``axis``.
    """

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        if isinstance(s, Stage):
            local_bound = bound | frozenset(ax.name for src in s.sources for ax in src.cache_axes)
        elif isinstance(s, StageBundle):
            local_bound = bound | frozenset(ax.name for stage in s.stages for src in stage.sources for ax in src.cache_axes)
        else:
            local_bound = bound
        local_bound = local_bound - {axis}
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in local_bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    keep: dict[str, bool] = {}
    for s in body.iter():
        has_axis = axis in deps[id(s)]
        for n in s.defines():
            keep[n] = keep.get(n, False) or has_axis

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
                for n in s.defines():
                    if not keep.get(n, False):
                        keep[n] = True
                        changed = True
    return keep


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
    *,
    extra_keep: dict[str, bool] | None = None,
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
    buffers, constants, axis-free producers).

    ``extra_keep`` is the body-global keep map for ``axis`` (see
    :func:`_global_keep_for_axis`): names True there must be suffixed
    regardless of the local body's def-use, so an ``Init(acc)`` /
    ``Write(C, acc)`` tower aligned with an ``Accum(acc, …N_r…)`` sibling
    tower replicates and renames consistently (acc → acc_0..N-1) in all
    three. OR-merged into the locally-derived keep."""

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        # Stage / StageBundle cache-axis Vars are smem-local — they don't vary
        # per replica. Mark them bound here so the staging IR isn't tagged for
        # replication; only the consumer Loads (which σ-rewrite cache-axis
        # Vars) multiply.
        if isinstance(s, Stage):
            local_bound = bound | frozenset(ax.name for src in s.sources for ax in src.cache_axes)
        elif isinstance(s, StageBundle):
            local_bound = bound | frozenset(ax.name for stage in s.stages for src in stage.sources for ax in src.cache_axes)
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
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in s.defines()}
    # Merge in body-global keep entries (the per-tower local fold can't see
    # that a name register-tiled by a sibling tower must also be suffixed
    # here). OR-merge: a name keep=True globally stays True locally.
    if extra_keep is not None:
        for n, v in extra_keep.items():
            if v:
                keep[n] = True

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
                for n in s.defines():
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
            # named the same). Stage/StageBundle hide their cache axes from
            # this check — those Vars are smem-local and shouldn't drive
            # replication of the wrapper.
            if isinstance(s, Stage):
                wrapper_bound = frozenset(ax.name for src in s.sources for ax in src.cache_axes)
            elif isinstance(s, StageBundle):
                wrapper_bound = frozenset(ax.name for stage in s.stages for src in stage.sources for ax in src.cache_axes)
            else:
                wrapper_bound = frozenset()
            own_refs_axis = axis not in wrapper_bound and any(axis in e.free_vars() for e in s.exprs())
            if nested and not own_refs_axis:
                # Wrap-body Stage's consumer body must be descended so the
                # consumer Loads inside the staged scope replicate across the
                # REGISTER axis. Stage's source-side state (cache_axes, origin)
                # stays untouched — the per-source Vars are smem-local and
                # the cache-axis 'bound' mask in the fold guard above prevents
                # 006a from tagging them for replication.
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
    return any(keep.get(n, False) for n in s.defines())
