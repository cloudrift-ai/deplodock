"""Planner-driven register tile — runs *after* ``010_stage_inputs``.

When ``000_partition_loops`` pre-splits a matmul's output Loops and
tags the inner halves ``Role.REGISTER``, this pass unwraps those
REGISTER Loops and replicates their bodies per-cell. By the time this
pass runs, ``010_stage_inputs`` has already emitted Stages with
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
from deplodock.compiler.ir.tile.ir import RegisterTile, Stage, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import replace_thread_tile_body, single_tile, thread_tile_of

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, outer = single_tile(body)
    tt = thread_tile_of(outer)

    new_body, factors = _replicate_register_tiles(tt.body)
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


def _replicate_register_tiles(body: Body) -> tuple[Body, list[int]]:
    """Inside-out unwrap of ``RegisterTile`` flavors. For each tile, recurse
    into nested RegisterTiles first, then replicate this layer's body by
    ``axis.extent`` with ``σ: axis → literal(i)`` for each of the tile's
    axes (outermost first). Returns ``(new_body, factors)`` with factors
    in outermost-first order. Caller stamps factors[0] → FM, factors[1] → FN.

    Walks into non-RegisterTile block stmts (e.g. ``SerialTile`` wrapping
    a softmax prologue + ``RegisterTile``-tiled matmul body for SDPA P@V)
    so deeply-nested RegisterTiles get replicated too.
    """
    out: list[Stmt] = []
    factors: list[int] = []
    for s in body:
        if isinstance(s, RegisterTile):
            inner_unwrapped, inner_factors = _replicate_register_tiles(s.body)
            current = inner_unwrapped
            local_factors: list[int] = []
            # Replicate from innermost axis outward.
            for ax in reversed(s.axes):
                factor = int(ax.extent)
                current = _replicate_along_axis(current, ax.name, factor, _sigma_to_literal(ax.name))
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
                new_sub, sub_factors = _replicate_register_tiles(sub)
                new_bodies.append(new_sub)
                factors.extend(sub_factors)
            out.append(s.with_bodies(tuple(new_bodies)))
        else:
            out.append(s)
    return Body(out), factors


def _sigma_to_literal(axis: str) -> Callable[[int], Sigma]:
    """σ-factory: ``axis → Literal(i)``."""

    def _f(i: int) -> Sigma:
        return Sigma({axis: Literal(i, "int")})

    return _f


def _replicate_along_axis(body: Body, axis: str, factor: int, sigma_for: Callable[[int], Sigma]) -> Body:
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
        # Stage cache-axis Vars are smem-local — they don't vary per replica.
        # Mark them bound here so Stages aren't tagged for replication; only
        # the consumer Loads (which σ-rewrite cache-axis Vars) multiply.
        if isinstance(s, Stage):
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
    keep: dict[str, bool] = {n: axis in deps[id(s)] for s in body.iter() for n in s.defines()}

    def rename_for(i: int):
        def _rename(name: str) -> str:
            return f"{name}_{i}" if keep.get(name, False) else name

        return _rename

    def go(b: Body) -> Body:
        out: list[Stmt] = []
        for s in b:
            nested = s.nested()
            if nested:
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
