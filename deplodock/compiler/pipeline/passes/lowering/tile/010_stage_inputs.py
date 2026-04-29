"""Stage frequently-reused external inputs into shared memory.

For each reduce ``Loop`` (or ``StridedLoop`` from cooperative reduce)
in the Tile body, examine its external Loads. A Load is *stage-worthy*
when its index does not reference at least one ``BIND_THREAD`` axis —
meaning multiple threads (across that axis) read the same element, so
a per-block smem cache eliminates the redundant DRAM traffic.

The pass walks scopes top-down. A "scope" is the Tile body or the body
of a free ``Loop`` nested inside it. For every reduce Loop / StridedLoop
sitting directly in a scope, classify each Load:

- Cache axes = the thread axes present in the index plus the reduce
  axis. Each cache axis must appear in exactly one source-buffer dim
  (no packed dims).
- Origin = the index with all cache axes substituted to 0.
- The cache part per dim must equal ``Var(cache_axis)`` exactly — the
  decomposition only handles coefficient-1 affine cases (after the
  div/mod simplifier folds collapsed-reshape indices).

Stage proposals at the same scope are grouped by
``(buf, origin, cache_axes, slab_dims)`` so siblings reading the same
slab share one Stage. Stages emit at the head of the scope's body —
their origin references only Vars bound at this scope, so placement is
always legal. Loads in the reduce loops are rewritten to read from the
staged smem at cache-local coordinates.

Skipped silently:
- Loads with no missing thread axis (no reuse).
- Loads where any cache axis appears in multiple source dims, or where
  the residue isn't ``Var(cache_axis)`` (non-affine / packed).
- Slabs whose float count exceeds ``_MAX_SLAB_FLOATS``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Load, Loop, Stmt, StridedLoop, Tile, iter_body, map_body
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_MAX_SLAB_FLOATS = 4096  # 16KB per Stage at fp32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple[Stmt, ...]) -> tuple[Stmt, ...] | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if any(isinstance(s, Stage) for s in iter_body(tile.body)):
        return None  # idempotence
    if not tile.thread_axes:
        return None

    used_names: set[str] = set()
    new_tile_body = _process_scope(tile, tile.thread_axes, tile.all_axes, used_names)
    if new_tile_body == tile.body:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    used_names: set[str],
) -> tuple[Stmt, ...]:
    stages: dict[tuple, Stage] = {}
    rewritten: list[Stmt] = []

    for s in scope.body:
        if isinstance(s, Loop) and not s.is_reduce:
            rewritten.append(Loop(axis=s.axis, body=_process_scope(s, thread_axes, (*in_scope_axes, s.axis), used_names)))
        elif isinstance(s, (Loop, StridedLoop)):
            rewritten.append(_stage_loop(s, thread_axes, in_scope_axes, stages, used_names))
        else:
            rewritten.append(s)

    if not stages:
        return tuple(rewritten)
    return tuple([*stages.values(), *rewritten])


def _stage_loop(
    loop: Loop | StridedLoop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    stages: dict[tuple, Stage],
    used_names: set[str],
) -> Loop | StridedLoop:
    """Stage reusable Loads from a ``Loop`` or ``StridedLoop`` body.

    For regular reduce ``Loop``s the iteration axis is the reduce axis
    and reuse is "thread axis missing from index" (matmul shape).
    For ``StridedLoop`` (cooperative-reduce: each thread strides through
    the iteration axis with step ``BLOCK_SIZE``), the same missing-thread-
    axis check applies — the cooperative thread axis ``t`` doesn't appear
    in the load index because the body uses the strided axis directly.
    Staging caches the row in smem so subsequent StridedLoops in the same
    scope (e.g. softmax's max / exp-sum / output passes) read from smem
    instead of DRAM.
    """
    scope_axes = (*in_scope_axes, loop.axis)
    rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    body_loads = tuple(s for s in loop.body if isinstance(s, Load))
    for stmt in body_loads:
        classified = _classify(stmt, thread_axes, loop.axis, scope_axes)
        if classified is None:
            continue
        origin, cache_axes, slab_dims, template = classified
        n_floats = 1
        for ax in cache_axes:
            n_floats *= int(ax.extent)
        if n_floats > _MAX_SLAB_FLOATS:
            continue
        origin_key = tuple(tuple(sorted(t.pretty() for t in _flatten_add(e))) for e in origin)
        # Cache axis names differ across sibling reduce loops in cooperative-
        # reduce kernels (softmax has three StridedLoops with axes a2, a3, a4
        # all sweeping the same row); key on extent + slab dim so the row
        # is staged once and shared.
        cache_key = tuple(int(ax.extent) for ax in cache_axes)
        template_key = tuple(e.pretty() for e in template) if template is not None else None
        key = (stmt.input, origin_key, cache_key, slab_dims, template_key)
        if key not in stages:
            smem_name = _gen_name(stmt.input, used_names)
            stages[key] = Stage(
                name=smem_name,
                buf=stmt.input,
                origin=origin,
                axes=cache_axes,
                slab_dims=slab_dims,
                source_index_template=template,
            )
        rewrites[stmt.name] = (stages[key].name, tuple(Var(ax.name) for ax in cache_axes))

    def replace(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.name in rewrites:
            smem_name, new_index = rewrites[s.name]
            return Load(name=s.name, input=smem_name, index=new_index)
        return s

    new_body = map_body(loop.body, replace)
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=new_body, unroll=loop.unroll)
    return Loop(axis=loop.axis, body=new_body, unroll=loop.unroll)


def _classify(
    load: Load,
    thread_axes: tuple[Axis, ...],
    reduce_axis: Axis,
    scope_axes: tuple[Axis, ...],
) -> tuple[tuple[Expr, ...], tuple[Axis, ...], tuple[int, ...], tuple[Expr, ...] | None] | None:
    """Returns (origin, cache_axes, slab_dims, source_index_template).

    The template is non-None when the additive ``origin + Var(cache_axis)``
    form can't represent the access — typically when multiple cache axes
    target the same source dim (post-rebalance kernels) or when the affine
    composition of cache vars in a single dim isn't coefficient-1. Each
    Load gets its own per-Load slab; a downstream merge pass (planned)
    can fuse contiguous siblings into a single larger slab.
    """
    idx = load.index
    candidates_by_name = {ax.name: ax for ax in (*thread_axes, reduce_axis)}
    cache_axes_list: list[Axis] = []
    seen: set[str] = set()
    for e in idx:
        for v in e.free_vars():
            if v in candidates_by_name and v not in seen:
                cache_axes_list.append(candidates_by_name[v])
                seen.add(v)
    if not cache_axes_list:
        return None
    if not ({ax.name for ax in thread_axes} - seen):
        return None  # every thread axis appears → no reuse

    var_to_dim: dict[str, int] = {}
    for ax in cache_axes_list:
        dims = [d for d, e in enumerate(idx) if ax.name in e.free_vars()]
        if len(dims) != 1:
            return None  # cache var spans multiple dims — can't represent
        var_to_dim[ax.name] = dims[0]

    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})
    cache_zero = Sigma({ax.name: Literal(0, "int") for ax in cache_axes_list})
    origin = tuple(cache_zero.apply(e).simplify(ctx) for e in idx)

    cache_axes = tuple(cache_axes_list)
    slab_dims = tuple(var_to_dim[ax.name] for ax in cache_axes)

    # Multi-cache-axis-per-dim → use template path (additive form can only
    # carry one cache axis per dim because ``decoded_per_dim`` overwrites).
    needs_template = len(set(slab_dims)) < len(slab_dims)

    # Verify each cache axis contributes exactly ``Var(ax)`` per dim under
    # the affine-decomposition form. If not, fall back to template too —
    # this catches non-coefficient-1 residues (e.g. ``a3*2``) that the
    # additive Stage path can't represent.
    if not needs_template:
        for ax in cache_axes_list:
            d = var_to_dim[ax.name]
            sigma = Sigma({other.name: Literal(0, "int") for other in cache_axes_list if other.name != ax.name})
            residue = sigma.apply(idx[d]).simplify(ctx)
            expected = (origin[d] + Var(ax.name)).simplify(ctx)
            if not _add_terms_equal(residue, expected):
                needs_template = True
                break

    template = tuple(idx) if needs_template else None
    return origin, cache_axes, slab_dims, template


def _flatten_add(e: Expr) -> list[Expr]:
    if isinstance(e, BinaryExpr) and e.op == "+":
        return _flatten_add(e.left) + _flatten_add(e.right)
    return [e]


def _add_terms_equal(a: Expr, b: Expr) -> bool:
    """Compare two Exprs for equality modulo ``+`` associativity / commutativity.
    The simplifier doesn't canonicalize ``+`` ordering, so a residue like
    ``X + (Y + Z)`` won't pretty-equal an expected ``(X + Y) + Z`` even though
    they denote the same value."""
    ta = sorted(t.pretty() for t in _flatten_add(a))
    tb = sorted(t.pretty() for t in _flatten_add(b))
    return ta == tb


def _gen_name(buf: str, used: set[str]) -> str:
    base = f"{buf}_smem"
    if base not in used:
        used.add(base)
        return base
    n = 1
    while f"{base}_{n}" in used:
        n += 1
    name = f"{base}_{n}"
    used.add(name)
    return name
