"""Stage frequently-reused external inputs into shared memory.

Runs early in the tile-lowering chain — *before* ``008_register_tile``
and ``009_rebalance_threads`` — so the classifier sees the clean PAT ×
PAT thread-axis layout from ``005_blockify_launch``. Downstream passes
keep already-staged Stages intact: ``register_tile`` σ-substitutes
cache-axis Vars in the consumer Loads (Stages stay singleton across
F²); ``rebalance_threads`` augments Stage cache-axes when carving a
referenced BLOCK axis.

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
- The cache part per dim should equal ``Var(cache_axis)`` exactly under
  the affine-decomposition form; if it doesn't (collapsed-reshape
  views with surviving div/mod residues), fall back to
  ``source_index_template`` — materialization decodes via Sigma at
  cooperative-load time.

Stage proposals at the same scope are grouped by
``(buf, origin, cache_axes, slab_dims)`` so siblings reading the same
slab share one Stage. Stages emit at the head of the scope's body —
their origin references only Vars bound at this scope, so placement is
always legal. Loads in the reduce loops are rewritten to read from the
staged smem at cache-local coordinates.

Skipped silently:
- Loads with no missing thread axis (no reuse).
- Loads where any cache axis appears in multiple source dims.
- Slabs whose float count exceeds ``_MAX_SLAB_FLOATS``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, SimplifyCtx, Var, affine_form
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_MAX_SLAB_FLOATS = 4096  # 16KB per Stage at fp32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    if any(isinstance(s, Stage) for s in tile.body.iter()):
        raise RuleSkipped("Tile body already has Stage stmts (idempotence)")
    if not tile.thread_axes:
        raise RuleSkipped("Tile has no thread_axes — no reuse to stage")

    used_names: set[str] = set()
    new_tile_body = _process_scope(tile, tile.thread_axes, tile.all_axes, used_names)
    if new_tile_body == tile.body:
        raise RuleSkipped("no Load qualifies for staging (no reuse, oversized slab, or unrepresentable cache var)")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    used_names: set[str],
) -> Body:
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

    new_body = loop.body.map(replace)
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

    Decompose each Load index entry as ``anchor + sum(coeffs[v] * Var(v))``
    over the candidate cache vars (thread axes + reduce axis). A cache var
    must (a) be in exactly one dim and (b) have coefficient 1 for the
    additive Stage form to apply; otherwise we fall back to
    ``source_index_template`` which materialization decodes via Sigma at
    cooperative-load time. Skip the Load entirely when no thread axis is
    missing from the index (no cross-thread reuse).
    """
    candidates_by_name = {ax.name: ax for ax in (*thread_axes, reduce_axis)}
    cache_var_set = frozenset(candidates_by_name)

    forms = [affine_form(e, cache_var_set) for e in load.index]
    if any(f is None for f in forms):
        return None
    forms = [f for f in forms if f is not None]  # narrow for type-checker

    seen_vars: list[str] = []
    var_to_dim: dict[str, int] = {}
    coeff_one: dict[str, bool] = {}
    for d, (_, coeffs) in enumerate(forms):
        for v, c in coeffs.items():
            if v in var_to_dim:
                return None  # cache var spans multiple dims
            seen_vars.append(v)
            var_to_dim[v] = d
            coeff_one[v] = c == 1

    if not seen_vars:
        return None
    if not ({ax.name for ax in thread_axes} - set(seen_vars)):
        return None  # every thread axis appears → no reuse

    cache_axes = tuple(candidates_by_name[v] for v in seen_vars)
    slab_dims = tuple(var_to_dim[v] for v in seen_vars)
    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})
    origin = tuple(anchor.simplify(ctx) for anchor, _ in forms)

    needs_template = not all(coeff_one.values())
    template = tuple(load.index) if needs_template else None
    return origin, cache_axes, slab_dims, template


def _flatten_add(e: Expr) -> list[Expr]:
    if isinstance(e, BinaryExpr) and e.op == "+":
        return _flatten_add(e.left) + _flatten_add(e.right)
    return [e]


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
