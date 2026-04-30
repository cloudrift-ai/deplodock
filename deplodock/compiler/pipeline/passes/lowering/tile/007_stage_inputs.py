"""Stage frequently-reused external inputs into shared memory.

Runs early in the tile-lowering chain — *before* ``008_register_tile``
and ``009_rebalance_threads`` — so the classifier sees the clean PAT ×
PAT thread-axis layout from ``005_blockify_launch``. Downstream passes
keep already-staged Stages intact: ``register_tile`` σ-substitutes
cache-axis Vars in the consumer Loads (Stages stay singleton across
F²); ``rebalance_threads`` augments Stage cache-axes when carving a
referenced BLOCK axis.

**Reuse decision.** For each Load inside a reduce Loop / StridedLoop,
compute ``reuse = work / index_set_size`` where ``work`` is the product
of bound-axis extents (threads × reduce iters) over the load's
iteration domain and ``index_set_size`` is the affine-bound projection
of the index (see :func:`index_set_size`). Stage when ``reuse > 1`` —
this captures *any* fan-in: a missing thread axis (matmul shape), a
load whose index doesn't reference the reduce var (loop-invariant in
the reduce), or correlated coefficients that alias rows.

**Decomposition for accepted Stages.** Each candidate cache var
(thread axis or reduce axis appearing in the index) must (a) live in
exactly one source dim and (b) have coefficient 1 for the additive
Stage form to apply; otherwise we fall back to
``source_index_template`` which materialization decodes via Sigma at
cooperative-load time.

Sibling Loads at the same scope sharing
``(buf, origin, cache_axes, slab_dims, template)`` collapse to one
Stage. ``009_rebalance_threads`` carries its own smem-budget gate, so
this pass admits any reuse-positive Load up to the per-Stage cap.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, SimplifyCtx, Var, affine_form, index_set_size
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_MAX_SLAB_FLOATS = 4096  # 16KB hard cap per Stage at fp32


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
    """Walk a reduce ``Loop`` / ``StridedLoop`` body. For each Load with
    ``reuse > 1`` (per :func:`index_set_size`), build or reuse a Stage at
    the surrounding scope and rewrite the Load to read from smem. ``stages``
    is shared across sibling reduces in the same scope so they collapse to
    one slab when their (buf, origin, cache_axes, slab_dims, template) keys
    match."""
    scope_axes = (*in_scope_axes, loop.axis)
    bound_extents = {ax.name: int(ax.extent) for ax in (*thread_axes, loop.axis)}
    work = 1
    for e in bound_extents.values():
        work *= e

    rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    for stmt in loop.body:
        if not isinstance(stmt, Load):
            continue
        built = _build_stage(stmt, thread_axes, loop.axis, scope_axes, bound_extents, work)
        if built is None:
            continue
        key, stage = built
        if key not in stages:
            stage = Stage(
                name=_gen_name(stage.buf, used_names),
                buf=stage.buf,
                origin=stage.origin,
                axes=stage.axes,
                slab_dims=stage.slab_dims,
                source_index_template=stage.source_index_template,
            )
            stages[key] = stage
        rewrites[stmt.name] = (stages[key].name, tuple(Var(ax.name) for ax in stages[key].axes))

    def replace(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.name in rewrites:
            smem_name, new_index = rewrites[s.name]
            return Load(name=s.name, input=smem_name, index=new_index)
        return s

    new_body = loop.body.map(replace)
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=new_body, unroll=loop.unroll)
    return Loop(axis=loop.axis, body=new_body, unroll=loop.unroll)


def _build_stage(
    load: Load,
    thread_axes: tuple[Axis, ...],
    reduce_axis: Axis,
    scope_axes: tuple[Axis, ...],
    bound_extents: dict[str, int],
    work: int,
) -> tuple[tuple, Stage] | None:
    """Return ``(key, Stage)`` when ``load`` qualifies for staging.

    Decompose each index entry into affine form over the candidate cache
    vars (thread axes + reduce axis). Reject when:

    - any entry is non-affine in those vars;
    - a cache var spans multiple source dims (collapsed-reshape that
      can't be split per-dim);
    - ``reuse = work / index_set_size <= 1`` — no fan-in across threads /
      reduce iters;
    - the slab exceeds ``_MAX_SLAB_FLOATS``.

    Coefficient-≠-1 cases fall back to ``source_index_template``. The
    Stage's ``name`` is a placeholder; ``_stage_loop`` assigns the real
    smem name on first admission.
    """
    candidates_by_name = {ax.name: ax for ax in (*thread_axes, reduce_axis)}
    cache_var_set = frozenset(candidates_by_name)

    forms = [affine_form(e, cache_var_set) for e in load.index]
    if any(f is None for f in forms):
        return None
    forms = [f for f in forms if f is not None]

    seen_vars: list[str] = []
    var_to_dim: dict[str, int] = {}
    coeff_one: dict[str, bool] = {}
    for d, (_, coeffs) in enumerate(forms):
        for v, c in coeffs.items():
            if v in var_to_dim:
                return None
            seen_vars.append(v)
            var_to_dim[v] = d
            coeff_one[v] = c == 1

    if not seen_vars:
        return None

    size = index_set_size(load.index, bound_extents)
    if size is None or work <= size:
        return None

    cache_axes = tuple(candidates_by_name[v] for v in seen_vars)
    n_floats = 1
    for ax in cache_axes:
        n_floats *= int(ax.extent)
    if n_floats > _MAX_SLAB_FLOATS:
        return None

    slab_dims = tuple(var_to_dim[v] for v in seen_vars)
    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})
    origin = tuple(anchor.simplify(ctx) for anchor, _ in forms)

    needs_template = not all(coeff_one.values())
    template = tuple(load.index) if needs_template else None

    origin_key = tuple(tuple(sorted(t.pretty() for t in _flatten_add(e))) for e in origin)
    # Sibling reduces in cooperative-reduce kernels use distinct cache-axis
    # names (softmax: a2 / a3 / a4 sweep the same row); key on extents +
    # slab dims so the row is staged once and shared.
    cache_key = tuple(int(ax.extent) for ax in cache_axes)
    template_key = tuple(e.pretty() for e in template) if template is not None else None
    key = (load.input, origin_key, cache_key, slab_dims, template_key)

    return key, Stage(
        name="",  # filled in by caller on first admission
        buf=load.input,
        origin=origin,
        axes=cache_axes,
        slab_dims=slab_dims,
        source_index_template=template,
    )


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
