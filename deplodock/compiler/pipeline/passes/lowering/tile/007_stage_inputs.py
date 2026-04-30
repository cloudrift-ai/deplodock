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

**Per-scope smem budget.** All Stage proposals at a scope are queued,
sorted by reuse (highest first), and admitted greedily until the smem
budget is exhausted. Loads whose proposals don't make the cut keep
reading DRAM. Sibling proposals with the same
``(buf, origin, cache_axes, slab_dims, template)`` key share one Stage.
"""

from __future__ import annotations

from typing import NamedTuple

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, SimplifyCtx, Var, affine_form, index_set_size
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_MAX_SLAB_FLOATS = 4096  # 16KB hard cap per Stage at fp32 (defensive — also bounded by per-scope budget)
_PER_SCOPE_FLOAT_BUDGET = 6144  # 24KB at fp32; downstream pad + double-buffer can ~2× this


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


# A ``_Proposal`` records a Stage candidate at a scope plus the metadata
# needed to admit / reject it under the per-scope budget. ``reuse`` drives
# the greedy admission order; ``stmt_name`` lets us rewrite the Load on
# admission. Multiple Loads with the same ``key`` share one Stage.
class _Proposal(NamedTuple):
    key: tuple
    stmt_name: str
    buf: str
    origin: tuple[Expr, ...]
    cache_axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]
    template: tuple[Expr, ...] | None
    reuse: float
    n_floats: int


def _process_scope(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    used_names: set[str],
) -> Body:
    proposals: list[_Proposal] = []
    rewritten: list[Stmt] = []

    for s in scope.body:
        if isinstance(s, Loop) and not s.is_reduce:
            rewritten.append(Loop(axis=s.axis, body=_process_scope(s, thread_axes, (*in_scope_axes, s.axis), used_names)))
        elif isinstance(s, (Loop, StridedLoop)):
            rewritten.append(_collect_proposals(s, thread_axes, in_scope_axes, proposals))
        else:
            rewritten.append(s)

    if not proposals:
        return tuple(rewritten)

    stages, name_rewrites = _admit_proposals(proposals, used_names)
    if not stages:
        return tuple(rewritten)

    def apply_rewrites(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.name in name_rewrites:
            smem_name, new_index = name_rewrites[s.name]
            return Load(name=s.name, input=smem_name, index=new_index)
        return s

    rewritten_body = tuple(s if isinstance(s, Stmt) and not _has_loads(s) else Body((s,)).map(apply_rewrites)[0] for s in rewritten)
    return tuple([*stages, *rewritten_body])


def _has_loads(s: Stmt) -> bool:
    return any(isinstance(c, Load) for c in Body((s,)).iter())


def _collect_proposals(
    loop: Loop | StridedLoop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    proposals: list[_Proposal],
) -> Loop | StridedLoop:
    """Walk a reduce ``Loop`` / ``StridedLoop`` body and append a Stage
    proposal for every Load with reuse > 1. The Loop body itself is left
    untouched here; the caller rewrites Loads of admitted Stages after
    the per-scope budget pass."""
    scope_axes = (*in_scope_axes, loop.axis)
    bound_extents = {ax.name: int(ax.extent) for ax in (*thread_axes, loop.axis)}
    work = 1
    for e in bound_extents.values():
        work *= e
    for stmt in loop.body:
        if not isinstance(stmt, Load):
            continue
        prop = _classify(stmt, thread_axes, loop.axis, scope_axes, bound_extents, work)
        if prop is None:
            continue
        proposals.append(prop)
    return loop


def _admit_proposals(proposals: list[_Proposal], used_names: set[str]) -> tuple[list[Stage], dict[str, tuple[str, tuple[Expr, ...]]]]:
    """Greedy admission under ``_PER_SCOPE_FLOAT_BUDGET``. Sibling proposals
    with the same key share one Stage (cost paid once). Highest reuse
    admitted first so partial admission still captures the biggest wins."""
    by_key: dict[tuple, list[_Proposal]] = {}
    for p in proposals:
        by_key.setdefault(p.key, []).append(p)

    # One representative per key; reuse = max across siblings (a shared
    # slab pays its cost once but services all sibling Loads).
    reps = sorted(
        ((max(p.reuse for p in group), group[0], group) for group in by_key.values()),
        key=lambda t: -t[0],
    )

    admitted: list[Stage] = []
    name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    used = 0
    for _, rep, group in reps:
        if rep.n_floats > _MAX_SLAB_FLOATS:
            continue
        if used + rep.n_floats > _PER_SCOPE_FLOAT_BUDGET:
            continue
        smem_name = _gen_name(rep.buf, used_names)
        admitted.append(
            Stage(
                name=smem_name,
                buf=rep.buf,
                origin=rep.origin,
                axes=rep.cache_axes,
                slab_dims=rep.slab_dims,
                source_index_template=rep.template,
            )
        )
        used += rep.n_floats
        for p in group:
            name_rewrites[p.stmt_name] = (smem_name, tuple(Var(ax.name) for ax in rep.cache_axes))
    return admitted, name_rewrites


def _classify(
    load: Load,
    thread_axes: tuple[Axis, ...],
    reduce_axis: Axis,
    scope_axes: tuple[Axis, ...],
    bound_extents: dict[str, int],
    work: int,
) -> _Proposal | None:
    """Build a Stage proposal for ``load`` when staging would be profitable.

    Decompose each index entry into affine form over the candidate cache
    vars (thread axes + reduce axis). Reject the Load when:

    - any entry is non-affine in those vars;
    - a cache var spans multiple source dims (collapsed-reshape that
      can't be split per-dim);
    - ``reuse = work / index_set_size <= 1`` (no fan-in across threads /
      reduce iters — staging would cost smem for no DRAM win).

    Coefficient-≠-1 cases fall back to ``source_index_template`` rather
    than being rejected — the additive Stage form can't carry them but
    the materializer's template path can.
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
        return None  # no reuse — every thread / iter sees its own element

    cache_axes = tuple(candidates_by_name[v] for v in seen_vars)
    slab_dims = tuple(var_to_dim[v] for v in seen_vars)
    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})
    origin = tuple(anchor.simplify(ctx) for anchor, _ in forms)

    needs_template = not all(coeff_one.values())
    template = tuple(load.index) if needs_template else None

    n_floats = 1
    for ax in cache_axes:
        n_floats *= int(ax.extent)

    origin_key = tuple(tuple(sorted(t.pretty() for t in _flatten_add(e))) for e in origin)
    # Sibling reduces in cooperative-reduce kernels use distinct cache-axis
    # names (softmax: a2 / a3 / a4 sweep the same row); key on extents +
    # slab dims so the row is staged once and shared.
    cache_key = tuple(int(ax.extent) for ax in cache_axes)
    template_key = tuple(e.pretty() for e in template) if template is not None else None
    key = (load.input, origin_key, cache_key, slab_dims, template_key)

    return _Proposal(
        key=key,
        stmt_name=load.name,
        buf=load.input,
        origin=origin,
        cache_axes=cache_axes,
        slab_dims=slab_dims,
        template=template,
        reuse=work / size,
        n_floats=n_floats,
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
