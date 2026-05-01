"""Stage frequently-reused external inputs into shared memory.

Runs early in the tile-lowering chain — *before* ``008_register_tile``
and ``009_rebalance_threads`` — so the classifier sees the clean PAT ×
PAT thread-axis layout from ``005_blockify_launch``. Downstream passes
keep already-staged Stages intact: ``register_tile`` σ-substitutes
cache-axis Vars in the consumer Loads (Stages stay singleton across
F²); ``rebalance_threads`` augments Stage cache-axes when carving a
referenced BLOCK axis.

**Pipeline:**

1. **Walk the scope**, collecting every Load from every reduce
   ``Loop`` / ``StridedLoop``, tagged with its reduce axis.
2. **Group by source buffer.** Sibling reduces (softmax max/sum/output)
   contribute their Loads to the same buffer's bucket; per-axis-name
   differences in the reduce axis are normalized away by the slab
   signature (reduce axes contribute extent only to the pattern).
3. **Per buffer, fit one slab.** Classify every Load. If they all agree
   on slab geometry, emit one Stage; if they disagree, bail — that
   buffer's Loads stay on DRAM. Bailing is preferred to per-pattern
   partitioning: the cost is missing optimization on rare buffers with
   multiple distinct access shapes within one scope.
4. **Admit & emit.** Greedily admit Stages until the per-scope smem
   budget is hit; emit at scope head; rewrite admitted Loads to read
   from staged smem.

**Slab geometry from a Load's index** (computed in ``_classify``):

- ``origin`` — the index with every cache var (thread + reduce axis)
  substituted to 0. The per-CTA anchor.
- For each cache var, find which source dim its Var appears in.
  Zero dims = fan-in axis (this var's threads/iterations all read the
  same staged value). One dim = cache axis at that dim. Multiple dims
  = bail (collapsed-reshape that can't be additively split).
- Coefficient-1 check: substitute the var → 1 (others → 0) and compare
  to ``origin[d] + 1``. If any axis fails (collapsed-reshape with a
  surviving stride), fall back to ``source_index_template``.

**Reuse.** A Load qualifies for staging iff at least one bound axis
(thread or reduce) doesn't appear in its index — that axis's threads
or iterations all read the same staged value. If every bound axis
appears, no fan-in, skip.
"""

from __future__ import annotations

from typing import NamedTuple

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

_MAX_SLAB_FLOATS = 4096  # 16 KB hard cap per Stage at fp32
# Per-scope budget across all admitted Stages. Consumer hardware static-
# smem cap is 48 KB = 12288 floats; downstream pad + double-buffer can
# ~2× this, so the pre-pad/db budget is ``12288 / 2 = 6144`` floats.
_PER_SCOPE_FLOAT_BUDGET = 6144


class _Slab(NamedTuple):
    """Slab geometry derived from one Load's index."""

    origin: tuple[Expr, ...]
    cache_axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]
    template: tuple[Expr, ...] | None
    n_floats: int


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
        raise RuleSkipped("no Load qualifies for staging")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(
    scope: Tile | Loop,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    used_names: set[str],
) -> Body:
    """Free Loops recurse; reduce loops contribute Loads to this scope's
    per-buffer bucket. Per buffer, derive one slab if all Loads agree;
    admit under budget; emit Stages at scope head; rewrite Loads."""
    rewritten_inner: list[Stmt] = []
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...]]]] = {}

    for s in scope.body:
        if isinstance(s, Loop) and not s.is_reduce:
            rewritten_inner.append(Loop(axis=s.axis, body=_process_scope(s, thread_axes, (*in_scope_axes, s.axis), used_names)))
            continue
        if isinstance(s, (Loop, StridedLoop)):
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes))
        rewritten_inner.append(s)

    stages, name_rewrites = _build_stages(loads_by_buf, thread_axes, used_names)
    if not stages:
        return tuple(rewritten_inner)

    rewritten = tuple(_rewrite_loads(s, name_rewrites) for s in rewritten_inner)
    return tuple([*stages, *rewritten])


def _build_stages(
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...]]]],
    thread_axes: tuple[Axis, ...],
    used_names: set[str],
) -> tuple[list[Stage], dict[str, tuple[str, tuple[Expr, ...]]]]:
    """Per buffer, partition Loads by structural index equality (within a
    loop they collapse only when textually identical — Python-scope
    shadowing provides this for sibling reduce loops). Each partition
    becomes a candidate Stage; classify the representative, admit if
    it fits the per-scope smem budget. Multiple distinct slabs per
    buffer are allowed (e.g. rotate-half kernels load the same buffer
    at multiple conditional positions, each requiring its own slab)."""
    stages: list[Stage] = []
    name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    used_floats = 0

    for buf, items in loads_by_buf.items():
        # Partition this buffer's Loads by structural index equality.
        # Each partition gets its own slab; admission decides which fit.
        partitions: list[tuple[Load, Axis, tuple[Axis, ...], list[Load]]] = []
        for load, reduce_axis, scope_axes in items:
            for rep_load, _, _, members in partitions:
                if load.index == rep_load.index:
                    members.append(load)
                    break
            else:
                partitions.append((load, reduce_axis, scope_axes, [load]))

        for rep_load, rep_reduce, rep_scope, members in partitions:
            slab = _classify(rep_load, thread_axes, rep_reduce, rep_scope)
            if slab is None:
                continue
            if used_floats + slab.n_floats > _PER_SCOPE_FLOAT_BUDGET:
                continue
            smem_name = _gen_name(buf, used_names)
            stages.append(
                Stage(
                    name=smem_name,
                    buf=buf,
                    origin=slab.origin,
                    axes=slab.cache_axes,
                    slab_dims=slab.slab_dims,
                    source_index_template=slab.template,
                )
            )
            used_floats += slab.n_floats
            smem_index = tuple(Var(ax.name) for ax in slab.cache_axes)
            for load in members:
                name_rewrites[load.name] = (smem_name, smem_index)
    return stages, name_rewrites


def _rewrite_loads(stmt: Stmt, name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]]) -> Stmt:
    if not name_rewrites:
        return stmt

    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.name in name_rewrites:
            smem_name, new_index = name_rewrites[s.name]
            return Load(name=s.name, input=smem_name, index=new_index)
        return s

    return Body((stmt,)).map(fn)[0]


def _classify(
    load: Load,
    thread_axes: tuple[Axis, ...],
    reduce_axis: Axis,
    scope_axes: tuple[Axis, ...],
) -> _Slab | None:
    """Derive the slab for ``load`` by evaluating ``load.index`` over
    the cache vars (thread axes + this loop's reduce axis):

    1. ``origin`` = ``index`` with every cache var → 0.
    2. For each cache var, find which source dim its Var appears in.
       Zero dims = fan-in axis (won't constrain the slab). One dim =
       cache axis at that dim. Multiple dims = bail.
    3. Coefficient-1 check: substitute one cache var → 1 (others → 0)
       and compare to ``origin[d] + 1``. If any axis disagrees, fall
       back to ``source_index_template``.

    Returns ``None`` when no fan-in axis exists (no reuse), when a
    cache var spans multiple dims, or when the slab exceeds the cap.
    """
    candidates = (*thread_axes, reduce_axis)
    ctx = SimplifyCtx({ax.name: Interval(0, int(ax.extent) - 1) for ax in scope_axes})

    zero_sigma = Sigma({ax.name: Literal(0, "int") for ax in candidates})
    origin = tuple(zero_sigma.apply(e).simplify(ctx) for e in load.index)

    var_to_dim: dict[str, int] = {}
    for ax in candidates:
        dims = [d for d, e in enumerate(load.index) if ax.name in e.free_vars()]
        if not dims:
            continue
        if len(dims) > 1:
            return None
        var_to_dim[ax.name] = dims[0]

    # No cache axis appears, or every bound axis appears (no fan-in) → no slab.
    if not var_to_dim or len(var_to_dim) == len(candidates):
        return None

    cache_axes = tuple(ax for ax in candidates if ax.name in var_to_dim)
    slab_dims = tuple(var_to_dim[ax.name] for ax in cache_axes)
    n_floats = 1
    for ax in cache_axes:
        n_floats *= int(ax.extent)
    if n_floats > _MAX_SLAB_FLOATS:
        return None

    needs_template = False
    for ax in cache_axes:
        d = var_to_dim[ax.name]
        sigma_one = Sigma({a.name: Literal(1 if a.name == ax.name else 0, "int") for a in candidates})
        actual = sorted(t.pretty() for t in _flatten_add(sigma_one.apply(load.index[d]).simplify(ctx)))
        expected = sorted(t.pretty() for t in _flatten_add((origin[d] + Literal(1, "int")).simplify(ctx)))
        if actual != expected:
            needs_template = True
            break
    template = tuple(load.index) if needs_template else None

    return _Slab(
        origin=origin,
        cache_axes=cache_axes,
        slab_dims=slab_dims,
        template=template,
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
