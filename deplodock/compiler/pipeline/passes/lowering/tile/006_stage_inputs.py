"""Stage frequently-reused external inputs into shared memory.

For each reduce ``Loop`` in the Tile body, examine its external Loads.
A Load is *stage-worthy* when its index does not reference at least one
``BIND_THREAD`` axis — meaning multiple threads (across that axis) read
the same element, so a per-block smem cache eliminates the redundant
DRAM traffic.

The pass walks scopes top-down. A "scope" is the Tile body or the body
of a free ``Loop`` nested inside it. For every reduce Loop sitting
directly in a scope, classify each Load:

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
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Load, Loop, Stmt, Tile, iter_body
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

    thread_names = {ba.axis.name for ba in tile.axes if ba.bind == BIND_THREAD}
    extents = {ba.axis.name: int(ba.axis.extent) for ba in tile.axes}
    used_names: set[str] = set()

    new_tile_body = _process_scope(tile.body, thread_names, extents, used_names)
    if new_tile_body == tile.body:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(
    body: tuple[Stmt, ...],
    thread_names: set[str],
    extents: dict[str, int],
    used_names: set[str],
) -> tuple[Stmt, ...]:
    walked: list[Stmt] = []
    for s in body:
        if isinstance(s, Loop) and not s.is_reduce:
            inner_extents = dict(extents)
            inner_extents[s.axis.name] = int(s.axis.extent)
            walked.append(Loop(axis=s.axis, body=_process_scope(s.body, thread_names, inner_extents, used_names)))
        else:
            walked.append(s)
    body = tuple(walked)

    stages: dict[tuple, Stage] = {}
    stage_order: list[tuple] = []
    load_rewrites: dict[tuple[int, str], tuple[str, tuple[Expr, ...]]] = {}

    for li, s in enumerate(body):
        if not (isinstance(s, Loop) and s.is_reduce):
            continue
        reduce_axis = s.axis.name
        scope_extents = dict(extents)
        scope_extents[reduce_axis] = int(s.axis.extent)
        for stmt in s.body:
            if not isinstance(stmt, Load):
                continue
            classified = _classify(stmt, thread_names, reduce_axis, scope_extents)
            if classified is None:
                continue
            origin, cache_axes, slab_dims = classified
            n_floats = 1
            for ax in cache_axes:
                n_floats *= int(ax.extent)
            if n_floats > _MAX_SLAB_FLOATS:
                continue
            origin_key = tuple(tuple(sorted(t.pretty() for t in _flatten_add(e))) for e in origin)
            cache_key = tuple((ax.name, int(ax.extent)) for ax in cache_axes)
            key = (stmt.input, origin_key, cache_key, slab_dims)
            if key not in stages:
                smem_name = _gen_name(stmt.input, used_names)
                stages[key] = Stage(name=smem_name, buf=stmt.input, origin=origin, axes=cache_axes, slab_dims=slab_dims)
                stage_order.append(key)
            stage = stages[key]
            new_index = tuple(Var(ax.name) for ax in cache_axes)
            load_rewrites[(li, stmt.name)] = (stage.name, new_index)

    if not stage_order:
        return body

    new_body: list[Stmt] = [stages[k] for k in stage_order]
    for li, s in enumerate(body):
        if isinstance(s, Loop) and s.is_reduce and any(k[0] == li for k in load_rewrites):
            inner: list[Stmt] = []
            for stmt in s.body:
                key = (li, stmt.name) if isinstance(stmt, Load) else None
                if key is not None and key in load_rewrites:
                    smem_name, new_index = load_rewrites[key]
                    inner.append(Load(name=stmt.name, input=smem_name, index=new_index))
                else:
                    inner.append(stmt)
            new_body.append(Loop(axis=s.axis, body=tuple(inner)))
        else:
            new_body.append(s)

    return tuple(new_body)


def _classify(
    load: Load,
    thread_names: set[str],
    reduce_axis: str,
    extents: dict[str, int],
) -> tuple[tuple[Expr, ...], tuple[Axis, ...], tuple[int, ...]] | None:
    idx = load.index
    cache_var_names: list[str] = []
    for e in idx:
        for v in e.free_vars():
            if (v in thread_names or v == reduce_axis) and v not in cache_var_names:
                cache_var_names.append(v)
    if not cache_var_names:
        return None
    thread_in_idx = set(cache_var_names) & thread_names
    if not (thread_names - thread_in_idx):
        return None  # every thread axis appears → no reuse

    var_to_dim: dict[str, int] = {}
    for v in cache_var_names:
        dims = [d for d, e in enumerate(idx) if v in e.free_vars()]
        if len(dims) != 1:
            return None
        var_to_dim[v] = dims[0]

    ctx = SimplifyCtx({n: Interval(0, ext - 1) for n, ext in extents.items()})
    cache_zero = Sigma({v: Literal(0, "int") for v in cache_var_names})
    origin = tuple(cache_zero.apply(e).simplify(ctx) for e in idx)

    for v in cache_var_names:
        d = var_to_dim[v]
        sigma = Sigma({other: Literal(0, "int") for other in cache_var_names if other != v})
        residue = sigma.apply(idx[d]).simplify(ctx)
        # cache part = residue - origin[d] should be exactly Var(v).
        # Build expected: origin[d] + Var(v), simplified. Compare via pretty.
        if _is_zero_lit(origin[d]):
            expected = Var(v)
        else:
            expected = (origin[d] + Var(v)).simplify(ctx)
        if not _add_terms_equal(residue, expected):
            return None

    if not all(v in extents for v in cache_var_names):
        return None
    cache_axes = tuple(Axis(name=v, extent=extents[v]) for v in cache_var_names)
    slab_dims = tuple(var_to_dim[v] for v in cache_var_names)
    return origin, cache_axes, slab_dims


def _is_zero_lit(e: Expr) -> bool:
    return isinstance(e, Literal) and e.value == 0


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
