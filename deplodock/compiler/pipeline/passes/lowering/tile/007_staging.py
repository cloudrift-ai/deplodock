"""Operand-staging — caches a buffer just outside the innermost loop
of a cooperative ``Tile`` body so a single cooperative load fills smem
once per outer iteration and serves every inner-loop step.

Algorithm:

- Find the innermost ``Loop`` / ``StridedLoop`` in each scope. The
  stage is inserted into the *parent scope* of that innermost loop,
  right before it (so any outer-loop axis it depends on is already
  bound).
- Cache axes = THREAD axes from ``Tile.axes`` plus the innermost
  loop's axis, restricted to those that actually appear in a Load's
  index. Outer-loop axes are treated as block-uniform and live in the
  stage's ``origin``.
- Per-axis extent is the max-min+1 across all Loads of the buf at that
  slab dim — handles unrolled cells with literal offsets.
- Footprint > ``STAGE_BYTES_LIMIT``: skip the buf, leave its Loads as
  direct global reads.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

STAGE_BYTES_LIMIT = 16 * 1024
DTYPE_BYTES = 4


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_stage(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_stage(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if not tile.block_axes:
        return None  # not cooperative — no smem for staging
    if _any_stage(tile.body):
        return None  # idempotence

    new_tile = _stage_tile(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _stage_tile(tile: Tile) -> Tile | None:
    thread_axes = {ba.axis.name: ba.axis for ba in tile.axes if ba.bind == BIND_THREAD}
    block_uniform = {ba.axis.name for ba in tile.axes if ba.bind != BIND_THREAD}
    new_body, changed = _stage_in_scope(tile.body, thread_axes, block_uniform)
    if not changed:
        return None
    return Tile(axes=tile.axes, body=new_body)


def _stage_in_scope(scope, thread_axes, outer_uniform):
    """Walk ``scope`` recursively, processing inner scopes first, then
    inserting stages at each scope where one or more 'innermost' loops
    live (loops whose body has no further Loop / StridedLoop)."""
    new_stmts: list[Stmt] = []
    changed = False
    for s in scope:
        if isinstance(s, Loop):
            inner_uniform = outer_uniform | {s.axis.name}
            new_inner, sub_changed = _stage_in_scope(s.body, thread_axes, inner_uniform)
            if sub_changed:
                new_stmts.append(Loop(axis=s.axis, body=new_inner))
                changed = True
            else:
                new_stmts.append(s)
        elif isinstance(s, StridedLoop):
            inner_uniform = outer_uniform | {s.axis.name}
            new_inner, sub_changed = _stage_in_scope(s.body, thread_axes, inner_uniform)
            if sub_changed:
                new_stmts.append(StridedLoop(axis=s.axis, start=s.start, step=s.step, body=new_inner))
                changed = True
            else:
                new_stmts.append(s)
        elif isinstance(s, Cond):
            b1, c1 = _stage_in_scope(s.body, thread_axes, outer_uniform)
            b2, c2 = _stage_in_scope(s.else_body, thread_axes, outer_uniform)
            if c1 or c2:
                new_stmts.append(Cond(cond=s.cond, body=b1, else_body=b2))
                changed = True
            else:
                new_stmts.append(s)
        else:
            new_stmts.append(s)

    inner_loops = [s for s in new_stmts if _is_innermost_loop(s)]
    if not inner_loops:
        return tuple(new_stmts), changed

    inner_axes = {lp.axis.name: lp.axis for lp in inner_loops}
    cacheable_axes = {**thread_axes, **inner_axes}

    loads_per_buf: dict[str, list[Load]] = {}
    for lp in inner_loops:
        for ld in _walk_loads(lp.body):
            loads_per_buf.setdefault(ld.input, []).append(ld)

    plans: list[Stage] = []
    for buf, loads in loads_per_buf.items():
        plan = _plan_stage(buf, loads, thread_axes, cacheable_axes, outer_uniform)
        if plan is not None:
            plans.append(plan)

    if not plans:
        return tuple(new_stmts), changed

    redirects = {p.buf: p for p in plans}
    rewritten = tuple(_rewrite_loads(s, redirects) for s in new_stmts)
    return tuple(plans) + rewritten, True


def _is_innermost_loop(s: Stmt) -> bool:
    if not isinstance(s, (Loop, StridedLoop)):
        return False
    return not any(isinstance(c, (Loop, StridedLoop)) for c in s.body)


def _plan_stage(buf, loads, thread_axes, cacheable_axes, outer_uniform):
    ref = loads[0]
    free_in_index: set[str] = set()
    for e in ref.index:
        free_in_index |= e.free_vars()

    thread_missing = thread_axes.keys() - free_in_index
    if len(loads) < 2 and not thread_missing:
        return None

    cache_entries: list[tuple[Axis, int, int]] = []
    seen: set[str] = set()
    for dim, e in enumerate(ref.index):
        free = e.free_vars() & cacheable_axes.keys()
        if not free:
            continue
        if len(free) > 1:
            return None
        ax_name = next(iter(free))
        if ax_name in seen:
            continue
        seen.add(ax_name)
        ax = cacheable_axes[ax_name]
        extent = _cache_extent_across_loads(loads, dim, ax_name, int(ax.extent))
        if extent is None:
            return None
        cache_entries.append((Axis(name=ax.name, extent=extent), extent, dim))

    if not cache_entries:
        return None

    footprint = 1
    for _, extent, _d in cache_entries:
        footprint *= extent
    if footprint * DTYPE_BYTES > STAGE_BYTES_LIMIT:
        return None

    cache_axes = tuple(ax for ax, _, _ in cache_entries)
    slab_dims = tuple(d for _, _, d in cache_entries)
    cache_names = {ax.name for ax in cache_axes}

    origin = tuple(_canonical_origin(e, cache_names) for e in ref.index)

    ref_origin_pretty = tuple(e.pretty() for e in origin)
    for ld in loads[1:]:
        ld_origin_pretty = tuple(_canonical_origin(e, cache_names).pretty() for e in ld.index)
        if ld_origin_pretty != ref_origin_pretty:
            return None

    allowed = cache_names | outer_uniform
    for ld in loads:
        for e in ld.index:
            if e.free_vars() - allowed:
                return None

    return Stage(name=f"{buf}_stage", buf=buf, origin=origin, axes=cache_axes, slab_dims=slab_dims)


def _walk_loads(stmts) -> Iterable[Load]:
    for s in stmts:
        if isinstance(s, Load):
            yield s
        elif isinstance(s, (Loop, StridedLoop)):
            yield from _walk_loads(s.body)
        elif isinstance(s, Cond):
            yield from _walk_loads(s.body)
            yield from _walk_loads(s.else_body)


def _any_stage(stmts) -> bool:
    for s in stmts:
        if isinstance(s, Stage):
            return True
        if isinstance(s, (Loop, StridedLoop, Cond)):
            if _any_stage(getattr(s, "body", ())):
                return True
            if _any_stage(getattr(s, "else_body", ())):
                return True
    return False


def _rewrite_loads(stmt: Stmt, redirects: dict[str, Stage]) -> Stmt:
    if isinstance(stmt, Load) and stmt.input in redirects:
        stage = redirects[stmt.input]
        smem_index = tuple(_smem_coord(ax, dim, stage, stmt.index) for ax, dim in zip(stage.axes, stage.slab_dims, strict=True))
        return Load(name=stmt.name, input=stage.name, index=smem_index)
    if isinstance(stmt, Loop):
        return Loop(axis=stmt.axis, body=tuple(_rewrite_loads(c, redirects) for c in stmt.body))
    if isinstance(stmt, StridedLoop):
        return StridedLoop(
            axis=stmt.axis,
            start=stmt.start,
            step=stmt.step,
            body=tuple(_rewrite_loads(c, redirects) for c in stmt.body),
        )
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_rewrite_loads(c, redirects) for c in stmt.body),
            else_body=tuple(_rewrite_loads(c, redirects) for c in stmt.else_body),
        )
    return stmt


def _smem_coord(stage_axis: Axis, slab_dim: int, stage: Stage, load_index: tuple) -> Expr:
    e = load_index[slab_dim]
    block_uniform = stage.origin[slab_dim].free_vars()
    if not block_uniform:
        return e
    sigma = Sigma({nc: Literal(0, "int") for nc in block_uniform})
    return _simplify(sigma.apply(e))


def _canonical_origin(e: Expr, cache_axis_names: set[str]) -> Expr:
    sigma = Sigma({n: Literal(0, "int") for n in cache_axis_names})
    e_zeroed = _simplify(sigma.apply(e))
    if isinstance(e_zeroed, BinaryExpr) and e_zeroed.op == "+":
        if isinstance(e_zeroed.right, Literal):
            return e_zeroed.left
        if isinstance(e_zeroed.left, Literal):
            return e_zeroed.right
    return e_zeroed


def _cache_extent_across_loads(loads, slab_dim: int, axis_name: str, axis_extent: int) -> int | None:
    """max(load_at_axis_extent-1) - min(load_at_axis_0) + 1 — covers
    contiguous (axis*F + lit) and interleaved (lit*F + axis) patterns."""
    max_top: list[int] = []
    min_bot: list[int] = []
    for ld in loads:
        e = ld.index[slab_dim]
        non_axis = e.free_vars() - {axis_name}
        zero_others = Sigma({n: Literal(0, "int") for n in non_axis})
        e_clean = zero_others.apply(e)
        e_top = _simplify(e_clean.substitute({axis_name: Literal(axis_extent - 1, "int")}))
        e_bot = _simplify(e_clean.substitute({axis_name: Literal(0, "int")}))
        if not (isinstance(e_top, Literal) and isinstance(e_bot, Literal)):
            return None
        max_top.append(int(e_top.value))
        min_bot.append(int(e_bot.value))
    return max(max_top) - min(min_bot) + 1


def _simplify(e: Expr) -> Expr:
    return e.simplify(SimplifyCtx.empty())
