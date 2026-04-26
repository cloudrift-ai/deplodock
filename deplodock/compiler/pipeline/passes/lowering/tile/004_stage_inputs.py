"""Operand-staging strategy — caches input buffers that get fetched
redundantly across threads / iterations of a cooperative ``Tile`` body.

Two reuse patterns trigger staging, both reducing to "this Load's
value is fetched many times for the same address":

- **Spatial reuse** (matmul): a single Load whose index doesn't depend
  on every THREAD axis from ``Tile.axes``. Threads that share the
  missing axis fetch the same global address.
- **Temporal reuse** (softmax / RMSNorm): the same buffer loaded by
  multiple loops in the body, with the same address pattern modulo
  the iterating axis.

Cache axes are derived from the Load index by intersecting its free
vars with the available "parallel" axes (THREAD axes from ``Tile.axes``
plus axes of any enclosing loop *below* the chosen Stage placement).

Stage placement: ``OUTERMOST`` scope (Tile body head) by default; if
the resulting smem footprint exceeds ``STAGE_BYTES_LIMIT``, descend
into the loop chain common to all Loads of the buffer. For matmul,
this descends into the K_o body so smem holds one K-chunk's worth of
tile, refilled per K_o iteration.

Body Loads of the staged buffer are rewritten in place to target the
staged name with cache-local indices.
"""

from __future__ import annotations

from collections.abc import Iterable

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

# Cache size budget per staged buffer (per CUDA block). 16 KB ≈ one
# 4096-fp32 row, comfortably fits with smem headroom for accumulators.
STAGE_BYTES_LIMIT = 16 * 1024
DTYPE_BYTES = 4


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    tile_op: TileOp = node.op

    new_body = _maybe_stage(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_stage(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if not tile.block_axes:
        return None  # not cooperative — no smem for staging
    if any(isinstance(s, Stage) for s in tile.body):
        return None  # already staged

    new_tile = _stage_tile(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _stage_tile(tile: Tile) -> Tile | None:
    """Find each stageable buf, decide its placement + cache axes, insert
    Stages and rewrite Loads."""
    thread_axes_by_name = {ba.axis.name: ba.axis for ba in tile.axes if ba.bind == BIND_THREAD}

    # Per buf, gather every Load with its enclosing-loop path.
    loads_per_buf: dict[str, list[tuple[Load, tuple]]] = {}
    for load, path in _walk_loads(tile.body):
        loads_per_buf.setdefault(load.input, []).append((load, path))

    # Decide a Stage for each buf that has reuse.
    plans: list[tuple[str, int, tuple, Stage]] = []
    for buf, entries in loads_per_buf.items():
        plan = _stage_plan(buf, entries, thread_axes_by_name)
        if plan is not None:
            plans.append(plan)

    if not plans:
        return None

    # Apply each plan to the body. Stages are inserted at their depths
    # (Tile head for d=0, descending into common loops for d>0). Loads
    # of staged bufs in scope are rewritten to cache-local form.
    new_body = tile.body
    for buf, depth, common_loops, stage in plans:
        new_body = _apply_plan(new_body, buf, depth, common_loops, stage)

    return Tile(axes=tile.axes, body=new_body)


def _walk_loads(stmts: tuple, path: tuple = ()) -> Iterable[tuple[Load, tuple]]:
    """Yield ``(Load, enclosing_loop_path)`` for every Load in the body.
    The path is a tuple of Loop / StridedLoop from outermost to immediate parent."""
    for s in stmts:
        if isinstance(s, Load):
            yield (s, path)
        elif isinstance(s, (Loop, StridedLoop)):
            yield from _walk_loads(s.body, path + (s,))
        elif isinstance(s, Cond):
            yield from _walk_loads(s.body, path)
            yield from _walk_loads(s.else_body, path)


def _stage_plan(
    buf: str,
    entries: list[tuple[Load, tuple]],
    thread_axes_by_name: dict[str, Axis],
) -> tuple[str, int, tuple, Stage] | None:
    """Compute a Stage plan for ``buf``. Returns ``(buf, depth,
    common_loops, stage)`` or None if not stageable.

    ``depth = 0`` → place Stage at Tile body head (most reuse, biggest
    footprint). Increasing ``depth`` descends into the common loop
    chain — fewer cache axes, smaller footprint, more refills."""
    paths = [path for _, path in entries]
    common = _common_loop_prefix(paths)

    ref_load = entries[0][0]
    thread_in_index = thread_axes_by_name.keys() & _index_free_vars(ref_load.index)

    # Try placement from outermost (d=0) to deepest common (d=len(common)).
    # Pick the first depth where smem footprint fits.
    for d in range(len(common) + 1):
        # Loops below the Stage = paths[i][d:] for each load.
        below_axes_by_name: dict[str, Axis] = {}
        for _, path in entries:
            for loop in path[d:]:
                below_axes_by_name[loop.axis.name] = loop.axis

        cache_axes_names = (thread_axes_by_name.keys() | below_axes_by_name.keys()) & _index_free_vars(ref_load.index)
        if not cache_axes_names:
            continue
        # Reuse signal:
        # - **Temporal**: ≥2 distinct Loads of the same buf (softmax / RMSNorm).
        # - **Spatial**: a Load with all enclosing-loop axes serial AND a
        #   THREAD axis absent from the index. Threads with distinct values
        #   of the missing THREAD axis access the same global address.
        #   A StridedLoop in the path means iteration is already
        #   thread-driven; a Load inside one doesn't statically share
        #   across threads (each thread visits distinct values of the
        #   strided axis), so we don't claim spatial reuse there.
        if len(entries) < 2:
            has_strided = any(isinstance(loop, StridedLoop) for path in paths for loop in path)
            thread_missing = thread_axes_by_name.keys() - thread_in_index
            if has_strided or not thread_missing:
                continue

        # Build cache_axes — every matching axis at every index position
        # must be cached. Order by appearance in the index. ``slab_dims``
        # records the source-buffer dim each axis lives in.
        # Slab form requires at most one cache axis per source dim (so
        # the source index reconstructs as ``origin[d] + decoded[d]``).
        # If any dim has >1 cache axes (e.g. matmul d=0 where a2_o and
        # a2_i both sit at K), skip — deeper placements have fewer
        # below-axes and resolve the conflict.
        cache_axes: list[Axis] = []
        slab_dims: list[int] = []
        seen: set[str] = set()
        per_dim_count: dict[int, int] = {}
        for dim, e in enumerate(ref_load.index):
            for ax_name in e.free_vars():
                if ax_name in cache_axes_names and ax_name not in seen:
                    seen.add(ax_name)
                    ax = thread_axes_by_name.get(ax_name) or below_axes_by_name[ax_name]
                    cache_axes.append(ax)
                    slab_dims.append(dim)
                    per_dim_count[dim] = per_dim_count.get(dim, 0) + 1
        if not cache_axes:
            continue
        if any(c > 1 for c in per_dim_count.values()):
            continue  # multi-cache-per-dim — deeper placement will resolve

        footprint = 1
        for ax in cache_axes:
            footprint *= int(ax.extent)
        if footprint * DTYPE_BYTES > STAGE_BYTES_LIMIT:
            continue

        # Build origin: per source dim, the block-uniform anchor =
        # ref_load.index[d] with all cache-axis Vars substituted to 0.
        origin_sigma = Sigma({ax.name: Literal(0, "int") for ax in cache_axes})
        origin = tuple(origin_sigma.apply(e) for e in ref_load.index)

        stage = Stage(
            name=f"{buf}_stage",
            buf=buf,
            origin=origin,
            axes=tuple(cache_axes),
            slab_dims=tuple(slab_dims),
        )
        return (buf, d, common[:d], stage)

    return None


def _common_loop_prefix(paths: list[tuple]) -> tuple:
    """Longest common prefix of loop-paths (compared by axis identity)."""
    if not paths:
        return ()
    common: list = []
    for level in zip(*paths, strict=False):
        first = level[0]
        if all(loop is first for loop in level):
            common.append(first)
        else:
            break
    return tuple(common)


def _index_free_vars(index: tuple) -> set[str]:
    out: set[str] = set()
    for e in index:
        out |= e.free_vars()
    return out


def _apply_plan(body: tuple, buf: str, depth: int, common_loops: tuple, stage: Stage) -> tuple:
    """Insert ``stage`` at ``depth`` (descending through ``common_loops``)
    and rewrite Loads of ``buf`` in scope to target ``stage.name``.

    Loops are matched by axis name (not identity) so multiple plans
    applied in sequence can each find the right loop even after the
    body has been mutated by an earlier plan."""
    if depth == 0:
        rewritten = tuple(_rewrite_loads(s, {buf: stage}) for s in body)
        return (stage,) + rewritten
    target_name = common_loops[0].axis.name
    new_body = []
    for s in body:
        if isinstance(s, (Loop, StridedLoop)) and s.axis.name == target_name:
            new_inner = _apply_plan(s.body, buf, depth - 1, common_loops[1:], stage)
            new_body.append(_clone_loop(s, new_inner))
        else:
            new_body.append(s)
    return tuple(new_body)


def _clone_loop(loop, body: tuple):
    if isinstance(loop, StridedLoop):
        return StridedLoop(axis=loop.axis, start=loop.start, step=loop.step, body=body)
    return Loop(axis=loop.axis, body=body)


def _rewrite_loads(stmt: Stmt, redirects: dict[str, Stage]) -> Stmt:
    """Recursively rewrite ``Load(buf, ...)`` to ``Load(stage.name,
    smem_index)`` for every staged buf. ``smem_index`` has one entry per
    cache axis (in stage.axes order); each entry is the load-site coord
    for that slab dim."""
    if isinstance(stmt, Load) and stmt.input in redirects:
        stage = redirects[stmt.input]
        smem_index = tuple(_smem_coord_for_axis(ax, dim, stage, stmt.index) for ax, dim in zip(stage.axes, stage.slab_dims, strict=True))
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


def _smem_coord_for_axis(stage_axis: Axis, slab_dim: int, stage: Stage, load_index: tuple):
    """Smem coord for one cache axis at a Load site. The slab axis lives
    in source dim ``slab_dim``; extract its value from ``load_index[slab_dim]``:

    - **Same naming** (matmul: load uses the same axis Vars as stage):
      ``Var(stage_axis.name)`` — already in scope.
    - **Aliasing** (softmax: load's Var at this dim is differently named
      e.g. ``a2`` for the cache-axis ``a1``): use the load's Var.
    - **Affine** (load index is ``outer*F + cache``): strip the
      block-uniform origin to leave the cache-local coord."""
    load_e = load_index[slab_dim]
    free = load_e.free_vars()
    if stage_axis.name in free:
        return Var(stage_axis.name)
    if isinstance(load_e, Var):
        return load_e
    origin_at_dim = stage.origin[slab_dim]
    non_cache = free - {stage_axis.name} - origin_at_dim.free_vars()
    sigma = Sigma({nc: Literal(0, "int") for nc in non_cache})
    return sigma.apply(load_e)
