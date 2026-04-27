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

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

# Cache size budget per staged buffer (per CUDA block). 16 KB ≈ one
# 4096-fp32 row, comfortably fits with smem headroom for accumulators.
STAGE_BYTES_LIMIT = 16 * 1024
DTYPE_BYTES = 4


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_stage(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_stage(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if not tile.block_axes:
        return None  # not cooperative — no smem for staging
    if _any_stage(tile.body):
        return None  # already staged (anywhere in the body, incl. inside loops)

    new_tile = _stage_tile(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _stage_tile(tile: Tile) -> Tile | None:
    """Find each stageable buf, decide its placement + cache axes, insert
    Stages and rewrite Loads."""
    thread_axes_by_name = {ba.axis.name: ba.axis for ba in tile.axes if ba.bind == BIND_THREAD}
    # Tid decode is row-major over ``tile.thread_axes`` in declaration
    # order — last axis is innermost (varies fastest within a warp).
    # The smem layout pass uses this to put the warp-varying cache axis
    # innermost in the smem buffer, avoiding bank conflicts.
    thread_axis_order = tuple(ba.axis.name for ba in tile.axes if ba.bind == BIND_THREAD)

    # Per buf, gather every Load with its enclosing-loop path.
    loads_per_buf: dict[str, list[tuple[Load, tuple]]] = {}
    for load, path in _walk_loads(tile.body):
        loads_per_buf.setdefault(load.input, []).append((load, path))

    # Decide a Stage for each buf that has reuse.
    plans: list[tuple[str, int, tuple, Stage]] = []
    for buf, entries in loads_per_buf.items():
        plan = _stage_plan(buf, entries, thread_axes_by_name, thread_axis_order)
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


def _any_stage(stmts: tuple) -> bool:
    """Recursively detect a ``Stage`` stmt anywhere in the body. Strategies
    that emit their own Stages (e.g. sub-tiled ``003_block_matmul``)
    place them inside loops, so a top-level scan misses them and 004
    would re-stage the staged buffer's Loads."""
    for s in stmts:
        if isinstance(s, Stage):
            return True
        if isinstance(s, (Loop, StridedLoop, Cond)):
            if _any_stage(getattr(s, "body", ())):
                return True
            if _any_stage(getattr(s, "else_body", ())):
                return True
    return False


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
    thread_axis_order: tuple[str, ...] = (),
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
        # Build cache_axes — every matching axis at every index position
        # must be cached. Per-axis ``scale`` is the affine multiplier on
        # the axis across this buf's Loads (1 for non-sub-tile patterns;
        # TM/TN for the ``axis*F + literal`` patterns sub-tiling emits).
        # The cache axis extent grows by ``scale`` so the smem buffer
        # holds the full sub-tile span.
        raw_cache_entries: list[tuple[Axis, int, int]] = []  # (axis, cache_extent, slab_dim)
        seen: set[str] = set()
        per_dim_count: dict[int, int] = {}
        for dim, e in enumerate(ref_load.index):
            for ax_name in e.free_vars():
                if ax_name in cache_axes_names and ax_name not in seen:
                    seen.add(ax_name)
                    ax = thread_axes_by_name.get(ax_name) or below_axes_by_name[ax_name]
                    extent = _cache_extent_across_loads(entries, dim, ax_name, int(ax.extent))
                    if extent is None:
                        extent = int(ax.extent)
                    raw_cache_entries.append((ax, extent, dim))
                    per_dim_count[dim] = per_dim_count.get(dim, 0) + 1
        if not raw_cache_entries:
            continue
        if any(c > 1 for c in per_dim_count.values()):
            continue  # multi-cache-per-dim — deeper placement will resolve

        # Layout reorder: put the warp-varying cache axis innermost so
        # adjacent threads in a warp access stride-1 smem (no bank
        # conflicts). The warp-varying cache axis is the one whose
        # source-dim's Load expression contains the *innermost* THREAD
        # axis (last in tile.thread_axes — that's threadIdx.x's low
        # bits in the row-major tid decode).
        raw_cache_entries = _reorder_for_layout(raw_cache_entries, ref_load, thread_axis_order)

        cache_axes: list[Axis] = [Axis(name=ax.name, extent=extent) for ax, extent, _ in raw_cache_entries]
        slab_dims: list[int] = [dim for _, _, dim in raw_cache_entries]

        footprint = 1
        for ax in cache_axes:
            footprint *= int(ax.extent)
        if footprint * DTYPE_BYTES > STAGE_BYTES_LIMIT:
            continue

        # Build origin: per source dim, the block-uniform anchor =
        # ref_load.index[d] with cache-axis Vars zeroed AND per-load
        # varying literal offsets stripped (those become part of the
        # cache coord at each Load site).
        cache_axis_names = {ax.name for ax in cache_axes}
        origin = tuple(_canonical_origin(e, cache_axis_names) for e in ref_load.index)

        # Multi-origin guard: every Load of this buf must reduce to the
        # same canonical origin (block-uniform vars only). Loads that
        # differ in block-uniform parts need separate staging — skip
        # and let a deeper placement resolve it.
        origins_per_load = {tuple(_canonical_origin(e, cache_axis_names).pretty() for e in ld.index) for ld, _ in entries}
        if len(origins_per_load) > 1:
            continue

        # Set source_index_template only for non-affine loads (``/`` or
        # ``%`` from collapsed-reshape views). Affine cases use the old
        # additive ``origin + decoded`` path because the cache extent is
        # already F-scaled — substituting the iter coord into the
        # unsubstituted ``axis*F + lit`` would over-multiply by F.
        non_affine = any(_has_div_or_mod(e) for e in ref_load.index)
        template = tuple(ref_load.index) if non_affine else None
        # When non-affine, force per-axis scale to 1 — cache extent must
        # equal the raw axis extent so iter coord = cache-relative
        # source position directly.
        if non_affine:
            cache_axes = [
                Axis(name=ax.name, extent=int(orig_ax.extent))
                for ax, orig_ax in zip(
                    cache_axes, [thread_axes_by_name.get(a.name) or below_axes_by_name[a.name] for a in cache_axes], strict=True
                )
            ]
        stage = Stage(
            name=f"{buf}_stage",
            buf=buf,
            origin=origin,
            axes=tuple(cache_axes),
            slab_dims=tuple(slab_dims),
            source_index_template=template,
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
    """Smem coord for one cache axis at a Load site — the cache-relative
    address (``load_e - origin_at_dim``). Implemented by zeroing every
    block-uniform Var in the load expression, leaving only cache-local
    parts:

    - **Same naming** (matmul, single-output-per-thread: load uses the
      same axis Vars as stage): collapses to ``Var(stage_axis.name)``.
    - **Affine sub-tile** (``m_o*BM_BLOCK + m_i_tg*TM + Lit(j)``):
      collapses to ``m_i_tg*TM + Lit(j)`` — the cache-relative coord
      including the per-load literal offset.
    - **Aliasing** (softmax: load uses ``Var(a2)`` while cache axis is
      ``a1``): falls through to the load's own Var since the stage
      axis name isn't in the load's free vars."""
    load_e = load_index[slab_dim]
    free = load_e.free_vars()
    if stage_axis.name not in free:
        # Aliasing — preserve the existing softmax-style behavior.
        if isinstance(load_e, Var):
            return load_e
        origin_at_dim = stage.origin[slab_dim]
        non_cache = free - {stage_axis.name} - origin_at_dim.free_vars()
        sigma = Sigma({nc: Literal(0, "int") for nc in non_cache})
        return sigma.apply(load_e)
    # Non-affine layout (``/`` or ``%`` from collapsed-reshape views):
    # smem is keyed by the raw cache axis Var. The complex layout
    # transform lives in the cooperative-load's source fetch, not in
    # the consumer's smem read. Without this, subtracting block-uniform
    # vars leaves a layout-transformed expression that doesn't match
    # the cooperative-load's smem-write coord.
    if _has_div_or_mod(load_e):
        return Var(stage_axis.name)
    block_uniform = stage.origin[slab_dim].free_vars()
    if not block_uniform:
        return load_e
    sigma = Sigma({nc: Literal(0, "int") for nc in block_uniform})
    return _simplify(sigma.apply(load_e))


def _reorder_for_layout(
    raw_cache_entries: list[tuple[Axis, int, int]],
    ref_load: Load,
    thread_axis_order: tuple[str, ...],
) -> list[tuple[Axis, int, int]]:
    """Reorder ``cache_axes`` so the warp-varying axis is innermost.

    Within a warp threads vary in the *innermost* THREAD axis (last in
    ``thread_axis_order``). The cache axis whose source-dim Load
    expression contains that thread axis is the warp-varying one — its
    smem stride should be 1 (innermost) to avoid bank conflicts.

    For axes whose Load expressions don't contain the innermost thread
    axis (e.g. the M side of a matmul where threads share an A row),
    layout doesn't affect bank-conflict behavior, so we leave their
    relative order alone."""
    if not thread_axis_order or len(raw_cache_entries) < 2:
        return raw_cache_entries
    inner_thread = thread_axis_order[-1]
    # Score each cache entry: 1 if its Load expression contains the
    # innermost thread axis (warp-varying), 0 otherwise. Stable sort
    # ascending puts non-warp-varying first, warp-varying last.
    scored = [
        (0 if inner_thread not in ref_load.index[dim].free_vars() else 1, i, entry)
        for i, entry in enumerate(raw_cache_entries)
        for dim in [entry[2]]
    ]
    scored.sort(key=lambda t: (t[0], t[1]))
    return [entry for _, _, entry in scored]


def _has_div_or_mod(e: Expr) -> bool:
    """Recursively detect ``/`` or ``%`` in an Expr — signature of a
    layout-transform like a collapsed-reshape view."""
    if isinstance(e, BinaryExpr):
        if e.op in ("/", "%"):
            return True
        return _has_div_or_mod(e.left) or _has_div_or_mod(e.right)
    return False


def _cache_extent_across_loads(entries: list, slab_dim: int, axis_name: str, axis_extent: int) -> int | None:
    """Cache buffer size needed to hold every distinct value the load
    can produce when the axis ranges over [0, axis_extent) across all
    Loads of the buf at this slab dim.

    Computed as ``max_value - min_value + 1`` after zeroing every var
    other than ``axis_name`` (so block-uniform vars and other cache axes
    contribute 0). Generalizes both the contiguous sub-tile pattern
    (``axis*F + lit``, lit in [0, F)) and the interleaved pattern
    (``lit*BM_TG + axis``, lit in [0, TM)) — both yield ``BM_BLOCK``
    here without needing to detect which addressing form is in use.
    """
    max_at_top: list[int] = []
    min_at_bottom: list[int] = []
    for load, _ in entries:
        e = load.index[slab_dim]
        non_axis = e.free_vars() - {axis_name}
        zero_others = Sigma({n: Literal(0, "int") for n in non_axis})
        e_clean = zero_others.apply(e)
        e_top = _simplify(e_clean.substitute({axis_name: Literal(axis_extent - 1, "int")}))
        e_bot = _simplify(e_clean.substitute({axis_name: Literal(0, "int")}))
        if not (isinstance(e_top, Literal) and isinstance(e_bot, Literal)):
            return None
        max_at_top.append(int(e_top.value))
        min_at_bottom.append(int(e_bot.value))
    return max(max_at_top) - min(min_at_bottom) + 1


def _scale_across_loads(entries: list, slab_dim: int, axis_name: str) -> int | None:
    """Largest affine scale ``F`` consistent with every Load:
    ``e[slab_dim] = axis_name*F + (block_uniform + per_load_literal)``.
    Returns ``None`` if any Load doesn't fit the affine pattern, or if
    Loads disagree on the scale."""
    scales = set()
    for load, _ in entries:
        s = _affine_scale(load.index[slab_dim], axis_name)
        if s is None:
            return None
        scales.add(s)
    if len(scales) != 1:
        return None
    f = scales.pop()
    return f if f > 0 else None


def _affine_scale(e: Expr, axis_name: str) -> int | None:
    """Coefficient on ``axis_name`` in ``e`` if ``e`` is affine in it.

    AST walk: ``Var(axis_name)`` bare → 1; ``Var(axis_name) * Literal(F)``
    (or commuted) → F; recursively under ``+`` / ``-``. Returns ``None``
    if axis appears in a multiplication with a non-Literal or under an
    op we don't model. Returns 0 if axis doesn't appear at all."""
    if isinstance(e, Var):
        return 1 if e.name == axis_name else 0
    if isinstance(e, Literal):
        return 0
    if isinstance(e, BinaryExpr):
        if e.op == "*":
            for left, right in ((e.left, e.right), (e.right, e.left)):
                if isinstance(left, Var) and left.name == axis_name and isinstance(right, Literal):
                    return int(right.value)
            if axis_name in e.free_vars():
                return None
            return 0
        if e.op in ("+", "-"):
            sl = _affine_scale(e.left, axis_name)
            sr = _affine_scale(e.right, axis_name)
            if sl is None or sr is None:
                return None
            return sl + (sr if e.op == "+" else -sr)
    if axis_name in e.free_vars():
        return None
    return 0


def _canonical_origin(e: Expr, cache_axis_names: set[str]) -> Expr:
    """``e`` with every cache axis zeroed AND any trailing per-load
    literal offset stripped. The stripped piece becomes part of the
    cache coord at the Load site, not the block-uniform origin — so
    canonical origins compare equal across Loads that differ only by
    per-iteration literal offsets."""
    zero_sigma = Sigma({n: Literal(0, "int") for n in cache_axis_names})
    e_zeroed = _simplify(zero_sigma.apply(e))
    return _strip_trailing_literal(e_zeroed)


def _strip_trailing_literal(e: Expr) -> Expr:
    if isinstance(e, BinaryExpr) and e.op == "+":
        if isinstance(e.right, Literal):
            return e.left
        if isinstance(e.left, Literal):
            return e.right
    return e


def _simplify(e: Expr) -> Expr:
    return e.simplify(SimplifyCtx.empty())
