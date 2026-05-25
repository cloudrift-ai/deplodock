"""Stage frequently-reused external inputs into shared memory (wrap-body).

Emits ``Stage(sources=[...], body=<consumer>)`` — the Stage *wraps* the
consumer subtree containing the rewritten Loads. Producer cooperative
``Load+Write`` is synthesized at materialize time from ``Source``
entries (cache axes, origin, source-dim mapping); no producer body is
stored on the Stage.

Runs *before* ``006a_register_tile_planned`` (pre-register-tile: there
is exactly one Load per ``(buffer, access-pattern)`` rather than F×F
duplicates). REGISTER axes in scope join cache axes via the
``register_axes`` channel — the slab spans ``BM·FM × BK`` (and similar)
with stride-1 Affine addressing, instead of ``BM × BK`` with
TemplateAddressing as it would post-006a. Stages emit wrapping the K_o
body; their cache-axis iteration vars shadow the outer REGISTER Loops,
and ``006a_register_tile_planned`` treats Stages as opaque (no
recursion into their producer-side state — only the consumer ``body``
descends).

**Pipeline:**

1. **Walk the scope**, collecting every Load from every reduce
   ``Loop`` / ``StridedLoop``, tagged with its reduce axis.
2. **Group by source buffer.** Sibling reduces contribute their Loads
   to the same buffer's bucket; per-axis-name differences in the
   reduce axis are normalized away by the slab signature.
3. **Per buffer, fit one slab.** Classify every Load. If they all agree
   on slab geometry, build one Source; if they disagree, bail.
4. **Admit & emit.** Greedily admit Sources until the per-scope smem
   budget is hit; emit a single Stage wrapping the consumer stmts;
   rewrite admitted Loads to read from staged smem.

**Slab geometry from a Load's index** (computed in ``_classify``):

- ``origin`` — the index with every cache var (thread + reduce axis)
  substituted to 0. The per-CTA anchor.
- For each cache var, find which source dim its Var appears in. Zero
  dims = fan-in axis; one dim = cache axis at that dim; multiple dims =
  bail (collapsed-reshape that can't be additively split).
- Coefficient-1 check: substitute the var → 1 (others → 0) and compare
  to ``origin[d] + 1``. If any axis fails, fall back to
  ``template_index`` carrying the original Load index.

**Reuse.** A Load qualifies for staging iff at least one bound axis
doesn't appear in its index (fan-in). When every bound axis appears,
no fan-in, skip — unless ``allow_no_fan_in`` is set (≥ 2 sibling Loads
sharing the same index ⇒ temporal reuse).
"""

from __future__ import annotations

import os
from typing import NamedTuple

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    CacheDim,
    GridTile,
    RegisterTile,
    SerialTile,
    Source,
    Stage,
    StridedTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    replace_thread_tile_body,
    single_tile,
    thread_tile_of,
)

PATTERN = [Pattern("root", TileOp)]

STAGE = Knob("STAGE", KnobType.BINMASK, help="Bitmask over ranked candidate buffers (char i = buffer i)")


def _tile_is_cooperative(tt: ThreadTile) -> bool:
    """True iff some Accum inside ``tt`` reduces over one of ``tt``'s
    thread axes — i.e. the kernel is cooperative-K. Replaces the legacy
    ``bool(tt.cooperative_axes)`` check now that cooperativity is derived
    from ``Accum.axes`` (see ``ir/tile/escape_analysis.py``)."""
    tt_axis_names = frozenset(ax.name for ax in tt.axes)
    for s in tt.body.iter():
        if isinstance(s, Accum) and tt_axis_names & frozenset(s.axes):
            return True
    return False


class _Slab(NamedTuple):
    """Slab geometry derived from one Load's index."""

    origin: tuple[Expr, ...]
    cache_axes: tuple[Axis, ...]
    slab_dims: tuple[int, ...]
    template: tuple[Expr, ...] | None
    n_bytes: int


def rewrite(ctx: Context, root: Node) -> list[TileOp] | None:
    """Emit one TileOp option per subset of stage-able input buffers,
    ordered most-staged first. Option-0 stages every qualifying buffer
    (best perf when smem fits). Subsequent options progressively drop
    buffers. The final option stages nothing.

    Idempotence is gated on the ``STAGE`` knob (stamped on every
    emitted variant) rather than on body structure, so the no-staging
    variant still has a distinct ``op_cache_key``.
    """
    if STAGE.name in root.op.knobs:
        raise RuleSkipped("stage already applied (idempotence via knob)")
    budget = ctx.max_dynamic_smem
    variants = _enumerate_variants(root.op.body, slab_cap=budget, scope_budget=budget, parent_op=root.op, warp_size=ctx.warp_size)
    if not variants:
        raise RuleSkipped("no Load qualifies for staging")
    return variants


def _forced_stage_mask(n: int) -> int | None:
    raw = os.environ.get(STAGE.env)
    if raw is None or raw == "":
        return None
    return STAGE.parse(raw, width=n)


def _enumerate_variants(body: Body, *, slab_cap: int, scope_budget: int, parent_op: TileOp, warp_size: int) -> list[TileOp]:
    candidates = _candidate_buffers(body, warp_size=warp_size)
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda kv: -kv[1])
    bufs_ranked = [b for b, _ in ranked]
    n = len(bufs_ranked)
    forced = _forced_stage_mask(n)
    if forced is not None:
        masks = [forced]
    else:
        masks = sorted(range(1 << n), key=lambda m: (-bin(m).count("1"), m))
    variants: list[TileOp] = []
    for mask in masks:
        allow = frozenset(b for i, b in enumerate(bufs_ranked) if mask & (1 << i))
        new_body = _maybe_rewrite(body, slab_cap=slab_cap, scope_budget=scope_budget, allowed_bufs=allow, warp_size=warp_size)
        if new_body is None:
            continue
        knobs = {**parent_op.knobs, STAGE.name: STAGE.pretty(mask, width=n)}
        variants.append(TileOp(body=new_body, name=parent_op.name, knobs=knobs))
    return variants


def _candidate_buffers(body: Body, *, warp_size: int) -> list[tuple[str, int]]:
    idx, outer = single_tile(body)
    tt = thread_tile_of(outer)
    if any(isinstance(s, Stage) for s in tt.body.iter()):
        return []
    if not tt.axes:
        return []
    n_thread = 1
    for ax in tt.axes:
        n_thread *= ax.extent.as_static()
    if n_thread <= warp_size:
        return []
    block_axes = outer.axes if isinstance(outer, GridTile) else ()
    block_axis_names = frozenset(ax.name for ax in block_axes)
    is_cooperative = _tile_is_cooperative(tt)
    all_axes = tuple(block_axes) + tuple(tt.axes)
    found: dict[str, int] = {}
    _collect_candidates(tt, tt.axes, all_axes, block_axis_names, found, slab_cap=10**12, is_cooperative=is_cooperative)
    return list(found.items())


def _collect_candidates(
    scope,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    found: dict[str, int],
    *,
    slab_cap: int,
    is_cooperative: bool = False,
    register_axes: tuple[Axis, ...] = (),
) -> None:
    """Preflight buffer-walk: record candidate buffers' slab sizes."""
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]] = {}
    for s in scope.body:
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "serial_outer":
            inner_stage = _peel_to_stage_inner(s)
            if inner_stage is not None:
                scope_axes = (*in_scope_axes, s.axis, inner_stage.axis)
                extra = (s.axis,) + register_axes
                for stmt in inner_stage.body:
                    if isinstance(stmt, Load):
                        loads_by_buf.setdefault(stmt.input, []).append((stmt, inner_stage.axis, scope_axes, extra))
                continue
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "stage_inner":
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
            continue
        if isinstance(s, RegisterTile):
            new_register_axes = register_axes + tuple(s.axes)
            _collect_candidates(
                s,
                thread_axes,
                tuple(in_scope_axes) + tuple(s.axes),
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=new_register_axes,
            )
            continue
        if isinstance(s, SerialTile) and not s.is_reduce:
            _collect_candidates(
                s,
                thread_axes,
                (*in_scope_axes, s.axis),
                block_axis_names,
                found,
                slab_cap=slab_cap,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
            )
            continue
        if isinstance(s, (SerialTile, StridedTile)):
            scope_axes = (*in_scope_axes, s.axis)
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
    for buf, items in loads_by_buf.items():
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _, _ in items):
            continue
        partitions_count: dict[tuple, int] = {}
        for load, _, _, _ in items:
            key = tuple(e.pretty() for e in load.index)
            partitions_count[key] = partitions_count.get(key, 0) + 1
        for load, reduce_axis, scope_axes, extra in items:
            key = tuple(e.pretty() for e in load.index)
            allow_no_fan_in = partitions_count[key] >= 2
            slab = _classify(
                load,
                thread_axes,
                reduce_axis,
                scope_axes,
                slab_cap=slab_cap,
                extra_candidates=extra,
                allow_no_fan_in=allow_no_fan_in,
            )
            if slab is None:
                continue
            found[buf] = found.get(buf, 0) + slab.n_bytes
            break


def _maybe_rewrite(
    body: Body, *, slab_cap: int, scope_budget: int, allowed_bufs: frozenset[str] | None = None, warp_size: int
) -> Body | None:
    idx, outer = single_tile(body)
    tt = thread_tile_of(outer)
    if not tt.axes:
        if allowed_bufs is None:
            raise RuleSkipped("ThreadTile has no axes — no reuse to stage")
        return None

    n_thread = 1
    for ax in tt.axes:
        n_thread *= ax.extent.as_static()
    if n_thread <= warp_size:
        if allowed_bufs is None:
            raise RuleSkipped(f"warp-only cooperative tile (n_threads={n_thread} ≤ {warp_size}); register-resident, no smem stage")
        return None

    used_names: set[str] = set()
    block_axes = outer.axes if isinstance(outer, GridTile) else ()
    block_axis_names = frozenset(ax.name for ax in block_axes)
    is_cooperative = _tile_is_cooperative(tt)
    all_axes = tuple(block_axes) + tuple(tt.axes)
    new_tile_body = _process_scope(
        tt,
        tt.axes,
        all_axes,
        block_axis_names,
        used_names,
        slab_cap=slab_cap,
        scope_budget=scope_budget,
        allowed_bufs=allowed_bufs,
        is_cooperative=is_cooperative,
    )
    if new_tile_body == tt.body:
        if allowed_bufs is None:
            raise RuleSkipped("no Load qualifies for staging")
        return body
    rebuilt = replace_thread_tile_body(outer, new_tile_body)
    return body[:idx] + (rebuilt,) + body[idx + 1 :]


def _peel_to_stage_inner(outer):
    """Descend single-stmt SerialTile chains until a stage_inner SerialTile
    is found. Returns ``None`` if the chain branches before reaching one.
    """
    cur = tuple(outer.body)
    while len(cur) == 1 and isinstance(cur[0], (SerialTile, StridedTile, RegisterTile)):
        s = cur[0]
        if isinstance(s, SerialTile) and s.kind == "stage_inner":
            return s
        cur = tuple(s.body)
    return None


def _process_scope(
    scope,
    thread_axes: tuple[Axis, ...],
    in_scope_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
    allowed_bufs: frozenset[str] | None = None,
    is_cooperative: bool = False,
    register_axes: tuple[Axis, ...] = (),
) -> Body:
    """Walk scope.body; recurse into non-reduce free tiles; collect Loads
    from reduce tiles into per-buffer buckets. Per buffer, build a Source
    if all Loads agree on slab geometry. Admit Sources under budget;
    emit one Stage wrapping the contiguous range of consumer stmts that
    contain rewritten Loads. Source name + index rewrites are applied
    inside the Stage's body."""
    rewritten_inner: list[Stmt] = []
    # For each top-level stmt, the list of (load, reduce_axis, scope_axes, extra_cache_axes) bucketed by buf.
    # We keep a per-stmt map (by index in rewritten_inner) so we know which top-level stmts must end up
    # inside the wrapping Stage's body.
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]] = {}
    # Track which indices in rewritten_inner contain Loads that may need rewriting.
    stmt_contains_loads_idx: list[int] = []

    for s in scope.body:
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "serial_outer":
            inner_stage = _peel_to_stage_inner(s)
            if inner_stage is not None:
                scope_axes = (*in_scope_axes, s.axis, inner_stage.axis)
                extra = (s.axis,) + register_axes
                for stmt in inner_stage.body:
                    if isinstance(stmt, Load):
                        loads_by_buf.setdefault(stmt.input, []).append((stmt, inner_stage.axis, scope_axes, extra))
                rewritten_inner.append(s)
                stmt_contains_loads_idx.append(len(rewritten_inner) - 1)
                continue
        if is_cooperative and isinstance(s, SerialTile) and not s.is_reduce and s.kind == "stage_inner":
            pass  # fall through to the collection branch
        elif isinstance(s, RegisterTile):
            new_register_axes = register_axes + tuple(s.axes)
            new_body = _process_scope(
                s,
                thread_axes,
                tuple(in_scope_axes) + tuple(s.axes),
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=new_register_axes,
            )
            rewritten_inner.append(s.with_bodies((new_body,)))
            continue
        elif isinstance(s, SerialTile) and not s.is_reduce:
            new_body = _process_scope(
                s,
                thread_axes,
                (*in_scope_axes, s.axis),
                block_axis_names,
                used_names,
                slab_cap=slab_cap,
                scope_budget=scope_budget,
                allowed_bufs=allowed_bufs,
                is_cooperative=is_cooperative,
                register_axes=register_axes,
            )
            rewritten_inner.append(s.with_bodies((new_body,)))
            continue
        if isinstance(s, (SerialTile, StridedTile)):
            scope_axes = (*in_scope_axes, s.axis)
            had_load = False
            for stmt in s.body:
                if isinstance(stmt, Load):
                    loads_by_buf.setdefault(stmt.input, []).append((stmt, s.axis, scope_axes, register_axes))
                    had_load = True
            rewritten_inner.append(s)
            if had_load:
                stmt_contains_loads_idx.append(len(rewritten_inner) - 1)
            continue
        rewritten_inner.append(s)

    if allowed_bufs is not None:
        loads_by_buf = {b: items for b, items in loads_by_buf.items() if b in allowed_bufs}
    sources, name_rewrites = _build_sources(
        loads_by_buf, thread_axes, block_axis_names, used_names, slab_cap=slab_cap, scope_budget=scope_budget
    )
    if not sources:
        return tuple(rewritten_inner)

    # Rewrite Loads inside the affected stmts to read from staged smem.
    rewritten = list(rewritten_inner)
    for i in stmt_contains_loads_idx:
        rewritten[i] = _rewrite_loads(rewritten[i], name_rewrites)

    # Wrap the contiguous range of stmt_contains_loads_idx in a single Stage.
    # If the affected stmts are not contiguous, wrap from the first to the last
    # (interleaved sibling stmts come along for the ride — they're typically
    # init/setup that's fine to live inside the staged scope).
    if not stmt_contains_loads_idx:
        return tuple(rewritten)
    lo, hi = stmt_contains_loads_idx[0], stmt_contains_loads_idx[-1]
    wrapped = Stage(sources=tuple(sources), body=Body(tuple(rewritten[lo : hi + 1])))
    return tuple([*rewritten[:lo], wrapped, *rewritten[hi + 1 :]])


def _build_sources(
    loads_by_buf: dict[str, list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...]]]],
    thread_axes: tuple[Axis, ...],
    block_axis_names: frozenset[str],
    used_names: set[str],
    *,
    slab_cap: int,
    scope_budget: int,
) -> tuple[list[Source], dict[str, tuple[str, tuple[Expr, ...]]]]:
    """Per buffer, partition Loads by index equality; per partition, derive
    slab geometry; admit Sources under budget. Returns (sources,
    name_rewrites) where name_rewrites maps original Load SSA names to
    (smem_buf_name, smem_index)."""
    sources: list[Source] = []
    name_rewrites: dict[str, tuple[str, tuple[Expr, ...]]] = {}
    used_bytes = 0

    for buf, items in loads_by_buf.items():
        if block_axis_names and all(_load_free_vars(load).isdisjoint(block_axis_names) for load, _, _, _ in items):
            continue

        partitions: list[tuple[Load, Axis, tuple[Axis, ...], tuple[Axis, ...], list[Load]]] = []
        for load, reduce_axis, scope_axes, extra_cache_axes in items:
            for rep_load, _, _, _, members in partitions:
                if load.index == rep_load.index:
                    members.append(load)
                    break
            else:
                partitions.append((load, reduce_axis, scope_axes, extra_cache_axes, [load]))

        for rep_load, rep_reduce, rep_scope, rep_extra, members in partitions:
            allow_no_fan_in = len(members) >= 2
            slab = _classify(
                rep_load,
                thread_axes,
                rep_reduce,
                rep_scope,
                slab_cap=slab_cap,
                extra_candidates=rep_extra,
                allow_no_fan_in=allow_no_fan_in,
            )
            if slab is None:
                continue
            if used_bytes + slab.n_bytes > scope_budget:
                continue
            smem_name = _gen_name(buf, used_names)
            cache_dims = tuple(CacheDim(axis=ax, source_dim=d) for ax, d in zip(slab.cache_axes, slab.slab_dims, strict=True))
            src = Source(
                name=smem_name,
                buf=buf,
                cache_dims=cache_dims,
                origin=slab.origin,
                pad=(),
                template_index=slab.template,
            )
            sources.append(src)
            used_bytes += slab.n_bytes
            smem_index = tuple(Var(ax.name) for ax in slab.cache_axes)
            for load in members:
                name_rewrites[load.name] = (smem_name, smem_index)
    return sources, name_rewrites


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
    *,
    slab_cap: int,
    extra_candidates: tuple[Axis, ...] = (),
    allow_no_fan_in: bool = False,
) -> _Slab | None:
    candidates = (*thread_axes, reduce_axis, *extra_candidates)
    ctx = SimplifyCtx({ax.name: Interval(0, ax.extent.as_static() - 1) for ax in scope_axes if ax.extent.is_static})
    candidate_names = tuple(ax.name for ax in candidates)

    zero_sigma = Sigma({n: Literal(0, "int") for n in candidate_names})
    origin = tuple(zero_sigma.reduce(e, ctx) for e in load.index)

    var_to_dim: dict[str, int] = {}
    for ax in candidates:
        dims = [d for d, e in enumerate(load.index) if ax.name in e.free_vars()]
        if not dims:
            continue
        if len(dims) > 1:
            return None
        var_to_dim[ax.name] = dims[0]

    if not var_to_dim:
        return None
    if len(var_to_dim) == len(candidates) and not allow_no_fan_in:
        return None

    cache_axes_unsorted = tuple(ax for ax in candidates if ax.name in var_to_dim)
    cache_axes = tuple(sorted(cache_axes_unsorted, key=lambda ax: var_to_dim[ax.name]))
    slab_dims = tuple(var_to_dim[ax.name] for ax in cache_axes)
    n_bytes = BYTES_PER_ELEM
    for ax in cache_axes:
        n_bytes *= ax.extent.as_static()
    if n_bytes > slab_cap:
        return None

    needs_template = False
    for ax in cache_axes:
        d = var_to_dim[ax.name]
        unit_sigma = Sigma({n: Literal(1 if n == ax.name else 0, "int") for n in candidate_names})
        actual = sorted(t.pretty() for t in _flatten_add(unit_sigma.reduce(load.index[d], ctx)))
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
        n_bytes=n_bytes,
    )


def _load_free_vars(load: Load) -> frozenset[str]:
    out: set[str] = set()
    for e in load.index:
        out |= e.free_vars()
    return frozenset(out)


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
