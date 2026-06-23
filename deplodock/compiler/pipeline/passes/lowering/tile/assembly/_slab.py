"""Slab synthesis — ``assemble`` materializes ``Schedule.staged`` into smem.

``plans/tile-ir-block-dag.md``: "Slabs, cooperative producers ... are all
assemble OUTPUTS — synthesized from the algorithm + Schedule, never stored in the
IR." This module is that synthesis for the ``stage`` move (R1, scalar tier): for
each ``Schedule.staged`` read-site it derives a :class:`Source` (the smem slab's
cache axes + per-CTA-per-stage origin + affine addressing) and wraps the K-tower
in one ``StageBundle(policy=SYNC)``, rewriting the staged consumer ``Load``s to
read the slab. The downstream kernel passes (``_stage_expand`` /
``100_materialize_tile``) expand the bundle into the cooperative ``Load``+``Write``
+ ``__syncthreads`` producer — untouched.

Slab geometry is read off the consumer ``Load``'s index, partitioned against the
*cache-eligible* axes — the ``THREAD`` / ``REGISTER`` free axes plus the within-stage
K axes (the ``stage_inner`` serial loop + any reduce ``RegisterTile`` ``FK`` strip).
``GRID`` axes are CTA-uniform and the serial-outer ``K_o`` is the loop *over*
stages, so both fold into the slab ``origin``. The per-cache-axis composite stride
is reconstructed by ``affine_decode_per_dim`` from the cache-axis extents
(``block=()``), so the cooperative producer's gmem index byte-matches the original
σ-rewritten ``Load``.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Var
from deplodock.compiler.ir.stmt import Body, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Block,
    RegisterTile,
    SerialTile,
    Source,
    StageBundle,
    StagePolicy,
    TileGraph,
    _add,
    _affine_terms,
    _mul_const,
)


def _cache_axis_names(block: Block, binding: dict) -> dict[str, Axis]:
    """The cache-eligible axes (name -> Axis): THREAD / WARP / REGISTER free axes
    (from ``binding``) + the within-stage K axes (the ``stage_inner`` serial loop
    and any reduce ``RegisterTile``). These span the smem slab; GRID and
    serial-outer K fold into the slab origin. The ATOM cell axes are
    non-addressable (excluded) — their per-cell extent rides the cache axes'
    ``AffineAddressing.block`` multiplier instead, derived from the σ coefficients
    in :func:`_source_from_load`."""
    from deplodock.compiler.ir.tile.ir import Binding  # noqa: PLC0415

    out: dict[str, Axis] = {a.name: a for a in block.domain if binding.get(a.name) in (Binding.THREAD, Binding.WARP, Binding.REGISTER)}
    for s in block.compute.iter():
        if isinstance(s, SerialTile) and s.kind == "stage_inner":
            out[s.axis.name] = s.axis
        elif isinstance(s, RegisterTile) and s.reduce:
            for ax in s.axes:
                out[ax.name] = ax
    return out


def _derive_block(group: list[tuple[str, int]], cache_axes: dict[str, Axis]) -> dict[str, int]:
    """Per-axis atom-cell stride multiplier for one source dim's cache axes.

    The cache vars of one dim are ordered most-significant-first (largest σ
    coefficient); walking right-to-left, ``block_ax = coef // suffix_product`` and
    ``suffix_product *= extent · block_ax``. For a scalar tile every ``block_ax``
    is ``1`` (the σ coefficient is exactly the suffix product of inner cache
    extents); a warp/atom tile carries the ``atom_m`` / ``atom_k`` factor on the
    innermost output / K cache axis (the ATOM cell is non-addressable, so its
    extent rides this multiplier). Raises on a non-divisible coefficient (a
    layout the additive slab can't size — TEMPLATE, out of R1 scope)."""
    out: dict[str, int] = {}
    suffix = 1
    for v, coef in reversed(group):  # group is most-significant-first → reverse = inner-first
        if coef % suffix != 0:
            raise NotImplementedError(f"stage: cache coefficient {coef} not a multiple of inner span {suffix}")
        block_ax = coef // suffix
        if block_ax < 1:
            raise NotImplementedError(f"stage: degenerate cache coefficient {coef}")
        out[v] = block_ax
        suffix *= cache_axes[v].extent.as_static() * block_ax
    return out


def _source_from_load(load: Load, src_name: str, cache_axes: dict[str, Axis], dtype) -> tuple[Source, tuple[Expr, ...]]:
    """Classify one consumer ``Load``'s index into a :class:`Source` (slab spec) +
    the rewritten consumer slab index. Each source dim is decomposed affinely; vars
    in ``cache_axes`` become cache axes (most-significant — highest coefficient —
    first within a dim), everything else folds into the per-dim origin anchor. The
    per-axis ``AffineAddressing.block`` multiplier is derived from the σ
    coefficients (:func:`_derive_block`) — ``()`` for a scalar tile, atom-strided
    for a warp/atom tile."""
    per_dim_cache: list[tuple[int, str, int]] = []  # (dim, var, coef)
    origin: list[Expr] = []
    for d, e in enumerate(load.index):
        terms = _affine_terms(e)
        if terms is None:
            raise NotImplementedError(f"stage: non-affine index for {load.input!r} — TEMPLATE slabs are not in R1 scope")
        coeffs, const = terms
        anchor = const
        for v, c in coeffs.items():
            if v in cache_axes:
                per_dim_cache.append((d, v, c))
            else:
                anchor = _add(anchor, _mul_const(Var(v), c))
        origin.append(anchor)
    # Cache-axis layout order: by source dim, most-significant (largest coef) first
    # within a dim — matches the σ-split's composite stride so ``affine_decode_per_dim``
    # reconstructs the original gmem index from the cache extents × block multipliers.
    per_dim_cache.sort(key=lambda t: (t[0], -t[2]))
    ordered_axes = tuple(cache_axes[v] for _, v, _ in per_dim_cache)
    dims = tuple(d for d, _, _ in per_dim_cache)
    # Derive the per-axis block multiplier per source dim, then align it to the
    # cache-axis order. Collapse to () when every multiplier is 1 (scalar tile).
    block_of: dict[str, int] = {}
    for d in {dd for dd, _, _ in per_dim_cache}:
        grp = [(v, c) for dd, v, c in per_dim_cache if dd == d]  # already most-sig-first (sorted by -coef)
        block_of.update(_derive_block(grp, cache_axes))
    block_tuple = tuple(block_of.get(v, 1) for _, v, _ in per_dim_cache)
    if all(b == 1 for b in block_tuple):
        block_tuple = ()
    source = Source(
        name=f"{src_name}_smem",
        buf=load.input,
        cache_axes=ordered_axes,
        origin=tuple(origin),
        addressing=AffineAddressing(dims=dims, block=block_tuple),
        dtype=dtype,
    )
    slab_index = tuple(Var(ax.name) for ax in ordered_axes)
    return source, slab_index


def _make_bundle(inner: tuple[Stmt, ...], staged_bufs: frozenset[str], cache_axes: dict[str, Axis], buffers: dict) -> StageBundle:
    """One SYNC ``StageBundle`` over ``inner``: a Source per staged buffer (from its
    first consumer ``Load``) + ``inner`` with every staged Load rewritten to the slab."""
    sources: list[Source] = []
    for ld in Body(inner).iter_of_type(Load):
        if ld.input not in staged_bufs or any(src.buf == ld.input for src in sources):
            continue  # one slab per buffer, from its first consumer Load
        dtype = buffers[ld.input].dtype if ld.input in buffers else None
        source, _ = _source_from_load(ld, ld.input, cache_axes, dtype)
        sources.append(source)
    by_buf = {src.buf: src for src in sources}
    return StageBundle(sources=tuple(sources), body=Body(_rewrite_loads(inner, by_buf)), policy=StagePolicy.SYNC)


def _wrap_k_body(stmts: tuple[Stmt, ...], staged_bufs: frozenset[str], cache_axes: dict[str, Axis], buffers: dict) -> tuple[Stmt, ...]:
    """Wrap the K-tower in a ``StageBundle``. With a multi-stage K loop the bundle
    sits *inside* the ``serial_outer`` loop (the slab reloads per stage); when ``BK
    == K`` collapses ``serial_outer`` away, the ``stage_inner`` loop alone is wrapped
    (the whole-K slab, loaded once)."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            bundle = _make_bundle(tuple(s.body), staged_bufs, cache_axes, buffers)
            out.append(SerialTile(axis=s.axis, body=Body((bundle,)), kind=s.kind))
        elif isinstance(s, SerialTile) and s.kind == "stage_inner":
            out.append(_make_bundle((s,), staged_bufs, cache_axes, buffers))
        else:
            out.append(s)
    return tuple(out)


def _rewrite_loads(stmts: tuple[Stmt, ...], by_buf: dict[str, Source]) -> tuple[Stmt, ...]:
    """Recursively rewrite every staged ``Load`` to read its smem slab; descend
    through nested tile bodies, leaving non-staged Loads and all other stmts intact."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.input in by_buf:
            src = by_buf[s.input]
            slab_index = tuple(Var(ax.name) for ax in src.cache_axes)
            out.append(Load(name=s.name, input=src.name, index=slab_index))
            continue
        nested = s.nested() if hasattr(s, "nested") else ()
        if nested:
            new_bodies = tuple(Body(_rewrite_loads(tuple(b), by_buf)) for b in nested)
            out.append(s.with_bodies(new_bodies))
        else:
            out.append(s)
    return tuple(out)


def synthesize_staging(graph: TileGraph) -> Body:
    """Return the single block's ``compute`` rewritten so each ``Schedule.staged``
    read-site reads an smem slab, with one ``StageBundle`` wrapping the K-tower. A
    no-op (returns ``block.compute`` unchanged) when nothing is staged."""
    staged = graph.schedule.staged
    block = graph.blocks[0]
    if not staged:
        return block.compute
    staged_bufs = frozenset(e.buffer for e in staged)
    cache_axes = _cache_axis_names(block, graph.schedule.binding)
    return Body(_wrap_k_body(tuple(block.compute), staged_bufs, cache_axes, graph.buffers))
