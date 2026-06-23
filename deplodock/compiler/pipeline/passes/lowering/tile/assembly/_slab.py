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

from dataclasses import replace

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Load, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    Binding,
    Block,
    RegisterTile,
    SerialTile,
    Source,
    StageBundle,
    StagePolicy,
    TileGraph,
    Transport,
    _add,
    _affine_terms,
    _mul_const,
    pick_swizzle_atom,
)
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import _identity_rename

# Double-buffer ring depth for TMA-staged K loops. R5 fixes the depth at 2 (the
# classical compute-current / fill-next double-buffer); the occupancy ``RING`` fork
# (depth 3-4) rides a later tier. The matching ``assembly/020_peel`` software-pipeline
# reads this off ``StageBundle.buffer_count``.
_TMA_RING = 2


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


def _source_inner_elems(src: Source) -> int:
    """Collapsed inner-row element span of ``src``'s slab — the product of
    ``(cache_extent × block)`` over every cache axis mapping to the innermost
    *source* dim (``max(dims)`` — the contiguous, fastest-varying source dim).
    Used to pick the TMA swizzle atom; mirrors the materializer's box reshape."""
    addressing = src.addressing
    assert isinstance(addressing, AffineAddressing)
    dims, block = addressing.dims, addressing.block
    inner_dim = max(dims)
    inner = 1
    for i, (d, ax) in enumerate(zip(dims, src.cache_axes, strict=True)):
        if d == inner_dim:
            b = block[i] if block else 1
            inner *= ax.extent.as_static() * b
    return inner


def _stamp_swizzle(src: Source) -> Source:
    """Stamp one TMA source's :class:`SwizzleMode` from its inner-row byte span (A
    64 B → B64, B 128 B → B128 — distinct modes on sources sharing one bundle). Only
    affine sources swizzle; the materializer reads ``src.swizzle`` into each
    descriptor and the matching ldmatrix XOR (``kernel/005_lower_atom_tile``)."""
    if not isinstance(src.addressing, AffineAddressing):
        return src
    elem_bytes = src.dtype.nbytes if src.dtype is not None else BYTES_PER_ELEM
    _, mode = pick_swizzle_atom(_source_inner_elems(src), elem_bytes)
    return replace(src, swizzle=mode)


def _bundle_sources(inner: tuple[Stmt, ...], staged_bufs: frozenset[str], cache_axes: dict[str, Axis], buffers: dict) -> list[Source]:
    """A :class:`Source` per staged buffer (from its first consumer ``Load``)."""
    sources: list[Source] = []
    for ld in Body(inner).iter_of_type(Load):
        if ld.input not in staged_bufs or any(src.buf == ld.input for src in sources):
            continue  # one slab per buffer, from its first consumer Load
        dtype = buffers[ld.input].dtype if ld.input in buffers else None
        source, _ = _source_from_load(ld, ld.input, cache_axes, dtype)
        sources.append(source)
    return sources


def _make_bundle(
    inner: tuple[Stmt, ...],
    staged_bufs: frozenset[str],
    cache_axes: dict[str, Axis],
    buffers: dict,
    *,
    tma_phase: Expr | None = None,
) -> StageBundle:
    """One ``StageBundle`` over ``inner``: a Source per staged buffer + ``inner`` with
    every staged Load rewritten to the slab. SYNC by default; when ``tma_phase`` is
    given (a ``Var(K_o) % RING`` ring slot) the bundle is the double-buffered TMA
    transport — each source swizzle-stamped, ``buffer_count = RING``, ``phase`` set —
    materialized as ``cp.async.bulk.tensor`` box copies (``assembly/020_peel`` then
    software-pipelines it)."""
    sources = _bundle_sources(inner, staged_bufs, cache_axes, buffers)
    by_buf = {src.buf: src for src in sources}
    if tma_phase is None:
        return StageBundle(sources=tuple(sources), body=Body(_rewrite_loads(inner, by_buf)), policy=StagePolicy.SYNC)
    # Double-buffered TMA: the slab is allocated ``[phase, *cache_axes]``; the
    # consumer slab Loads carry the ring slot as a leading index dim so the
    # ldmatrix lowering (``kernel/005_lower_atom_tile._mma_src_index``) reads the
    # right slot (it detects the prefix via ``len(index) > len(cache_axes)``).
    return StageBundle(
        sources=tuple(_stamp_swizzle(s) for s in sources),
        body=Body(_rewrite_loads(inner, by_buf, phase=tma_phase)),
        policy=StagePolicy.TMA,
        buffer_count=_TMA_RING,
        phase=tma_phase,
    )


def _wrap_k_body(
    stmts: tuple[Stmt, ...],
    staged_bufs: frozenset[str],
    cache_axes: dict[str, Axis],
    buffers: dict,
    *,
    tma_bufs: frozenset[str] = frozenset(),
) -> tuple[Stmt, ...]:
    """Wrap the K-tower in a ``StageBundle``. With a multi-stage K loop the bundle
    sits *inside* the ``serial_outer`` loop (the slab reloads per stage); when ``BK
    == K`` collapses ``serial_outer`` away, the ``stage_inner`` loop alone is wrapped
    (the whole-K slab, loaded once). When every staged buffer is in ``tma_bufs`` the
    ``serial_outer`` bundle is the double-buffered TMA transport (phase ``K_o % RING``)
    — only the ringable multi-stage loop is TMA-promoted (the single-stage whole-K
    slab can't pipeline, so it stays SYNC)."""
    all_tma = bool(staged_bufs) and staged_bufs <= tma_bufs
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            phase = BinaryExpr("%", Var(s.axis.name), Literal(_TMA_RING, "int")) if all_tma else None
            bundle = _make_bundle(tuple(s.body), staged_bufs, cache_axes, buffers, tma_phase=phase)
            out.append(SerialTile(axis=s.axis, body=Body((bundle,)), kind=s.kind))
        elif isinstance(s, SerialTile) and s.kind == "stage_inner":
            out.append(_make_bundle((s,), staged_bufs, cache_axes, buffers))
        else:
            out.append(s)
    return tuple(out)


def _rewrite_loads(stmts: tuple[Stmt, ...], by_buf: dict[str, Source], *, phase: Expr | None = None) -> tuple[Stmt, ...]:
    """Recursively rewrite every staged ``Load`` to read its smem slab; descend
    through nested tile bodies, leaving non-staged Loads and all other stmts intact.
    When ``phase`` is given (a double-buffered ring slot) it is prepended as the
    leading slab index dim."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.input in by_buf:
            src = by_buf[s.input]
            slab_index = tuple(Var(ax.name) for ax in src.cache_axes)
            if phase is not None:
                slab_index = (phase, *slab_index)
            out.append(Load(name=s.name, input=src.name, index=slab_index))
            continue
        nested = s.nested() if hasattr(s, "nested") else ()
        if nested:
            new_bodies = tuple(Body(_rewrite_loads(tuple(b), by_buf, phase=phase)) for b in nested)
            out.append(s.with_bodies(new_bodies))
        else:
            out.append(s)
    return tuple(out)


def _drop_size1_registers(block: Block, binding: dict) -> Body:
    """Substitute every static extent-1 REGISTER axis var → 0 in ``compute``, the
    same drop ``_wrap_tower`` applies later (a size-1 REGISTER cell is inlined).
    Doing it BEFORE slab classification keeps the slab in lock-step with the
    materialized tower: an atom-strided block multiplier on a dropped ``A_r`` cell
    (the ``FM=1`` warp case) migrates onto the surviving ``A_w`` warp axis instead
    of dangling on an axis the tower won't emit (an undefined ldmatrix index var)."""
    sub = {
        a.name: Literal(0, "int")
        for a in block.domain
        if binding.get(a.name) is Binding.REGISTER and a.extent.is_static and a.extent.as_static() == 1
    }
    if not sub:
        return block.compute
    sigma = Sigma(sub)
    return Body(tuple(s.rewrite(_identity_rename, sigma) for s in block.compute))


def prospective_sources(graph: TileGraph) -> list[Source]:
    """The smem ``Source``s ``assemble`` *would* synthesize for ``graph``'s staged
    read-sites — the derived projection the ``promote_transport`` fork's eligibility
    oracle (``enumeration/052_transport``) reads, without materializing the tower. Empty
    when nothing is staged."""
    staged = graph.schedule.staged
    block = graph.blocks[0]
    if not staged:
        return []
    staged_bufs = frozenset(e.buffer for e in staged)
    compute = _drop_size1_registers(block, graph.schedule.binding)
    cache_axes = _cache_axis_names(block, graph.schedule.binding)
    return _bundle_sources(tuple(compute), staged_bufs, cache_axes, graph.buffers)


def _is_k_stmt(s: Stmt) -> bool:
    """The hoistable K-tower stmts: a ``serial_outer`` / ``stage_inner`` ``SerialTile``
    (post-``_wrap_k_body`` the bundle rides inside one) or a bare ``StageBundle``."""
    return isinstance(s, StageBundle) or (isinstance(s, SerialTile) and s.kind in ("serial_outer", "stage_inner"))


def _input_extents(buffers: dict) -> dict[str, tuple]:
    """``{gmem buffer name -> per-dim extents}`` for the masked cooperative-load
    clamp. A static dim contributes its ``int`` extent; a symbolic dim its ``Expr``
    (``Var('seq_len')``), which renders the clamp ternary against the runtime kernel
    arg. Buffers with an unparseable dim are skipped (no clamp)."""
    out: dict[str, tuple] = {}
    for name, buf in buffers.items():
        exts: list[int | Expr] = []
        for d in buf.shape:
            if getattr(d, "is_static", False):
                exts.append(d.as_static())
            elif getattr(d, "expr", None) is not None:
                exts.append(d.expr)
            else:
                break
        else:
            out[name] = tuple(exts)
    return out


def _stamp_gmem_extents(stmt: Stmt, input_extents: dict) -> Stmt:
    """Recursively stamp ``Source.gmem_extents`` on every SYNC ``StageBundle`` source
    whose ``buf`` is a kernel input — so ``_stage_expand.emit_stage`` clamps the
    hoisted cooperative gmem read to ``[0, extent)``. Only the SYNC transport needs
    it: a TMA bundle relies on the hardware OOB zero-fill (descriptor globalDim =
    runtime extent), so its sources are left unstamped."""
    if isinstance(stmt, StageBundle):
        if stmt.policy is StagePolicy.SYNC:
            new_sources = tuple(
                replace(src, gmem_extents=input_extents[src.buf]) if src.buf in input_extents and src.gmem_extents is None else src
                for src in stmt.sources
            )
            stmt = replace(stmt, sources=new_sources)
        return stmt
    nested = stmt.nested()
    if not nested:
        return stmt
    new_bodies = tuple(Body(tuple(_stamp_gmem_extents(s, input_extents) for s in body)) for body in nested)
    return stmt.with_bodies(new_bodies)


def _hoist_masked(stmts: tuple[Stmt, ...], staged_bufs, cache_axes, buffers, tma_bufs) -> tuple[Stmt, ...] | None:
    """Masked-tile staging: a symbolic / non-divisor matmul wraps its K tower +
    output ``Write`` in a boundary ``Cond(σ(M|N) < bound)``. The cooperative load
    must be hoisted **above** the guard so every thread issues it uniformly (a SYNC
    ``__syncthreads`` / cp.async / TMA inside divergent control flow is a hang, and a
    skipping overhang thread leaves the slab half-filled). The overhang then reads
    past the buffer: SYNC sources are clamped to the gmem bounds
    (``_stamp_gmem_extents`` → ``_stage_expand``), TMA sources rely on the hardware
    OOB zero-fill. Either way the masked rows accumulate harmless values the gated
    ``Write`` never stores. Returns the hoisted body (K tower above, ``Cond`` gating
    just the epilogue/``Write``) when the pattern matches, else ``None`` (caller falls
    back to the plain in-place wrap — a clean-divisor tile has no boundary ``Cond``).

    Gated on a ``<`` boundary predicate: the ``==`` split-K invariant guard
    (``Cond(K_s == 0, ...)``) must NOT hoist its loads above the guard (it would
    re-run the cooperative load on the CTAs the guard skips)."""
    if len(stmts) != 1 or not isinstance(stmts[0], Cond) or stmts[0].else_body:
        return None
    cond = stmts[0]
    if not (isinstance(cond.cond, BinaryExpr) and cond.cond.op == "<"):
        return None
    inner = tuple(cond.body)
    if not any(isinstance(s, SerialTile) and s.kind in ("serial_outer", "stage_inner") for s in inner):
        return None
    wrapped = _wrap_k_body(inner, staged_bufs, cache_axes, buffers, tma_bufs=tma_bufs)
    k_part = tuple(s for s in wrapped if _is_k_stmt(s))
    rest = tuple(s for s in wrapped if not _is_k_stmt(s))
    if not k_part:
        return None
    # SSA-dependency safety: refuse the lift when a hoisted K-tower stmt reads a
    # name defined by a stmt staying inside the ``Cond`` (the fused-prologue shape
    # — e.g. a matmul consuming the rsqrt of its row stats). Hoisting would order
    # the consumer ABOVE its definition (undefined identifier at render). Leaving
    # the ``Cond`` intact (caller falls back to the in-place wrap) keeps the body
    # well-formed. Defense-in-depth: the planner doesn't emit liftable masked
    # prologue ``Cond``s today (static-K prologue kernels stay degenerate,
    # symbolic-K ones never stage), but a future planner change / exotic pin could.
    k_reads = {name for s in k_part for st in Body((s,)).iter() for name in st.deps()}
    rest_defs = {name for s in rest for st in Body((s,)).iter() for name in st.defines()}
    if k_reads & rest_defs:
        return None
    extents = _input_extents(buffers)
    k_part = tuple(_stamp_gmem_extents(s, extents) for s in k_part)
    return (*k_part, Cond(cond=cond.cond, body=Body(rest), else_body=()))


def synthesize_staging(graph: TileGraph) -> Body:
    """Return the single block's ``compute`` rewritten so each ``Schedule.staged``
    read-site reads an smem slab, with one ``StageBundle`` wrapping the K-tower. A
    no-op (returns ``block.compute`` unchanged) when nothing is staged."""
    staged = graph.schedule.staged
    block = graph.blocks[0]
    if not staged:
        return block.compute
    staged_bufs = frozenset(e.buffer for e in staged)
    tma_bufs = frozenset(e.buffer for e, t in staged.items() if t is Transport.TMA)
    compute = _drop_size1_registers(block, graph.schedule.binding)
    cache_axes = _cache_axis_names(block, graph.schedule.binding)
    stmts = tuple(compute)
    hoisted = _hoist_masked(stmts, staged_bufs, cache_axes, graph.buffers, tma_bufs)
    if hoisted is not None:
        return Body(hoisted)
    return Body(_wrap_k_body(stmts, staged_bufs, cache_axes, graph.buffers, tma_bufs=tma_bufs))
