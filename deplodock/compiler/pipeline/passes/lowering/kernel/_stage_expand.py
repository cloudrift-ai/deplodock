"""Stage-expansion helpers for ``100_materialize_tile``.

Producer scaffolding for a ``StageBundle``: its transport ``sources`` become
cooperative ``Load + Write`` (or ``CpAsyncCopy``) nests, and its optional
``compute`` phase becomes a σ-substituted cooperative ``StridedLoop``. Both
flatten multi-axis cache slabs via a row-major flat-iter decode. Pure
functions — no shared materializer state — so they live here rather than
inside the pass.

The leading-underscore module name keeps the pass loader (which globs
``*.py`` skipping ``_``-prefixed files) from mistaking this for a rule.
"""

from __future__ import annotations

from deplodock.compiler.dim import to_dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync
from deplodock.compiler.ir.stmt import Assign, Load, Select, SelectBranch, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, StagePolicy, TemplateAddressing


def _cp_async_width(elem_size: int, padded_extents: tuple[int, ...], addressing, n_origin_dims: int, gmem_inner: int | Expr | None) -> int:
    """Elements per ``cp.async`` for a cooperative stage load — the widest
    contiguous vector whose byte size is a legal cp.async width (4 / 8 / 16).

    Returns ``V`` (elements/thread), or ``0`` when no cp.async is safe (fp16/bf16
    that isn't vector-contiguous — cp.async's min copy is 4 B = 2 halves, so a
    half tile can't fall back to a 1-element copy and must use the sync path).
    fp32 keeps a scalar 4-byte copy (``V = 1``) when not vectorizable.

    Safe wide vectorization needs: (a) the inner (last) cache/slab axis maps
    **stride-1 to the gmem inner dim** — ``dims[-1] == last source dim`` — so the
    inner slab run is row-major-contiguous in *both* smem and gmem (the per-axis
    ``block`` atom-stride is irrelevant here: the cooperative load walks raw slab
    coords and the inner slab dim is always the row-major fastest one);
    (b) ``V`` divides the inner alloc extent (a chunk never straddles a padded
    row, so ``PAD_SMEM``'s ``+1`` disables wide cp.async); and (c) the gmem inner
    stride is ``V``-aligned, so each row's chunk start stays ``V*elem``-byte
    aligned (cp.async faults on a misaligned 16-byte copy). Only
    ``AffineAddressing`` is analyzed; anything else is treated non-contiguous.
    A *symbolic* gmem inner stride (an ``Expr`` extent from a runtime-sized
    buffer) can't be V-alignment-checked at compile time, so it conservatively
    takes the scalar fallback."""
    if elem_size not in (2, 4) or not padded_extents:
        return 1 if elem_size == 4 else 0
    if gmem_inner is not None and not isinstance(gmem_inner, int):
        return 1 if elem_size == 4 else 0
    inner = int(padded_extents[-1])
    contiguous = isinstance(addressing, AffineAddressing) and bool(addressing.dims) and addressing.dims[-1] == n_origin_dims - 1
    if not contiguous:
        return 1 if elem_size == 4 else 0
    for nbytes in (16, 8, 4):
        v = nbytes // elem_size
        if v < 1 or inner % v != 0:
            continue
        if gmem_inner is not None and gmem_inner % v != 0:
            continue  # row stride not V-aligned → chunk start would misalign
        return v
    return 1 if elem_size == 4 else 0


def _ext_expr(ext: int | Expr) -> Expr:
    """An extent as an ``Expr``: static ints become ``Literal``s, symbolic
    extents (e.g. ``Var('seq_len')``) pass through and render against the
    runtime kernel arg."""
    return Literal(ext, "int") if isinstance(ext, int) else ext


def _ext_minus_one(ext: int | Expr) -> Expr:
    return Literal(ext - 1, "int") if isinstance(ext, int) else BinaryExpr("-", ext, Literal(1, "int"))


def _clamp_source_index(source_index: tuple[Expr, ...], gmem_extents: tuple[int | Expr, ...] | None) -> tuple[Expr, ...]:
    """Clamp each cooperative-load gmem index dim to ``[0, extent)``.

    Set only by ``021_hoist_staged_loads_above_mask`` for sources whose
    cooperative load was hoisted above a masked-tile boundary ``Cond``. A
    masked output axis tiles past the real extent (N=256 tiled at 192 → the
    boundary tile spans [192, 384); a symbolic axis tiles at its hint, so the
    boundary tile overruns whenever the runtime extent isn't tile-aligned),
    so the producer's gmem read overruns the buffer for the overhang columns
    — the original cause of the ``CUDA_ERROR_ILLEGAL_ADDRESS`` in masked
    linear-projection kernels. The overhang slab slots get the clamped
    (duplicate) value, which is harmless: they feed masked output cells that
    the boundary ``Cond`` never writes.

    Extents are ``int`` for static dims or the dim's symbolic ``Expr``
    (``Var('seq_len')``) for runtime-sized ones — the ternary's bound then
    renders against the runtime kernel arg.

    Two index shapes occur, both handled:

    - **Per-dim (affine)**: ``source_index`` has one coord per gmem dim, so
      each is clamped to ``[0, extent_d)`` via ``idx < extent ? idx :
      extent-1``. Only the masked dim's ternary survives constant-folding; the
      clean dims clamp against their own extent (no-op at runtime).
    - **Collapsed (template / reshape)**: ``source_index`` is a single
      pre-flattened ``row*N + col`` expression while ``gmem_extents`` is
      multi-dim. Clamp the flat index to ``[0, ∏ extents)`` — this kills the
      only true OOB (the boundary tile's last row reading past the buffer
      end). An overhang column that wraps into the next row is in-bounds and
      harmless (it feeds a masked-out output cell).

    On clean-divisor tiles ``gmem_extents`` is ``None`` and the index passes
    through untouched — no perf cost on the common path."""
    if gmem_extents is None:
        return source_index
    if len(source_index) == len(gmem_extents):
        out: list[Expr] = []
        for idx, ext in zip(source_index, gmem_extents, strict=True):
            cond = BinaryExpr("<", idx, _ext_expr(ext))
            out.append(TernaryExpr(cond=cond, if_true=idx, if_false=_ext_minus_one(ext)))
        return tuple(out)
    if len(source_index) == 1:
        # ∏ extents, folding the static factors into one Literal so the common
        # all-static case keeps its single-literal bound.
        total_static = 1
        total_sym: Expr | None = None
        for e in gmem_extents:
            if isinstance(e, int):
                total_static *= e
            else:
                total_sym = e if total_sym is None else BinaryExpr("*", total_sym, e)
        if total_sym is None:
            bound: int | Expr = total_static
        elif total_static == 1:
            bound = total_sym
        else:
            bound = BinaryExpr("*", total_sym, Literal(total_static, "int"))
        idx = source_index[0]
        cond = BinaryExpr("<", idx, _ext_expr(bound))
        return (TernaryExpr(cond=cond, if_true=idx, if_false=_ext_minus_one(bound)),)
    # Unexpected rank shape (multi-dim source_index that doesn't match the
    # buffer rank) — leave untouched rather than mis-clamp.
    return source_index


def smem_cuda_dtype(src) -> str:  # noqa: ANN001 — Source carries DataType | None
    """C type spelling for a Source's smem slab, derived from the
    stamped ``Source.dtype`` (``030_stamp_types``). Defaults to the
    legacy ``"float"`` when unstamped — handwritten test fixtures
    rely on the fallback."""
    from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

    if src.dtype is None:
        return "float"
    return cuda_name(src.dtype)


def emit_stage(
    sources,  # noqa: ANN001 — tuple[Source, ...]
    tid_expr,
    n_threads: int,
    *,
    policy: StagePolicy,
    buffer_count: int,
    phase,
    pipeline_depth: int,
) -> list[Stmt]:
    """Emit producer scaffolding for a bundle's transport ``sources``.

    For each ``Source``, emits per-source cooperative ``Load + Write``
    (``CpAsyncCopy`` for ASYNC). Smem decls are hoisted to kernel scope in
    ``_materialize`` (we skip them via the dedup filter). Leading + trailing
    ``Sync`` bracket the cooperative load for SYNC policy; BUFFERED/ASYNC/TMA
    drop the leading sync (different physical slab per iter).

    Bundle context (policy, buffer_count, phase, pipeline_depth) is passed in.
    """
    # Masked-K (symbolic reduce) sources need a per-element ZERO-fill of the
    # final partial K slab — the mma accumulates the overhang, so a clamped
    # duplicate (the M/N edge-clamp) would corrupt the result. cp.async /
    # buffered transports copy raw bytes and can't ternary a value, so a bundle
    # carrying any kmask source is pinned to the SYNC transport (the promotion
    # passes 040/050/060/080 also skip it, so the consumer reads the same
    # un-phased slab this sync writer fills).
    has_kmask = any(getattr(s, "kmask", None) is not None for s in sources)
    if has_kmask:
        policy = StagePolicy.SYNC
    is_buffered = policy != StagePolicy.SYNC
    is_async = policy == StagePolicy.ASYNC

    prelude: list[Stmt] = [] if is_buffered else [Sync()]
    body_out: list[Stmt] = list(prelude)

    for src in sources:
        # Per-Source iteration axis: synthesize a unique flat-iter axis
        # so the StridedLoop's loop variable doesn't collide with outer
        # thread-decode variables.
        cache_axes = src.cache_axes
        if not cache_axes:
            raise ValueError(f"Source {src.name!r} has no cache axes")
        padded_extents = src.alloc_extents
        # M3+: a blocked source's slab is sized by ``alloc_extents`` (cache
        # extent × per-axis block factor). The cooperative producer must
        # fill *every* slab position, so the flat-iter axis spans the
        # block-scaled total and the per-axis decode resolves into
        # ``0..alloc_extent_i`` (raw slab coords) rather than the bare
        # cache-extent range. Source-side stride collapses to 1 once the
        # iteration covers the full block, so we bypass
        # ``AffineAddressing.source_index`` for the producer and emit
        # ``origin + slab_coord`` directly.
        addressing = src.addressing
        is_blocked = isinstance(addressing, AffineAddressing) and bool(addressing.block)
        if is_blocked:
            iter_extents = tuple(ax.extent.as_static() * b for ax, b in zip(cache_axes, addressing.block, strict=True))
        else:
            iter_extents = tuple(ax.extent.as_static() for ax in cache_axes)
        total = 1
        for e in iter_extents:
            total *= e
        iter_axis = Axis(name=f"{src.name}_flat", extent=total)
        if is_blocked:
            # Block-aware flat decode: each cache axis is treated as if
            # its extent were the block-scaled value, so the decoded
            # coords land in the raw slab range.
            scaled_axes = tuple(Axis(name=ax.name, extent=e) for ax, e in zip(cache_axes, iter_extents, strict=True))
            coord_for = flat_decode(scaled_axes, iter_axis.name) if len(scaled_axes) > 1 else {scaled_axes[0].name: Var(iter_axis.name)}
        elif len(cache_axes) == 1:
            coord_for = {cache_axes[0].name: Var(iter_axis.name)}
        else:
            coord_for = flat_decode(cache_axes, iter_axis.name)
        smem_index = tuple(coord_for[ax.name] for ax in cache_axes)
        # Per-source source-index reconstruction.
        if isinstance(addressing, TemplateAddressing):
            from deplodock.compiler.ir.sigma import Sigma as _Sigma  # noqa: PLC0415

            cache_sigma = _Sigma({ax.name: coord_for[ax.name] for ax in cache_axes})
            source_index = tuple(cache_sigma.apply(e) for e in addressing.exprs)
        elif is_blocked:
            # Per-source-dim: row-major composite over the slab-coord
            # cache vars (block already absorbed by the iter range; the
            # composite stride here is the *block-scaled* extents).
            # ``affine_decode_per_dim`` with ``block=()`` and the
            # block-scaled axes does exactly that.
            from deplodock.compiler.ir.tile.ir import affine_decode_per_dim as _decode  # noqa: PLC0415

            decoded_per_dim = _decode(scaled_axes, addressing.dims, coord_for)
            source_index = tuple(
                src.origin[d] + decoded_per_dim[d] if d in decoded_per_dim else src.origin[d] for d in range(len(src.origin))
            )
        else:
            # Per source dim: composite-decode the cache axes mapping to
            # it. ``AffineAddressing.source_index`` threads
            # ``addressing.block`` through ``affine_decode_per_dim`` so
            # MMA atom-strided slabs (block != ()) get the right per-axis
            # stride. Scalar paths keep block=(), behavior identical to
            # pre-M4.
            source_index = addressing.source_index(cache_axes, coord_for, src.origin)
        # Masked-tile overhang clamp: when ``021_hoist_staged_loads_above_mask``
        # hoisted this cooperative load above a boundary ``Cond``, the gmem
        # read can overrun the buffer for the overhang columns. Clamp each
        # source dim to its gmem extent so the producer never reads OOB
        # (no-op / None on clean-divisor tiles).
        # Masked-K zero-fill: this operand's reduce (K) gmem dim is symbolic, so
        # the final K_o tile overruns the runtime extent. Build the in-bounds
        # predicate from the PRE-clamp K coordinate — when masked M is also
        # present, ``gmem_extents`` (stamped by 021) covers the K dim too, so the
        # clamp below would otherwise fold the K index to ``bound-1`` and make
        # the predicate vacuously true (the masked-M + masked-K bug). The clamp
        # then makes the read in-bounds; the value is zeroed where the predicate
        # fails so the mma accumulates zero past seq_len.
        kmask = getattr(src, "kmask", None)
        k_inbounds: Expr | None = None
        k_bound_e: Expr | None = None
        k_dim_idx = -1
        if kmask is not None:
            k_dim_idx, k_bound = kmask
            if k_dim_idx < len(source_index):
                k_bound_e = _ext_expr(k_bound)
                k_inbounds = BinaryExpr("<", source_index[k_dim_idx], k_bound_e)
        source_index = _clamp_source_index(source_index, getattr(src, "gmem_extents", None))
        if k_inbounds is not None and getattr(src, "gmem_extents", None) is None:
            # masked-K only (static M/N → no 021 clamp): clamp the K dim for a
            # safe in-bounds read ourselves (the value is zeroed below anyway).
            source_index = tuple(
                TernaryExpr(cond=BinaryExpr("<", e, k_bound_e), if_true=e, if_false=_ext_minus_one(kmask[1])) if d == k_dim_idx else e
                for d, e in enumerate(source_index)
            )
        # Buffered: prepend phase dim to write index (writes the current
        # ring slot).
        if is_buffered:
            smem_index = (phase, *smem_index)
        # Per-source dtype: use gmem source's CUDA C type so fp16 inputs
        # stage into __half smem. ``Source.dtype`` is stamped by
        # ``030_stamp_types`` from the matching graph node's dtype.
        smem_dtype = smem_cuda_dtype(src)
        smem_align = 16 if smem_dtype == "__half" else 0
        full_extents = (buffer_count, *padded_extents) if is_buffered else padded_extents

        # cp.async path (sm_80+): each thread copies a contiguous ``V``-element
        # vector (``V*elem_size`` ∈ {4,8,16} B) gmem→smem in one instruction.
        # ``_cp_async_width`` picks the widest safe V, so an fp16 tile streams 8
        # halves as one 16-byte ``cp.async.cg`` (the CUTLASS / cuBLAS shape).
        # The chunk loop covers ``total // V`` chunk indices; ``σ(iter → iter*V)``
        # rewrites the per-element decode to the chunk-start coords (the inner
        # slab decode ``(iter*V) % inner`` folds to ``V * (iter % (inner/V))`` —
        # a V-aligned, contiguous chunk). V=0 ⇒ no legal cp.async (fp16
        # non-contiguous) → sync path; V=1 ⇒ fp32's scalar 4-byte copy.
        elem_size = {"float": 4, "__half": 2, "__nv_bfloat16": 2}.get(smem_dtype, 0)
        gmem_inner = src.gmem_extents[-1] if getattr(src, "gmem_extents", None) else None
        cp_v = _cp_async_width(elem_size, padded_extents, addressing, len(src.origin), gmem_inner) if is_async else 0
        if k_inbounds is not None:
            cp_v = 0  # masked-K zero-fill needs the per-value sync ternary, not a raw cp.async copy
        if cp_v >= 1:
            nbytes = cp_v * elem_size
            if cp_v > 1:
                from deplodock.compiler.ir.sigma import Sigma as _Sigma  # noqa: PLC0415

                scale = _Sigma({iter_axis.name: Var(iter_axis.name) * Literal(cp_v, "int")})
                smem_index = tuple(scale.apply(e) for e in smem_index)
                source_index = tuple(scale.apply(e) for e in source_index)
            cooperative_load = StridedLoop(
                axis=Axis(name=iter_axis.name, extent=total // cp_v),
                start=tid_expr,
                step=Literal(n_threads, "int"),
                body=(CpAsyncCopy(smem=src.name, smem_index=smem_index, src=src.buf, src_index=source_index, nbytes=nbytes),),
            )
            body_out.append(Smem(name=src.name, extents=full_extents, dtype=smem_dtype, align=max(smem_align, nbytes)))
            body_out.append(cooperative_load)
            continue

        # Sync path: cooperative Load + Write. For a masked-K source the loaded
        # value is zeroed where the K coord is out of the runtime extent — the
        # overhang feeds the mma accumulation, so it must be 0, not the clamped
        # duplicate the index read returns. ``zero = v - v`` synthesizes a typed
        # zero (no literal-cast guesswork) and ``Select`` binds the in-bounds
        # value or that zero.
        load_name = f"{src.name}_v"
        if k_inbounds is not None:
            zero_name = f"{src.name}_z"
            sel_name = f"{src.name}_kf"
            load_body: tuple[Stmt, ...] = (
                Load(name=load_name, input=src.buf, index=source_index),
                Assign(name=zero_name, op="subtract", args=(load_name, load_name)),
                Select(
                    name=sel_name,
                    branches=(
                        SelectBranch(value=load_name, select=k_inbounds),
                        SelectBranch(value=zero_name, select=Literal(1, "int")),
                    ),
                ),
                Write(output=src.name, index=smem_index, value=sel_name),
            )
        else:
            load_body = (
                Load(name=load_name, input=src.buf, index=source_index),
                Write(output=src.name, index=smem_index, value=load_name),
            )
        cooperative_load = StridedLoop(
            axis=iter_axis,
            start=tid_expr,
            step=Literal(n_threads, "int"),
            body=load_body,
        )
        body_out.append(Smem(name=src.name, extents=full_extents, dtype=smem_dtype, align=smem_align))
        body_out.append(cooperative_load)

    # Trailing transport: cp.async stages emit Commit; for the
    # unpipelined wrap-body shape (pipeline_depth == 1), follow with the
    # implicit CpAsyncWait(0) + Sync so the consumer body sees the
    # committed copy at the wrap boundary. Pipelined stages
    # (pipeline_depth > 1) get expanded by
    # 080_pipeline_stages before materialize and emit their
    # own waits at the pipelined schedule positions.
    # Sync stages just emit __syncthreads so the slab is CTA-visible.
    if is_async:
        body_out.append(CpAsyncCommit())
        if pipeline_depth == 1:
            body_out.append(CpAsyncWait(group=0))
            body_out.append(Sync())
    else:
        body_out.append(Sync())
    return body_out


def compute_phase_info(compute, sources):  # noqa: ANN001 — Body, tuple[Source, ...]
    """Recover the fused-slab descriptors from a self-describing compute body.

    The compute phase (``StageBundle.compute``, set by
    ``030_hoist_invariant_compute``) carries no structured output fields —
    its single ``Write`` names the output slab and its cache-axis index
    ``Var``s, and the cache-axis ``Axis`` objects (with extents) live on the
    sibling cone ``Source``s the compute ``Load``s read. Returns
    ``(slab_name, cache_axes, value_dtype)`` where ``value_dtype`` is the
    ``Write``'s stamped dtype (``None`` until ``030_stamp_types`` runs)."""
    write = None
    for s in compute:
        if isinstance(s, Write):
            write = s
    if write is None:
        raise ValueError("compute phase body has no Write — not a hoisted-compute bundle")
    axis_map: dict[str, Axis] = {}
    for src in sources:
        for ax in src.cache_axes:
            axis_map[ax.name] = ax
    # Each index entry must be a bare ``Var`` naming one of the sibling cone's
    # cache axes. A sibling-cell fusion (``012_fuse_sibling_register_cells``)
    # can σ-collapse a cache-axis ``Var`` to a constant ``Literal`` in this
    # self-describing Write (the compute body is exposed via
    # ``StageBundle.nested()``, so generic index substitution reaches it) —
    # the slab is then co-filled by N sibling bundles, each writing one pinned
    # cell. The hoisted-compute materializer derives ONE iteration domain /
    # slab shape from a single Write, so it can't represent that multi-bundle
    # fill; flag it as un-lowerable. Under ``tune`` this prunes the offending
    # search branch (``Run.drive`` containment); the deployable greedy pick
    # never reaches this fork.
    bad = [v for v in write.index if not (isinstance(v, Var) and v.name in axis_map)]
    if bad:
        from deplodock.compiler.pipeline.pipeline import LoweringError  # noqa: PLC0415

        raise LoweringError(
            f"compute phase {write.output!r}: hoisted-compute Write index {write.index!r} has non-cache-axis "
            f"entr{'ies' if len(bad) > 1 else 'y'} {bad!r} (axes {tuple(axis_map)!r}) — a sibling-cell-fused "
            f"slab fill the single-Write materializer can't represent"
        )
    cache_axes = tuple(axis_map[v.name] for v in write.index)
    if not cache_axes:
        raise ValueError(f"compute phase {write.output!r}: needs at least one cache axis")
    return write.output, cache_axes, write.value_dtype


def compute_phase_extents(compute, sources) -> tuple[int, ...]:  # noqa: ANN001
    """The fused slab's **physical** per-dim extents — the input slabs' ``alloc_extents``
    (cache extent × atom-stride ``block``) when the compute reads a block-multiplied
    (warp-tier) operand slab, else the bare cache extents (scalar tier, ``block=()``).

    The SMEM fused edge (``assembly/_fused``) builds every input slab by projecting one
    operand ``Source``, so the **full** operand's slab shares the output slab's physical
    layout; iterating that full physical slab (not the logical cache cell) makes the
    producer a layout-agnostic element-wise smem→smem transform — correct for a 32×32
    warp micro-tile where the bare cache cell is only 2×2. The match is by the output
    cache axes (the full operand, not a broadcast sub-slab like ``rs[m]``/``nw[k]``). For
    the scalar tier ``alloc_extents == cache extents``, so it is a no-op there."""
    _name, out_axes, _dtype = compute_phase_info(compute, sources)
    out_names = [ax.name for ax in out_axes]
    by_name = {s.name: s for s in sources}
    for ld in compute.iter_of_type(Load):
        src = by_name.get(ld.input)
        if (
            src is not None
            and isinstance(src.addressing, AffineAddressing)
            and src.addressing.block
            and [ax.name for ax in src.cache_axes] == out_names
        ):
            return tuple(src.alloc_extents)
    return tuple(ax.extent.as_static() for ax in out_axes)


def emit_compute_phase(compute, sources, tid_expr, n_threads: int, *, buffer_count: int) -> list[Stmt]:  # noqa: ANN001
    """Emit producer scaffolding for a ``StageBundle.compute`` phase.

    Wraps the compute body in a cooperative ``StridedLoop`` over the fused slab's
    **physical** extents (:func:`compute_phase_extents` — the full warp micro-tile, not
    the logical cache cell), σ-substituting the cache-axis Vars with row-major flat-iter
    decoded coords. Because the fused edge's input and output slabs share one physical
    layout, the same physical coords index every slab → a correct element-wise fill.
    The fused slab name / cache axes are recovered from the body via
    :func:`compute_phase_info` (no output ``Source`` is carried).

    The fused slab's smem decl is hoisted to kernel scope by
    ``_materialize``'s prologue walk; the dedup filter drops the redundant
    in-line decl. Leading ``Sync`` ensures sibling-stage smem writes are
    visible; trailing ``Sync`` makes the freshly computed slab CTA-visible to
    the consumer.
    """
    from deplodock.compiler.ir.sigma import Sigma  # noqa: PLC0415

    name, cache_axes, _dtype = compute_phase_info(compute, sources)
    extents = compute_phase_extents(compute, sources)
    # Re-stamp the cache axes with the physical extents for the decode + iteration
    # (same names, so the σ-substitution still reaches the compute body's index Vars).
    axes = tuple(Axis(name=ax.name, extent=to_dim(e)) for ax, e in zip(cache_axes, extents, strict=True))
    total = 1
    for e in extents:
        total *= e
    iter_axis = Axis(name=f"{name}_flat", extent=total)
    if len(axes) == 1:
        coord_for = {axes[0].name: Var(iter_axis.name)}
    else:
        coord_for = flat_decode(axes, iter_axis.name)
    sigma = Sigma(coord_for)

    # σ-substitute cache vars in every stmt of the compute body. Loads /
    # Assigns / Writes are leaves so a flat ``map`` over the body is
    # enough; no recursion into nested bodies is needed for the cone
    # shape this pass produces.
    new_stmts: list[Stmt] = []
    for s in compute:
        new_stmts.append(s.rewrite(lambda n: n, sigma))

    full_extents = (buffer_count, *extents) if buffer_count > 1 else extents
    body_out: list[Stmt] = [
        Sync(),
        Smem(name=name, extents=full_extents),
        StridedLoop(
            axis=iter_axis,
            start=tid_expr,
            step=Literal(n_threads, "int"),
            body=tuple(new_stmts),
        ),
        Sync(),
    ]
    return body_out


def flat_decode(cache_axes: tuple[Axis, ...], flat_name: str) -> dict:
    """Row-major decode of a flat index into per-axis coordinates.

    Innermost axis: ``flat % extent``. Middle axes:
    ``(flat / inner_stride) % extent``. Outermost axis: ``flat /
    outer_stride`` (mod is redundant — flat < total)."""
    flat = Var(flat_name)
    coords: dict = {}
    inner_stride = 1
    for ax in reversed(cache_axes):
        ext = ax.extent.as_static()
        coords[ax.name] = flat % Literal(ext, "int") if inner_stride == 1 else (flat / Literal(inner_stride, "int")) % Literal(ext, "int")
        inner_stride *= ext
    outer = cache_axes[0]
    outer_stride = inner_stride // outer.extent.as_static()
    coords[outer.name] = flat if outer_stride == 1 else flat / Literal(outer_stride, "int")
    return coords
