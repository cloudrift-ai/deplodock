"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

The wrapper stays as ``Tile`` (shared with Tile IR via ``ir.stmt``);
only the body content changes — a ``StageBundle`` becomes ``Smem`` +
cooperative loads, cooperative ``Accum`` escapes become smem tree-halve /
warp-shuffle via the escape-analysis helper (``ir/tile/escape_analysis.py``),
``Loop`` / ``StridedLoop`` pass through. Two paths:

- **Non-cooperative** (no ``BIND_BLOCK`` axes): every BoundAxis is
  ``BIND_THREAD`` (pointwise / per-thread serial) — ``axes`` passed
  through, inner ``Loop``s pass through.

- **Cooperative** (one or more ``BIND_BLOCK`` axes): the Tile's THREAD
  axes are the cooperative thread set (synthesized by the strategy:
  ``cooperative-reduce`` adds a single ``t`` axis; ``blockify`` uses
  the per-block tile dims ``m_i`` / ``n_i``). Materialization passes
  ``Tile.axes`` through, computes a linear thread index ``tid_expr``
  from the THREAD axes, then walks the body:

    * ``StageBundle`` → per-source smem decl + cooperative load driven by
      ``tid_expr`` (multi-axis slabs flatten via row-major decode).
    * ``Loop`` / ``StridedLoop`` → passed through (recursive walk for
      bundle / Write handling inside).
    * Cooperative ``Accum`` (per ``accum_cooperative_axes``) → smem
      tree-halve + broadcast emitted right after the enclosing reduce.
    * ``Write`` whose index references a THREAD axis is emitted
      unconditionally (each thread owns a unique output slot). Writes
      that don't reference any THREAD axis get a ``tid==0`` guard at
      Kernel render time (``Write.render`` reads ``ctx.broadcast_writes``).

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    CpAsyncWait,
    KernelOp,
    MbarrierArrive,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    SetMaxNReg,
    Smem,
    Sync,
    TmaDescriptor,
    TmaLoad,
)
from deplodock.compiler.ir.stmt import Accum, Cond, Monoid, Stmt
from deplodock.compiler.ir.tile.ir import (
    _SWIZZLE_BY_BYTES,
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncWait,
    GridTile,
    RegisterTile,
    SerialTile,
    StageBundle,
    StagePolicy,
    StridedTile,
    SwizzleMode,
    ThreadTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
    pick_swizzle_atom,
)
from deplodock.compiler.pipeline import Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import (
    cooperative_combine_geometry,
    emit_combine,
    emit_combine_states,
    find_nested_monoids,
    find_nested_reduce_accums,
)
from deplodock.compiler.pipeline.passes.lowering.kernel._stage_expand import (
    compute_phase_extents,
    compute_phase_info,
    emit_compute_phase,
    emit_stage,
    smem_cuda_dtype,
)
from deplodock.compiler.pipeline.passes.lowering.kernel._tma_groups import partition_tma_groups

PATTERN = [Pattern("root", TileOp)]


# Standard TMA destination alignment. 16 B is the hardware minimum;
# 128 B is what NVIDIA's TMA programming guide recommends for max
# throughput on box copies.
_TMA_ALIGN_BYTES = 128


def _swizzle_align_bytes(mode: SwizzleMode) -> int:
    """Swizzled TMA smem aligns to its full swizzle atom (8 rows × width):
    B128→1024, B64→512, B32→256 — the coordinate-only ldmatrix XOR only
    reproduces the hardware deposit when the base zeroes the swizzle's
    source address bits. NONE keeps NVIDIA's 128 B box recommendation."""
    for wb, m in _SWIZZLE_BY_BYTES:
        if m == mode:
            return 8 * wb
    return _TMA_ALIGN_BYTES


def rewrite(ctx: Context, root: Node) -> Graph | None:
    escape = root.op.body.coordination
    new_body: list[Stmt] = []
    for s in root.op.body:
        if isinstance(s, (GridTile, ThreadTile, WarpTile)):
            new_body.append(_materialize_top(s, warp_size=ctx.warp_size, escape=escape))
        else:
            new_body.append(s)
    return KernelOp(body=new_body, name=root.op.name)


def _materialize_top(top: Stmt, *, warp_size: int, escape=None) -> Stmt:
    """Dispatch the outermost tile of a TileOp body to materialization.

    Three shapes are possible coming out of ``001_launch_geometry`` /
    downstream emitters:

    - ``GridTile(... body=[ThreadTile(... body=actual)])``: cooperative
      kernel (matmul / fused-reduce). The ThreadTile's body is what
      ``_materialize`` walks; the GridTile wrapper preserved unchanged
      so kernel render emits the ``blockIdx`` decode.
    - ``GridTile(... body=[WarpTile(... body=actual)])``: warp-cooperative
      kernel (MMA fragment factorization, future consumer). ``_materialize``
      walks the WarpTile body with a warp-id flatten tid_expr and
      ``n_threads = prod(warp_extents) * 32``; the rebuilt wrapper stays
      a ``WarpTile`` so the kernel renderer emits the ``warp_id`` /
      ``lane`` decode.
    - ``ThreadTile(... body=actual)``: pointwise/standalone. Materialize
      the body directly; the kernel renderer's linear-tid path handles
      launch geometry from the ThreadTile's extents.
    """
    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    if isinstance(top, GridTile):
        new_outer: list[Stmt] = []
        for child in top.body:
            if isinstance(child, (ThreadTile, WarpTile)):
                new_outer.append(_materialize(child, warp_size=warp_size, escape=escape))
            else:
                new_outer.append(child)
        return GridTile(axes=top.axes, body=Body(new_outer), swizzle_group_m=top.swizzle_group_m)
    if isinstance(top, (ThreadTile, WarpTile)):
        return _materialize(top, warp_size=warp_size, escape=escape)
    raise ValueError(f"unexpected top-level tile flavor: {type(top).__name__}")


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: ThreadTile | WarpTile, *, warp_size: int, escape=None) -> Stmt:
    """Materialize a ``ThreadTile`` or ``WarpTile`` body. The inner tile
    carries the per-CTA scope axes directly (no BoundAxis filtering
    needed); strategies set this up — this pass commits no axis
    decisions of its own.

    Two flavors:

    - ``ThreadTile`` (one coord = one thread): ``tid_expr`` is the row-
      major flatten of ``axes``; ``n_threads = prod(extents)``.
    - ``WarpTile`` (one coord = one warp, 32 lanes): ``tid_expr`` is the
      row-major flatten of warp axes (the ``warp_id`` value); ``n_threads
      = prod(extents) * 32``. Cooperative scaffolding (Stage loads, Accum
      combines) that wants per-CTA thread cooperation must address all
      threads — those code paths are reserved for the follow-up MMA
      consumer plan and do not fire on M2-shaped (Write-only) bodies.

    Strategies that need single-thread Writes (e.g. cooperative scalar
    output) wrap them in ``Cond(thread_var == 0)`` themselves —
    materialization passes Writes through unchanged."""
    axes = blk.axes
    # Each ``StageBundle`` is materialized in place: ``emit_bundle_producer``
    # emits the per-source cooperative-load scaffolding (and the optional
    # compute phase) ahead of the bundle's consumer ``body`` stmts.
    body = blk.body
    thread_axes = axes
    if not thread_axes:
        raise ValueError(f"{type(blk).__name__} must have at least one axis")
    is_warp = isinstance(blk, WarpTile)
    if is_warp:
        # WarpTile-context cooperative loads need the per-LANE thread id so
        # every lane in every warp picks a distinct slab position. Using
        # the warp_id-only form fanned only one writer per warp and left
        # 31/32 of the slab uninitialized (caught while probing a staged-MMA matmul).
        # ``Builtin("thread_idx.x")`` renders via ``ctx.builtins`` to the
        # CUDA ``threadIdx.x`` spelling and stays opaque to numerical
        # simplifiers.
        tid_expr = Builtin("thread_idx.x")
        n_threads = 32
    else:
        tid_expr = _build_linear_tid(thread_axes)
        n_threads = 1
    for ax in thread_axes:
        n_threads *= ax.extent.as_static()

    rename: dict[str, str] = {}

    def transform(s: Stmt) -> Stmt:
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    declared_smem: set[str] = set()

    # TMA hoist state. ``descriptors`` and ``mbar_prologue`` collect the
    # per-stage ``TmaDescriptor`` + per-group mbar Smem + ``MbarrierInit``
    # so the post-walk prepends them to the Tile body.
    #
    # TMA stages are partitioned into "pipeline-unit groups": one group
    # per K-loop containing TmaBufferedStages, plus any prologue stages
    # immediately before that loop and the trailing epilogue AsyncWait.
    # Each group gets its own mbarrier array
    # ``tma_mbar_<gid>[buffer_count]`` initialised with
    # ``count = num distinct stage names in the group`` — so a tile with
    # multiple K-loops over different stage sets (e.g. SDPA P@V whose
    # softmax-max + softmax-sum + weighted-V reduces have different stage
    # multiplicities) gets per-loop arrive counts and its mbar waits
    # don't deadlock. ``050_use_tma`` enforces all-or-nothing TMA
    # promotion per tile, so any tile with TMA stages is guaranteed to
    # have no cp.async stages in the same pipelined K-loop and the
    # AsyncWait lowering can stay as a pure ``MbarrierWait``.
    descriptors: dict[str, TmaDescriptor] = {}
    mbar_prologue: list[Stmt] = []
    declared_mbar: set[str] = set()

    # Smem decls inside a loop don't reach kernel scope in CUDA. Walk the
    # body once, pre-emit Smem decls at kernel scope, and mark them
    # ``declared_smem`` so per-stage emits are dedup'd. The bundle's
    # ``buffer_count`` scales the leading extent for BUFFERED/ASYNC/TMA;
    # SYNC has buffer_count == 1 (no leading slot dim).
    compute_stage_prologue: list[Stmt] = []
    for stmt in body.iter():
        if not isinstance(stmt, StageBundle):
            continue
        buf_count = stmt.buffer_count if stmt.policy != StagePolicy.SYNC else 1
        # TMA destinations need 128-byte alignment per NVIDIA's TMA programming
        # guide — ``cp.async.bulk.tensor`` against a misaligned smem dst raises
        # ``Misaligned shared or local address`` mid-kernel (compute-sanitizer
        # surfaces it; the in-flight TMA never completes, the consumer
        # mbarrier.wait spins forever, and the bench watchdog times out at
        # 1000 ms with the kernel pinned ``bench_fail``). Two things needed
        # together — base alignment AND slot-stride padding — when the ring
        # buffer's natural slot stride is < 128 B (BK·sizeof < 128 on FP32 →
        # BK < 32). The per-stage TMA emit later (``emit_tma_stage`` →
        # ``_TMA_ALIGN_BYTES``) sets ``align=128`` on its own Smem, but
        # ``filter_emit`` drops it as a name-dup of this pre-emit; without
        # both, the second slot lands at a 64 B offset from a 128 B base —
        # silently misaligned for cp.async.bulk.tensor.
        is_tma = stmt.policy == StagePolicy.TMA
        for src in stmt.sources:
            if src.name in declared_smem:
                continue
            extents = src.alloc_extents
            smem_dtype = smem_cuda_dtype(src)
            if is_tma:
                # Round the slot's inner BOX EXTENT up to make the slot
                # bytes a multiple of 128 B. The check is on the
                # COLLAPSED inner box (product of cache_axes mapping to
                # the innermost source dim), not just ``extents[-1]`` —
                # for the multi-axis collapse case (e.g. cache
                # ``(BK=32, BN_thread=32, FN_reg=4)`` mapping to dims
                # ``(0, 1, 1)``), the natural flat layout already packs
                # each K row as ``BN_thread × FN_reg = 128`` cells =
                # 512 B, well 128 B-aligned; padding the last axis (4 →
                # 32) would 8x the smem with zero correctness gain.
                # Only pad when the collapsed inner is genuinely
                # sub-128 B — single-axis ``BN=16`` etc.
                inner_per_align = _TMA_ALIGN_BYTES // BYTES_PER_ELEM
                addressing = src.addressing
                if isinstance(addressing, AffineAddressing):
                    inner_dim = addressing.dims[-1]
                    block = addressing.block
                    collapsed_inner = 1
                    for i, (d, ax) in enumerate(zip(addressing.dims, src.cache_axes, strict=True)):
                        if d == inner_dim:
                            b = block[i] if block else 1
                            collapsed_inner *= ax.extent.as_static() * b
                else:
                    collapsed_inner = extents[-1]
                if collapsed_inner % inner_per_align != 0:
                    padded_inner = (extents[-1] + inner_per_align - 1) // inner_per_align * inner_per_align
                    extents = (*extents[:-1], padded_inner)
            if buf_count > 1:
                extents = (buf_count, *extents)
            smem_align = _swizzle_align_bytes(src.swizzle) if is_tma else (16 if smem_dtype == "__half" else 0)
            compute_stage_prologue.append(Smem(name=src.name, extents=extents, dtype=smem_dtype, align=smem_align))
            declared_smem.add(src.name)
        # Hoisted-compute phase: the fused slab is no longer carried as a
        # ``Source``, so pre-emit its kernel-scope Smem decl here — right
        # after the transport sources (byte-identical first-seen order).
        # Name / cache axes / dtype are recovered from the self-describing
        # compute body (its single Write + the cone sources it reads).
        if stmt.compute is not None:
            from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

            fused_name, _fused_axes, fused_value_dtype = compute_phase_info(stmt.compute, stmt.sources)
            if fused_name not in declared_smem:
                # PHYSICAL extents (cache × atom-stride block) so the fused slab matches
                # the input operand slabs' layout — the same sizing emit_compute_phase
                # iterates (scalar tier: == cache extents).
                fused_extents = compute_phase_extents(stmt.compute, stmt.sources)
                if buf_count > 1:
                    fused_extents = (buf_count, *fused_extents)
                fused_dtype = cuda_name(fused_value_dtype) if fused_value_dtype is not None else "float"
                fused_align = 16 if fused_dtype == "__half" else 0
                compute_stage_prologue.append(Smem(name=fused_name, extents=fused_extents, dtype=fused_dtype, align=fused_align))
                declared_smem.add(fused_name)

    tma = partition_tma_groups(body)

    def filter_emit(stmts: list[Stmt]) -> list[Stmt]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Smem):
                if s.name in declared_smem:
                    continue
                declared_smem.add(s.name)
            out.append(s)
        return out

    def emit_async_wait(stmt: AsyncWait, *, ws_consumer: WarpSpecialize | None = None) -> list[Stmt]:
        # TMA path: wait carries the explicit consumer-side phase + slot
        # set by 015_pipeline_k_outer. The wait targets its pipeline-unit
        # group's mbar (each group has its own mbar with arrive count
        # == num distinct stages in that group). A trailing ``Sync()``
        # backs up the mbarrier's CTA-wide visibility guarantee — nvcc
        # treats the wait's inline-PTX asm as opaque, so without an
        # explicit ``__syncthreads()`` the compiler is free to reorder
        # smem Loads across iterations of the K loop, reading stale
        # bytes from the previous iter's slot. Surfaces on small tiles
        # (BM=16, BN=16 + stage) where the inner-loop schedule makes
        # the reorder profitable; on larger tiles the schedule is
        # dense enough that no useful hoist is possible.
        # The trailing fence routes to a named ``bar.sync`` when we're
        # inside a WarpSpecialize consumer subtree (``__syncthreads()``
        # is CUDA UB on the warp-divergent producer/consumer branch).
        # ``ws_consumer`` is threaded in by ``emit_warp_specialize`` and
        # carries the consumer-thread-axis structure, from which the
        # materializer derives the consumer participant count for
        # ``bar.sync N, M`` — product of the WarpSpecialize's
        # ``consumer_thread_axes.extent``.
        if ws_consumer is not None:
            n_cons = 1
            for ax in ws_consumer.consumer_thread_axes:
                n_cons *= ax.extent.as_static()
            # Warp-tier consumer axes count *warps* (32 lanes each), so the
            # ``bar.sync`` participant count must be warps × warp_size — a bare
            # warp count is not a multiple of 32 and ptxas rejects it.
            if ws_consumer.consumer_is_warp:
                n_cons *= warp_size
            trailing_sync = Sync(barrier_id=1, count=n_cons)
        else:
            trailing_sync = Sync()
        if stmt.phase is not None and tma.has_tma:
            gid = tma.wait_group.get(id(stmt))
            if gid is not None:
                return [MbarrierWait(mbar=tma.mbar_name(gid), phase=stmt.phase, slot=stmt.slot), trailing_sync]
        # cp.async fallback (or pre-pipelining synchronous-style wait,
        # or AsyncWait whose pipeline group couldn't be inferred).
        return [CpAsyncWait(group=stmt.keep), trailing_sync]

    def emit_tma_stage(bundle: StageBundle, src) -> list[Stmt]:  # noqa: ANN001 — src: Source
        # One TMA box-copy per source: the materializer emits a descriptor +
        # elected-thread arrive per ``Source`` directly (no per-source stage
        # wrapper), all arriving against the bundle's group mbarrier.
        addressing = src.addressing
        assert isinstance(addressing, AffineAddressing), f"TMA stage source {src.name!r} must use AffineAddressing"
        desc_name = f"{src.name}_desc"
        # Key by smem name (Source.name) — id(bundle) isn't stable across
        # Body.map's wrapper-rebuild on descent, but Source.name is set
        # by 020_stage_inputs and survives every downstream rewrite.
        gid = tma.stage_group[src.name]
        mbar_name = tma.mbar_name(gid)
        # Box extents per source dim: product of (cache_extent × block) of
        # every cache axis mapping to that dim (multiple cache axes can sweep
        # the same source dim — e.g. an outer-block axis and a per-thread
        # fragment axis both decoded into the M dim of a matmul slab). The
        # per-axis ``AffineAddressing.block`` multiplier encodes per-cell
        # strides (e.g. ``atom_n = 8`` for the m16n8k16 atom), which the eligibility
        # check at 050_use_tma.py:_stage_eligible already mirrors. Without
        # threading block through here the TMA descriptor's box width would
        # under-report the actual slab inner width for warp-tier MMA slabs.
        # Source dims not covered by any cache axis get extent 1.
        dims = addressing.dims
        block = addressing.block
        box_per_dim: dict[int, int] = {}
        for i, (d, ax) in enumerate(zip(dims, src.cache_axes, strict=True)):
            b = block[i] if block else 1
            box_per_dim[d] = box_per_dim.get(d, 1) * ax.extent.as_static() * b
        full_box = tuple(box_per_dim.get(d, 1) for d in range(len(src.origin)))
        # Drop *gap* inert dims (``box == 1`` AND ``origin`` is literal
        # 0) between the first and last swept source dims. Leading
        # singletons stay in the descriptor — keeping them matches the
        # working linear-matmul rank-3 shape. Gap singletons (e.g. GQA
        # V's kv_head=1 between seq and head_dim) must be dropped to
        # avoid the rank-4 pipelined-TMA deadlock at seq=512. The
        # runtime encoder reconstructs the same collapse from
        # ``arr.shape`` + ``box_extents`` alone.
        outer, inner = dims[0], dims[-1]
        kept = tuple(
            d
            for d in range(len(src.origin))
            if d < outer
            or d > inner
            or d in dims
            or not (full_box[d] == 1 and isinstance(src.origin[d], Literal) and int(src.origin[d].value) == 0)
        )
        box = tuple(full_box[d] for d in kept)
        coords = tuple(src.origin[d] for d in kept)
        # Swizzle-atom box reshape: split the innermost box dim down to the
        # swizzle atom width so the descriptor's innermost dim in bytes equals
        # the swizzle width (TMA rejects swizzle when the inner box-dim byte
        # span exceeds the atom). Gated on ``src.swizzle`` — the per-Source
        # mode 050_use_tma stamps (B64 / B128 on mma.sync sources, NONE
        # otherwise), so the box reshape and the descriptor's swizzle mode are
        # driven by the same field and can't disagree. The box rank grows by
        # one; the runtime ``_collapse_inert_dims`` reconstructs the matching
        # rank+1 globalDim by splitting the array's inner dim. The matching
        # ldmatrix consumer XOR is emitted by 005_lower_atom_tile.
        if src.swizzle != SwizzleMode.NONE and box:
            elem_bytes = src.dtype.nbytes if src.dtype is not None else BYTES_PER_ELEM
            inner_elems = box[-1]
            atom_elems, _ = pick_swizzle_atom(inner_elems, elem_bytes)
            if atom_elems < inner_elems:
                inner_coord = coords[-1]
                outer_coord = BinaryExpr("/", inner_coord, Literal(atom_elems, "int"))
                box = (*box[:-1], inner_elems // atom_elems, atom_elems)
                coords = (*coords[:-1], outer_coord, Literal(0, "int"))
        if desc_name not in descriptors:
            # Source shape is unknown at materialization time — the
            # backend resolves it from the bound array at launch.
            descriptors[desc_name] = TmaDescriptor(
                name=desc_name,
                src_buf=src.buf,
                src_shape=(),
                box_extents=box,
                # Per-Source swizzle: A (64 B inner) and B (128 B inner) share
                # one bundle but need distinct modes, so the mode rides on the
                # Source (there is no bundle-level mode). 050_use_tma stamps
                # B64 / B128 here on every mma.sync source; it stays NONE for
                # the non-swizzled TMA paths (e.g. SDPA).
                swizzle=src.swizzle.value,
                dtype=smem_cuda_dtype(src),
            )
        if mbar_name not in declared_mbar:
            declared_mbar.add(mbar_name)
            # Per-group mbarrier array: one mbar per ring-buffer slot,
            # with arrive count = number of distinct stages in *this*
            # group (so all of the group's stages must arrive before its
            # phase flips).
            bc = tma.group_buffer_count[gid]
            mbar_prologue.append(
                Smem(name=mbar_name, extents=(bc,), dtype="unsigned long long"),
            )
            for s in range(bc):
                mbar_prologue.append(MbarrierInit(mbar=mbar_name, count=tma.arrive_count(gid), slot=Literal(s, "int")))
        # Use the stamped source dtype's byte count so TMA arrive-expect
        # bytes match the actual copy size on fp16 inputs (legacy
        # ``BYTES_PER_ELEM`` over-counted fp16 by 2x). Multiply through the
        # per-axis ``AffineAddressing.block`` multiplier for the same reason
        # ``box_per_dim`` does — without it, warp-tier MMA slabs report a
        # ``slab_bytes`` of ``∏ cache_extents`` (e.g. 2·2·4 = 16 elements =
        # 32 B for fp16) while the actual TMA box delivers
        # ``∏ cache_extents · ∏ block`` (e.g. 32·128 = 4096 elements =
        # 8192 B). The mbarrier's ``arrive.expect_tx`` would then expect
        # 32 B and never satisfy the actual ``complete_tx::bytes`` payload,
        # hanging the consumer's ``mbarrier.wait``.
        slab_bytes = src.dtype.nbytes if src.dtype is not None else BYTES_PER_ELEM
        block = addressing.block
        for i, ax in enumerate(src.cache_axes):
            b = block[i] if block else 1
            slab_bytes *= ax.extent.as_static() * b
        # Smem allocation: leading phase dim + cache extents (with pad).
        # buffer_count / phase live on the StageBundle now (collapsed Stage
        # hierarchy — Stage.{buffer_count,phase,swizzle} are no longer
        # carried per-Stage; pull them from the enclosing bundle).
        full_extents = (bundle.buffer_count, *src.alloc_extents)
        smem_index = (bundle.phase, *([Literal(0, "int")] * len(src.cache_axes)))
        # Distribute issuer threads across stages within a group so each
        # stage's arrive+TMA pair issues from a different thread (stage
        # 0 → tid 0, stage 1 → tid 1, ...) rather than serializing on tid 0.
        cond = Cond(
            cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(tma.issuer_tid[src.name], "int")),
            body=(
                MbarrierArriveExpectTx(mbar=mbar_name, bytes_=slab_bytes, slot=bundle.phase),
                TmaLoad(
                    smem=src.name,
                    smem_index=smem_index,
                    desc=desc_name,
                    coords=coords,
                    mbar=mbar_name,
                    mbar_slot=bundle.phase,
                ),
            ),
        )
        # Swizzled TMA smem aligns to its full swizzle atom (8 rows × width):
        # B128→1024, B64→512, B32→256. The coordinate-only ldmatrix XOR (emitted
        # by 005_lower_atom_tile) only reproduces the hardware's absolute-address
        # swizzle when the buffer base zeroes the swizzle's source address bits.
        # Non-swizzled (NONE) keeps NVIDIA's 128 B box-copy recommendation.
        align = _swizzle_align_bytes(src.swizzle)
        smem_dtype = smem_cuda_dtype(src)
        out: list[Stmt] = [
            Smem(name=src.name, extents=full_extents, align=align, dtype=smem_dtype),
            cond,
        ]
        # No trailing wait here — for unpipelined multi-stage bundles
        # the mbarrier is shared across stages with ``arrive_count = N``,
        # so a wait emitted between stages would block forever (only one
        # arrive has happened by then). ``emit_bundle_producer`` emits
        # one wait+sync after ALL stages in the bundle for the
        # unpipelined case.
        return out

    def emit_bundle_producer(bundle: StageBundle) -> list[Stmt]:
        """Emit producer-side scaffolding for a bundle. Dispatch:

        - ``bundle.policy == TMA`` → one TMA box copy per source.
        - otherwise (SYNC / BUFFERED / ASYNC) → cooperative
          ``Load + Write`` (or ``CpAsyncCopy`` for ASYNC) over all sources
          behind one barrier pair.

        Then, if ``bundle.compute`` is set, the hoisted-invariant
        cooperative compute phase is emitted after the transport sources
        (reads the just-staged sibling slabs, fills the fused slab).

        For unpipelined TMA bundles (``pipeline_depth == 1``) with multiple
        sources, the group mbarrier is shared with ``arrive_count = N`` (one
        per source), so a single trailing ``MbarrierWait`` after every source
        has arrived/issued is what flips the phase. Emitting the wait inside
        ``emit_tma_stage`` would deadlock (only one arrive has happened).
        """
        out: list[Stmt] = []
        if bundle.policy == StagePolicy.TMA:
            for src in bundle.sources:
                out.extend(emit_tma_stage(bundle, src))
        else:
            out.extend(
                emit_stage(
                    bundle.sources,
                    tid_expr,
                    n_threads,
                    policy=bundle.policy,
                    buffer_count=bundle.buffer_count,
                    phase=bundle.phase,
                    pipeline_depth=bundle.pipeline_depth,
                )
            )
        if bundle.compute is not None:
            out.extend(emit_compute_phase(bundle.compute, bundle.sources, tid_expr, n_threads, buffer_count=bundle.buffer_count))
        # Unpipelined TMA: trailing wait+sync once after all sources
        # arrived (mbarrier ``arrive_count = N`` is met).
        if bundle.policy == StagePolicy.TMA and bundle.pipeline_depth == 1:
            tma_srcs = list(bundle.sources)
            if tma_srcs:
                first_src = tma_srcs[0]
                gid = tma.stage_group[first_src.name]
                mbar_name = tma.mbar_name(gid)
                mbar_phase = _mbar_wait_phase(bundle.phase, bundle.buffer_count)
                out.append(MbarrierWait(mbar=mbar_name, phase=mbar_phase, slot=bundle.phase))
                out.append(Sync())
        return out

    def emit_warp_specialize(ws: WarpSpecialize) -> list[Stmt]:
        """Lower a Tile-IR ``WarpSpecialize`` into the full mbarrier
        handshake: empty-mbarrier ring + per-K_o ``MbarrierWait`` /
        ``MbarrierArrive`` pairs + named ``bar.sync`` consumer fences +
        ``SetMaxNReg`` register-budget redistribution + producer /
        consumer ``Cond`` wrapper.

        Post-WarpTile-refactor shape:

        - The enclosing ``WarpTile`` carries a single ``role`` axis (warp-
          granularity coord ∈ ``[0, total_warps)``). We read its name from
          ``thread_axes[0]`` (``_materialize`` passes the WarpTile.axes
          through to its caller-named ``thread_axes`` slot).
        - The role-split ``Cond`` predicate is ``Var(role) <
          n_producer_warps`` — structural, no arithmetic shift.
        - The consumer branch wraps the materialized stmts in a
          ``ThreadTile(consumer_thread_axes, tid_offset=n_producer_threads)``
          so the renderer emits ``int <axis> = ((threadIdx.x - n_producer)
          / stride) % extent`` per consumer axis. The consumer body
          references the original axis Vars unshifted; the nested
          ThreadTile rebinds them to the consumer-relative tid range.
        """
        from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

        # Role axis = the single axis on the enclosing WarpTile.
        role = thread_axes[0]
        consumer_thread_axes = ws.consumer_thread_axes
        if not consumer_thread_axes:
            raise ValueError("WarpSpecialize.consumer_thread_axes must be non-empty for the WarpTile-based materializer arm")
        n_consumer_threads = 1
        for ax in consumer_thread_axes:
            n_consumer_threads *= ax.extent.as_static()
        # Warp-tier consumer axes count *warps* (32 lanes each), so the named
        # ``bar.sync N, count`` participant count is warps × warp_size.
        if ws.consumer_is_warp:
            n_consumer_threads *= warp_size
        if ws.n_producer_threads % warp_size != 0:
            raise ValueError(f"WarpSpecialize: n_producer_threads ({ws.n_producer_threads}) must be a multiple of warp_size ({warp_size})")
        n_producer_warps = ws.n_producer_threads // warp_size
        first_consumer_tid = ws.n_producer_threads
        bc = ws.ring_depth
        empty_mbar = "tma_mbar_empty"

        # Empty-mbarrier prologue: Smem + per-slot MbarrierInit (single-
        # thread gated) + Sync. Init count = 1 — exactly one consumer
        # thread arrives per slot release.
        empty_decl = Smem(name=empty_mbar, extents=(bc,), dtype="unsigned long long")
        empty_inits = tuple(MbarrierInit(mbar=empty_mbar, count=1, slot=Literal(s, "int")) for s in range(bc))
        init_cond = Cond(
            cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(0, "int")),
            body=empty_inits,
        )

        # Wire producer-side wait + consumer-side arrive into each
        # branch's serial_outer K_o body. Each helper reads k_var off the
        # matched SerialTile.axis (post-normalize canonical names like
        # ``a2`` rather than the WS-pass-time ``k_outer``).
        wired_producer = _wire_producer_wait(ws.producer_body, empty_mbar, bc)
        wired_consumer = _wire_consumer_arrive(ws.consumer_body, empty_mbar, bc, first_consumer_tid, n_consumer_threads)

        # Materialize each branch. Consumer-side AsyncWaits route their
        # trailing fence through the named ``bar.sync 1, n_consumer``
        # path — pass ws_consumer=ws into emit_async_wait via the ews
        # plumbing.
        def cons_ews(stmt: AsyncWait) -> list[Stmt]:
            return emit_async_wait(stmt, ws_consumer=ws)

        prod_out: list[Stmt] = []
        for s in wired_producer:
            prod_out.extend(_process_stmt(s))
        cons_out: list[Stmt] = []
        for s in wired_consumer:
            cons_out.extend(_process_stmt(s, ews=cons_ews))

        # Consumer-side body wraps in a tid-offset tile so the original consumer
        # axes decode against ``(threadIdx.x - n_producer_threads)``. The
        # warp-tier MMA consumer uses a ``WarpTile`` (``warp_id = (threadIdx.x -
        # off) / 32`` + ``lane``), the scalar path a ``ThreadTile``.
        decode_cls = WarpTile if ws.consumer_is_warp else ThreadTile
        consumer_decode = decode_cls(
            axes=consumer_thread_axes,
            body=Body(tuple(cons_out)),
            tid_offset=ws.n_producer_threads,
        )

        ws_cond = Cond(
            cond=BinaryExpr("<", Var(role.name), Literal(n_producer_warps, "int")),
            body=Body((SetMaxNReg(24, "dec"), *prod_out)),
            else_body=Body((SetMaxNReg(240, "inc"), consumer_decode)),
        )
        return [empty_decl, init_cond, Sync(), ws_cond]

    def _wire_producer_wait(stmts, empty_mbar: str, bc: int):
        """Prepend a ``MbarrierWait(empty[slot], phase)`` inside each
        top-level ``SerialTile(serial_outer)``, gated by ``Cond(K_o >=
        bc-1)`` so the first ``bc-1`` iters skip (those slots were
        unfilled by the prologue). The K_o axis name is read off the
        matched ``SerialTile.axis`` (canonical post-normalize)."""
        from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, SerialTile) and s.kind == "serial_outer":
                k_var = s.axis.name
                k_plus_1 = Var(k_var) + Literal(1, "int")
                slot_expr = k_plus_1 % Literal(bc, "int")
                phase_expr = (k_plus_1 / Literal(bc, "int") - Literal(1, "int")) % Literal(2, "int")
                wait_cond = Cond(
                    cond=BinaryExpr(">=", Var(k_var), Literal(bc - 1, "int")),
                    body=(MbarrierWait(mbar=empty_mbar, phase=phase_expr, slot=slot_expr),),
                )
                new_inner = Body((wait_cond, *s.body))
                out.append(s.with_bodies((new_inner,)))
            else:
                out.append(s)
        return out

    def _wire_consumer_arrive(stmts, empty_mbar: str, bc: int, first_cons_tid: int, n_cons: int):
        """Recursively descend stmts; inside every
        ``SerialTile(serial_outer)``, append a named
        ``Sync(barrier_id=1, count=n_cons)`` + a single-thread
        ``MbarrierArrive``. The Sync is critical: without it the chosen
        arriving thread (``threadIdx.x == first_cons_tid``) can race
        ahead of slower consumers still reading the slot's smem; the
        arrive then flips the empty mbarrier and the producer is free
        to overwrite the slot mid-read."""
        from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

        def _augment(stmt: Stmt) -> Stmt:
            if isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
                k_var = stmt.axis.name
                slot_expr = Var(k_var) % Literal(bc, "int")
                barrier_sync = Sync(barrier_id=1, count=n_cons)
                arrive_cond = Cond(
                    cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(first_cons_tid, "int")),
                    body=(MbarrierArrive(mbar=empty_mbar, slot=slot_expr),),
                )
                return stmt.with_bodies((Body((*stmt.body, barrier_sync, arrive_cond)),))
            nested = stmt.nested()
            if nested:
                return stmt.with_bodies(tuple(Body(tuple(_augment(c) for c in b)) for b in nested))
            return stmt

        return [_augment(s) for s in stmts]

    def _process_stmt(stmt: Stmt, *, ews=None) -> list[Stmt]:
        """Materialize one body Stmt — bundles inline their producer code
        then recursively process their consumer body.

        ``ews`` overrides the per-walk ``emit_async_wait`` flavor — used
        by ``emit_warp_specialize`` to thread a ``ws_consumer`` context
        through the consumer subtree so AsyncWaits route their trailing
        fence to the named ``bar.sync``."""
        if ews is None:
            ews = emit_async_wait
        if isinstance(stmt, WarpSpecialize):
            return list(emit_warp_specialize(stmt))
        if isinstance(stmt, StageBundle):
            out_local: list[Stmt] = list(filter_emit(emit_bundle_producer(stmt)))
            for inner in stmt.body:
                out_local.extend(_process_stmt(inner, ews=ews))
            return out_local
        if isinstance(stmt, AsyncWait):
            return list(ews(stmt))
        if isinstance(stmt, Cond):
            # Top-level Cond wrapping StageBundles / AsyncWaits — both
            # branches may contain stmts that need recursive processing.
            # (Pre-WarpSpecialize 085 emitted bare role-split Conds here;
            # left in place because user-emitted Conds and other rules
            # may still reach this point.)
            from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

            body_out: list[Stmt] = []
            for inner in stmt.body:
                body_out.extend(_process_stmt(inner, ews=ews))
            else_out: list[Stmt] = []
            for inner in stmt.else_body:
                else_out.extend(_process_stmt(inner, ews=ews))
            return [Cond(cond=stmt.cond, body=Body(tuple(body_out)), else_body=Body(tuple(else_out)))]
        if isinstance(stmt, (SerialTile, StridedTile, RegisterTile)):
            single = _emit_loop(stmt, tid_expr, n_threads, transform, filter_emit, emit_bundle_producer, ews)
            extra: list[Stmt] = []
            if isinstance(stmt, (SerialTile, StridedTile)) and stmt.is_reduce:
                accums_in_scope = {a.name: a for a in stmt.body if isinstance(a, Accum)}
                combines_in_scope = {c.state[0]: c for c in stmt.body if isinstance(c, Monoid)}
            else:
                accums_in_scope = find_nested_reduce_accums(stmt.body)
                combines_in_scope = find_nested_monoids(stmt.body)
            for acc_name, acc in accums_in_scope.items():
                coop_names = escape.accum_cooperative_axes.get(acc_name) if escape is not None else None
                if coop_names:
                    tid_var, n_coop = cooperative_combine_geometry(thread_axes, coop_names, warp_size=warp_size)
                    dt = acc.dtype or F32
                    extra.extend(emit_combine(acc_name, acc.op, tid_var, n_coop, dt, warp_size=warp_size))
                    rename[acc_name] = f"{acc_name}_b"
            # Monoid (Monoid) carriers: the tuple-valued cross-thread combine via
            # combine_states. Keyed by the first state name; reassigns the state in
            # place (no _b rename — the butterfly / tree leaves every thread holding
            # the full reduction in the carried SSA names).
            for first_state, carrier in combines_in_scope.items():
                coop_names = escape.accum_cooperative_axes.get(first_state) if escape is not None else None
                if coop_names:
                    tid_var, n_coop = cooperative_combine_geometry(thread_axes, coop_names, warp_size=warp_size)
                    extra.extend(emit_combine_states(carrier, tid_var, n_coop, warp_size=warp_size))
            return [single, *extra]
        if isinstance(stmt, Accum):
            out_local = [transform(stmt)]
            coop_names = escape.accum_cooperative_axes.get(stmt.name) if escape is not None else None
            if coop_names:
                tid_var, n_coop = cooperative_combine_geometry(thread_axes, coop_names, warp_size=warp_size)
                dt = stmt.dtype or F32
                out_local.extend(emit_combine(stmt.name, stmt.op, tid_var, n_coop, dt, warp_size=warp_size))
                rename[stmt.name] = f"{stmt.name}_b"
            return out_local
        if isinstance(stmt, Monoid):
            out_local = [transform(stmt)]
            coop_names = escape.accum_cooperative_axes.get(stmt.state[0]) if escape is not None else None
            if coop_names:
                tid_var, n_coop = cooperative_combine_geometry(thread_axes, coop_names, warp_size=warp_size)
                out_local.extend(emit_combine_states(stmt, tid_var, n_coop, warp_size=warp_size))
            return out_local
        return [transform(stmt)]

    for stmt in body:
        new_body.extend(_process_stmt(stmt))

    # Init scoping for accumulators is handled by the upstream
    # ``020_place_inits`` pass — explicit ``Init`` Stmts already sit at
    # the correct scope (Tile body head for reduce-only nesting, inside
    # a free Loop body when one wraps the Accum). Materialize is purely
    # mechanical from here.
    #
    # TMA prologue: descriptor decls (declarative — render to nothing in
    # body) + mbar smem + single-thread MbarrierInit + Sync. Hoisted to
    # the head of the Tile so every consumer of the mbar sees an
    # initialized barrier.
    if descriptors or mbar_prologue:
        mbar_smems = [s for s in mbar_prologue if isinstance(s, Smem)]
        mbar_inits = [s for s in mbar_prologue if isinstance(s, MbarrierInit)]
        prologue: list[Stmt] = [*descriptors.values(), *mbar_smems]
        if mbar_inits:
            prologue.append(Cond(cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(0, "int")), body=tuple(mbar_inits)))
            prologue.append(Sync())
        new_body = prologue + new_body
    if compute_stage_prologue:
        new_body = compute_stage_prologue + new_body
    # Redundant-Sync cleanup runs as the separate Kernel-IR pass
    # ``110_drop_redundant_syncs`` after this lowering. The inner-tile
    # flavor is preserved so kernel render emits the matching coord
    # decode (threadIdx vs warp_id + lane).
    if is_warp:
        return WarpTile(axes=axes, body=new_body)
    return ThreadTile(axes=axes, body=new_body)


def _emit_loop(loop, tid_expr, n_threads, transform, filter_emit, emit_bundle_producer, emit_async_wait) -> Stmt:
    """Translate a body SerialTile / StridedTile / RegisterTile. Recurses
    so nested staging / loops / writes inside the body get the same
    uniform treatment. The wrapper type is preserved — strategies
    decided the iteration shape; materialization just walks.

    ``filter_emit`` dedupes ``Smem`` decls by name across the whole
    KernelOp body — software-pipelined ``AsyncBufferedStage``s share
    a buffer name with their prologue counterparts, and only the first
    decl should reach the rendered kernel.

    ``emit_tma_stage`` / ``emit_async_wait`` are closures that share
    TMA hoist + active-mbar state with the top-level walker."""

    def materialize_leaf(s: Stmt):
        if isinstance(s, StageBundle):
            # Producer-side stmts followed by the bundle's consumer body
            # inlined as siblings. Bundle body has already been mapped
            # post-order, so its contents are kernel-IR.
            return [*filter_emit(emit_bundle_producer(s)), *s.body]
        if isinstance(s, AsyncWait):
            return emit_async_wait(s)
        if s.nested():
            # Block wrapper (SerialTile / StridedTile / RegisterTile / Cond / ...) —
            # body.map has already mapped its child bodies post-order.
            return s
        return transform(s)

    return loop.with_bodies((loop.body.map(materialize_leaf),))


def _build_linear_tid(thread_axes: tuple[Axis, ...]):
    """Linear row-major thread index from the THREAD axes.

    Single-axis (softmax) → ``Var(name)``.
    Multi-axis (matmul) → ``m_i * BN + n_i`` for ``(m_i, n_i)``."""
    if len(thread_axes) == 1:
        return Var(thread_axes[0].name)
    inner_stride = 1
    parts: list = []
    for ax in reversed(thread_axes):
        ext = ax.extent.as_static()
        if inner_stride == 1:
            parts.append(Var(ax.name))
        else:
            parts.append(Var(ax.name) * Literal(inner_stride, "int"))
        inner_stride *= ext
    expr = parts[0]
    for p in parts[1:]:
        expr = p + expr
    return expr


def _build_warp_id_expr(warp_axes: tuple[Axis, ...]):
    """Linear row-major warp index from the WARP axes — the warp granularity
    counterpart of :func:`_build_linear_tid`.

    Single-axis → ``Var(name)``; multi-axis → ``m_w * N_w + n_w`` (row-
    major). Lane is implicit — the renderer emits ``int lane = threadIdx.x &
    31;`` unconditionally inside ``WarpTile.render``. Callers that need
    a single linear *thread* id keep using ``threadIdx.x`` directly (it's
    a builtin).
    """
    # Same row-major flatten as _build_linear_tid; share the implementation
    # to keep the warp-axis decode and thread-axis decode shapes aligned.
    return _build_linear_tid(warp_axes)


# ---------------------------------------------------------------------------
# TMA mbarrier phase
# ---------------------------------------------------------------------------


def _mbar_wait_phase(stage_phase, buffer_count: int):
    """Derive the mbarrier-test phase from a ``TmaBufferedStage.phase``.

    050_use_tma stamps ``stage.phase = Var(K_o.name) % buffer_count``
    (the ring slot). The matching mbarrier phase rotates one bit per
    full ring sweep — ``(K_o / buffer_count) % 2``. For the degenerate
    single-shot case (``phase`` is a literal) the mbar phase starts at
    0 and never flips.
    """
    if isinstance(stage_phase, BinaryExpr) and stage_phase.op == "%":
        k_expr = stage_phase.left
        return BinaryExpr("%", BinaryExpr("/", k_expr, Literal(buffer_count, "int")), Literal(2, "int"))
    return Literal(0, "int")
