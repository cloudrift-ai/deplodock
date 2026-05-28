"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

The wrapper stays as ``Tile`` (shared with Tile IR via ``ir.stmt``);
only the body content changes — ``Stage`` becomes ``Smem`` + cooperative
load, cooperative ``Accum`` escapes become smem tree-halve / warp-shuffle
via the escape-analysis helper (``ir/tile/escape_analysis.py``),
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

    * ``Stage`` → smem decl + cooperative load driven by ``tid_expr``
      (multi-axis stages flatten via row-major decode).
    * ``Loop`` / ``StridedLoop`` → passed through (recursive walk for
      Stage / Write handling inside).
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
    TreeHalve,
)
from deplodock.compiler.ir.stmt import Accum, Cond, Stmt
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncWait,
    GridTile,
    RegisterTile,
    SerialTile,
    Stage,
    StageBundle,
    StagePolicy,
    StridedTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import emit_combine, find_nested_reduce_accums, single_thread_var
from deplodock.compiler.pipeline.passes.lowering.kernel._stage_expand import emit_compute_stage, emit_stage, smem_cuda_dtype
from deplodock.compiler.pipeline.passes.lowering.kernel._tma_groups import partition_tma_groups

PATTERN = [Pattern("root", TileOp)]


# Standard TMA destination alignment. 16 B is the hardware minimum;
# 128 B is what NVIDIA's TMA programming guide recommends for max
# throughput on box copies.
_TMA_ALIGN_BYTES = 128


def rewrite(ctx: Context, root: Node) -> Graph | None:
    escape = root.op.body.coordination
    ws = int(root.op.knobs.get("WS", 0))

    new_body: list[Stmt] = []
    for s in root.op.body:
        if isinstance(s, (GridTile, ThreadTile)):
            new_body.append(
                _materialize_top(s, warp_size=ctx.warp_size, escape=escape, ws_producer_warps=1 if ws else 0)
            )
        else:
            new_body.append(s)

    return KernelOp(body=new_body, name=root.op.name)


def _materialize_top(top: Stmt, *, warp_size: int, escape=None, ws_producer_warps: int = 0) -> Stmt:
    """Dispatch the outermost tile of a TileOp body to materialization.

    Two shapes are possible coming out of ``001_launch_geometry``:

    - ``GridTile(... body=[ThreadTile(... body=actual)])``: cooperative
      kernel (matmul / fused-reduce). The ThreadTile's body is what
      ``_materialize`` walks; the GridTile wrapper preserved unchanged
      so kernel render emits the ``blockIdx`` decode.
    - ``ThreadTile(... body=actual)``: pointwise/standalone. Materialize
      the body directly; the kernel renderer's linear-tid path handles
      launch geometry from the ThreadTile's extents.

    ``ws_producer_warps`` (default 0 = WS off) selects the warp-specialized
    materializer path. When non-zero the ThreadTile body lowers to a
    ``Cond(warp < P)`` split: producer warps issue TMA, consumer warps
    wait + reduce, coordinated through ``tma_mbar`` (full) +
    ``tma_mbar_empty`` (empty) mbarrier pairs.
    """
    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    materializer = _materialize_ws if ws_producer_warps > 0 else _materialize
    kwargs = {"warp_size": warp_size, "escape": escape}
    if ws_producer_warps > 0:
        kwargs["producer_warps"] = ws_producer_warps

    if isinstance(top, GridTile):
        # Locate the (sole) ThreadTile child.
        new_outer: list[Stmt] = []
        for child in top.body:
            if isinstance(child, ThreadTile):
                new_outer.append(materializer(child, **kwargs))
            else:
                new_outer.append(child)
        return GridTile(axes=top.axes, body=Body(new_outer))
    if isinstance(top, ThreadTile):
        return materializer(top, **kwargs)
    raise ValueError(f"unexpected top-level tile flavor: {type(top).__name__}")


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: ThreadTile, *, warp_size: int, escape=None) -> Stmt:
    """Materialize a ThreadTile body. The ThreadTile carries the per-CTA
    thread axes directly (no BoundAxis filtering needed); strategies set
    this up — this pass commits no axis decisions of its own.

    Strategies that need single-thread Writes (e.g. cooperative scalar
    output) wrap them in ``Cond(thread_var == 0)`` themselves —
    materialization passes Writes through unchanged."""
    axes = blk.axes
    # Wrap-body Stages are already flattened to the legacy shape
    # ([Stage(empty body), *consumer_stmts]) by ``090_flatten_wrap_stages``,
    # so the walker sees producer scaffolding (Stage.sources) followed by
    # the consumer stmts as siblings.
    body = blk.body
    thread_axes = axes
    if not thread_axes:
        raise ValueError("ThreadTile must have at least one axis")
    tid_expr = _build_linear_tid(thread_axes)
    n_threads = 1
    for ax in thread_axes:
        n_threads *= ax.extent.as_static()

    rename: dict[str, str] = {}

    # TMA stages always emit with ``swizzle=NONE`` — the post-refactor
    # pipeline doesn't carry a swizzle-picker pass (012 was dropped).

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
        for member in stmt.stages:
            for src in member.sources:
                if src.name in declared_smem:
                    continue
                extents = src.alloc_extents
                if buf_count > 1:
                    extents = (buf_count, *extents)
                smem_dtype = smem_cuda_dtype(src)
                smem_align = 16 if smem_dtype == "__half" else 0
                compute_stage_prologue.append(Smem(name=src.name, extents=extents, dtype=smem_dtype, align=smem_align))
                declared_smem.add(src.name)

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

    def emit_async_wait(stmt: AsyncWait) -> list[Stmt]:
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
        if stmt.phase is not None and tma.has_tma:
            gid = tma.wait_group.get(id(stmt))
            if gid is not None:
                return [MbarrierWait(mbar=tma.mbar_name(gid), phase=stmt.phase, slot=stmt.slot), Sync()]
        # cp.async fallback (or pre-pipelining synchronous-style wait,
        # or AsyncWait whose pipeline group couldn't be inferred).
        return [CpAsyncWait(group=stmt.keep), Sync()]

    def emit_tma_stage(bundle: StageBundle, stage: Stage) -> list[Stmt]:
        # 050_use_tma only promotes single-Source bundles, so the box-copy
        # here issues exactly one TMA load per group activation per member.
        assert len(stage.sources) == 1, f"TMA stage requires one Source, got {len(stage.sources)}"
        src = stage.sources[0]
        addressing = src.addressing
        assert isinstance(addressing, AffineAddressing), f"TMA stage source {src.name!r} must use AffineAddressing"
        desc_name = f"{src.name}_desc"
        gid = tma.stage_group[id(bundle)]
        mbar_name = tma.mbar_name(gid)
        # Box extents per source dim: product of cache extents that map
        # to that dim (multiple cache axes can sweep the same source dim
        # — e.g. an outer-block axis and a per-thread fragment axis both
        # decoded into the M dim of a matmul slab). Source dims not
        # covered by any cache axis get extent 1.
        dims = addressing.dims
        box_per_dim: dict[int, int] = {}
        for d, ax in zip(dims, src.cache_axes, strict=True):
            box_per_dim[d] = box_per_dim.get(d, 1) * ax.extent.as_static()
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
        if desc_name not in descriptors:
            # Source shape is unknown at materialization time — the
            # backend resolves it from the bound array at launch.
            descriptors[desc_name] = TmaDescriptor(
                name=desc_name,
                src_buf=src.buf,
                src_shape=(),
                box_extents=box,
                swizzle=stage.swizzle.value,
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
        # ``BYTES_PER_ELEM`` over-counted fp16 by 2x).
        slab_bytes = src.dtype.nbytes if src.dtype is not None else BYTES_PER_ELEM
        for ax in src.cache_axes:
            slab_bytes *= ax.extent.as_static()
        # Smem allocation: leading phase dim + cache extents (with pad).
        full_extents = (stage.buffer_count, *src.alloc_extents)
        smem_index = (stage.phase, *([Literal(0, "int")] * len(src.cache_axes)))
        # Distribute issuer threads across stages within a group so each
        # stage's arrive+TMA pair issues from a different thread (stage
        # 0 → tid 0, stage 1 → tid 1, ...) rather than serializing on tid 0.
        cond = Cond(
            cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(tma.issuer_tid[src.name], "int")),
            body=(
                MbarrierArriveExpectTx(mbar=mbar_name, bytes_=slab_bytes, slot=stage.phase),
                TmaLoad(
                    smem=src.name,
                    smem_index=smem_index,
                    desc=desc_name,
                    coords=coords,
                    mbar=mbar_name,
                    mbar_slot=stage.phase,
                ),
            ),
        )
        # 128 B = NVIDIA's recommended TMA-destination alignment for box
        # copies. Swizzle modes would need wider alignment but the swizzle
        # picker (012) was dropped from the wrap-body pipeline, so every
        # TMA stage runs at the base recommendation.
        align = _TMA_ALIGN_BYTES
        smem_dtype = smem_cuda_dtype(src)
        out: list[Stmt] = [
            Smem(name=src.name, extents=full_extents, align=align, dtype=smem_dtype),
            cond,
        ]
        # Implicit wait at the wrap boundary for unpipelined bundles
        # (pipeline_depth == 1). The bundle wrapping this stage carries
        # the phase / buffer_count / pipeline_depth.
        if bundle.pipeline_depth == 1:
            mbar_phase = _mbar_wait_phase(bundle.phase, bundle.buffer_count)
            out.append(MbarrierWait(mbar=mbar_name, phase=mbar_phase, slot=bundle.phase))
            out.append(Sync())
        return out

    def emit_bundle_producer(bundle: StageBundle) -> list[Stmt]:
        """Emit producer-side scaffolding for every member Stage of a
        bundle. Dispatch:

        - ``member.compute is not None`` → cooperative compute template
          (formerly ``ComputeStage``).
        - ``bundle.policy == TMA`` → TMA box copy.
        - otherwise (SYNC / BUFFERED / ASYNC) → cooperative
          ``Load + Write`` (or ``CpAsyncCopy`` for ASYNC).
        """
        out: list[Stmt] = []
        for member in bundle.stages:
            if member.compute is not None:
                out.extend(emit_compute_stage(member, tid_expr, n_threads, buffer_count=bundle.buffer_count))
            elif bundle.policy == StagePolicy.TMA:
                out.extend(emit_tma_stage(bundle, member))
            else:
                out.extend(
                    emit_stage(
                        member,
                        tid_expr,
                        n_threads,
                        policy=bundle.policy,
                        buffer_count=bundle.buffer_count,
                        phase=bundle.phase,
                        pipeline_depth=bundle.pipeline_depth,
                    )
                )
        return out

    def _process_stmt(stmt: Stmt) -> list[Stmt]:
        """Materialize one body Stmt — bundles inline their producer code
        then recursively process their consumer body."""
        if isinstance(stmt, StageBundle):
            out_local: list[Stmt] = list(filter_emit(emit_bundle_producer(stmt)))
            for inner in stmt.body:
                out_local.extend(_process_stmt(inner))
            return out_local
        if isinstance(stmt, AsyncWait):
            return list(emit_async_wait(stmt))
        if isinstance(stmt, (SerialTile, StridedTile, RegisterTile)):
            single = _emit_loop(stmt, tid_expr, n_threads, transform, filter_emit, emit_bundle_producer, emit_async_wait)
            extra: list[Stmt] = []
            if isinstance(stmt, (SerialTile, StridedTile)) and stmt.is_reduce:
                accums_in_scope = {a.name: a for a in stmt.body if isinstance(a, Accum)}
            else:
                accums_in_scope = find_nested_reduce_accums(stmt.body)
            for acc_name, acc in accums_in_scope.items():
                if escape is not None and escape.accum_cooperative_axes.get(acc_name):
                    tid_var = single_thread_var(thread_axes)
                    dt = acc.dtype or F32
                    extra.extend(emit_combine(acc_name, acc.op, tid_var, n_threads, dt, warp_size=warp_size))
                    rename[acc_name] = f"{acc_name}_b"
            return [single, *extra]
        if isinstance(stmt, Accum):
            out_local = [transform(stmt)]
            if escape is not None and escape.accum_cooperative_axes.get(stmt.name):
                tid_var = single_thread_var(thread_axes)
                dt = stmt.dtype or F32
                out_local.extend(emit_combine(stmt.name, stmt.op, tid_var, n_threads, dt, warp_size=warp_size))
                rename[stmt.name] = f"{stmt.name}_b"
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
    # ``110_drop_redundant_syncs`` after this lowering.
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


def _materialize_ws(blk: ThreadTile, *, warp_size: int, escape=None, producer_warps: int) -> Stmt:
    """Warp-specialized materialization. Reuses ``_materialize`` to get a
    baseline kernel body, then post-passes it into producer + consumer
    subtrees wrapped in ``Cond(warp < producer_warps)``.

    Producer warps issue TMA loads + ``MbarrierArriveExpectTx``; consumer
    warps wait on the full mbarrier, run the per-iter reduce, and arrive
    on an extra ``tma_mbar_empty`` mbarrier so the producer can refill
    a slot only after the consumer drains it. Inside the consumer branch
    every ``Sync`` becomes a named-barrier sync (``bar.sync 1,
    n_consumer_threads``) — ``__syncthreads`` would be UB on the
    warp-divergent ``Cond``.

    Currently implemented for kernels without cooperative ``Accum``s
    (matmul shape). SDPA cooperative reduces inside the consumer branch
    need additional consumer-tid remapping; raises NotImplementedError
    if `escape.accum_cooperative_axes` is non-empty."""
    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    # Eligibility: cooperative reduces need consumer-tid remap (Stage 5,
    # not yet implemented). Surface loudly.
    if escape is not None:
        coop = {n: ax for n, ax in escape.accum_cooperative_axes.items() if ax}
        if coop:
            raise NotImplementedError(
                f"WS materialization with cooperative Accum(s) {sorted(coop)} not yet implemented "
                "(consumer-tid remap pending — see plan Phase D Stage 5)"
            )

    # 1. Run the regular materialization to get a baseline ThreadTile.
    regular = _materialize(blk, warp_size=warp_size, escape=escape)
    if not isinstance(regular, ThreadTile):
        raise ValueError(f"_materialize returned {type(regular).__name__}, expected ThreadTile")

    axes = regular.axes
    n_threads = 1
    for ax in axes:
        n_threads *= ax.extent.as_static()
    n_producer_threads = producer_warps * warp_size
    n_consumer_threads = n_threads - n_producer_threads
    if n_producer_threads <= 0 or n_consumer_threads <= 0:
        raise ValueError(
            f"WS requires both producer and consumer warps; got P={producer_warps} ({n_producer_threads} threads), "
            f"total {n_threads} threads ({n_threads // warp_size} warps)"
        )

    # 2. Split into shared prologue (Smem/MbarrierInit/etc.) and main body.
    prologue, main = _split_prologue(tuple(regular.body))

    # 3. Walk main, classify each stmt and build producer + consumer lists.
    producer_main, consumer_main, mbar_groups = _split_roles(main)

    # 4. Add empty-mbarrier wiring: extra mbarrier per TMA group, init +
    # per-iter wait in producer / arrive in consumer.
    prologue_ws, producer_main_ws, consumer_main_ws = _wire_empty_mbarriers(
        prologue, producer_main, consumer_main, mbar_groups, n_producer_threads
    )

    # 5. Rename consumer-side Syncs to named-barrier.
    consumer_main_named = _rename_consumer_syncs(consumer_main_ws, n_consumer_threads)

    # 6. Wrap in Cond.
    ws_cond = Cond(
        cond=BinaryExpr("<", Var("warp"), Literal(producer_warps, "int")),
        body=Body((SetMaxNReg(24, "dec"), *producer_main_ws)),
        else_body=Body((SetMaxNReg(240, "inc"), *consumer_main_named)),
    )
    new_body: list[Stmt] = [*prologue_ws, ws_cond]
    return ThreadTile(axes=axes, body=Body(new_body))


# ---------------------------------------------------------------------------
# WS helpers
# ---------------------------------------------------------------------------


def _split_prologue(body: tuple[Stmt, ...]) -> tuple[list[Stmt], list[Stmt]]:
    """Split a materialized ThreadTile body into (prologue, main).

    Prologue is the leading run of kernel-scope declarations + mbarrier
    init: ``TmaDescriptor`` / ``Smem`` / ``Cond(thread_idx==0, [MbarrierInit
    ...])`` / the trailing ``Sync`` after init. Main is everything that
    follows (the actual work)."""
    cut = 0
    for i, s in enumerate(body):
        if isinstance(s, (TmaDescriptor, Smem)):
            cut = i + 1
            continue
        # MbarrierInit init block: Cond(tid==0, [MbarrierInits])
        if isinstance(s, Cond) and all(isinstance(b, MbarrierInit) for b in s.body):
            cut = i + 1
            continue
        # Sync immediately after the MbarrierInit Cond.
        if isinstance(s, Sync) and cut > 0 and isinstance(body[cut - 1], Cond):
            cut = i + 1
            continue
        break
    return list(body[:cut]), list(body[cut:])


def _is_producer_stmt(s: Stmt) -> bool:
    """A stmt belongs in the producer branch if it carries TMA-issue
    scaffolding: ``Cond(thread_idx==N, [MbarrierArriveExpectTx, TmaLoad])``."""
    if not isinstance(s, Cond):
        return False
    return any(isinstance(b, (MbarrierArriveExpectTx, TmaLoad)) for b in s.body)


def _split_roles(main: list[Stmt]) -> tuple[list[Stmt], list[Stmt], dict[str, int]]:
    """Walk the main body, classifying each top-level stmt as producer or
    consumer. Returns (producer_body, consumer_body, mbar_groups) where
    mbar_groups maps each TMA mbarrier name → its buffer_count (recovered
    from the prologue's ``Smem(name, extents=(bc,))``).

    For ``SerialTile(serial_outer)`` (the K_o loop), recurses: builds two
    K_o loops, one per role, each iterating the same axis."""
    producer: list[Stmt] = []
    consumer: list[Stmt] = []
    mbar_groups: dict[str, int] = {}

    def _collect_mbar(stmts: tuple[Stmt, ...]):
        for s in stmts:
            if isinstance(s, Cond):
                for b in s.body:
                    if isinstance(b, MbarrierArriveExpectTx):
                        mbar_groups.setdefault(b.mbar, 1)
                    if isinstance(b, TmaLoad):
                        mbar_groups.setdefault(b.mbar, 1)
            if isinstance(s, MbarrierWait):
                mbar_groups.setdefault(s.mbar, 1)

    for stmt in main:
        if _is_producer_stmt(stmt):
            producer.append(stmt)
            _collect_mbar((stmt,))
        elif isinstance(stmt, SerialTile) and stmt.kind == "serial_outer":
            # Recurse into K_o body, build two role-specific loops.
            inner_prod, inner_cons, inner_mbar = _split_roles(list(stmt.body))
            mbar_groups.update(inner_mbar)
            from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

            if inner_prod:
                producer.append(stmt.with_bodies((Body(tuple(inner_prod)),)))
            if inner_cons:
                consumer.append(stmt.with_bodies((Body(tuple(inner_cons)),)))
        elif isinstance(stmt, (MbarrierWait, Sync)):
            consumer.append(stmt)
            _collect_mbar((stmt,))
        elif isinstance(stmt, (SerialTile, StridedTile, RegisterTile)):
            # Inner reduce + epilogue reduce — consumer.
            consumer.append(stmt)
        else:
            # Default: consumer (output Writes, Accums, etc.)
            consumer.append(stmt)
    return producer, consumer, mbar_groups


def _wire_empty_mbarriers(
    prologue: list[Stmt],
    producer: list[Stmt],
    consumer: list[Stmt],
    mbar_groups: dict[str, int],
    n_producer_threads: int,
) -> tuple[list[Stmt], list[Stmt], list[Stmt]]:
    """Allocate ``<mbar_name>_empty[bc]`` per TMA group, init in prologue,
    insert ``MbarrierWait`` in producer K_o body before each issue,
    insert ``MbarrierArrive`` in consumer K_o body after each reduce.

    Producer wait is gated by ``Cond(K_o >= bc-1, ...)`` so the first
    ``bc-1`` iters of the producer skip the wait — they fill slots
    that weren't touched by the prologue's ``σ_first`` (which uses slot
    0). No pre-arrives needed; the conditional skip is cleaner than
    a pre-arrival scheme that needs different formulas per slot.

    Consumer arrive is gated by ``Cond(threadIdx.x == n_producer_threads, ...)``
    — a single consumer thread arrives so arrive_count stays 1.

    Phase formula for producer wait (for K_o >= bc-1):
        slot  = (K_o + 1) % bc
        phase = ((K_o + 1) / bc - 1) % 2

    Currently single-group only — multi-group TMA kernels skip the
    empty-mbarrier wiring (raises a warning). Most matmul/SDPA kernels
    are single-group post-080."""
    bc_per_mbar: dict[str, int] = {}
    for s in prologue:
        if isinstance(s, Smem) and s.dtype == "unsigned long long":
            bc_per_mbar[s.name] = int(s.extents[0])

    # Filter to mbars that actually appear in mbar_groups (collected
    # during _split_roles). This drops any prologue Smem leftovers from
    # non-TMA paths.
    relevant = {n: bc for n, bc in bc_per_mbar.items() if n in mbar_groups}
    if not relevant:
        return prologue, producer, consumer

    # Single-group simplification for v1.
    if len(relevant) > 1:
        # Multi-group kernels (e.g., SDPA's multi-K-loop shape) need
        # per-group wait/arrive insertion keyed by which K_o loop owns
        # which mbar. Defer; today's matmul is single-group.
        return prologue, producer, consumer

    full_mbar = next(iter(relevant))
    bc = relevant[full_mbar]
    empty_mbar = f"{full_mbar}_empty"

    # 1. Augment prologue: empty Smem + per-slot MbarrierInit.
    extra_smem = Smem(name=empty_mbar, extents=(bc,), dtype="unsigned long long")
    extra_inits = tuple(MbarrierInit(mbar=empty_mbar, count=1, slot=Literal(s, "int")) for s in range(bc))

    new_prologue: list[Stmt] = []
    smem_inserted = False
    init_inserted = False
    for s in prologue:
        # Insert empty Smem right after the full mbar Smem.
        if not smem_inserted and isinstance(s, Smem) and s.name == full_mbar:
            new_prologue.append(s)
            new_prologue.append(extra_smem)
            smem_inserted = True
            continue
        # Splice empty inits into the MbarrierInit Cond.
        if not init_inserted and isinstance(s, Cond) and all(isinstance(b, MbarrierInit) for b in s.body):
            new_prologue.append(Cond(cond=s.cond, body=(*s.body, *extra_inits)))
            init_inserted = True
            continue
        new_prologue.append(s)

    # 2. Producer: insert MbarrierWait before each K_o issue.
    def _augment_producer(stmts: list[Stmt]) -> list[Stmt]:
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
                inner = Body((wait_cond, *s.body))
                out.append(s.with_bodies((inner,)))
            else:
                out.append(s)
        return out

    new_producer = _augment_producer(producer)

    # 3. Consumer: append MbarrierArrive after each K_o reduce.
    # arrive_count=1 on empty mbar init means exactly one consumer
    # thread arrives per K_o iter. We pick the first thread of the
    # consumer range — ``threadIdx.x == n_producer_threads``.
    consumer_first_tid = Literal(n_producer_threads, "int")

    def _augment_consumer(stmts: list[Stmt]) -> list[Stmt]:
        from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, SerialTile) and s.kind == "serial_outer":
                k_var = s.axis.name
                slot_expr = Var(k_var) % Literal(bc, "int")
                arrive_cond = Cond(
                    cond=BinaryExpr("==", Builtin("thread_idx.x"), consumer_first_tid),
                    body=(MbarrierArrive(mbar=empty_mbar, slot=slot_expr),),
                )
                inner = Body((*s.body, arrive_cond))
                out.append(s.with_bodies((inner,)))
            else:
                out.append(s)
        return out

    new_consumer = _augment_consumer(consumer)
    return new_prologue, new_producer, new_consumer


def _rename_consumer_syncs(stmts: list[Stmt], n_consumer_threads: int) -> list[Stmt]:
    """Recursively rewrite plain ``Sync()`` to ``Sync(barrier_id=1,
    count=n_consumer_threads)`` inside the consumer subtree.

    ``__syncthreads`` (the default Sync) inside the warp-divergent
    ``Cond(warp<P)`` consumer branch is CUDA UB; a named bar.sync
    synchronizes exactly the consumer threads.

    Also descends into ``TreeHalve`` (which carries its own per-iter
    barrier in the rendered loop): switches barrier_id/barrier_count
    to match. ``WarpShuffle`` is intra-warp via ``__shfl_xor_sync``
    and doesn't need a CTA-wide sync — unchanged."""
    out: list[Stmt] = []
    for s in stmts:
        out.append(_rename_one_consumer_sync(s, n_consumer_threads))
    return out


def _rename_one_consumer_sync(s: Stmt, n: int) -> Stmt:
    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    if isinstance(s, Sync) and s.barrier_id == 0:
        return Sync(barrier_id=1, count=n)
    if isinstance(s, TreeHalve) and s.barrier_id == 0:
        return TreeHalve(
            buf=s.buf,
            op=s.op,
            length=s.length,
            tid_var=s.tid_var,
            dtype=s.dtype,
            barrier_id=1,
            barrier_count=n,
        )
    nested = s.nested()
    if nested:
        new_bodies = tuple(Body(tuple(_rename_one_consumer_sync(c, n) for c in b)) for b in nested)
        return s.with_bodies(new_bodies)
    return s


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
