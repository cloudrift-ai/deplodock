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
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    Sync,
    TmaDescriptor,
    TmaLoad,
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
    new_body: list[Stmt] = []
    for s in root.op.body:
        if isinstance(s, (GridTile, ThreadTile)):
            new_body.append(_materialize_top(s, warp_size=ctx.warp_size, escape=escape))
        else:
            new_body.append(s)
    return KernelOp(body=new_body, name=root.op.name)


def _materialize_top(top: Stmt, *, warp_size: int, escape=None) -> Stmt:
    """Dispatch the outermost tile of a TileOp body to materialization.

    Two shapes are possible coming out of ``001_launch_geometry``:

    - ``GridTile(... body=[ThreadTile(... body=actual)])``: cooperative
      kernel (matmul / fused-reduce). The ThreadTile's body is what
      ``_materialize`` walks; the GridTile wrapper preserved unchanged
      so kernel render emits the ``blockIdx`` decode.
    - ``ThreadTile(... body=actual)``: pointwise/standalone. Materialize
      the body directly; the kernel renderer's linear-tid path handles
      launch geometry from the ThreadTile's extents.
    """
    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    if isinstance(top, GridTile):
        new_outer: list[Stmt] = []
        for child in top.body:
            if isinstance(child, ThreadTile):
                new_outer.append(_materialize(child, warp_size=warp_size, escape=escape))
            else:
                new_outer.append(child)
        return GridTile(axes=top.axes, body=Body(new_outer))
    if isinstance(top, ThreadTile):
        return _materialize(top, warp_size=warp_size, escape=escape)
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
        # AsyncWait.barrier_id / barrier_count let the caller (085 for WS
        # kernels) route the trailing CTA fence to a named barrier
        # (``bar.sync N, count``) instead of ``__syncthreads()``. Default
        # 0 keeps every legacy emission on ``__syncthreads()``.
        trailing_sync = Sync(barrier_id=stmt.barrier_id, count=stmt.barrier_count)
        if stmt.phase is not None and tma.has_tma:
            gid = tma.wait_group.get(id(stmt))
            if gid is not None:
                return [MbarrierWait(mbar=tma.mbar_name(gid), phase=stmt.phase, slot=stmt.slot), trailing_sync]
        # cp.async fallback (or pre-pipelining synchronous-style wait,
        # or AsyncWait whose pipeline group couldn't be inferred).
        return [CpAsyncWait(group=stmt.keep), trailing_sync]

    def emit_tma_stage(bundle: StageBundle, stage: Stage) -> list[Stmt]:
        # 050_use_tma only promotes single-Source bundles, so the box-copy
        # here issues exactly one TMA load per group activation per member.
        assert len(stage.sources) == 1, f"TMA stage requires one Source, got {len(stage.sources)}"
        src = stage.sources[0]
        addressing = src.addressing
        assert isinstance(addressing, AffineAddressing), f"TMA stage source {src.name!r} must use AffineAddressing"
        desc_name = f"{src.name}_desc"
        # Key by smem name (Source.name) — id(bundle) isn't stable across
        # Body.map's wrapper-rebuild on descent, but Source.name is set
        # by 020_stage_inputs and survives every downstream rewrite.
        gid = tma.stage_group[src.name]
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
                swizzle=bundle.swizzle.value,
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
        if isinstance(stmt, Cond):
            # 085_warp_specialize wraps producer + consumer subtrees in
            # a top-level ``Cond(warp_role_check, prod, cons)`` — both
            # branches may contain StageBundles / AsyncWaits / SerialTiles
            # that need recursive processing.
            from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

            body_out: list[Stmt] = []
            for inner in stmt.body:
                body_out.extend(_process_stmt(inner))
            else_out: list[Stmt] = []
            for inner in stmt.else_body:
                else_out.extend(_process_stmt(inner))
            return [Cond(cond=stmt.cond, body=Body(tuple(body_out)), else_body=Body(tuple(else_out)))]
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
