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
    ComputeStage,
    GridTile,
    RegisterTile,
    SerialTile,
    Stage,
    StridedTile,
    ThreadTile,
    TileOp,
    TmaBufferedStage,
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
        # Locate the (sole) ThreadTile child.
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
        n_threads *= int(ax.extent)

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

    # Compute-Stage Smem hoist: a ComputeStage (produced by
    # 030_hoist_invariant_compute when FUSED_PIPELINE=True) and an
    # inline-fuse multi-source Stage (FUSED_PIPELINE=False) both emit
    # their body inside the K-outer loop body — Smem decls inside a
    # loop don't reach kernel scope in CUDA. Walk the body once, pre-
    # emit Smem decls at kernel scope, and mark them ``declared_smem``
    # so ``_emit_stage``'s in-loop emit is dedup'd. Single-source
    # transport stages are hoisted to prologue naturally by 015.
    # Hoist every Stage's per-Source Smem decl to kernel scope so the
    # Stages' producer side (cooperative load) can emit the Smem decl
    # in-line without escaping the Stage.body scope. The new wrap-body
    # Stage's body IS the consumer; in CUDA, an Smem decl inside a Loop
    # body doesn't reach kernel scope, so the hoist happens here.
    compute_stage_prologue: list[Stmt] = []
    for stmt in body.iter():
        if isinstance(stmt, Stage):
            for src in stmt.sources:
                if src.name in declared_smem:
                    continue
                extents = src.alloc_extents
                buf_count = getattr(stmt, "buffer_count", 1)
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

    def emit_tma_stage(stage: TmaBufferedStage) -> list[Stmt]:
        # Wrap-body invariant: 050_use_tma only promotes single-Source
        # stages, so the box-copy here issues exactly one TMA load per
        # group activation.
        assert len(stage.sources) == 1, f"TmaBufferedStage requires one Source, got {len(stage.sources)}"
        src = stage.sources[0]
        addressing = src.addressing
        assert isinstance(addressing, AffineAddressing), f"TmaBufferedStage source {src.name!r} must use AffineAddressing"
        desc_name = f"{src.name}_desc"
        gid = tma.stage_group[id(stage)]
        mbar_name = tma.mbar_name(gid)
        # Box extents per source dim: product of cache extents that map
        # to that dim (multiple cache axes can sweep the same source dim
        # — e.g. an outer-block axis and a per-thread fragment axis both
        # decoded into the M dim of a matmul slab). Source dims not
        # covered by any cache axis get extent 1.
        dims = addressing.dims
        box_per_dim: dict[int, int] = {}
        for d, ax in zip(dims, src.cache_axes, strict=True):
            box_per_dim[d] = box_per_dim.get(d, 1) * int(ax.extent)
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
            slab_bytes *= int(ax.extent)
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
        # Implicit wait at the wrap boundary for unpipelined stages
        # (pipeline_depth == 1). Mirrors the AsyncBufferedStage flow in
        # ``_emit_stage`` — the consumer body sees the committed copy
        # before reading. ``080_pipeline_stages`` (when it
        # lands) expands depth > 1 stages and emits its own waits.
        if stage.pipeline_depth == 1:
            mbar_phase = _mbar_wait_phase(stage.phase, stage.buffer_count)
            out.append(MbarrierWait(mbar=mbar_name, phase=mbar_phase, slot=stage.phase))
            out.append(Sync())
        return out

    for stmt in body:
        if isinstance(stmt, TmaBufferedStage):
            new_body.extend(filter_emit(emit_tma_stage(stmt)))
        elif isinstance(stmt, ComputeStage):
            new_body.extend(filter_emit(emit_compute_stage(stmt, tid_expr, n_threads)))
        elif isinstance(stmt, Stage):
            new_body.extend(filter_emit(emit_stage(stmt, tid_expr, n_threads)))
        elif isinstance(stmt, AsyncWait):
            new_body.extend(emit_async_wait(stmt))
        elif isinstance(stmt, (SerialTile, StridedTile, RegisterTile)):
            new_body.append(_emit_loop(stmt, tid_expr, n_threads, transform, filter_emit, emit_tma_stage, emit_async_wait))
            # Locate Accums whose value escapes this loop scope so we
            # can emit helper-driven Combines for cooperative ones.
            if isinstance(stmt, (SerialTile, StridedTile)) and stmt.is_reduce:
                # Single-loop reduce: Accums live at the immediate-body level.
                accums_in_scope = {a.name: a for a in stmt.body if isinstance(a, Accum)}
            else:
                # Non-reduce wrapper (e.g. ``SerialTile(K_o, kind="serial_outer",
                # body=[SerialTile(K_i, kind="stage_inner", reduce, [Accum])])``
                # for cooperative-K reduce after the partition planner's σ-split).
                # Descend into nested reduce subtrees so sibling Combines match
                # their Accums' dtypes.
                accums_in_scope = find_nested_reduce_accums(stmt.body)
            for acc_name, acc in accums_in_scope.items():
                if escape is not None and escape.accum_cooperative_axes.get(acc_name):
                    tid_var = single_thread_var(thread_axes)
                    dt = acc.dtype or F32
                    new_body.extend(emit_combine(acc_name, acc.op, tid_var, n_threads, dt, warp_size=warp_size))
                    rename[acc_name] = f"{acc_name}_b"
        elif isinstance(stmt, Accum):
            # Bare Accum at the ThreadTile scope — degenerate cooperative
            # reduce where K_i collapsed to size-1 (e.g. K=warp_size with
            # BR=warp_size cooperative threads each handling one element).
            new_body.append(transform(stmt))
            if escape is not None and escape.accum_cooperative_axes.get(stmt.name):
                tid_var = single_thread_var(thread_axes)
                dt = stmt.dtype or F32
                new_body.extend(emit_combine(stmt.name, stmt.op, tid_var, n_threads, dt, warp_size=warp_size))
                rename[stmt.name] = f"{stmt.name}_b"
        else:
            new_body.append(transform(stmt))

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


def _emit_loop(loop, tid_expr, n_threads, transform, filter_emit, emit_tma_stage, emit_async_wait) -> Stmt:
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
        if isinstance(s, TmaBufferedStage):
            return filter_emit(emit_tma_stage(s))
        if isinstance(s, ComputeStage):
            return filter_emit(emit_compute_stage(s, tid_expr, n_threads))
        if isinstance(s, Stage):
            return filter_emit(emit_stage(s, tid_expr, n_threads))
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
        ext = int(ax.extent)
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
