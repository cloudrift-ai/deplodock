"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

The wrapper stays as ``Tile`` (shared with Tile IR via ``ir.stmt``);
only the body content changes — ``Stage`` becomes
``Smem`` + cooperative load, ``Combine`` becomes smem tree-halve,
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
    * ``Combine`` after a reduce loop → smem tree-halve + broadcast.
    * ``Write`` whose index references a THREAD axis is emitted
      unconditionally (each thread owns a unique output slot). Writes
      that don't reference any THREAD axis are guarded by ``tid==0`` so
      only one thread writes.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Builtin, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    KernelOp,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    Sync,
    TmaDescriptor,
    TmaLoad,
    TreeHalve,
    WarpShuffle,
)
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncBufferedStage,
    AsyncWait,
    BufferedStage,
    Combine,
    ComputeStage,
    GridTile,
    RegisterTile,
    SerialTile,
    Stage,
    StridedTile,
    TemplateAddressing,
    ThreadTile,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", TileOp)]


# Standard TMA destination alignment. 16 B is the hardware minimum;
# 128 B is what NVIDIA's TMA programming guide recommends for max
# throughput on box copies.
_TMA_ALIGN_BYTES = 128


def _smem_cuda_dtype(src) -> str:  # noqa: ANN001 — Source carries DataType | None
    """C type spelling for a Source's smem slab, derived from the
    stamped ``Source.dtype`` (``001_stamp_types``). Defaults to the
    legacy ``"float"`` when unstamped — handwritten test fixtures
    rely on the fallback."""
    from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

    if src.dtype is None:
        return "float"
    return cuda_name(src.dtype)


def _flatten_wrap_stages(body) -> tuple[Stmt, ...]:
    """Pre-flatten wrap-body Stages: ``Stage(sources, body=[consumer])`` becomes
    ``[Stage(sources, body=Body(())), *consumer_stmts]`` so the existing
    materializer's flat walker can emit producer scaffolding then process
    the consumer stmts as siblings.

    Recurses into Loop / StridedLoop / Cond / Tile bodies so nested Stages
    flatten too. ComputeStage's ``compute`` body is kept attached to the
    stage; materializer emits it specially.
    """
    from dataclasses import replace as _replace  # noqa: PLC0415

    from deplodock.compiler.ir.stmt.body import Body  # noqa: PLC0415

    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Stage):
            inner = _flatten_wrap_stages(s.body)
            # Recursively flatten ComputeStage.compute too (in case future
            # passes nest stages inside it).
            if isinstance(s, ComputeStage):
                compute_inner = _flatten_wrap_stages(s.compute)
                out.append(_replace(s, body=Body(()), compute=Body(compute_inner)))
            else:
                out.append(_replace(s, body=Body(())))
            out.extend(inner)
        elif isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            new_body = _flatten_wrap_stages(s.body)
            out.append(s.with_bodies((Body(new_body),)))
        elif isinstance(s, Cond):
            new_body = _flatten_wrap_stages(s.body)
            new_else = _flatten_wrap_stages(s.else_body) if s.else_body else ()
            out.append(_replace(s, body=Body(new_body), else_body=Body(new_else)))
        else:
            out.append(s)
    return tuple(out)


def rewrite(ctx: Context, root: Node) -> Graph | None:
    # M2 soundness check: assert the escape-analysis helper agrees with
    # the planner-emitted tags and coordination-pass-emitted markers on
    # every TileOp the materializer processes. Will be removed in M3
    # when the helper becomes the sole source of truth.
    from deplodock.compiler.ir.tile.escape_analysis import analyze as _analyze_escape  # noqa: PLC0415
    from deplodock.compiler.ir.tile.escape_analysis import cross_check_against_tags  # noqa: PLC0415

    cross_check_against_tags(root.op)
    escape = _analyze_escape(root.op)

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
        # splitk_axes is intentionally NOT propagated — coordination has
        # already used the tag to stamp Write.reduce_op (which is the
        # only signal codegen needs). The output GridTile carries axes
        # only.
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
    # Pre-flatten wrap-body Stages so the rest of the walker sees the
    # legacy flat shape ([Stage(empty body), *consumer_stmts]) it was
    # designed for. _emit_stage consumes Stage.sources for the producer
    # scaffolding; consumer stmts are handled as siblings.
    from deplodock.compiler.ir.stmt.body import Body as _Body  # noqa: PLC0415

    body = _Body(_flatten_wrap_stages(blk.body))
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
    # don't deadlock. ``011_tma_copy`` enforces all-or-nothing TMA
    # promotion per tile, so any tile with TMA stages is guaranteed to
    # have no cp.async stages in the same pipelined K-loop and the
    # AsyncWait lowering can stay as a pure ``MbarrierWait``.
    descriptors: dict[str, TmaDescriptor] = {}
    mbar_prologue: list[Stmt] = []
    declared_mbar: set[str] = set()

    # Compute-Stage Smem hoist: a ComputeStage (produced by
    # 007b_hoist_invariant_compute when FUSED_PIPELINE=True) and an
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
                smem_dtype = _smem_cuda_dtype(src)
                smem_align = 16 if smem_dtype == "__half" else 0
                compute_stage_prologue.append(Smem(name=src.name, extents=extents, dtype=smem_dtype, align=smem_align))
                declared_smem.add(src.name)

    stage_group, wait_group, group_stage_names, group_buffer_count = _partition_tma_groups(body)
    has_tma = bool(group_stage_names)
    # Per-stage issuer thread = position of stage name within its group
    # (so two stages in the same group issue from tid 0 and tid 1
    # respectively, distributing the arrive+TMA work).
    issuer_tid: dict[str, int] = {}
    for names in group_stage_names:
        for idx, name in enumerate(sorted(names)):
            issuer_tid[name] = idx

    def _mbar_name(gid: int) -> str:
        return f"tma_mbar_{gid}" if len(group_stage_names) > 1 else "tma_mbar"

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
        if stmt.phase is not None and has_tma:
            gid = wait_group.get(id(stmt))
            if gid is not None:
                return [MbarrierWait(mbar=_mbar_name(gid), phase=stmt.phase, slot=stmt.slot), Sync()]
        # cp.async fallback (or pre-pipelining synchronous-style wait,
        # or AsyncWait whose pipeline group couldn't be inferred).
        return [CpAsyncWait(group=stmt.keep), Sync()]

    def emit_tma_stage(stage: TmaBufferedStage) -> list[Stmt]:
        # Wrap-body invariant: 011_tma_copy only promotes single-Source
        # stages, so the box-copy here issues exactly one TMA load per
        # group activation.
        assert len(stage.sources) == 1, f"TmaBufferedStage requires one Source, got {len(stage.sources)}"
        src = stage.sources[0]
        addressing = src.addressing
        assert isinstance(addressing, AffineAddressing), f"TmaBufferedStage source {src.name!r} must use AffineAddressing"
        desc_name = f"{src.name}_desc"
        gid = stage_group[id(stage)]
        mbar_name = _mbar_name(gid)
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
                dtype=_smem_cuda_dtype(src),
            )
        if mbar_name not in declared_mbar:
            declared_mbar.add(mbar_name)
            # Per-group mbarrier array: one mbar per ring-buffer slot,
            # with arrive count = number of distinct stages in *this*
            # group (so all of the group's stages must arrive before its
            # phase flips).
            bc = group_buffer_count[gid]
            mbar_prologue.append(
                Smem(name=mbar_name, extents=(bc,), dtype="unsigned long long"),
            )
            for s in range(bc):
                mbar_prologue.append(MbarrierInit(mbar=mbar_name, count=len(group_stage_names[gid]), slot=Literal(s, "int")))
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
            cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(issuer_tid[src.name], "int")),
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
        smem_dtype = _smem_cuda_dtype(src)
        out: list[Stmt] = [
            Smem(name=src.name, extents=full_extents, align=align, dtype=smem_dtype),
            cond,
        ]
        # Implicit wait at the wrap boundary for unpipelined stages
        # (pipeline_depth == 1). Mirrors the AsyncBufferedStage flow in
        # ``_emit_stage`` — the consumer body sees the committed copy
        # before reading. ``015_lower_pipelined_async_stage`` (when it
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
            new_body.extend(filter_emit(_emit_compute_stage(stmt, tid_expr, n_threads)))
        elif isinstance(stmt, Stage):
            new_body.extend(filter_emit(_emit_stage(stmt, tid_expr, n_threads)))
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
                accums_in_scope = _find_nested_reduce_accums(stmt.body)
            for acc_name, acc in accums_in_scope.items():
                if escape is not None and escape.accum_cooperative_axes.get(acc_name):
                    tid_var = _single_thread_var(thread_axes)
                    dt = acc.dtype or F32
                    new_body.extend(_emit_combine(acc_name, acc.op, tid_var, n_threads, dt, warp_size=warp_size))
                    rename[acc_name] = f"{acc_name}_b"
        elif isinstance(stmt, Accum):
            # Bare Accum at the ThreadTile scope — degenerate cooperative
            # reduce where K_i collapsed to size-1 (e.g. K=warp_size with
            # BR=warp_size cooperative threads each handling one element).
            new_body.append(transform(stmt))
            if escape is not None and escape.accum_cooperative_axes.get(stmt.name):
                tid_var = _single_thread_var(thread_axes)
                dt = stmt.dtype or F32
                new_body.extend(_emit_combine(stmt.name, stmt.op, tid_var, n_threads, dt, warp_size=warp_size))
                rename[stmt.name] = f"{stmt.name}_b"
        elif isinstance(stmt, Combine):
            # Skip — helper-driven Combine emission above already handled
            # cooperative Accums. Kept as a no-op branch so coordination's
            # legacy Combine stmts don't fall into the ``else`` (which
            # would copy them into the kernel body verbatim — they have
            # no render() in Kernel IR). Will be removed in M5 when the
            # Combine class is deleted.
            pass
        else:
            new_body.append(transform(stmt))

    # Init scoping for accumulators is handled by the upstream
    # ``000_place_inits`` pass — explicit ``Init`` Stmts already sit at
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
    return ThreadTile(axes=axes, body=_drop_redundant_syncs(new_body), cooperative_axes=blk.cooperative_axes)


def _drop_redundant_syncs(body: list[Stmt]) -> list[Stmt]:
    """Drop ``Sync`` stmts that are guaranteed no-ops at the body level:

    * Two consecutive ``Sync`` stmts collapse to one.
    * A leading ``Sync`` before any smem access is unnecessary — at kernel
      entry no thread can hold a stale view of smem because nothing has
      been written yet.

    Only handles the body-level (no descent into nested ``Loop`` / ``Cond``
    bodies); the bulk of redundant syncs come from materializer templates
    that emit a defensive ``Sync()`` at template boundaries, and those
    surface here.
    """
    smem_seen = False
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Sync):
            if not smem_seen:
                continue  # nothing for the sync to fence yet
            if out and isinstance(out[-1], Sync):
                continue  # back-to-back sync collapses
        else:
            if isinstance(
                s,
                (
                    Smem,
                    MbarrierInit,
                    MbarrierArriveExpectTx,
                    MbarrierWait,
                    TmaLoad,
                    CpAsyncCopy,
                    CpAsyncCommit,
                    CpAsyncWait,
                    TreeHalve,
                    WarpShuffle,
                ),
            ):
                smem_seen = True
        out.append(s)
    return out


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
            return filter_emit(_emit_compute_stage(s, tid_expr, n_threads))
        if isinstance(s, Stage):
            return filter_emit(_emit_stage(s, tid_expr, n_threads))
        if isinstance(s, AsyncWait):
            return emit_async_wait(s)
        if s.nested():
            # Block wrapper (SerialTile / StridedTile / RegisterTile / Cond / ...) —
            # body.map has already mapped its child bodies post-order.
            return s
        return transform(s)

    return loop.with_bodies((loop.body.map(materialize_leaf),))


def _find_nested_reduce_accums(stmts) -> dict[str, Accum]:
    """All ``Accum``s sitting at the immediate-body level of the first
    nested reduce ``SerialTile`` / ``StridedTile`` subtree, keyed by
    Accum name. Used by the materializer when a non-reduce outer tile
    wraps a deeper reduce — e.g. the cooperative-K shape
    ``SerialTile(K_o, "serial_outer", body=[SerialTile(K_i, "stage_inner",
    reduce, [Accum, ...])])`` produced by the partition planner's σ-split,
    possibly with F-replicated sibling Accums from
    ``006a_register_tile_planned``. Returns ``{}`` when no reduce-with-
    immediate-Accum is found, preserving the existing "stray Combine
    raises" safety net."""
    for s in stmts:
        if isinstance(s, (SerialTile, StridedTile)) and s.is_reduce:
            accums = {a.name: a for a in s.body if isinstance(a, Accum)}
            if accums:
                return accums
        if isinstance(s, (SerialTile, StridedTile, RegisterTile)):
            found = _find_nested_reduce_accums(s.body)
            if found:
                return found
    return {}


def _single_thread_var(thread_axes: tuple[Axis, ...]) -> str:
    """Combine + TreeHalve emit a single ``tid_var`` string. Only valid
    when there's exactly one THREAD axis — softmax-style cooperation
    (matmul has multi-axis THREAD set but doesn't emit Combine)."""
    if len(thread_axes) != 1:
        raise ValueError(f"Combine requires a single THREAD axis; got {len(thread_axes)}")
    return thread_axes[0].name


def _partition_tma_groups(
    body: tuple[Stmt, ...],
) -> tuple[dict[int, int], dict[int, int], list[set[str]], list[int]]:
    """Partition this Tile body's TMA stages + waits into pipeline-unit
    groups. Each K-loop containing TmaBufferedStages is one group, plus
    any prologue stages immediately before the loop and the immediately-
    following epilogue AsyncWait. Synchronous (pre-pipelining) stages
    (a TmaBufferedStage at body level paired with a trailing AsyncWait)
    each form a singleton group.

    Returns:
      stage_group_by_id : id(TmaBufferedStage) → group_id
      wait_group_by_id  : id(AsyncWait) → group_id
      group_stage_names : group_id → set of distinct stage names
      group_buffer_count: group_id → max buffer_count across the group
    """
    stage_group_by_id: dict[int, int] = {}
    wait_group_by_id: dict[int, int] = {}
    group_stage_names: list[set[str]] = []
    group_buffer_count: list[int] = []

    def _new_group() -> int:
        group_stage_names.append(set())
        group_buffer_count.append(0)
        return len(group_stage_names) - 1

    def _add_stage(gid: int, stage: TmaBufferedStage) -> None:
        stage_group_by_id[id(stage)] = gid
        # 011_tma_copy promotes single-source stages only; group_stage_names
        # tracks per-Source smem name so issuer-tid allocation distributes
        # the arrive+TmaLoad across distinct elected threads.
        for src in stage.sources:
            group_stage_names[gid].add(src.name)
        group_buffer_count[gid] = max(group_buffer_count[gid], stage.buffer_count)

    pending_prologue: list[TmaBufferedStage] = []
    last_pipeline_group: int | None = None

    def _flush_prologues_as_singletons() -> None:
        for st in pending_prologue:
            gid = _new_group()
            _add_stage(gid, st)
        pending_prologue.clear()

    for stmt in body:
        if isinstance(stmt, TmaBufferedStage):
            pending_prologue.append(stmt)
            continue
        if isinstance(stmt, SerialTile) and any(isinstance(s, TmaBufferedStage) for s in stmt.body):
            gid = _new_group()
            for st in pending_prologue:
                _add_stage(gid, st)
            pending_prologue.clear()
            for s in stmt.body:
                if isinstance(s, TmaBufferedStage):
                    _add_stage(gid, s)
                elif isinstance(s, AsyncWait):
                    wait_group_by_id[id(s)] = gid
            last_pipeline_group = gid
            continue
        if isinstance(stmt, AsyncWait):
            # Trailing epilogue wait pairs with the most recent
            # pipeline-unit group (015_pipeline_k_outer always emits one
            # epilogue AsyncWait after its main loop). If there's no
            # pipeline group in scope but pending synchronous stages
            # exist, the wait pairs them with the next-to-be-flushed
            # singleton group (whichever stage was most recently added).
            if last_pipeline_group is not None and not pending_prologue:
                wait_group_by_id[id(stmt)] = last_pipeline_group
            elif pending_prologue:
                # Synchronous stage(s) followed by their wait — collapse
                # into one group with arrive count = num pending.
                gid = _new_group()
                for st in pending_prologue:
                    _add_stage(gid, st)
                pending_prologue.clear()
                wait_group_by_id[id(stmt)] = gid
                last_pipeline_group = gid
            continue
        # Any other statement breaks the prologue chain — flush.
        _flush_prologues_as_singletons()
        last_pipeline_group = None

    _flush_prologues_as_singletons()
    return stage_group_by_id, wait_group_by_id, group_stage_names, group_buffer_count


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


def _emit_combine(name: str, op, t: str, n_threads: int, dtype: DataType = F32, *, warp_size: int) -> list[Stmt]:
    """Emit the cross-thread combine producing ``<name>_b``.

    Three paths, picked by ``n_threads``:

    - **Warp** (``n_threads ≤ WARP_SIZE`` and power of two): a single
      ``WarpShuffle`` butterfly via ``__shfl_xor_sync``. No smem, no
      syncthreads.
    - **Hierarchical** (``n_threads`` a power-of-two multiple of
      ``WARP_SIZE``): each warp first shuffle-reduces its lanes into
      register-resident ``<acc>_w`` (broadcast within the warp); lane 0
      of each warp writes ``<acc>_w`` to a tiny ``smem[n_warps]`` slab;
      one ``Sync`` + ``TreeHalve(length=n_warps)`` collapses across
      warps; broadcast load delivers ``<acc>_b``. The ``TreeHalve``
      runs on the ``warp`` index — sized to ``n_warps`` (4 / 8 / etc.)
      rather than ``n_threads`` (128 / 256 / etc.), so the cross-warp
      reduce is one round of compare-sync instead of five.
    - **Block** (otherwise — n_threads not a clean multiple of 32):
      legacy path. Each thread writes its partial to a smem buffer
      indexed by ``t``, a single ``TreeHalve`` over ``n_threads``
      reduces in place, broadcast load.

    ``dtype`` flows from the parent ``Accum.dtype`` (set by the
    Init-placement pass) so the per-warp register, the inter-warp smem
    slab, and the TreeHalve combine all render in the accumulator's
    element type — fp16 reductions stay fp16 across the inter-warp
    step instead of promoting back to fp32 in the broadcast.

    The Tile renderer emits ``int lane = threadIdx.x & 31;`` and
    ``int warp = threadIdx.x >> 5;`` for any cooperative Tile with
    ``n_threads > WARP_SIZE`` so the hierarchical path's ``Var("lane")``
    / ``Var("warp")`` references resolve.
    """
    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c_name = _cuda_name(dtype)
    broadcast_name = f"{name}_b"
    if n_threads <= warp_size and (n_threads & (n_threads - 1)) == 0:
        return [WarpShuffle(name=broadcast_name, value=name, op=op, length=n_threads, dtype=dtype)]
    if n_threads % warp_size == 0 and (n_threads & (n_threads - 1)) == 0:
        n_warps = n_threads // warp_size
        smem_name = f"{name}_smem"
        warp_w = f"{name}_w"
        return [
            WarpShuffle(name=warp_w, value=name, op=op, length=warp_size, dtype=dtype),
            Smem(name=smem_name, extents=(n_warps,), dtype=smem_c_name),
            Cond(
                cond=BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(Write(output=smem_name, index=(Var("warp"),), value=warp_w),)
            ),
            Sync(),
            TreeHalve(buf=smem_name, op=op, length=n_warps, tid_var="warp", dtype=dtype),
            # TreeHalve's render ends each loop iter with __syncthreads(), so a
            # trailing Sync here would be a no-op pair with the loop's last sync.
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ]
    smem_name = f"{name}_smem"
    return [
        Smem(name=smem_name, extents=(n_threads,), dtype=smem_c_name),
        Write(output=smem_name, index=(Var(t),), value=name),
        Sync(),
        TreeHalve(buf=smem_name, op=op, length=n_threads, tid_var=t, dtype=dtype),
        # See note above on TreeHalve's trailing sync.
        Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
    ]


# ---------------------------------------------------------------------------
# Stage expansion
# ---------------------------------------------------------------------------


def _mbar_wait_phase(stage_phase, buffer_count: int):
    """Derive the mbarrier-test phase from a ``TmaBufferedStage.phase``.

    011_tma_copy stamps ``stage.phase = Var(K_o.name) % buffer_count``
    (the ring slot). The matching mbarrier phase rotates one bit per
    full ring sweep — ``(K_o / buffer_count) % 2``. For the degenerate
    single-shot case (``phase`` is a literal) the mbar phase starts at
    0 and never flips.
    """
    if isinstance(stage_phase, BinaryExpr) and stage_phase.op == "%":
        k_expr = stage_phase.left
        return BinaryExpr("%", BinaryExpr("/", k_expr, Literal(buffer_count, "int")), Literal(2, "int"))
    return Literal(0, "int")


def _emit_stage(stage: Stage, tid_expr, n_threads: int) -> list[Stmt]:
    """Emit producer scaffolding for a wrap-body Stage.

    For each ``Source`` in ``stage.sources``, emits per-source
    cooperative ``Load + Write`` (or ``CpAsyncCopy`` for async transport).
    Smem decls are hoisted to kernel scope in ``_materialize`` (we skip
    them here unless not yet declared, in which case the dedup filter
    will pass them through). Leading + trailing ``Sync`` bracket the
    cooperative load.

    Wrap-body semantics: ``stage.body`` is the *consumer* and has been
    pre-flattened to siblings by ``_flatten_wrap_stages`` before this
    function runs. ``stage.body`` is empty at this point.
    """
    # Buffered: ping-pong-style ring with phase indexing. Leading Sync
    # dropped (different physical buffer per iter).
    is_buffered = isinstance(stage, BufferedStage)
    is_async = isinstance(stage, AsyncBufferedStage)

    prelude: list[Stmt] = [] if is_buffered else [Sync()]
    body_out: list[Stmt] = list(prelude)

    for src in stage.sources:
        # Per-Source iteration axis: synthesize a unique flat-iter axis
        # so the StridedLoop's loop variable doesn't collide with outer
        # thread-decode variables.
        cache_axes = src.cache_axes
        if not cache_axes:
            raise ValueError(f"Source {src.name!r} has no cache axes")
        extents = tuple(int(ax.extent) for ax in cache_axes)
        padded_extents = src.alloc_extents
        total = 1
        for e in extents:
            total *= e
        iter_axis = Axis(name=f"{src.name}_flat", extent=total)
        if len(cache_axes) == 1:
            coord_for = {cache_axes[0].name: Var(iter_axis.name)}
        else:
            coord_for = _flat_decode(cache_axes, iter_axis.name)
        smem_index = tuple(coord_for[ax.name] for ax in cache_axes)
        # Per-source source-index reconstruction.
        addressing = src.addressing
        if isinstance(addressing, TemplateAddressing):
            from deplodock.compiler.ir.sigma import Sigma as _Sigma  # noqa: PLC0415

            cache_sigma = _Sigma({ax.name: coord_for[ax.name] for ax in cache_axes})
            source_index = tuple(cache_sigma.apply(e) for e in addressing.exprs)
        else:
            decoded_per_dim = {dim: coord_for[ax.name] for ax, dim in zip(cache_axes, addressing.dims, strict=True)}
            source_index = tuple(o if d not in decoded_per_dim else o + decoded_per_dim[d] for d, o in enumerate(src.origin))
        # Buffered: prepend phase dim to write index (writes the current
        # ring slot).
        if is_buffered:
            smem_index = (stage.phase, *smem_index)
        # Per-source dtype: use gmem source's CUDA C type so fp16 inputs
        # stage into __half smem. ``Source.dtype`` is stamped by
        # ``001_stamp_types`` from the matching graph node's dtype.
        smem_dtype = _smem_cuda_dtype(src)
        smem_align = 16 if smem_dtype == "__half" else 0
        full_extents = (stage.buffer_count, *padded_extents) if is_buffered else padded_extents

        # cp.async path (sm_80+): fp32 only — fp16 falls through to sync
        # path because cp.async.ca's 4-byte size isn't a clean fit for
        # per-thread stride-1 __half writes.
        if is_async and smem_dtype == "float":
            cooperative_load = StridedLoop(
                axis=iter_axis,
                start=tid_expr,
                step=Literal(n_threads, "int"),
                body=(CpAsyncCopy(smem=src.name, smem_index=smem_index, src=src.buf, src_index=source_index),),
            )
            body_out.append(Smem(name=src.name, extents=full_extents, dtype=smem_dtype, align=smem_align))
            body_out.append(cooperative_load)
            continue

        # Sync path: cooperative Load + Write.
        load_name = f"{src.name}_v"
        cooperative_load = StridedLoop(
            axis=iter_axis,
            start=tid_expr,
            step=Literal(n_threads, "int"),
            body=(
                Load(name=load_name, input=src.buf, index=source_index),
                Write(output=src.name, index=smem_index, value=load_name),
            ),
        )
        body_out.append(Smem(name=src.name, extents=full_extents, dtype=smem_dtype, align=smem_align))
        body_out.append(cooperative_load)

    # Trailing transport: cp.async stages emit Commit; for the
    # unpipelined wrap-body shape (pipeline_depth == 1), follow with the
    # implicit CpAsyncWait(0) + Sync so the consumer body sees the
    # committed copy at the wrap boundary. Pipelined stages
    # (pipeline_depth > 1) get expanded by
    # 015_lower_pipelined_async_stage before materialize and emit their
    # own waits at the pipelined schedule positions.
    # Sync stages just emit __syncthreads so the slab is CTA-visible.
    if is_async:
        body_out.append(CpAsyncCommit())
        if stage.pipeline_depth == 1:
            body_out.append(CpAsyncWait(group=0))
            body_out.append(Sync())
    else:
        body_out.append(Sync())
    return body_out


def _emit_compute_stage(stage: ComputeStage, tid_expr, n_threads: int) -> list[Stmt]:
    """Emit producer scaffolding for a ``ComputeStage``.

    Wraps ``stage.compute`` in a cooperative ``StridedLoop`` over the
    fused cache axes — each thread executes the compute template once
    per cell it owns, σ-substituting cache-axis Vars with row-major
    flat-iter decoded coords. ``stage.body`` (the consumer subtree) is
    pre-flattened to siblings by ``_flatten_wrap_stages`` before we run,
    so it's empty here.

    The output Source's smem decl is hoisted to kernel scope by
    ``_materialize``'s prologue walk; the dedup filter drops the
    redundant in-line decl. Leading ``Sync`` ensures sibling-Stage
    smem writes are visible; trailing ``Sync`` makes the freshly
    computed slab CTA-visible to the consumer.
    """
    from deplodock.compiler.ir.sigma import Sigma  # noqa: PLC0415

    # ComputeStage has exactly one output Source (the slab it fills).
    if len(stage.sources) != 1:
        raise ValueError(f"ComputeStage: expected 1 output Source, got {len(stage.sources)}")
    out = stage.sources[0]
    cache_axes = out.cache_axes
    if not cache_axes:
        raise ValueError(f"ComputeStage {out.name!r}: needs at least one cache axis")
    padded_extents = out.alloc_extents
    total = 1
    for ax in cache_axes:
        total *= int(ax.extent)
    iter_axis = Axis(name=f"{out.name}_flat", extent=total)
    if len(cache_axes) == 1:
        coord_for = {cache_axes[0].name: Var(iter_axis.name)}
    else:
        coord_for = _flat_decode(cache_axes, iter_axis.name)
    sigma = Sigma(coord_for)

    # σ-substitute cache vars in every stmt of the compute body. Loads /
    # Assigns / Writes are leaves so a flat ``map`` over the body is
    # enough; no recursion into nested bodies is needed for the cone
    # shape this pass produces.
    new_stmts: list[Stmt] = []
    for s in stage.compute:
        new_stmts.append(s.rewrite(lambda n: n, sigma))

    full_extents = (stage.buffer_count, *padded_extents) if stage.buffer_count > 1 else padded_extents
    body_out: list[Stmt] = [
        Sync(),
        Smem(name=out.name, extents=full_extents),
        StridedLoop(
            axis=iter_axis,
            start=tid_expr,
            step=Literal(n_threads, "int"),
            body=tuple(new_stmts),
        ),
        Sync(),
    ]
    return body_out


def _flat_decode(cache_axes: tuple[Axis, ...], flat_name: str) -> dict:
    """Row-major decode of a flat index into per-axis coordinates.

    Innermost axis: ``flat % extent``. Middle axes:
    ``(flat / inner_stride) % extent``. Outermost axis: ``flat /
    outer_stride`` (mod is redundant — flat < total)."""
    flat = Var(flat_name)
    coords: dict = {}
    inner_stride = 1
    for ax in reversed(cache_axes):
        ext = int(ax.extent)
        coords[ax.name] = flat % Literal(ext, "int") if inner_stride == 1 else (flat / Literal(inner_stride, "int")) % Literal(ext, "int")
        inner_stride *= ext
    outer = cache_axes[0]
    outer_stride = inner_stride // int(outer.extent)
    coords[outer.name] = flat if outer_stride == 1 else flat / Literal(outer_stride, "int")
    return coords
