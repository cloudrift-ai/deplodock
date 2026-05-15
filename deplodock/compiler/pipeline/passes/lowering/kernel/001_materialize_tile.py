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

from dataclasses import replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
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
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Loop, Stmt, StridedLoop, Tile, Write
from deplodock.compiler.ir.tile.ir import (
    BYTES_PER_ELEM,
    AffineAddressing,
    AsyncBufferedStage,
    AsyncWait,
    BufferedStage,
    Combine,
    Stage,
    SwizzleMode,
    TemplateAddressing,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", TileOp)]


# Standard TMA destination alignment. 16 B is the hardware minimum;
# 128 B is what NVIDIA's TMA programming guide recommends for max
# throughput on box copies.
_TMA_ALIGN_BYTES = 128


def rewrite(root: Node) -> Graph | None:
    new_body: list[Stmt] = []
    for s in root.op.body:
        if isinstance(s, Tile):
            new_body.append(_materialize(s))
        else:
            new_body.append(s)

    return KernelOp(body=new_body, name=root.op.name)


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Tile) -> Stmt:
    """Materialize a Tile. ``Tile.axes`` carries the launch geometry
    (THREAD + optional BLOCK axes); strategies set this up — this pass
    commits no axis decisions of its own. The body is walked uniformly
    whether or not the Tile is cooperative: pointwise (no BLOCK axes)
    is the degenerate case where Stage / Combine / StridedLoop don't
    appear and no Accum nesting exists. The same walker handles both.

    Strategies that need single-thread Writes (e.g. cooperative scalar
    output) wrap them in ``Cond(thread_var == 0)`` themselves —
    materialization passes Writes through unchanged."""
    axes = blk.axes
    body = blk.body
    thread_axes = tuple(ba for ba in axes if ba.bind == BIND_THREAD)
    if not thread_axes:
        raise ValueError("Tile must have at least one BIND_THREAD axis")
    tid_expr = _build_linear_tid(thread_axes)
    n_threads = 1
    for ba in thread_axes:
        n_threads *= int(ba.extent)

    rename: dict[str, str] = {}

    # Body-Load XOR decode for TMA-swizzled stages is now handled by
    # ``012_split_inner_for_swizzle`` (which both picks the swizzle mode
    # AND rewrites Loads). Materialize just sees the post-rewrite IR.

    def transform(s: Stmt) -> Stmt:
        if rename:
            s = s.rewrite(lambda n: rename.get(n, n))
        return s

    new_body: list[Stmt] = []
    pending_reduce: Accum | None = None
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
        _assert_trivial_stage_body(stage)
        desc_name = f"{stage.name}_desc"
        gid = stage_group[id(stage)]
        mbar_name = _mbar_name(gid)
        assert isinstance(stage.addressing, AffineAddressing)
        # Box extents per source dim: product of cache extents that map
        # to that dim (a single source dim can be spanned by multiple
        # cache axes, e.g. matmul fragments where the M-dim is split
        # across an outer-block axis and a per-thread fragment axis).
        # Dims not covered by any cache axis get extent 1 — the coord
        # supplies the origin and the slab is scalar in that dim.
        # Per-source-dim box product. When two or more cache axes map to
        # the same source dim, multiplying yields the total slab span on
        # that dim — except when ``012_split_inner_for_swizzle`` has
        # split the inner axis: there we want the descriptor to *see* the
        # split as separate inner box dims (so the innermost matches the
        # swizzle width). Detect a tail repeat in ``addressing.dims`` and
        # don't collapse it.
        dims = stage.addressing.dims
        split_tail = len(dims) >= 2 and dims[-1] == dims[-2] and stage.swizzle != SwizzleMode.NONE
        if split_tail:
            collapse_axes = stage.axes[:-2]
            collapse_dims = dims[:-2]
            tail_axes = stage.axes[-2:]
        else:
            collapse_axes = stage.axes
            collapse_dims = dims
            tail_axes = ()
        box_per_dim: dict[int, int] = {}
        for d, ax in zip(collapse_dims, collapse_axes, strict=True):
            box_per_dim[d] = box_per_dim.get(d, 1) * int(ax.extent)
        full_box = tuple(box_per_dim.get(d, 1) for d in range(len(stage.origin)))
        # Drop *gap* inert dims (``box == 1`` AND ``origin`` is literal
        # 0) that sit between the first and last swept source dims.
        # Leading singletons stay in the descriptor — keeping them
        # matches the rank-3 shape the working linear-matmul TMA kernels
        # emit. Gap singletons (e.g. the kv_head=1 axis between seq and
        # head_dim in GQA's V tensor) must be dropped — without it, the
        # rank-4 descriptor with the gap singleton deadlocks pipelined
        # TMA at seq=512. The runtime encoder reconstructs the same
        # collapse from ``arr.shape`` + ``box_extents`` alone (a literal-
        # 0 origin coord can only be emitted for a singleton arr dim, so
        # at runtime "drop extent-1 arr dims that align with box dims of
        # extent > 1" recovers exactly the materializer's decision).
        swept = collapse_dims if collapse_dims else dims
        outer, inner = swept[0], swept[-1]
        kept = tuple(
            d
            for d in range(len(stage.origin))
            if d < outer
            or d > inner
            or d in swept
            or not (full_box[d] == 1 and isinstance(stage.origin[d], Literal) and int(stage.origin[d].value) == 0)
        )
        box = tuple(full_box[d] for d in kept)
        coords = tuple(stage.origin[d] for d in kept)
        # Append the split-tail box dims + decomposed coords. The split
        # source dim is the last one in ``swept``; the inner cache axis
        # has extent IPS, the outer factor = orig_extent / IPS. Coord on
        # the new outer dim = ``origin[split_dim] / IPS``; coord on the
        # new inner dim = ``origin[split_dim] % IPS``. The previous
        # ``coords`` entry for the split dim already holds origin —
        # replace it with the split pair.
        if tail_axes:
            split_dim = dims[-1]
            ips = int(tail_axes[-1].extent)
            # Find where split_dim sits in `kept` and remove it; its place
            # is taken by the (outer_coord, inner_coord) pair.
            kept_idx = kept.index(split_dim)
            orig_coord = stage.origin[split_dim]
            outer_coord = BinaryExpr("/", orig_coord, Literal(ips, "int"))
            inner_coord = BinaryExpr("%", orig_coord, Literal(ips, "int"))
            coords = (
                *coords[:kept_idx],
                *coords[kept_idx + 1 :],
                outer_coord,
                inner_coord,
            )
            box = (
                *box[:kept_idx],
                *box[kept_idx + 1 :],
                int(tail_axes[0].extent),
                int(tail_axes[1].extent),
            )
        if desc_name not in descriptors:
            # Source shape is unknown at materialization time — the
            # backend resolves it from the bound array at launch. Pass
            # an empty tuple as a sentinel; descriptor encoding fills it.
            descriptors[desc_name] = TmaDescriptor(
                name=desc_name,
                src_buf=stage.buf,
                src_shape=(),
                box_extents=box,
                swizzle=stage.swizzle.value,
                dtype="float",
            )
        if mbar_name not in declared_mbar:
            declared_mbar.add(mbar_name)
            # Per-group mbarrier array: one mbar per ring-buffer slot,
            # with arrive count = number of distinct stages in *this*
            # group (so all of the group's stages must arrive before its
            # phase flips). Multiple expect_tx on the same mbar
            # accumulate transaction bytes for the current phase,
            # exactly the semantics we want.
            bc = group_buffer_count[gid]
            mbar_prologue.append(
                Smem(name=mbar_name, extents=(bc,), dtype="unsigned long long"),
            )
            for s in range(bc):
                mbar_prologue.append(MbarrierInit(mbar=mbar_name, count=len(group_stage_names[gid]), slot=Literal(s, "int")))
        slab_bytes = BYTES_PER_ELEM
        for ax in stage.axes:
            slab_bytes *= int(ax.extent)
        # Smem allocation: leading phase dim + cache extents (with pad).
        full_extents = (stage.buffer_count, *stage.alloc_extents)
        smem_index = (stage.phase, *([Literal(0, "int")] * len(stage.axes)))
        # Distribute issuer threads across stages so each stage's
        # arrive+TMA pair issues from a different thread (stage 0 → tid 0,
        # stage 1 → tid 1, ...) rather than serializing on tid 0.
        cond = Cond(
            cond=BinaryExpr("==", Builtin("thread_idx.x"), Literal(issuer_tid[stage.name], "int")),
            body=(
                MbarrierArriveExpectTx(mbar=mbar_name, bytes_=slab_bytes, slot=stage.phase),
                TmaLoad(
                    smem=stage.name,
                    smem_index=smem_index,
                    desc=desc_name,
                    coords=coords,
                    mbar=mbar_name,
                    mbar_slot=stage.phase,
                ),
            ),
        )
        # B128 swizzle uses byte-address bits 7..9 to drive the XOR, so the
        # buffer base must be 1024-aligned (next multiple of the swizzle
        # group size: 8 rows × 128 B). Otherwise the row-bit position
        # leaks bits from a non-zero base offset and the body decoder's
        # ``(row & 7) * shift`` reads from off-by-N permuted positions.
        # B64 needs 512-byte alignment, B32 needs 256.
        align = _TMA_ALIGN_BYTES
        if stage.swizzle == SwizzleMode.B128:
            align = 1024
        elif stage.swizzle == SwizzleMode.B64:
            align = 512
        elif stage.swizzle == SwizzleMode.B32:
            align = 256
        return [Smem(name=stage.name, extents=full_extents, align=align), cond]

    for stmt in body:
        if isinstance(stmt, TmaBufferedStage):
            new_body.extend(filter_emit(emit_tma_stage(stmt)))
            pending_reduce = None
        elif isinstance(stmt, Stage):
            new_body.extend(filter_emit(_emit_stage(stmt, tid_expr, n_threads)))
            pending_reduce = None
        elif isinstance(stmt, AsyncWait):
            new_body.extend(emit_async_wait(stmt))
            pending_reduce = None
        elif isinstance(stmt, (Loop, StridedLoop)):
            new_body.append(_emit_loop(stmt, tid_expr, n_threads, transform, filter_emit, emit_tma_stage, emit_async_wait))
            if stmt.is_reduce:
                # Combine matching needs the Accum at the immediate-body
                # level (single-loop reduce). For nested-reduce shapes
                # (K-chunked matmul: outer reduce wraps inner reduce, no
                # immediate Accum) there's no Combine to match anyway, so
                # ``None`` here is safe — a stray Combine would error in
                # the Combine branch as before.
                pending_reduce = next((a for a in stmt.body if isinstance(a, Accum)), None)
            else:
                pending_reduce = None
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce loop")
            if pending_reduce.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({pending_reduce.name!r})")
            new_body.extend(_emit_combine(pending_reduce, _single_thread_var(thread_axes), n_threads))
            rename[pending_reduce.name] = f"{pending_reduce.name}_b"
            pending_reduce = None
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
    return Tile(axes=axes, body=_drop_redundant_syncs(new_body))


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
    """Translate a body Loop or StridedLoop. Recurses so nested staging
    / loops / writes inside the body get the same uniform treatment.
    The wrapper type (Loop vs StridedLoop) is preserved — strategies
    decided the iteration shape; materialization just walks.

    ``filter_emit`` dedupes ``Smem`` decls by name across the whole
    KernelOp body — software-pipelined ``AsyncBufferedStage``s share
    a buffer name with their prologue counterparts, and only the first
    decl should reach the rendered kernel.

    ``emit_tma_stage`` / ``emit_async_wait`` are closures that share
    TMA hoist + active-mbar state with the top-level walker."""
    inner: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, TmaBufferedStage):
            inner.extend(filter_emit(emit_tma_stage(s)))
        elif isinstance(s, Stage):
            inner.extend(filter_emit(_emit_stage(s, tid_expr, n_threads)))
        elif isinstance(s, AsyncWait):
            inner.extend(emit_async_wait(s))
        elif isinstance(s, (Loop, StridedLoop)):
            inner.append(_emit_loop(s, tid_expr, n_threads, transform, filter_emit, emit_tma_stage, emit_async_wait))
        else:
            inner.append(transform(s))
    return replace(loop, body=inner)


def _single_thread_var(thread_axes: tuple) -> str:
    """Combine + TreeHalve emit a single ``tid_var`` string. Only valid
    when there's exactly one THREAD axis — softmax-style cooperation
    (matmul has multi-axis THREAD set but doesn't emit Combine)."""
    if len(thread_axes) != 1:
        raise ValueError(f"Combine requires a single THREAD axis; got {len(thread_axes)}")
    return thread_axes[0].axis.name


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
        group_stage_names[gid].add(stage.name)
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
        if isinstance(stmt, Loop) and any(isinstance(s, TmaBufferedStage) for s in stmt.body):
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


def _build_linear_tid(thread_axes: tuple[BoundAxis, ...]):
    """Linear row-major thread index from the THREAD axes.

    Single-axis (softmax) → ``Var(name)``.
    Multi-axis (matmul) → ``m_i * BN + n_i`` for ``(m_i, n_i)``."""
    if len(thread_axes) == 1:
        return Var(thread_axes[0].axis.name)
    inner_stride = 1
    parts: list = []
    for ba in reversed(thread_axes):
        ext = int(ba.extent)
        if inner_stride == 1:
            parts.append(Var(ba.axis.name))
        else:
            parts.append(Var(ba.axis.name) * Literal(inner_stride, "int"))
        inner_stride *= ext
    expr = parts[0]
    for p in parts[1:]:
        expr = p + expr
    return expr


_WARP_SIZE = 32


def _emit_combine(accum: Accum, t: str, n_threads: int) -> list[Stmt]:
    """Emit the cross-thread combine producing ``<accum>_b``.

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

    The Tile renderer emits ``int lane = threadIdx.x & 31;`` and
    ``int warp = threadIdx.x >> 5;`` for any cooperative Tile with
    ``n_threads > WARP_SIZE`` so the hierarchical path's ``Var("lane")``
    / ``Var("warp")`` references resolve.
    """
    broadcast_name = f"{accum.name}_b"
    if n_threads <= _WARP_SIZE and (n_threads & (n_threads - 1)) == 0:
        return [WarpShuffle(name=broadcast_name, value=accum.name, op=accum.op, length=n_threads)]
    if n_threads % _WARP_SIZE == 0 and (n_threads & (n_threads - 1)) == 0:
        n_warps = n_threads // _WARP_SIZE
        smem_name = f"{accum.name}_smem"
        warp_w = f"{accum.name}_w"
        return [
            WarpShuffle(name=warp_w, value=accum.name, op=accum.op, length=_WARP_SIZE),
            Smem(name=smem_name, extents=(n_warps,)),
            Cond(
                cond=BinaryExpr("==", Var("lane"), Literal(0, "int")), body=(Write(output=smem_name, index=(Var("warp"),), value=warp_w),)
            ),
            Sync(),
            TreeHalve(buf=smem_name, op=accum.op, length=n_warps, tid_var="warp"),
            # TreeHalve's render ends each loop iter with __syncthreads(), so a
            # trailing Sync here would be a no-op pair with the loop's last sync.
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ]
    smem_name = f"{accum.name}_smem"
    return [
        Smem(name=smem_name, extents=(n_threads,)),
        Write(output=smem_name, index=(Var(t),), value=accum.name),
        Sync(),
        TreeHalve(buf=smem_name, op=accum.op, length=n_threads, tid_var=t),
        # See note above on TreeHalve's trailing sync.
        Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
    ]


# ---------------------------------------------------------------------------
# Stage expansion
# ---------------------------------------------------------------------------


def _assert_trivial_stage_body(stage: Stage) -> None:
    """Phase-1 trip-wire: materializer still reads legacy (buf, origin,
    addressing) fields, so the Stage body must be the auto-synthesized
    trivial pair ``Load(input=buf, ...) ; Write(output=name, ...)``.
    Anything richer (fused producer compute) requires phase 2.
    """
    if len(stage.body) != 2:
        raise AssertionError(
            f"Stage {stage.name!r}: expected trivial 2-stmt body (Load, Write), "
            f"got {len(stage.body)} stmts — fused stage bodies are not yet supported "
            f"by the materializer (phase 2 required)."
        )
    load, write = stage.body
    if not isinstance(load, Load) or load.input != stage.buf:
        raise AssertionError(f"Stage {stage.name!r}: body[0] must be Load(input={stage.buf!r}), got {load!r}")
    if not isinstance(write, Write) or write.output != stage.name:
        raise AssertionError(f"Stage {stage.name!r}: body[1] must be Write(output={stage.name!r}), got {write!r}")


def _emit_stage(stage: Stage, tid_expr, n_threads: int) -> list[Stmt]:
    """Expand a ``Stage`` Stmt into ``Smem`` decl + cooperative load + sync.

    The cooperative load reads a contiguous slab of ``stage.buf``
    starting at ``stage.origin`` (block-uniform) and spanning
    ``stage.axes`` extents. Each thread fetches one or more elements
    via a StridedLoop driven by ``tid_expr``: for a 1D slab, the loop
    iterates the cache axis directly; for N-D, it iterates a synthetic
    flat axis decoded into per-axis coords.

    Source index reconstruction depends on ``stage.addressing``:

    - ``AffineAddressing(dims)`` — fast path. ``source_index[d] =
      origin[d] + decoded[d]`` (the decoded slab coord, if any cache
      axis maps to ``d`` via ``dims``); else just ``origin[d]``.
    - ``TemplateAddressing(exprs)`` — escape hatch. Sigma-substitute
      cache-axis Vars → iter-decoded coords into ``exprs``.

    Emits a leading ``Sync`` so iterations 2+ of an enclosing serial
    loop (chunked-K matmul) wait for the prior iteration's compute to
    finish reading smem before this iteration overwrites it. Iteration
    1's leading Sync is harmless (no prior state)."""
    if not stage.axes:
        raise ValueError(f"Stage {stage.name!r} has no cache axes")
    _assert_trivial_stage_body(stage)
    extents = tuple(int(ax.extent) for ax in stage.axes)
    padded_extents = stage.alloc_extents

    # Iteration axis + per-cache-axis coord. Always synthesize a fresh
    # iter axis name so the cooperative-load ``for`` variable can't
    # collide with a same-named outer thread-decode variable (C++
    # init-expression scoping rules read the freshly-declared inner var,
    # giving an undefined initial value when the inner name shadows an
    # outer one). 1D cache: trivial map to the single cache axis. N-D:
    # row-major decode of the flat iter into per-axis coords.
    total = 1
    for e in extents:
        total *= e
    iter_axis = Axis(name=f"{stage.name}_flat", extent=total)
    if len(stage.axes) == 1:
        coord_for = {stage.axes[0].name: Var(iter_axis.name)}
    else:
        coord_for = _flat_decode(stage.axes, iter_axis.name)

    smem_index = tuple(coord_for[ax.name] for ax in stage.axes)
    if isinstance(stage.addressing, TemplateAddressing):
        # Non-affine path (``/``, ``%`` from collapsed-reshape views).
        # Cache extent equals the raw axis extent (no F-scale baked in),
        # so substituting cache-axis Vars with their iter coords into
        # the original Load index gives the right source address per
        # cache position. Affine cases stay on the additive
        # ``origin + decoded`` path because their cache extent IS scaled
        # by F, so iter coord directly equals the cache-relative source
        # position — no need to re-multiply via F.
        from deplodock.compiler.ir.sigma import Sigma as _Sigma

        cache_sigma = _Sigma({ax.name: coord_for[ax.name] for ax in stage.axes})
        source_index = tuple(cache_sigma.apply(e) for e in stage.addressing.exprs)
    else:
        assert isinstance(stage.addressing, AffineAddressing)
        decoded_per_dim = {dim: coord_for[ax.name] for dim, ax in zip(stage.addressing.dims, stage.axes, strict=True)}
        source_index = tuple(o if d not in decoded_per_dim else o + decoded_per_dim[d] for d, o in enumerate(stage.origin))

    # Buffered stages prepend a phase dim to the smem allocation and to
    # both the cooperative-load write index and (downstream) every body
    # Load. The leading Sync is dropped — ping-pong avoids the
    # prev-compute / next-load conflict by using different physical buffers.
    if isinstance(stage, BufferedStage):
        full_extents = (stage.buffer_count, *padded_extents)
        smem_index = (stage.phase, *smem_index)
        prelude: list[Stmt] = []  # leading Sync omitted
    else:
        full_extents = padded_extents
        prelude = [Sync()]

    if isinstance(stage, AsyncBufferedStage):
        # Async transport: emit cooperative cp.async + commit only. The
        # sibling ``AsyncWait`` Stmt that dominates every consumer
        # lowers to ``CpAsyncWait + Sync``; no implicit wait here.
        cooperative_load = StridedLoop(
            axis=iter_axis,
            start=tid_expr,
            step=Literal(n_threads, "int"),
            body=(CpAsyncCopy(smem=stage.name, smem_index=smem_index, src=stage.buf, src_index=source_index),),
        )
        return [Smem(name=stage.name, extents=full_extents), cooperative_load, CpAsyncCommit()]

    load_name = f"{stage.name}_v"
    cooperative_load = StridedLoop(
        axis=iter_axis,
        start=tid_expr,
        step=Literal(n_threads, "int"),
        body=(
            Load(name=load_name, input=stage.buf, index=source_index),
            Write(output=stage.name, index=smem_index, value=load_name),
        ),
    )
    return [*prelude, Smem(name=stage.name, extents=full_extents), cooperative_load, Sync()]


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
