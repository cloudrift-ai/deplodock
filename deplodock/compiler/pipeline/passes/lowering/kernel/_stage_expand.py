"""Stage-expansion helpers for ``100_materialize_tile``.

Producer scaffolding for wrap-body Stages: a transport ``Stage`` becomes a
cooperative ``Load + Write`` (or ``CpAsyncCopy``) nest, a ``ComputeStage``
becomes a σ-substituted cooperative ``StridedLoop``. Both flatten multi-axis
cache slabs via a row-major flat-iter decode. Pure functions — no shared
materializer state — so they live here rather than inside the pass.

The leading-underscore module name keeps the pass loader (which globs
``*.py`` skipping ``_``-prefixed files) from mistaking this for a rule.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync
from deplodock.compiler.ir.stmt import Load, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Stage, StagePolicy, TemplateAddressing


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
    stage: Stage,
    tid_expr,
    n_threads: int,
    *,
    policy: StagePolicy,
    buffer_count: int,
    phase,
    pipeline_depth: int,
) -> list[Stmt]:
    """Emit producer scaffolding for a Stage member of a StageBundle.

    For each ``Source`` in ``stage.sources``, emits per-source
    cooperative ``Load + Write`` (``CpAsyncCopy`` for ASYNC). Smem decls
    are hoisted to kernel scope in ``_materialize`` (we skip them via
    the dedup filter). Leading + trailing ``Sync`` bracket the
    cooperative load for SYNC policy; BUFFERED/ASYNC/TMA drop the
    leading sync (different physical slab per iter).

    Bundle context (policy, buffer_count, phase, pipeline_depth) is
    passed in; the Stage itself no longer carries those fields.
    """
    is_buffered = policy != StagePolicy.SYNC
    is_async = policy == StagePolicy.ASYNC

    prelude: list[Stmt] = [] if is_buffered else [Sync()]
    body_out: list[Stmt] = list(prelude)

    for src in stage.sources:
        # Per-Source iteration axis: synthesize a unique flat-iter axis
        # so the StridedLoop's loop variable doesn't collide with outer
        # thread-decode variables.
        cache_axes = src.cache_axes
        if not cache_axes:
            raise ValueError(f"Source {src.name!r} has no cache axes")
        extents = tuple(ax.extent.as_static() for ax in cache_axes)
        padded_extents = src.alloc_extents
        total = 1
        for e in extents:
            total *= e
        iter_axis = Axis(name=f"{src.name}_flat", extent=total)
        if len(cache_axes) == 1:
            coord_for = {cache_axes[0].name: Var(iter_axis.name)}
        else:
            coord_for = flat_decode(cache_axes, iter_axis.name)
        smem_index = tuple(coord_for[ax.name] for ax in cache_axes)
        # Per-source source-index reconstruction.
        addressing = src.addressing
        if isinstance(addressing, TemplateAddressing):
            from deplodock.compiler.ir.sigma import Sigma as _Sigma  # noqa: PLC0415

            cache_sigma = _Sigma({ax.name: coord_for[ax.name] for ax in cache_axes})
            source_index = tuple(cache_sigma.apply(e) for e in addressing.exprs)
        else:
            # Per source dim: composite-decode the cache axes mapping to it.
            # ``020_stage_inputs`` admits multi-axis-per-dim AffineAddressing
            # for the matmul ``(BN_thread, FN_register)`` collapse (and the M
            # analogue); the legacy ``{dim: coord_for[ax.name] for …}`` dict
            # comprehension silently OVERWROTE the entry when two axes shared
            # a dim, keeping only the last axis's decoded coord and producing
            # wrong gmem addresses on the cp.async path. Accumulate instead,
            # each cache axis weighted by its composite stride (product of
            # extents of subsequent cache axes that ALSO map to its dim) —
            # mirrors the materializer's ``box_per_dim`` collapse on the TMA
            # path and the composite-stride check in ``_derive_slab``.
            dims_tuple = addressing.dims
            decoded_per_dim: dict[int, Expr] = {}
            for i, (ax, d) in enumerate(zip(cache_axes, dims_tuple, strict=True)):
                stride = 1
                for j in range(i + 1, len(cache_axes)):
                    if dims_tuple[j] == d:
                        stride *= cache_axes[j].extent.as_static()
                term: Expr = coord_for[ax.name] if stride == 1 else BinaryExpr("*", coord_for[ax.name], Literal(stride, "int"))
                decoded_per_dim[d] = term if d not in decoded_per_dim else BinaryExpr("+", decoded_per_dim[d], term)
            source_index = tuple(o if d not in decoded_per_dim else o + decoded_per_dim[d] for d, o in enumerate(src.origin))
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


def emit_compute_stage(stage: Stage, tid_expr, n_threads: int, *, buffer_count: int) -> list[Stmt]:
    """Emit producer scaffolding for a ``ComputeStage``.

    Wraps ``stage.compute`` in a cooperative ``StridedLoop`` over the
    fused cache axes — each thread executes the compute template once
    per cell it owns, σ-substituting cache-axis Vars with row-major
    flat-iter decoded coords. ``stage.body`` (the consumer subtree) is
    pre-flattened to siblings by ``flatten_wrap_stages`` before we run,
    so it's empty here.

    The output Source's smem decl is hoisted to kernel scope by
    ``_materialize``'s prologue walk; the dedup filter drops the
    redundant in-line decl. Leading ``Sync`` ensures sibling-Stage
    smem writes are visible; trailing ``Sync`` makes the freshly
    computed slab CTA-visible to the consumer.
    """
    from deplodock.compiler.ir.sigma import Sigma  # noqa: PLC0415

    # Compute Stage member has exactly one output Source (the slab it fills).
    if stage.compute is None:
        raise ValueError("emit_compute_stage: stage.compute is None — not a compute member")
    if len(stage.sources) != 1:
        raise ValueError(f"compute Stage: expected 1 output Source, got {len(stage.sources)}")
    out = stage.sources[0]
    cache_axes = out.cache_axes
    if not cache_axes:
        raise ValueError(f"ComputeStage {out.name!r}: needs at least one cache axis")
    padded_extents = out.alloc_extents
    total = 1
    for ax in cache_axes:
        total *= ax.extent.as_static()
    iter_axis = Axis(name=f"{out.name}_flat", extent=total)
    if len(cache_axes) == 1:
        coord_for = {cache_axes[0].name: Var(iter_axis.name)}
    else:
        coord_for = flat_decode(cache_axes, iter_axis.name)
    sigma = Sigma(coord_for)

    # σ-substitute cache vars in every stmt of the compute body. Loads /
    # Assigns / Writes are leaves so a flat ``map`` over the body is
    # enough; no recursion into nested bodies is needed for the cone
    # shape this pass produces.
    new_stmts: list[Stmt] = []
    for s in stage.compute:
        new_stmts.append(s.rewrite(lambda n: n, sigma))

    full_extents = (buffer_count, *padded_extents) if buffer_count > 1 else padded_extents
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
