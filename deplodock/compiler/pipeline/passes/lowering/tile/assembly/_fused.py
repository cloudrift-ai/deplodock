"""The SMEM fused-edge assemble (``plans/dag-edge-placement-split-as-enumeration.md``).

The fused realization of an `SMEM`-placed edge: a MONOID/MAP producer `--xn-->` SEMIRING
consumer kept in **one kernel**, the `xn` intermediate riding an smem slab (the producer
fills it, the consumer `ldmatrix`/scalar-reads it — no gmem round-trip, the form that
*beats* the cut). The mechanism reuses the existing `StageBundle.compute` phase (the
"sibling-smem → own-smem producer template", lowered end-to-end by
`kernel/_stage_expand.emit_compute_phase`):

1. the consumer (matmul) is tiled normally and its `xn` operand staged (`synthesize_staging`
   gives a `StageBundle` whose source loads `xn` from gmem into `xn_smem`);
2. `_fuse_producers` then **patches** that bundle — the `xn` source becomes an `x_smem`
   source (the producer's gmem *input*), and the producer's transform becomes the bundle's
   `compute` phase writing `xn_smem` from `x_smem`;
3. the consumer body already reads `xn_smem` — unchanged.

So the producer rides the consumer's tiling (the slab cache axes), which is why the fused
edge is **shared-knob** (one kernel, one knob set) — no knob-namespace problem.

v1 scope: a **single-input MAP producer** (e.g. `relu(x)` / `scale·x`) — `emit_compute_phase`
lowers a pointwise compute body. A MONOID producer (rmsnorm) needs the compute phase
generalized to carry a reduce; this raises `NotImplementedError` for now.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    Binding,
    Block,
    CoopReduce,
    Edge,
    Source,
    StageBundle,
    TileGraph,
    TileOp,
    Transport,
)
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import _free_layers
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import synthesize_staging
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _wrap_tower


def is_fused_graph(graph: TileGraph) -> bool:
    """True iff ``graph`` is a multi-block DAG whose blocks share one launch group —
    the `SMEM`/`INLINE` fused case (one kernel), distinct from a `GMEM` cut (separate
    groups, the multi-launch `assemble`)."""
    if len(graph.blocks) < 2:
        return False
    launch = graph.schedule.launch
    groups = {launch.get(b.name, b.name) for b in graph.blocks}
    return len(groups) == 1


def fused_producer_blocks(graph: TileGraph) -> set[str]:
    """The names of the producer blocks in a fused graph — those writing an
    intermediate another block reads. They stay logical (un-tiled): the consumer rides
    them as its slab ``compute`` phase, so the assembly readiness check exempts them."""
    writer = {p.buffer: b.name for b in graph.blocks for p in b.writes}
    read_any = {p.buffer for b in graph.blocks for p in b.reads}
    return {writer[buf] for buf in writer if buf in read_any}


def _map_transform(block: Block):
    """A MAP producer block → ``(input_loads, assigns, write)`` — the pointwise transform
    ``xn = f(x, y, …)``: the input ``Load`` stmts (each indexing some subset of the
    output axes — a full operand, or a broadcast like ``rs[m]`` / ``nw[k]``), the
    transform ``Assign``\\ s, and the output ``Write``. ``None`` for a non-pointwise body
    (a reduce loop / extra stmt — the MONOID rmsnorm reduce, a compute-phase reduce being
    future work)."""
    stmts = list(block.compute)
    loads = [s for s in stmts if isinstance(s, Load)]
    writes = [s for s in stmts if isinstance(s, Write)]
    assigns = [s for s in stmts if isinstance(s, Assign)]
    if len(writes) != 1 or len(stmts) != len(loads) + len(writes) + len(assigns):
        return None  # not a flat pointwise body (a reduce loop / extra stmt)
    if not loads or any(len(ld.names) != 1 for ld in loads):
        return None
    return tuple(loads), tuple(assigns), writes[0]


def _project_source(xn_src: Source, load: Load, dim_of_axis: dict[str, int], dtype) -> Source | None:
    """The ``Source`` for one producer-input gmem ``Load``, by **projecting** the ``xn``
    slab's source onto the output dims the input varies over. A full operand (``x[m,k]``)
    keeps all cache axes; a broadcast (``rs[m]`` / ``nw[k]``) keeps only its dims' cache
    axes and pins the others to their constant gmem index — so ``rs`` is staged as an
    ``[m]`` slab read at the M sub-coords, ``nw`` as ``[k]`` over K. The operand shares
    the output's per-dim layout (anchor + atom-stride block), so the xn source's
    ``origin``/``block`` carry over per dim. ``None`` for an index dim with more than one
    free axis (a collapsed/transposed operand — not a simple broadcast)."""
    by_xn_dim: dict[int, list[tuple]] = {}  # xn source dim -> [(Axis, block)]
    block = xn_src.addressing.block
    for i, (ax, d) in enumerate(zip(xn_src.cache_axes, xn_src.addressing.dims, strict=True)):
        by_xn_dim.setdefault(d, []).append((ax, block[i] if block else 1))
    new_origin: list = []
    new_cache: list = []
    new_dims: list[int] = []
    new_block: list[int] = []
    for d, e in enumerate(load.index):
        fv = e.free_vars()
        if not fv:
            new_origin.append(e)  # a constant (broadcast / size-1) gmem dim
            continue
        if len(fv) != 1 or next(iter(fv)) not in dim_of_axis:
            return None  # >1 axis in one dim, or an axis not in the output's index
        xn_dim = dim_of_axis[next(iter(fv))]
        new_origin.append(xn_src.origin[xn_dim])
        for ax, blk in by_xn_dim.get(xn_dim, []):
            new_cache.append(ax)
            new_dims.append(d)
            new_block.append(blk)
    block_t = tuple(new_block) if any(b != 1 for b in new_block) else ()
    return replace(
        xn_src,
        name=f"{load.input}_smem",
        buf=load.input,
        dtype=dtype,
        cache_axes=tuple(new_cache),
        origin=tuple(new_origin),
        addressing=AffineAddressing(dims=tuple(new_dims), block=block_t),
    )


def _fuse_producers(body: Body, producer_of: dict[str, Block], graph: TileGraph) -> Body:
    """Patch every ``StageBundle`` whose source loads an intermediate ``xn``: swap the
    ``xn`` source for the producer's gmem input sources (each projected onto the axes it
    reads — full or broadcast) and emit the producer's transform as the bundle's
    ``compute`` phase writing the ``xn_smem`` slab."""

    def patch(stmt: Stmt) -> Stmt:
        nested = stmt.nested() if hasattr(stmt, "nested") else ()
        if nested:
            stmt = stmt.with_bodies(tuple(Body(tuple(patch(s) for s in b)) for b in nested))
        if not isinstance(stmt, StageBundle):
            return stmt
        new_sources: list[Source] = []
        compute: list[Stmt] = list(stmt.compute) if stmt.compute else []
        for src in stmt.sources:
            if src.buf not in producer_of:
                new_sources.append(src)
                continue
            t = _map_transform(producer_of[src.buf])
            if t is None:
                raise NotImplementedError(
                    f"fused SMEM edge: producer of {src.buf!r} is not a flat MAP transform "
                    "(the MONOID rmsnorm reduce needs a compute-phase reduce — not yet supported)"
                )
            input_loads, assigns, write = t
            # Map each producer output axis to the xn slab's source dim (read off the
            # producer Write index), so a broadcast operand projects onto the right axes.
            dim_of_axis = {v: d for d, e in enumerate(write.index) for v in e.free_vars()}
            for ld in input_loads:
                in_dtype = graph.buffers[ld.input].dtype if ld.input in graph.buffers else src.dtype
                op_src = _project_source(src, ld, dim_of_axis, in_dtype)
                if op_src is None:
                    raise NotImplementedError(f"fused SMEM edge: cannot project operand {ld.input!r} (collapsed/transposed index)")
                idx = tuple(Var(ax.name) for ax in op_src.cache_axes)
                if ld.input in graph.buffers:
                    new_sources.append(op_src)  # a gmem operand — stage it
                    compute.append(Load(names=ld.names, input=op_src.name, index=idx))
                else:
                    # an INTERNAL slab (the rmsnorm prologue's v4_smem, produced by the
                    # CoopReduce) — read it directly at its projected (M) cache axes, no
                    # gmem source to stage.
                    compute.append(Load(names=ld.names, input=ld.input, index=idx))
            compute += [*assigns, Write(output=src.name, index=tuple(Var(ax.name) for ax in src.cache_axes), value=write.values[0])]
        return replace(stmt, sources=tuple(new_sources), compute=Body(tuple(compute)) if compute else None)

    return Body(tuple(patch(s) for s in body))


def assemble_fused(graph: TileGraph, *, knobs: dict, base_knobs: dict, kernel_name: str, leading: tuple = ()) -> TileOp:
    """Assemble an `SMEM`-fused multi-block ``TileGraph`` into one ``TileOp``: the
    tiled consumer with each producer folded into its ``xn`` slab's ``compute`` phase."""
    writer = {p.buffer: b.name for b in graph.blocks for p in b.writes}
    read_any = {p.buffer for b in graph.blocks for p in b.reads}
    intermediates = {buf for buf in writer if buf in read_any}
    producer_of = {buf: graph.block(writer[buf]) for buf in intermediates}
    consumers = [b for b in graph.blocks if any(p.buffer in intermediates for p in b.reads)]
    if len(consumers) != 1:
        raise NotImplementedError(f"fused SMEM edge: expected one consumer block, got {[b.name for b in consumers]}")
    consumer = consumers[0]

    # A MONOID (rmsnorm) producer needs a reduce PROLOGUE — split it off as a CoopReduce,
    # the scale-application stays the compute phase (with the per-row scale read from the
    # prologue's smem slab as a broadcast operand).
    prologues: list[CoopReduce] = []
    producer_of = dict(producer_of)
    for buf, blk in list(producer_of.items()):
        split = _split_monoid_producer(blk)
        if split is None:
            continue
        prologue, scale_block = _build_reduce_prologue(split, buf, consumer, graph.schedule.binding)
        prologues.append(prologue)
        producer_of[buf] = scale_block  # the MAP scale-application (reads the v4 slab)

    # Stage each intermediate edge into the consumer, then fold in the producers.
    staged = {e: t for e, t in graph.schedule.staged.items() if e.dst == consumer.name}
    for buf in intermediates:
        staged.setdefault(Edge(src=writer[buf], dst=consumer.name, buffer=buf), Transport.SYNC)
    sub = replace(graph, blocks=(consumer,), schedule=replace(graph.schedule, staged=staged))
    staged_body = synthesize_staging(sub)
    fused_body = _fuse_producers(staged_body, producer_of, graph)

    layers = _free_layers(consumer, graph.schedule)
    if prologues:
        # Emit the prologue(s) as GridTile-level siblings before the matmul tower: build
        # the inner (sub-grid) tower, then wrap [CoopReduce…, inner] in the GRID layer.
        grid = [ll for ll in layers if ll[1] is Role.BLOCK]
        inner = [ll for ll in layers if ll[1] is not Role.BLOCK]
        inner_chain = _wrap_tower(inner, tuple(fused_body), atom=consumer.atom)
        chain_body = _wrap_tower(grid, (*prologues, *inner_chain))
    else:
        chain_body = _wrap_tower(layers, tuple(fused_body), atom=consumer.atom)
    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)


def _split_monoid_producer(block: Block):
    """A MONOID (rmsnorm) producer block → ``(leading, reduce_loop, scalars, scale_body,
    scale_indexed, v4_name)`` or ``None`` for a plain MAP. The block is ``[leading
    consts] [reduce Loop → acc] [scalar chain → v4] [scale Loop → xn]``; ``scale_body``
    is the scale loop's per-element body and ``v4_name`` the scale SSA the compute phase
    reads from the prologue slab."""
    stmts = list(block.compute)
    leading: list[Stmt] = []
    reduce_loop: Loop | None = None
    scalars: list[Stmt] = []
    scale_loop: Loop | None = None
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and reduce_loop is None:
            reduce_loop = s
        elif isinstance(s, Loop) and not s.is_reduce:
            scale_loop = s
        elif reduce_loop is None:
            leading.append(s)
        else:
            scalars.append(s)
    if reduce_loop is None or scale_loop is None or not scalars:
        return None
    v4_name = scalars[-1].defines()[0] if scalars[-1].defines() else None
    if v4_name is None:
        return None
    return tuple(leading), reduce_loop, tuple(scalars), tuple(scale_loop.body), v4_name


def _build_reduce_prologue(split, out_buf: str, consumer: Block, binding: dict) -> tuple[CoopReduce, Block]:
    """From a MONOID producer split, build the :class:`CoopReduce` prologue (the per-row
    reduce → ``<out_buf>__rscale`` smem slab) and the rewritten MAP scale block (the
    scale-application that reads that slab). The producer's logical row axis is σ-mapped
    to the consumer's M index (read off the consumer's output ``Write``), whose
    THREAD/REGISTER cache axes become the prologue's cooperative cells + slab index."""
    leading, reduce_loop, scalars, scale_body, v4_name = split
    # The consumer reads the intermediate ``xn[M, K]`` (2D) as its A operand — its M
    # index is the row the prologue must reduce. (The kernel output ``o`` may be batched
    # 3D, so its index[0] is the batch, not M — read M off the xn load instead.)
    xn_load = next(ld for ld in consumer.compute.iter_of_type(Load) if ld.input == out_buf)
    m_expr = xn_load.index[0]
    row_axis = _producer_row_axis(reduce_loop)
    cells = tuple(a for a in consumer.domain if a.name in m_expr.free_vars() and binding.get(a.name) in (Binding.THREAD, Binding.REGISTER))
    slab = f"{out_buf}__rscale"
    sigma = Sigma({row_axis: m_expr})
    body = (
        reduce_loop.rewrite(_id, sigma),
        *(s.rewrite(_id, sigma) for s in scalars),
        Write(output=slab, index=tuple(Var(c.name) for c in cells), value=v4_name),
    )
    prologue = CoopReduce(cells=cells, leading=Body(leading), body=Body(body), out_slab=slab, out_dtype=F32)
    # The scale block: the per-element scale-application reading v4 from the slab (an
    # internal-slab operand the broadcast machinery reads directly, no gmem staging).
    scale_block = Block(
        name=f"{out_buf}__scale",
        domain=(),
        compute=Body((Load(names=(v4_name,), input=slab, index=(Var(row_axis),)), *scale_body)),
    )
    return prologue, scale_block


def _producer_row_axis(reduce_loop: Loop) -> str:
    """The producer's logical row (M) axis — the non-reduce free var of the reduce's
    ``x`` load (its index is ``x[…, row, k]``; the reduce axis is the K it loops)."""
    k = reduce_loop.axis.name
    for ld in reduce_loop.body.iter_of_type(Load):
        rows = [v for e in ld.index for v in e.free_vars() if v != k]
        if rows:
            return rows[0]
    raise ValueError("reduce loop has no row-indexed load")


def _id(n: str) -> str:
    return n
