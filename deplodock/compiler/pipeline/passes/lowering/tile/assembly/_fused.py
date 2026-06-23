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

from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import Block, Edge, Source, StageBundle, TileGraph, TileOp, Transport
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import _free_layers
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import synthesize_staging
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import _wrap_tower


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
    """A MAP producer block → ``(inputs, assigns, out_buf, out_value)`` — the pointwise
    transform ``xn = f(x, y, …)``. ``inputs`` is ``[(buf, name), …]``, one per input
    ``Load``. ``None`` for anything else: a reduce-bearing producer (the MONOID rmsnorm
    — a compute-phase reduce is future work) or an input whose index differs from the
    output's (a ``nw[k]``-style broadcast — different cache axes, also future work)."""
    stmts = list(block.compute)
    loads = [s for s in stmts if isinstance(s, Load)]
    writes = [s for s in stmts if isinstance(s, Write)]
    assigns = [s for s in stmts if isinstance(s, Assign)]
    if len(writes) != 1 or len(stmts) != len(loads) + len(writes) + len(assigns):
        return None  # not a flat pointwise body (a reduce loop / extra stmt)
    w = writes[0]
    inputs: list[tuple[str, str]] = []
    for ld in loads:
        if len(ld.names) != 1 or ld.index != w.index:
            return None  # broadcast / differing-shape operand — different cache axes
        inputs.append((ld.input, ld.names[0]))
    if not inputs:
        return None
    return inputs, tuple(assigns), w.output, w.values[0]


def _fuse_producers(body: Body, producer_of: dict[str, Block], graph: TileGraph) -> Body:
    """Patch every ``StageBundle`` whose source loads an intermediate ``xn``: swap the
    ``xn`` source for an ``x_smem`` source (the producer's gmem input) and emit the
    producer's transform as the bundle's ``compute`` phase writing the ``xn_smem`` slab."""

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
                    f"fused SMEM edge: producer of {src.buf!r} is not a MAP transform over same-shape operands "
                    "(MONOID/reduce or broadcast producers need a compute-phase reduce / multi-cache-axis fill — not yet supported)"
                )
            inputs, assigns, _out_buf, out_value = t
            cache_idx = tuple(Var(ax.name) for ax in src.cache_axes)
            # The producer transform over the slab: stage each input into its own
            # slab (same cache axes as xn), read them, apply the assigns, and write
            # the xn_smem slab (the consumer body already reads src.name).
            for in_buf, in_name in inputs:
                in_dtype = graph.buffers[in_buf].dtype if in_buf in graph.buffers else src.dtype
                in_src = replace(src, buf=in_buf, name=f"{in_buf}_smem", dtype=in_dtype)
                new_sources.append(in_src)
                compute.append(Load(names=(in_name,), input=in_src.name, index=cache_idx))
            compute += [*assigns, Write(output=src.name, index=cache_idx, value=out_value)]
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

    # Stage each intermediate edge into the consumer, then fold in the producers.
    staged = {e: t for e, t in graph.schedule.staged.items() if e.dst == consumer.name}
    for buf in intermediates:
        staged.setdefault(Edge(src=writer[buf], dst=consumer.name, buffer=buf), Transport.SYNC)
    sub = replace(graph, blocks=(consumer,), schedule=replace(graph.schedule, staged=staged))
    staged_body = synthesize_staging(sub)
    fused_body = _fuse_producers(staged_body, producer_of, graph)

    layers = _free_layers(consumer, graph.schedule)
    chain_body = _wrap_tower(layers, tuple(fused_body), atom=consumer.atom)
    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)
