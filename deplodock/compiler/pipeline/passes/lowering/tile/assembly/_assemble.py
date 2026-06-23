"""``assemble`` — the one deterministic step (block-DAG ``TileGraph`` → tower).

``plans/tile-ir-block-dag.md`` makes staging / pipelining / warp-spec / register
tiling / split-K / placement all the same kind of operation: a :class:`Schedule`
annotation over an invariant algorithm. By the time a ``TileGraph`` reaches here the
enumeration body moves have already σ-split it (F3-b: ``reduce_decomp`` re-bracketed
K, ``free_tile`` split the free axes); ``assemble`` **does no build** — it applies the
``Schedule`` to the stored algorithm and emits the ``TileOp`` tower (the migration
oracle is byte-identical CUDA — the downstream kernel/cuda passes stay untouched).

**Covered today: pointwise + scalar ``SEMIRING`` matmul (incl. masked / symbolic free
axes, split-K, ``FK`` strip-mine), with smem staging (R1).** ``assemble`` synthesizes
``Schedule.staged`` into slabs (``_slab``) then reconstructs the binding tower in the
layer order the legacy ``materialize._assemble`` produced (``REGISTER`` cells
innermost, then THREAD, then GRID, extra-outer GRID axes last) via the shared
:func:`_wrap_tower`, so the output is the same ``TileOp``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import Binding, Block, Edge, Schedule, TileGraph, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import synthesize_staging
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _wrap_tower
from deplodock.compiler.tensor import Tensor

# Schedule ``Binding`` → tower ``Role``. SERIAL has no free-axis use yet (the K
# re-bracket emits its own SERIAL_OUTER / STAGE_INNER layers); mapped to plain
# serial for completeness.
_ROLE_OF: dict[Binding, Role] = {
    Binding.GRID: Role.BLOCK,
    Binding.THREAD: Role.THREAD,
    Binding.REGISTER: Role.REGISTER,
    Binding.WARP: Role.WARP,
    Binding.ATOM: Role.ATOM,
}


def _free_layers(block: Block, sched) -> list[tuple]:
    """The innermost-first ``(axis, Role)`` layers for one block, in the exact
    order ``materialize._assemble`` emitted them: ATOM cells, REGISTER cells,
    WARP, THREAD, then GRID — each tier in ``block.domain`` order (so the inner
    ``N`` axis precedes the outer ``M`` axis, the split-K ``K_s`` and extra-outer
    GRID axes trail last). The K serial tower (``K_o`` / ``K_i``) is NOT a layer —
    the ``tile_axis`` reduce move embeds it directly in ``block.compute``."""
    binding = sched.binding

    def tier(b: Binding) -> list[tuple]:
        return [(a, _ROLE_OF[b]) for a in block.domain if binding.get(a.name) is b]

    return [
        *tier(Binding.ATOM),
        *tier(Binding.REGISTER),
        *tier(Binding.WARP),
        *tier(Binding.THREAD),
        *tier(Binding.GRID),
    ]


def assemble_block(
    graph: TileGraph,
    *,
    knobs: dict,
    base_knobs: dict,
    kernel_name: str,
    leading: tuple = (),
):
    """Assemble a ``TileGraph`` into a ``TileOp`` (single block) or a ``Graph`` of
    ``TileOp`` kernels (multi-block, one per launch group).

    Covers pointwise + scalar/warp matmul + cooperative reduce: each block's
    ``compute`` is the σ-rewritten inner body with any K serial tower already
    embedded (the ``tile_axis`` / ``partition_reduce`` body moves ran in
    ``build_dag``); ``assemble`` only reconstructs the binding tower via the shared
    :func:`_wrap_tower`. ``knobs`` / ``base_knobs`` / ``kernel_name`` are the
    deployed-variant stamp the downstream passes + perf DB key on (not part of the
    pure algorithm).

    For a single-block graph the return is the byte-identical ``TileOp`` the
    pipeline has always emitted. For a multi-block DAG (``Schedule.launch``
    partitions blocks into kernels — the edge-placement ``GMEM`` cut, R7) the return
    is a ``Graph`` fragment: one ``TileOp`` node per launch group, every cross-group
    edge materialized as a graph-node intermediate tensor (shape/dtype derived from
    the producer ``Write``). The partition is **deterministic** — same ``TileGraph``
    → byte-identical kernel set — per the RF invariant-guard discipline."""
    if len(graph.blocks) == 1:
        return _assemble_one(graph, graph.blocks[0], graph, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=leading)
    return _assemble_multi(graph, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=leading)


def _assemble_one(
    graph: TileGraph,
    block: Block,
    sub: TileGraph,
    *,
    knobs: dict,
    base_knobs: dict,
    kernel_name: str,
    leading: tuple,
) -> TileOp:
    """Assemble one ``block`` of ``graph`` into its ``TileOp`` tower. ``sub`` is the
    single-block ``TileGraph`` the slab synthesizer sees (the whole ``graph`` for a
    single-block input — byte-identical to the historical path; a per-block
    restriction for a multi-block DAG so each kernel only stages its own edges)."""
    atom = block.atom
    # Materialize ``Schedule.staged`` into smem slabs + a cooperative StageBundle
    # (a no-op when nothing is staged), then wrap the binding tower around it.
    staged_body = synthesize_staging(sub)
    layers = _free_layers(block, sub.schedule)
    chain_body = _wrap_tower(layers, tuple(staged_body), atom=atom)

    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)


def _restrict_schedule(sched: Schedule, block_name: str) -> Schedule:
    """The ``Schedule`` as one block sees it: the edge-keyed maps narrowed to edges
    consumed by this block (``dst == block_name``) so the slab synthesizer stages
    only this kernel's reads. The axis-/block-keyed fields (binding / scope / role /
    launch / unroll / grid_swizzle / reg_budget) are harmless to a block that does
    not reference them and pass through unchanged."""

    def keep(d: dict[Edge, object]) -> dict:
        return {e: v for e, v in d.items() if e.dst == block_name}

    return replace(
        sched,
        staged=keep(sched.staged),
        distance=keep(sched.distance),
        cohort=keep(sched.cohort),
        ring_depth=keep(sched.ring_depth),
        pad=keep(sched.pad),
    )


def _launch_groups(graph: TileGraph) -> dict[object, list[Block]]:
    """Partition ``blocks`` by ``Schedule.launch`` (one group = one kernel). A block
    with no explicit launch assignment is its own group (the default two-launch cut:
    every block a separate kernel)."""
    launch = graph.schedule.launch
    groups: dict[object, list[Block]] = {}
    for b in graph.blocks:
        groups.setdefault(launch.get(b.name, b.name), []).append(b)
    return groups


def _topo_blocks(graph: TileGraph) -> list[Block]:
    """``blocks`` in producer-before-consumer order, ties broken by the stored block
    order (so the kernel set is deterministic — same ``TileGraph`` → same order)."""
    block_of = {b.name: b for b in graph.blocks}
    names = [b.name for b in graph.blocks]
    deps: dict[str, set[str]] = {n: set() for n in names}
    for e in graph.edges:
        if e.src in deps and e.dst in deps:  # a block→block edge (not an input source)
            deps[e.dst].add(e.src)
    order: list[Block] = []
    done: set[str] = set()
    while len(order) < len(names):
        ready = [n for n in names if n not in done and deps[n] <= done]
        if not ready:
            raise ValueError(f"cycle in block DAG {graph.name}")
        order.append(block_of[ready[0]])
        done.add(ready[0])
    return order


def _assemble_multi(graph: TileGraph, *, knobs: dict, base_knobs: dict, kernel_name: str, leading: tuple) -> Graph:
    """Assemble a multi-block DAG into a ``Graph`` of ``TileOp`` kernels — one per
    launch group, cross-group edges materialized as intermediate graph tensors.

    v1 scope (``plans/dag-edge-placement-split-as-enumeration.md`` → "Sync scope vs.
    barrier mechanism"): every launch group is a single block (the two-launch cut —
    the kernel boundary is the grid barrier). A group with more than one block would
    be the cooperative ``grid.sync`` mechanism, a later enumeration field."""
    for gid, blocks in _launch_groups(graph).items():
        if len(blocks) != 1:
            raise NotImplementedError(
                f"multi-block launch group {gid!r} ({[b.name for b in blocks]}): the cooperative one-kernel grid.sync "
                "mechanism is a later field — v1 cuts take two launches (one block per group)"
            )
    order = _topo_blocks(graph)
    writer = {p.buffer: b.name for b in graph.blocks for p in b.writes}
    read_any = {p.buffer for b in graph.blocks for p in b.reads}
    out_bufs = [bn for bn in writer if bn not in read_any]  # written but never read internally = a graph output

    frag = Graph()
    # InputOp for every external read (a buffer no block writes), in first-read order.
    for b in order:
        for p in b.reads:
            if p.buffer in writer or p.buffer in frag.nodes:
                continue
            buf = graph.buffers[p.buffer]
            frag.add_node(InputOp(), [], Tensor(p.buffer, buf.shape, buf.dtype), node_id=p.buffer)
            frag.inputs.append(p.buffer)
    # One TileOp per block, node id = its single output buffer (so a consumer's
    # read of that buffer wires straight to the producer node).
    for b in order:
        writes = {p.buffer for p in b.writes}
        if len(writes) != 1:
            raise NotImplementedError(f"multi-block assemble: block {b.name!r} must write exactly one buffer, got {sorted(writes)}")
        out_buf = next(iter(writes))
        kname = kernel_name if out_buf in out_bufs else (b.name or f"{kernel_name}__{out_buf}")
        sub = replace(graph, blocks=(b,), schedule=_restrict_schedule(graph.schedule, b.name))
        tile_op = _assemble_one(graph, b, sub, knobs=knobs, base_knobs=base_knobs, kernel_name=kname, leading=leading)
        inputs = list(dict.fromkeys(p.buffer for p in b.reads))
        buf = graph.buffers[out_buf]
        frag.add_node(tile_op, inputs, Tensor(out_buf, buf.shape, buf.dtype), node_id=out_buf)
    frag.outputs = list(out_bufs)
    return frag


# Back-compat alias (the pointwise equivalence test + early callers).
assemble_pointwise = assemble_block
