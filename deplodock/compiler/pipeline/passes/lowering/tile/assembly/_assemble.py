"""``assemble`` — the one deterministic step (block-DAG ``TileGraph`` → tower).

The block-DAG ``TileGraph`` makes staging / pipelining / warp-spec / register
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

import math
from dataclasses import replace

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import FragmentScale, LdmatrixLoad, RegFragment, RegStore, Smem, Sync
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Load, Loop, Mma, Monoid, Select, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    AffineAddressing,
    AtomTile,
    Binding,
    Block,
    CoopReduce,
    Edge,
    Schedule,
    SerialTile,
    Source,
    StageBundle,
    TileGraph,
    TileOp,
    Transport,
)
from deplodock.compiler.pipeline.passes.lowering._addr import add as _fadd
from deplodock.compiler.pipeline.passes.lowering._addr import mul as _fmul
from deplodock.compiler.pipeline.passes.lowering._predicates import map_transform, split_monoid_producer
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._frag_softmax import (
    FragGeom,
    realize_boundary_mask,
    realize_fragment_softmax,
    realize_score_mask,
)
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._slab import synthesize_staging
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import CarryScope, Role, _wrap_tower
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
    pipeline has always emitted. For a multi-block DAG ``assemble_block`` is the single
    entry that **realizes the placement lattice**:
    a same-launch-group multi-block graph (the ``SMEM``/``INLINE`` fused edge) assembles
    to **one** fused ``TileOp`` (``_assemble_group`` — the producer folds into the
    consumer's slab ``compute`` phase); a multi-launch DAG (``Schedule.launch`` partitions
    blocks into kernels — the ``GMEM`` cut, R7) returns a ``Graph`` fragment: one ``TileOp``
    node per launch group, every cross-group edge materialized as a graph-node intermediate
    tensor (shape/dtype derived from the producer ``Write``). The partition is
    **deterministic** — same ``TileGraph`` → byte-identical kernel set — per the RF
    invariant-guard discipline."""
    if graph.schedule.carry:
        # Warp-tier streaming flash: the ``Schedule.carry`` representation marks the kv-stream axis a
        # fragment-tier online-softmax carrier. The same single-block path, with the carrier realized
        # at the fragment tier (the produce/consume cells read off the graph) — funnels through the
        # one ``assemble_carry`` like every other kernel.
        return carry_scope_from_graph(graph, kernel_name=kernel_name)
    if is_fused_graph(graph):
        return _assemble_group(graph, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=leading)
    if len(graph.blocks) == 1:
        return _assemble_one(graph, graph.blocks[0], graph, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=leading)
    return _assemble_multi(graph, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=leading)


def assembly_ready(graph: TileGraph) -> bool:
    """Whether every block that needs tiling has a populated ``domain`` (the
    materialization-side readiness the assembly pass gates on — never a search-side knob).
    In a same-launch-group fused graph the producer blocks stay **logical** (they fold into
    the consumer's slab ``compute`` phase), so only the non-producer blocks must be tiled;
    otherwise every block must be."""
    skip = fused_producer_blocks(graph) if is_fused_graph(graph) else set()
    return all(b.domain for b in graph.blocks if b.name not in skip)


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
    # (a no-op when nothing is staged), then wrap the binding tower around it through the
    # generalized carry assembler. A matmul / scalar reduce / pointwise is the **embedded**
    # carry (``axis=None``): the enumeration K re-bracket already embedded the K serial tower
    # in the body, so the whole staged body is the degenerate ``consume`` phase — the SAME
    # ``assemble_carry`` flash uses, differing only in the carrier (here: no phase-built loop).
    staged_body = synthesize_staging(sub)
    layers = _free_layers(block, sub.schedule)
    chain_body = assemble_carry(CarryScope(consume=tuple(staged_body)), parallel_layers=layers, atom=atom)

    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)


def assemble_carry(carry: CarryScope, *, parallel_layers: list[tuple], atom=None) -> tuple[Stmt, ...]:
    """Materialize a :class:`CarryScope` into the tower — the **one** materialization every
    assembly path funnels through: the single-block kernels (``_assemble_one`` — pointwise /
    matmul / reduce), the SMEM-fused edge (``_assemble_group``), the multi-launch DAG
    (``_assemble_multi`` → ``_assemble_one``), and the warp-tier flash (``carry_scope_from_graph``).
    It is the sole caller of :func:`_wrap_tower`.

    The per-iteration phases concatenate in carrier order
    (``produce`` → ``merge`` → ``rescale`` → ``handoff`` → ``consume`` → ``update``),
    ``init`` brackets them above and ``epilogue`` below, and the parallel tower
    (GRID / WARP / THREAD / REGISTER, innermost-first ``(axis, Role)`` layers — the
    carry-axis analogue of :func:`_free_layers`) wraps the whole via the shared
    :func:`_wrap_tower`. Returns the tower stmt tuple.

    The carrier algebra + the presence of a reduction are what differ, not the
    materialization:

    - **embedded carry** (``carry.axis is None``) — a matmul / scalar reduce / pointwise:
      the enumeration K re-bracket already embedded the ``SERIAL_OUTER`` K-tower inside
      ``consume`` and the per-tile work is one in-place ``Accum`` / ``Mma``, so the phases
      ARE the body — no loop is built here (a ``MAP`` pointwise has no reduction at all,
      just ``consume`` = the σ-rewritten body).
    - **phase-built carry** (``carry.axis`` set) — a flat monoid reduce or flash: the phases
      are wrapped in a ``SERIAL_OUTER`` loop over ``carry.axis`` so the accumulator persists
      across the stream (flash populates every phase; a monoid reduce only ``merge`` /
      ``update``)."""
    phases = carry.produce + carry.merge + carry.rescale + carry.handoff + carry.consume + carry.update
    if carry.axis is not None:
        phases = (SerialTile(axis=carry.axis, body=Body(phases), kind="serial_outer"),)
    inner = carry.init + phases + carry.epilogue
    return _wrap_tower(parallel_layers, inner, atom=atom)


def synthesize_frag_handoff(prob_frags, *, slab: str, slab_dtype, a_frag: str, shape) -> list[Stmt]:
    """A **register-fragment → smem → register-fragment** staged edge (capability 2) — the
    fragment-source analog of the gmem cooperative stage (``_slab``). Stores each producer
    fragment (``prob_frags``, the m16n8 C-fragments) into ``slab`` at its column offset, a
    ``Sync``, then ``ldmatrix``-loads it back as the consumer A fragment ``a_frag`` in the A
    layout — a relayout-through-smem (what CUTLASS does between fused-GEMM stages). Reused by
    the flash C→A handoff; the eventual ``Schedule.staged`` ``FRAG`` transport will call it."""
    out: list[Stmt] = [
        RegStore(dst_buffer=slab, dst_index=(Literal(0, "int"), Literal(i * 8, "int")), frag=pf, shape=shape, ldm=16)
        for i, pf in enumerate(prob_frags)
    ]
    out.append(Sync())
    out.append(RegFragment(name=a_frag, role="a", shape=shape, dtype=slab_dtype))
    out.append(LdmatrixLoad(frag=a_frag, src_buffer=slab, src_index=(Literal(0, "int"),), role="a", ldm=16, staged=True, b_trans=False))
    return out


def _relabel_tile(tile: AtomTile, *, c: str, sfx: str, a: str | None = None, guards: dict | None = None) -> AtomTile:
    """Rename a graph-sourced flash AtomTile to the ``carry_scope_from_graph`` fragment scheme:
    suffix the cell's ``Load`` names (so sibling N-atom tiles don't collide), set the ``Mma``'s
    output fragment ``c`` (→ ``Sf{nt}`` / ``Of{n}``), optionally pin ``a`` (the P@V A = the handoff
    ``pa``) and merge symbolic edge guards. Recurses through the ``D>atom_k`` re-bracket SerialTile."""

    def walk(stmts: tuple[Stmt, ...], rename: dict[str, str]) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Load):
                nn = f"{s.names[0]}_{sfx}"
                rename[s.names[0]] = nn
                out.append(replace(s, names=(nn,)))
            elif isinstance(s, SerialTile):
                out.append(replace(s, body=Body(walk(tuple(s.body), rename))))
            elif isinstance(s, Mma):
                m = replace(s, c=c, a=(a if a is not None else rename.get(s.a, s.a)), b=rename.get(s.b, s.b))
                out.append(replace(m, **guards) if guards else m)
            else:
                out.append(s)
        return tuple(out)

    return replace(tile, body=Body(walk(tuple(tile.body), {})))


def carry_scope_from_graph(graph: TileGraph, *, kernel_name: str) -> TileOp:
    """The warp-tier-flash branch of :func:`assemble_block` — NOT a separate assembly entry. Reads
    the ``warp_chain_build`` atomized streaming ``TileGraph`` (the QK^T / P@V cells already fused by
    the generic ``atomize_cell``, σ-tiled to the warp geometry) and assembles the warp-chain
    ``CarryScope``: the produce / consume ``Mma`` cells come from the graph (no hand-authored
    ``Mma``), and only the fragment-tier phases (softmax via ``realize_fragment_softmax`` / scale /
    causal+boundary masks / the C→A handoff / the epilogue ``RegStore``) are realized here from the
    carrier + the geometry read off the ``TileGraph`` (``graph.buffers`` / ``block.domain``), then
    handed to the one ``assemble_carry`` every kernel funnels through. ``assemble_block`` dispatches
    here off ``Schedule.carry``. This is the graph walk that replaced the hand-assembled
    ``realize_flash``."""
    block = graph.blocks[0]
    out = block.writes[0].buffer  # the output buffer — the RegStore target

    # The kv-stream serial-outer carry + its full twisted carrier (the online-softmax Monoid that
    # ``warp_chain_build`` placed) + the produce (QK^T, transposed-B) / consume (P@V) AtomTiles.
    kv_loop = next(s for s in block.compute if isinstance(s, SerialTile) and s.kind == "serial_outer")
    carrier = next(s for s in kv_loop.body if isinstance(s, Monoid))
    causal = any(isinstance(s, Select) for s in kv_loop.body)  # the score-mask Select — structural, not a flag
    is_qkt = lambda t: any(isinstance(s, Mma) and s.b_trans for s in t.body.iter())  # noqa: E731
    produce_tiles = [t for t in kv_loop.body if isinstance(t, AtomTile) and is_qkt(t)]
    consume_tiles = [t for t in kv_loop.body if isinstance(t, AtomTile) and not is_qkt(t)]
    nt_count = len(produce_tiles)

    # Geometry off the graph, no flash descriptor: the atom off the Mma cell; the head dim ``D`` off
    # the P@V output-atom count (``nd·atom_n``); the seq extent off the kv-stream axis (its
    # ``source_axis`` carries the un-tiled extent — static ``S`` or the symbolic ``seq_len``).
    atom = produce_tiles[0].atom
    atom_m, atom_n, _atom_k = atom.shape
    ab_dt = atom.operand_dtype("a")
    nd = len(consume_tiles)
    D = nd * atom_n
    scale = f"{1.0 / math.sqrt(D)!r}f"
    seq_ext = (kv_loop.axis.source_axis or kv_loop.axis).extent
    symbolic = not seq_ext.is_static
    seq_var = next(iter(seq_ext.expr.free_vars())) if symbolic else None
    S = None if symbolic else seq_ext.as_static()
    seq = Var(seq_var) if symbolic else Literal(S, "int")

    qb = Var(block.domain[0].name)  # the query block axis (GRID), one warp / 16 query rows
    kv = Var(kv_loop.axis.name)  # the kv stream var
    qload = next(s for s in produce_tiles[0].body.iter() if isinstance(s, Load))
    bh_dims = tuple(qload.index[:2])  # (batch, head) — the leading 4D dims, shared with the output

    geom = FragGeom(
        atom_m=atom_m,
        atom_n=atom_n,
        score_frags=tuple(f"Sf{nt}_frag" for nt in range(nt_count)),
        prob_frags=tuple(f"Pf{nt}" for nt in range(nt_count)),
        accum_frags=tuple(f"Of{n}" for n in range(nd)),
    )
    fs = realize_fragment_softmax(carrier, geom=geom)

    init: list = list(fs.init)
    init += [RegFragment(name=f"Of{n}", role="c", shape=atom.shape, dtype=F32) for n in range(nd)]
    init.append(Smem(name="flash_pv_smem", extents=(16, 16), dtype=cuda_name(ab_dt), align=16))

    def _qk_guards(nt: int) -> dict:
        if not symbolic:
            return {}
        return {"m_guard": (_fmul(qb, 16), seq), "n_guard": (_fadd(_fmul(kv, 16), nt * atom_n), seq)}

    # Realize the score fragments + their fragment-tier transforms. The causal mask fires off the
    # graph's ``Select`` (``causal`` above), the boundary mask off the symbolic ``seq`` — both are
    # structural facts (a graph op / a shape extent), never an attention flag; the scale is keyed on
    # the shape fact ``D``.
    produce: list = [_relabel_tile(t, c=f"Sf{nt}", sfx=f"q{nt}", guards=_qk_guards(nt)) for nt, t in enumerate(produce_tiles)]
    produce += [FragmentScale(frag=f"Sf{nt}_frag", top=scale, bot=scale) for nt in range(nt_count)]
    kv_col_bases = tuple(_fadd(_fmul(kv, 16), nt * atom_n) for nt in range(nt_count))
    if causal:
        produce += realize_score_mask(geom, q_row_base=_fmul(qb, 16), kv_col_bases=kv_col_bases)
    if symbolic:
        produce += realize_boundary_mask(geom, kv_col_bases=kv_col_bases, bound=seq)

    merge: list = list(fs.merge)
    rescale: list = list(fs.rescale)
    handoff = synthesize_frag_handoff(geom.prob_frags, slab="flash_pv_smem", slab_dtype=ab_dt, a_frag="pa", shape=atom.shape)

    pv_kzero = {"k_zero": (_fmul(kv, 16), seq)} if symbolic else {}
    consume: list = [_relabel_tile(t, c=f"Of{n}", a="pa", sfx=f"v{n}", guards=pv_kzero) for n, t in enumerate(consume_tiles)]
    consume.append(Sync())

    store_mguard = (_fmul(qb, 16), seq) if symbolic else None
    epilogue: list = []
    for n in range(nd):
        epilogue.append(fs.epilogue[n])
        out_idx = (*bh_dims, _fmul(qb, 16), Literal(n * atom_n, "int"))
        epilogue.append(RegStore(dst_buffer=out, dst_index=out_idx, frag=f"Of{n}", shape=atom.shape, ldm=D, m_guard=store_mguard))

    carry = CarryScope(
        axis=kv_loop.axis,
        init=tuple(init),
        produce=tuple(produce),
        merge=tuple(merge),
        rescale=tuple(rescale),
        handoff=tuple(handoff),
        consume=tuple(consume),
        update=tuple(fs.update),
        epilogue=tuple(epilogue),
    )
    parallel_layers = [(Axis("w", Dim(1)), Role.WARP)] + [(a, Role.BLOCK) for a in block.domain]
    name = kernel_name if kernel_name.startswith("k_") else f"k_{kernel_name}"
    return TileOp(body=assemble_carry(carry, parallel_layers=parallel_layers), name=name, knobs={})


def _restrict_schedule(sched: Schedule, block_name: str) -> Schedule:
    """The ``Schedule`` as one block sees it: the edge-keyed ``staged`` map narrowed
    to edges consumed by this block (``dst == block_name``) so the slab synthesizer
    stages only this kernel's reads. The axis-/block-keyed fields (binding / launch)
    are harmless to a block that does not reference them and pass through unchanged."""

    def keep(d: dict[Edge, Transport]) -> dict[Edge, Transport]:
        return {e: v for e, v in d.items() if e.dst == block_name}

    return replace(sched, staged=keep(sched.staged))


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

    v1 scope: every launch group is a single block (the two-launch cut —
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


# ---------------------------------------------------------------------------
# The SMEM/INLINE fused edge.
#
# A MONOID/MAP producer ``--xn-->`` SEMIRING consumer kept in **one kernel**, the ``xn``
# intermediate riding an smem slab (the producer fills it, the consumer
# ``ldmatrix``/scalar-reads it — no gmem round-trip, the form that *beats* the cut). The
# mechanism reuses the existing ``StageBundle.compute`` phase (the "sibling-smem → own-smem
# producer template", lowered end-to-end by ``kernel/_stage_expand.emit_compute_phase``):
#
# 1. the consumer (matmul) is tiled normally and its ``xn`` operand staged
#    (``synthesize_staging`` gives a ``StageBundle`` whose source loads ``xn`` from gmem);
# 2. ``_fuse_producers`` then **patches** that bundle — the ``xn`` source becomes an
#    ``x_smem`` source (the producer's gmem *input*), and the producer's transform becomes
#    the bundle's ``compute`` phase writing ``xn_smem`` from ``x_smem``;
# 3. the consumer body already reads ``xn_smem`` — unchanged.
#
# So the producer rides the consumer's tiling (the slab cache axes), which is why the fused
# edge is **shared-knob** (one kernel, one knob set). This is the same-launch-group branch of
# ``assemble_block``'s placement-lattice dispatch (the GMEM cut takes separate launches above).
# ---------------------------------------------------------------------------


def is_fused_graph(graph: TileGraph) -> bool:
    """True iff ``graph`` is a multi-block DAG whose blocks share one launch group —
    the `SMEM`/`INLINE` fused case (one kernel), distinct from a `GMEM` cut (separate
    groups, the multi-launch ``_assemble_multi``)."""
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
            t = map_transform(producer_of[src.buf])
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
                if op_src.cache_axes == ():
                    # A fully-constant operand (a scalar like ``0.5`` in ``0.5·x``) varies
                    # over no output axis, so there is nothing to stage — read it straight
                    # from gmem at its (constant) index, per compute-phase element.
                    compute.append(Load(names=ld.names, input=ld.input, index=ld.index))
                elif ld.input in graph.buffers:
                    new_sources.append(op_src)  # a gmem operand — stage it
                    compute.append(Load(names=ld.names, input=op_src.name, index=idx))
                else:
                    # an INTERNAL slab (the rmsnorm prologue's v4_smem, produced by the
                    # CoopReduce) — read it directly at its projected (M) cache axes, no
                    # gmem source to stage.
                    compute.append(Load(names=ld.names, input=ld.input, index=idx))
            # Stamp the slab Write with the xn buffer dtype (``src.dtype``) so the fused
            # slab declares that dtype, not the value SSA's: a MONOID scale chain computes
            # in f32 (rsqrt / mean) but xn is f16, and the warp tier's ``ldmatrix`` reads
            # the slab as b16 — a float slab would feed it garbage. ``040_demote`` casts the
            # f32 result on store. (A MAP producer's value is already the slab dtype.)
            compute += [
                *assigns,
                Write(output=src.name, index=tuple(Var(ax.name) for ax in src.cache_axes), value=write.values[0], value_dtype=src.dtype),
            ]
        return replace(stmt, sources=tuple(new_sources), compute=Body(tuple(compute)) if compute else None)

    return Body(tuple(patch(s) for s in body))


def _assemble_group(graph: TileGraph, *, knobs: dict, base_knobs: dict, kernel_name: str, leading: tuple = ()) -> TileOp:
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
        split = split_monoid_producer(blk)
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
    # Materialize through the generalized carry assembler (embedded carrier — the fused
    # consumer's K tower is already in the body), the SAME path the single-block kernels and
    # flash use.
    if prologues:
        # Emit the prologue(s) as GridTile-level siblings before the matmul tower: build
        # the inner (sub-grid) tower, then wrap [CoopReduce…, inner] in the GRID layer.
        grid = [ll for ll in layers if ll[1] is Role.BLOCK]
        inner = [ll for ll in layers if ll[1] is not Role.BLOCK]
        inner_chain = assemble_carry(CarryScope(consume=tuple(fused_body)), parallel_layers=inner, atom=consumer.atom)
        chain_body = assemble_carry(CarryScope(consume=(*prologues, *inner_chain)), parallel_layers=grid)
    else:
        chain_body = assemble_carry(CarryScope(consume=tuple(fused_body)), parallel_layers=layers, atom=consumer.atom)
    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)


def _build_reduce_prologue(split, out_buf: str, consumer: Block, binding: dict) -> tuple[CoopReduce, Block]:
    """From a MONOID producer split, build the :class:`CoopReduce` prologue (the per-row
    reduce → ``<out_buf>__rscale`` smem slab) and the rewritten MAP scale block (the
    scale-application that reads that slab).

    The prologue fills one scale per M row of the **per-CTA M tile**, indexed by a single
    cooperative ``local_m`` axis of extent ``BM`` (= the product of the consumer's NON-grid
    M-axis extents — warp · register · atom at the warp tier, thread · register at the
    scalar tier). This decouples the reduce from how the consumer assigns rows to
    warps/threads: the producer's logical row σ-maps to ``m_grid·BM + local_m`` (``m_grid``
    the M GRID/block coord, in scope at the kernel top), so the reduce loads
    ``x[m_grid·BM + local_m, k]`` and writes ``rscale[local_m]``. The matmul's
    scale-application then reads ``rscale`` at its own within-tile M sub-coord (the
    materializer maps the producer row to the ``xn`` slab's M cache axis = the same
    ``[0, BM)`` index), so prologue fill and consumer read share one index space at every
    tier — fixing the warp tier, where the old THREAD/REGISTER-cell slab sized to 1 entry
    and the reduce referenced the not-yet-bound warp coord."""
    leading, reduce_loop, scalars, scale_body, v4_name = split
    # The consumer reads the intermediate ``xn[M, K]`` (2D) as its A operand — its M index
    # is the global row. (The kernel output ``o`` may be batched 3D, so its index[0] is the
    # batch, not M — read M off the xn load instead.) The M source axis + its per-CTA tile
    # come off that index's domain axes.
    xn_load = next(ld for ld in consumer.compute.iter_of_type(Load) if ld.input == out_buf)
    m_expr = xn_load.index[0]
    m_grid, bm = _m_tile(m_expr, consumer, binding)
    row_axis = _producer_row_axis(reduce_loop)
    slab = f"{out_buf}__rscale"
    local_m = Axis(f"{slab}_m", bm)
    global_row = (
        Var(local_m.name) if m_grid is None else BinaryExpr("+", BinaryExpr("*", Var(m_grid), Literal(bm, "int")), Var(local_m.name))
    )
    sigma = Sigma({row_axis: global_row})
    body = (
        reduce_loop.rewrite(_id, sigma),
        *(s.rewrite(_id, sigma) for s in scalars),
        Write(output=slab, index=(Var(local_m.name),), value=v4_name),
    )
    # The prologue and the consumer matmul are independently-numbered blocks fused into
    # one kernel; their SSA namespaces collide (both reuse ``in0``/``v0``/…). That breaks
    # the SSA invariant the renderer relies on — its literal-constant env is keyed by SSA
    # name and kernel-global, so the prologue's constant operand loads (``xn_mean_count`` =
    # 1024, ``xn_eps`` = 1e-6) would clobber the consumer's identically-named operand loads
    # (the matmul's ``v0 = in0·in1`` rendering as ``1024·1e-6``). Prefix the prologue's SSA
    # value names so the producer side is disjoint; its only cross-block link is the slab
    # buffer (the scale block re-reads it under its own name), so renaming internal SSA is
    # sound. Restrict the rename to actual defs — axis/grid Vars in index exprs share the
    # Var namespace and must NOT be touched.
    pfx = f"{slab}__"
    defined = {n for s in (*leading, *body) for st in Body.coerce((s,)).iter() for n in st.defines()}
    ren = lambda n: f"{pfx}{n}" if n in defined else n  # noqa: E731
    leading = tuple(s.rewrite(ren) for s in leading)
    body = tuple(s.rewrite(ren) for s in body)
    prologue = CoopReduce(cells=(local_m,), leading=Body(leading), body=Body(body), out_slab=slab, out_dtype=F32)
    # The scale block: the per-element scale-application reading v4 from the slab (an
    # internal-slab operand the broadcast machinery reads directly, no gmem staging). It
    # indexes by the producer row, which the materializer resolves to the xn slab's M
    # sub-coord — the same ``[0, BM)`` index the prologue filled.
    scale_block = Block(
        name=f"{out_buf}__scale",
        domain=(),
        compute=Body((Load(names=(v4_name,), input=slab, index=(Var(row_axis),)), *scale_body)),
    )
    return prologue, scale_block


def _m_tile(m_expr, consumer: Block, binding: dict) -> tuple[str | None, int]:
    """The consumer's M tiling for the reduce prologue: ``(m_grid_axis_name | None, BM)``,
    where ``m_grid`` is the M axis bound GRID (the block coord, in scope at the kernel top)
    and ``BM`` is the product of the NON-grid M-axis extents (the per-CTA M tile —
    warp·register·atom at the warp tier, thread·register at the scalar tier; the atom M
    lane is a domain axis even though it carries no σ term, so it counts toward ``BM``).
    The M source axis is read off any free var of ``m_expr`` (the xn-load M index)."""
    fv = m_expr.free_vars()
    m_src = next(((a.source_axis or a).name for a in consumer.domain if a.name in fv), None)
    m_axes = [a for a in consumer.domain if (a.source_axis or a).name == m_src]
    m_grid = next((a.name for a in m_axes if binding.get(a.name) is Binding.GRID), None)
    bm = 1
    for a in m_axes:
        if binding.get(a.name) is not Binding.GRID:
            bm *= a.extent.as_static()
    return m_grid, bm


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
