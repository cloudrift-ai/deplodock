"""Build the block-DAG ``TileGraph`` from the iteration DAG + a move choice.

The composer's front half: instead of
materializing the ``TileOp`` tower, ``build_dag`` emits the invariant algorithm (a
:class:`Block`) + a reference :class:`Schedule` (the binding), and ``lower`` wraps
it in the ``TileGraphOp`` the enumeration pass returns. One ``build_dag`` serves
every regime — a ``MAP`` nest tiles its free axes; a ``SEMIRING`` / ``MONOID``
reduce additionally re-brackets its contraction axis (``target_names``). There is
no per-shape code: the moves are algebra-licensed, applied uniformly.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32, DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Monoid, Select, SelectBranch, Stmt, Write
from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier
from deplodock.compiler.ir.tile.ir import Atom, Binding, Block, RegisterTile, Schedule, TileGraph
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag


def _free_axes(dag: IterDag) -> tuple[Axis, Axis | None, tuple[Loop, ...]]:
    """The free (PARALLEL) axes to tile, read off the DAG's parallel chain:
    ``(inner_n axis, outer_m axis | None, extra-outer loops)``."""
    parallel = dag.parallel
    inner_n = parallel[-1].loop.axis
    outer_m = parallel[-2].loop.axis if len(parallel) >= 2 else None
    extra_outer = tuple(n.loop for n in parallel[:-2])
    return inner_n, outer_m, extra_outer


def _split_free_axis(axis: Axis, thread: int, reg: int, *, interleave_when_masked: bool):
    """Build the block/thread/register axes + σ entry + optional store-guard bound
    for one free axis (``A → A_b·(T·R) + A_t·R + A_r``; interleaved on a masked
    inner axis)."""
    tr = thread * reg
    masked = (not axis.extent.is_static) or (axis.extent.as_static() % tr != 0)
    src = axis.source_axis or axis
    a_b = Axis(
        f"{axis.name}_b",
        axis.extent.ceil_div(tr) if masked else axis.extent // tr,
        source_axis=src,
        real_extent=axis.extent.as_static() if masked and axis.extent.is_static else None,
    )
    a_t = Axis(f"{axis.name}_t", thread, source_axis=src)
    a_r = Axis(f"{axis.name}_r", reg, source_axis=src)
    block = Var(a_b.name) * Literal(tr, "int")
    if masked and interleave_when_masked:
        expr = block + Var(a_r.name) * Literal(thread, "int") + Var(a_t.name)
    else:
        expr = block + Var(a_t.name) * Literal(reg, "int") + Var(a_r.name)
    bound = axis.extent.expr if masked else None
    return a_b, a_t, a_r, expr, bound


def _replace_k_scalar(
    stmts: tuple[Stmt, ...], target_names: frozenset[str], k_extent: int, bk: int, fk: int, splitk: int, k_s: Axis | None
) -> tuple[Stmt, ...]:
    """Replace every contraction loop named in ``target_names`` with a ``K_o``
    (serial-outer) / ``K_i`` (stage-inner) tower, σ-mapping
    ``K → K_s·(K_o_ext·bk·fk) + K_o·(bk·fk) + K_f·bk + K_i`` (``K_f`` only when
    ``fk > 1``, ``K_s`` only when ``splitk > 1``). The ``K_s`` grid axis is added to
    the outer BLOCK layers by the caller."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            this_extent = s.axis.extent.as_static() if s.axis.extent.is_static else k_extent
            k_o_ext = this_extent // (splitk * bk * fk)
            k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{kn}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * fk, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{kn}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(bk, "int")
            expr = expr + Var(k_i.name)
            if k_s is not None:
                expr = Var(k_s.name) * Literal(k_o_ext * bk * fk, "int") + expr
            sigma_k = Sigma({kn: expr})
            inner = _replace_k_scalar(tuple(s.body), target_names, k_extent, bk, fk, splitk, k_s)
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
            if k_f is not None:
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def _apply_masked_guards(body: tuple, bounds: list, sigma_outer: Sigma) -> tuple:
    """Wrap ``body`` in a boundary ``Cond`` per masked free axis — the derived
    store guard (masked-ness is ``real_extent`` vs
    tile, a derived ``Cond``). Mirrors ``materialize._assemble`` exactly, so the
    σ-split + guarded body stays byte-identical."""
    for name, bound in bounds:
        pred = sigma_outer.reduce(Var(name), SimplifyCtx({}))
        body = (Cond(cond=BinaryExpr("<", pred, bound), body=Body(body)),)
    return body


def _k_s_axis(dag: IterDag, knobs: dict, target_names: frozenset[str]) -> Axis | None:
    """The split-K GRID axis ``K_s`` (the cross-CTA partition factor), or ``None``
    when there is no contraction (``MAP``) or ``SPLITK == 1``. Computed identically
    by the reduce-decomp body move (it σ-maps ``K`` through ``K_s``) and the
    free-tile move (it binds ``K_s`` GRID) — a pure function of the knobs, so the
    two agree without sharing state."""
    if not target_names:
        return None
    splitk = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)]).cta
    if splitk <= 1:
        return None
    kax = dag.k_node.loop.axis
    return Axis(f"{kax.name}_s", splitk, source_axis=kax.source_axis or kax)


# === The per-pass body moves. ===
# ``build_dag`` is no longer a monolith called at one site: it is the COMPOSITION of
# the incremental moves the enumeration passes apply one at a time to the stored,
# knob-invariant algorithm. ``010_build`` seeds the logical block; ``060_reduce_tile``
# applies ``reduce_decomp`` (the K re-bracket); ``100_register_tile`` applies
# ``free_tile`` (the free-axis σ-split — needs both the thread + register knobs, so it
# is the one free-axis body move, at the last free fork). The composition order
# (reduce **then** free) is the reverse of the old monolith (free then reduce), but
# the two σ-rewrites touch disjoint axis sets (K vs the free N/M), so they commute and
# the built block is byte-identical — the migration oracle.


def seed_graph(dag: IterDag, *, kernel_name: str, buffers: dict | None = None) -> TileGraph:
    """The logical (un-tiled) algorithm: one ``Block`` whose ``compute`` is the DAG's
    inner body verbatim and whose ``domain`` / ``Schedule`` are empty. The moves below
    refine it in place; nothing is built all-at-once."""
    block = Block(name=kernel_name, domain=(), compute=Body(tuple(dag.inner_body)))
    return TileGraph(name=kernel_name, buffers=dict(buffers or {}), blocks=(block,), schedule=Schedule(binding={}))


def reduce_decomp(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The reduce-decomposition body move (``060_reduce_tile``): re-bracket each
    contraction axis named in ``target_names`` into the ``K_o`` (serial-outer) /
    ``K_i`` (stage-inner) tower with an optional ``K_f`` strip-mine, σ-mapped through
    the split-K factor ``K_s`` — embedded directly in ``compute``. No-op for a ``MAP``
    nest. Touches only the body (the ``K_s`` GRID binding is added by ``free_tile``)."""
    if not target_names:
        return graph
    k_s = _k_s_axis(dag, knobs, target_names)
    block = graph.blocks[0]
    d = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)])
    compute = _replace_k_scalar(tuple(block.compute), target_names, dag.k_extent, d.serial, d.fold, d.cta, k_s)
    return replace(graph, blocks=(replace(block, compute=Body(compute)), *graph.blocks[1:]))


def free_tile(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The free-axis ``tile_axis`` body move (``100_register_tile`` — the last free
    fork, where both the thread + register knobs are pinned): split each free
    (``PARALLEL`` / ``MAP``) axis ``A → A_b·(T·R) + A_t·R + A_r``, σ-rewriting the
    body, laying ``A_b, A_t, A_r`` into ``Block.domain`` (bound GRID / THREAD /
    REGISTER) with the split-K ``K_s`` + extra-outer axes trailing GRID, and wrapping
    each masked free axis in its boundary ``Cond``. This is the only free-axis split;
    ``090_thread_tile`` just pins the thread knob (the layout needs the register knob
    too, so a byte-identical split can't be staged across the two forks)."""
    inner_n, outer_m, extra_outer = _free_axes(dag)
    n_par, n_reg = fam.dec_split(knobs[fam.split_key(inner_n.name)])
    free_specs = [(inner_n, n_par, n_reg, True)]
    if outer_m is not None:
        m_par, m_reg = fam.dec_split(knobs[fam.split_key(outer_m.name)])
        free_specs.append((outer_m, m_par, m_reg, False))

    sigma_map: dict = {}
    bounds: list = []
    domain: list = []
    binding: dict[str, Binding] = {}
    for axis, thread, reg, interleave in free_specs:
        a_b, a_t, a_r, expr, bound = _split_free_axis(axis, thread, reg, interleave_when_masked=interleave)
        sigma_map[axis.name] = expr
        if bound is not None:
            bounds.append((axis.name, bound))
        domain.extend((a_b, a_t, a_r))
        binding[a_b.name] = Binding.GRID
        binding[a_t.name] = Binding.THREAD
        binding[a_r.name] = Binding.REGISTER

    k_s = _k_s_axis(dag, knobs, target_names)
    if k_s is not None:
        domain.append(k_s)
        binding[k_s.name] = Binding.GRID
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    block = graph.blocks[0]
    compute = tuple(s.rewrite(_identity_rename, sigma_outer) for s in block.compute)
    compute = _apply_masked_guards(compute, bounds, sigma_outer)
    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(compute))
    return replace(
        graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding={**graph.schedule.binding, **binding})
    )


# === MONOID build move — one move for the cooperative reduce AND the streaming flash ===
# A ``MONOID`` nest — a plain reduce (softmax LSE / rmsnorm stat / mean / max) **or** a
# streaming-flash nest (online-softmax over a nested QK^T contraction) — lowers through
# the SAME body move: σ-split the free axes (``free_tile``, register forced to 1) + apply
# the reduce-decomposition tower to **each** contraction axis the DAG exposes. A flat
# monoid exposes one contraction; a nested (streaming) monoid exposes the outer KV stream
# plus the inner QK^T reduce — the same move per axis, the cooperative ``K_c`` THREAD lane
# (the carrier's commutative-licensed partition, ``BR`` lanes per row) placed on the
# PRIMARY axis (``dag.k_node``) and the rest serial. The cross-thread / cross-lane combine
# is NOT in the body — the carrier's ``axes`` (``Accum.axes`` / ``Monoid.axes``) carry
# ``K_c`` through σ and ``kernel/100_materialize_tile`` + ``kernel/_combine`` synthesize
# the warp-shuffle / tree / online-softmax-rescale from ``carrier.axes ∩ ThreadTile``.


def _mask_partial(partial: str, op: ElementwiseImpl, dtype: DataType, pred: object) -> tuple[str, list[Stmt]]:
    """Neutralize one folded ``partial`` past a masked-K boundary: ``Init(kid, op)``
    (the op's neutral element — ``0`` for add, ``-inf`` for max) +
    ``Select(km, partial if pred else kid)``. Returns the masked name + the two
    stmts to splice before the carrier (the Load was already index-clamped for a
    safe read; this only neutralizes the contribution)."""
    ident, masked = f"{partial}_kid", f"{partial}_km"
    return masked, [
        Init(name=ident, op=op, dtype=dtype),
        Select(
            name=masked,
            branches=(SelectBranch(value=partial, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
        ),
    ]


def _mask_carrier(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask a symbolic-K reduce loop's body past the runtime bound, dispatching on
    the carrier the loop folds. A scalar ``Accum`` masks its folded ``value`` to the
    op's identity. A streaming ``Monoid`` (flash online-softmax) masks its score
    (``partial[0]``) to ``-inf`` (maximum's identity) so the masked key contributes
    nothing (``max(m, -inf) = m``, ``exp(-inf) = 0``) while the in-place state update
    stays unconditional; ``partial[1:]`` (the value) is left untouched. Both share
    the :func:`_mask_partial` skeleton; the carrier stays a direct child of the
    reduce loop (``is_reduce`` + the cross-thread combine intact)."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Accum):
            masked, stmts = _mask_partial(c.value, c.op, c.dtype or F32, pred)
            out.extend([*stmts, replace(c, value=masked)])
        elif isinstance(c, Monoid):
            masked, stmts = _mask_partial(c.partial[0], ElementwiseImpl("maximum"), F32, pred)
            out.extend([*stmts, replace(c, partial=(masked, *c.partial[1:]))])
        else:
            out.append(c)
    return tuple(out)


def _replace_k_monoid(
    stmts: tuple[Stmt, ...], target_names: frozenset[str], *, bk: int, fk: int, br: int, k_c: Axis | None, coop_axis: str
) -> tuple[Stmt, ...]:
    """Re-bracket each contraction loop named in ``target_names`` into the reduce
    tower — the one move serving both the flat cooperative reduce and the nested
    streaming flash. **Recursive**: a nested contraction (flash's QK^T reduce inside
    the KV stream) is towered before its enclosing loop. The cooperative ``K_c``
    THREAD lane (``br`` lanes per row) rides only the PRIMARY axis ``coop_axis``
    (``dag.k_node``); every other contraction is serial (``br = 1``). σ-mapping
    ``K → K_o·(br·bk·fk) + K_f·(br·bk) + K_i·br + K_c`` (``K_f`` only when ``fk > 1``;
    ``K_c`` only on ``coop_axis`` when ``br > 1``); the carrier's ``axes`` propagate
    ``K_c`` through σ → ``kernel/100`` emits the combine.

    A **symbolic** loop (dynamic ``seq_len`` KV) ceil-divides ``K_o`` and masks the
    final partial tile: a reduce clamps its load index for a safe read and folds the
    carrier identity past the bound (``_mask_carrier`` — ``Monoid`` score → ``-inf``,
    ``Accum`` → op identity); a second-pass map loop guards its store with
    ``Cond(decoded_k < bound)``."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            this_br = br if kn == coop_axis else 1
            this_kc = k_c if kn == coop_axis else None
            ext = s.axis.extent
            symbolic = not ext.is_static
            bound = ext.expr if symbolic else None
            inner = _replace_k_monoid(tuple(s.body), target_names, bk=bk, fk=fk, br=br, k_c=k_c, coop_axis=coop_axis)
            stride = this_br * bk * fk
            k_o_ext = ext.ceil_div(stride) if symbolic else ext // stride
            k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{kn}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(stride, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{kn}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(this_br * bk, "int")
            expr = expr + Var(k_i.name) * Literal(this_br, "int")
            if this_kc is not None:
                expr = expr + Var(this_kc.name)
            sigma_k = Sigma({kn: expr})
            if symbolic:
                decoded_k = sigma_k.apply(Var(kn))
                pred = BinaryExpr("<", decoded_k, bound)
                if s.is_reduce:
                    clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", bound, Literal(1, "int")))
                    body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in inner)
                    new_body = _mask_carrier(body_c, pred)
                else:
                    body_u = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
                    new_body = (Cond(cond=pred, body=Body(body_u)),)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
            if k_f is not None:
                # ``reduce`` follows the loop: the reduce loops accumulate (FK
                # multiple-accumulator), a second-pass map loop only writes.
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def monoid_build(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The MONOID build move — one move for the cooperative reduce (softmax / rmsnorm
    / mean / max) AND the streaming flash (online-softmax over a nested QK^T). Apply
    the reduce-decomposition tower to **each** contraction axis the DAG exposes
    (``_replace_k_monoid``, recursive for a nested stream) with the cooperative ``K_c``
    THREAD lane on the primary axis, then σ-split the free axes (``free_tile``, register
    forced to 1 by the caller). The ``K_c`` axis is laid FIRST in ``Block.domain`` so it
    sits innermost in the THREAD tier (fastest threadIdx bits); ``assemble`` reconstructs
    the tower from ``domain`` + ``Schedule.binding`` and the downstream combine rides the
    carrier's ``axes``.

    The caller pins the knobs per regime: a flat cooperative reduce searches ``(bk, fk,
    br)``; a streaming flash defaults ``bk = fk = 1`` and ``br`` over the static KV axis
    (cooperative-KV). ``BR > 1`` lays the cooperative lane — for the flat reduce the
    carrier's commutative partition, for the stream each lane's strided slice into its own
    online-softmax partial, merged at materialize via the carrier's ``combine_states``."""
    d = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)])
    bk, fk, br = d.serial, d.fold, d.coop
    targets = target_names or frozenset({dag.k_node.loop.axis.name})

    kax = dag.k_node.loop.axis
    # The cooperative lane rides the primary axis only. A symbolic streaming axis stays
    # serial (the offer set already returns ``br = 1`` for it); a flat symbolic reduce
    # tiles the masked K at the hint and may carry ``br`` (ceil-div, masked fill).
    k_c = Axis(f"{kax.name}_c", br, source_axis=kax.source_axis or kax) if br > 1 else None

    block = graph.blocks[0]
    compute = _replace_k_monoid(tuple(block.compute), targets, bk=bk, fk=fk, br=br, k_c=k_c, coop_axis=kax.name)
    g = free_tile(replace(graph, blocks=(replace(block, compute=Body(compute)),)), dag, knobs, target_names=target_names)
    if k_c is not None:
        # Lay K_c FIRST in domain (innermost THREAD bits), so the segmented cross-lane
        # combine stays an aligned intra-warp segment.
        nb = g.blocks[0]
        g = replace(
            g,
            blocks=(replace(nb, domain=(k_c, *nb.domain)),),
            schedule=replace(g.schedule, binding={**g.schedule.binding, k_c.name: Binding.THREAD}),
        )
    return g


# === The carried-contraction-chain build move (the shared-axis reduce_decomp) ===
# A ``MONOID(SEMIRING)``
# nest — a twisted carrier streaming over a nested contraction whose combine embeds a
# SECOND contraction (flash: online softmax over QK^T, with P@V embedded in
# ``O = O·α + p·v``) — is restructured so the **P@V free output ``d``** rides a register
# vector ``O[BM, D]`` *inside* the stream and the score is computed ONCE per KV step and
# **shared** across ``d`` (the INLINE score edge), instead of being recomputed per ``d``
# block. This is the FA-2 nest the warp tier needs: the score becomes a register
# fragment, the inner QK^T and the embedded P@V become two cells, and the twisted
# carrier splits into a **scalar stats carrier** (row max / denom, ``d``-invariant) +
# a **register-tiled accumulation carrier** (``O[d]``, reading the stats' rescale ``α``
# and probability ``p``). Not flash-specific: it dispatches on the compositional
# ``MONOID(SEMIRING)`` algebra the chain view exposes (``IterDag.chain``), the
# ``MONOID(SEMIRING)`` analog of ``monoid_build``.


def _chain_axes(dag: IterDag, value_load: Load) -> tuple[Axis, Axis, tuple[Loop, ...]]:
    """Classify the free axes of a chain nest off the def-use: the **P@V output ``d``**
    (a free axis in the value load's index but NOT in the inner QK^T contraction — the
    score is independent of it), the **query row ``m``** (in the inner contraction, not
    the value), and the shared **grid** axes (in both — batch / head). Structural, not
    a named-shape match."""
    parallel = {n.axis.name: n.axis for n in dag.parallel}
    inner_free = {v for ld in dag.chain.inner.body for v in _index_vars(ld)} & parallel.keys()
    value_free = {v for e in value_load.index for v in e.free_vars()} & parallel.keys()
    d_names = value_free - inner_free
    m_names = inner_free - value_free
    if len(d_names) != 1 or len(m_names) != 1:
        raise ValueError(f"chain_build: ambiguous chain free axes (d={d_names}, m={m_names})")
    d_axis = parallel[next(iter(d_names))]
    m_axis = parallel[next(iter(m_names))]
    grid = tuple(n.loop for n in dag.parallel if n.axis.name not in d_names | m_names)
    return d_axis, m_axis, grid


def _index_vars(stmt: Stmt) -> set[str]:
    """The index Vars a ``Load`` references (empty for a non-Load) — used to read a
    cell's free-axis footprint."""
    if isinstance(stmt, Load):
        return {v for e in stmt.index for v in e.free_vars()}
    return set()


def chain_build(graph: TileGraph, dag: IterDag, knobs: dict) -> TileGraph:
    """The shared-axis reduce_decomp (Phase 1c): restructure a ``MONOID(SEMIRING)``
    chain nest into the FA-2 form — the P@V output ``d`` as a register vector ``O[d]``
    inside the stream, the score computed once per KV step (the INLINE score edge,
    shared across ``d``), the twisted carrier split into a scalar stats carrier + a
    register-tiled accumulation carrier.

    ``d`` becomes a **REGISTER domain axis** (so the tower wraps the whole block once);
    the register-replication pass (``kernel/010_split_register_axes``) then replicates
    ONLY the statements that depend on the ``d`` var — the value load + the accumulation
    carrier ``O[d]`` — while the score + the scalar stats carrier (which the split made
    ``d``-independent) pass through shared, once per KV step. ``m`` binds THREAD, the
    shared axes GRID. The reduce axes stay serial (``BN=1`` per step — the per-KV online
    fold)."""
    chain = dag.chain
    if chain is None:
        raise ValueError("chain_build requires a carried-contraction-chain (streaming MONOID(SEMIRING)) nest")
    block = graph.blocks[0]
    body = tuple(block.compute)
    carrier = chain.carrier
    value_name = carrier.partial[1]

    # Locate the KV stream loop + its carrier / value load.
    kv_loop = next(s for s in body if isinstance(s, Loop) and s.axis.name == chain.hinge_name)
    kv_body = tuple(kv_loop.body)
    value_load = next(s for s in kv_body if isinstance(s, Load) and s.name == value_name)
    monoid = next(s for s in kv_body if isinstance(s, Monoid))
    prefix = tuple(s for s in kv_body if s is not value_load and s is not monoid)  # d-invariant score chain

    stats, accum, _d_state = split_carrier(carrier, value_name)
    d_axis, m_axis, grid = _chain_axes(dag, value_load)

    # ``d`` -> a REGISTER domain axis; ``m`` -> a THREAD tile (reg forced to 1).
    d_r = Axis(f"{d_axis.name}_r", d_axis.extent, source_axis=d_axis.source_axis or d_axis)
    m_split = knobs.get(fam.split_key(m_axis.name))
    m_thread = fam.dec_split(m_split)[0] if m_split is not None else 1
    m_b, m_t, m_r, m_expr, m_bound = _split_free_axis(m_axis, m_thread, 1, interleave_when_masked=True)

    # The carrier-state Inits + epilogue stay where they are; the register-replication
    # pass derives which (the ``O`` init / normalize / write) depend on ``d``.
    head_inits = tuple(s for s in body if isinstance(s, Init))
    epilogue = tuple(s for s in body if isinstance(s, (Assign, Write)))
    new_kv = replace(kv_loop, body=Body((*prefix, stats, value_load, accum)))
    compute: tuple[Stmt, ...] = (*head_inits, new_kv, *epilogue)

    # σ-rewrite: ``d`` -> the register axis var; ``m`` -> its block/thread split.
    sigma = Sigma({d_axis.name: Var(d_r.name), m_axis.name: m_expr})
    compute = tuple(s.rewrite(_identity_rename, sigma) for s in compute)
    if m_bound is not None:
        compute = _apply_masked_guards(compute, [(m_axis.name, m_bound)], sigma)

    # Domain ordered register..thread..grid (inner→outer) so the tower groups
    # GridTile > ThreadTile > RegisterTile cleanly; ``d_r`` is the innermost register.
    domain: list[Axis] = [d_r, m_r, m_t, m_b]
    binding: dict[str, Binding] = {
        d_r.name: Binding.REGISTER,
        m_r.name: Binding.REGISTER,
        m_t.name: Binding.THREAD,
        m_b.name: Binding.GRID,
    }
    for lp in reversed(grid):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID
    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(compute))
    return replace(graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding=binding))


# === Warp-tier (tensor-core ``atomize``) build move. ===
# The warp tower is the same kind of body move as the scalar ``free_tile`` /
# ``reduce_decomp``, but it splits each output axis FOUR ways
# (``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom``, bound GRID/WARP/REGISTER/ATOM)
# and re-brackets K at ``atom_k`` granularity, then FUSES the cell
# ``[Load,Load,mul,Accum]`` into one ``Mma`` via ``_atom.atomize_cell`` (the atom
# layer's body edit — provenance-agnostic, naming A/B by SSA value; ``Block.atom``
# then derives from that ``Mma``). The staging geometry below (free-axis σ-split,
# K re-bracket, gmem I/O) is the matmul-staging layer ``warp_build`` composes with
# the atom layer. ``assemble`` materializes the AtomTile/WarpTile tower around it
# via the shared ``_free_layers`` + ``_wrap_tower``.


def _warp_axis(axis: Axis, warp: int, reg: int, atom_cell: int):
    """4-level output-axis split for the warp tier:
    ``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom`` (the per-lane ``A_a`` offset
    is owned by ``mma.sync``, so it is NOT in σ — the ``A_a`` ATOM axis carries
    the cell extent but binds no σ term). A symbolic / non-divisible axis is
    masked: ``A_b`` ceil-divides and carries ``real_extent``; the boundary ``Expr``
    is returned for the per-cell store ``Cond``."""
    src = axis.source_axis or axis
    per_block = warp * reg * atom_cell
    masked = (not axis.extent.is_static) or (axis.extent.as_static() % per_block != 0)
    b_ext = axis.extent.ceil_div(per_block) if masked else axis.extent // per_block
    a_b = Axis(
        f"{axis.name}_b",
        b_ext,
        source_axis=src,
        real_extent=axis.extent.as_static() if (masked and axis.extent.is_static) else None,
    )
    a_w = Axis(f"{axis.name}_w", warp, source_axis=src)
    a_r = Axis(f"{axis.name}_r", reg, source_axis=src)
    a_a = Axis(f"{axis.name}_a", atom_cell, source_axis=src)
    expr = (
        Var(a_b.name) * Literal(per_block, "int")
        + Var(a_w.name) * Literal(reg * atom_cell, "int")
        + Var(a_r.name) * Literal(atom_cell, "int")
    )
    bound = axis.extent.expr if masked else None
    return a_b, a_w, a_r, a_a, expr, bound


def _replace_k_warp(stmts: tuple[Stmt, ...], k_name: str, k_dim, bk: int, atom_k: int, k_bound=None) -> tuple[Stmt, ...]:
    """Replace the K reduce loop with the ``atom_k``-strided ``K_o`` / ``K_i``
    tower: ``σ(K) = K_o·(bk·atom_k) + K_i·atom_k`` (each ``K_i`` step is one
    ``mma.sync`` over ``atom_k`` K-elements). No split-K / strip-mine (v1).

    For a **symbolic (masked) K** (``k_bound`` set — SDPA P@V over ``seq_len``),
    ``K_o`` ceil-divides the runtime extent (``ceil(seq_len/(bk·atom_k))`` serial
    steps, so the loop bound — and thus ``seq_len`` — enters the kernel signature
    and seq > hint is covered); the partial final tile past ``k_bound`` is
    ZERO-filled in smem / by the ``dpl_mma_load_*_kzero`` gmem-direct helper
    (``kernel/005_lower_atom_tile`` + ``_stage_expand``), so the mma accumulates 0
    past the runtime extent (a clamped duplicate would corrupt the reduction)."""
    out: list[Stmt] = []
    stride = bk * atom_k
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name == k_name:
            src = s.axis.source_axis or s.axis
            k_o_ext = k_dim.ceil_div(stride) if k_bound is not None else k_dim // stride
            k_o = Axis(f"{k_name}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{k_name}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * atom_k, "int") + Var(k_i.name) * Literal(atom_k, "int")
            new_body = tuple(c.rewrite(_identity_rename, Sigma({k_name: expr})) for c in s.body)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def warp_build(graph: TileGraph, dag: IterDag, knobs: dict, *, atom: Atom) -> TileGraph:
    """The warp-tier ``atomize`` build move: take the logical seed graph and σ-split
    the free axes four ways (GRID/WARP/REGISTER/ATOM) + re-bracket K at ``atom_k``
    granularity + fuse the cell into an ``Mma``, laying the domain axes + bindings.
    ``assemble`` reconstructs the AtomTile/WarpTile tower from ``domain`` + ``Mma``."""
    atom_m, atom_n, atom_k = atom.shape
    bk = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)]).serial

    inner_n, outer_m, extra_outer = _free_axes(dag)
    wn, fn = fam.dec_split(knobs[fam.split_key(inner_n.name)])
    wm, fm = fam.dec_split(knobs[fam.split_key(outer_m.name)])
    n_b, n_w, n_r, n_a, n_expr, n_bound = _warp_axis(inner_n, wn, fn, atom_n)
    m_b, m_w, m_r, m_a, m_expr, m_bound = _warp_axis(outer_m, wm, fm, atom_m)
    sigma_outer = Sigma({inner_n.name: n_expr, outer_m.name: m_expr})

    block = graph.blocks[0]
    new_inner = tuple(s.rewrite(_identity_rename, sigma_outer) for s in block.compute)
    new_inner = _replace_k_warp(new_inner, dag.k_node.loop.axis.name, dag.k_node.loop.axis.extent, bk, atom_k, dag.k_bound)
    new_inner = atomize_cell(new_inner, atom=atom, k_name=None, write=None)
    bounds = [(n, b) for n, b in ((inner_n.name, n_bound), (outer_m.name, m_bound)) if b is not None]
    new_inner = _apply_masked_guards(new_inner, bounds, sigma_outer)

    # Domain: N then M, each split b/w/r/a. ``_free_layers`` orders the tiers
    # (ATOM innermost … GRID outermost) for ``assemble``; extra-outer trail GRID.
    domain: list[Axis] = []
    binding: dict[str, Binding] = {}
    for a_b, a_w, a_r, a_a in ((n_b, n_w, n_r, n_a), (m_b, m_w, m_r, m_a)):
        domain.extend((a_b, a_w, a_r, a_a))
        binding[a_b.name] = Binding.GRID
        binding[a_w.name] = Binding.WARP
        binding[a_r.name] = Binding.REGISTER
        binding[a_a.name] = Binding.ATOM
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(new_inner))
    return replace(
        graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding={**graph.schedule.binding, **binding})
    )


def build_dag(
    dag: IterDag,
    knobs: dict,
    *,
    kernel_name: str,
    target_names: frozenset[str] = frozenset(),
    buffers: dict | None = None,
) -> TileGraph:
    """The full single-block ``TileGraph`` for a knob choice — now the COMPOSITION of
    the per-pass moves (``seed_graph`` → ``reduce_decomp`` → ``free_tile``), kept as
    one entry point for unit / equivalence callers. The enumeration passes apply the
    same three moves incrementally (F3-b); this wrapper makes the composition explicit
    and is the byte-identity oracle for that distribution."""
    graph = seed_graph(dag, kernel_name=kernel_name, buffers=buffers)
    graph = reduce_decomp(graph, dag, knobs, target_names=target_names)
    return free_tile(graph, dag, knobs, target_names=target_names)
