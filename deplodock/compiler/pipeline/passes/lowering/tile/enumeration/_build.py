"""Build the block-DAG ``TileGraph`` from the iteration DAG + a move choice.

The composer's front half (``plans/tile-ir-block-dag.md``): instead of
materializing the ``TileOp`` tower, ``build_dag`` emits the invariant algorithm (a
:class:`Block`) + a reference :class:`Schedule` (the binding), and ``lower`` wraps
it in the ``TileGraphOp`` the enumeration pass returns. One ``build_dag`` serves
every regime — a ``MAP`` nest tiles its free axes; a ``SEMIRING`` / ``MONOID``
reduce additionally re-brackets its contraction axis (``target_names``). There is
no per-shape code: the moves are algebra-licensed, applied uniformly.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Loop, Monoid, Select, SelectBranch, Stmt
from deplodock.compiler.ir.tile.ir import Atom, Binding, Block, RegisterTile, Schedule, TileGraph
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    COOP_BR,
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    RED_SPLITK,
)


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
    store guard (``plans/tile-ir-block-dag.md``: masked-ness is ``real_extent`` vs
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
    splitk = knobs[RED_SPLITK.name]
    if splitk <= 1:
        return None
    kax = dag.k_node.loop.axis
    return Axis(f"{kax.name}_s", splitk, source_axis=kax.source_axis or kax)


# === The per-pass body moves (``plans/tile-ir-block-dag.md`` F3-b). ===
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
    compute = _replace_k_scalar(
        tuple(block.compute), target_names, dag.k_extent, knobs[RED_BK.name], knobs[RED_FK.name], knobs[RED_SPLITK.name], k_s
    )
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
    free_specs = [(inner_n, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True)]
    if outer_m is not None:
        free_specs.append((outer_m, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False))

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


# === Cooperative-reduce (MONOID) build move (``plans/tile-ir-block-dag.md`` R2). ===
# The coop tower is the same kind of body move as the scalar ``reduce_decomp`` +
# ``free_tile``, but the carrier's commutative-licensed partition lands on a THREAD
# axis (``K_c`` — ``BR`` cooperative lanes per row) rather than split-K's BLOCK axis,
# and the free-axis register tile is forced to 1 (one element per cell-owner; the
# rows thread- or grid-parallelize). The cross-thread combine is NOT in the body —
# the reduce ``Accum.axes`` carry ``K_c`` through σ, and ``kernel/100_materialize_tile``
# synthesizes the warp-shuffle / hierarchical tree from ``Accum.axes ∩ ThreadTile``.


def _mask_reduce_accums(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask each ``Accum``'s folded value to the carrier's identity past a
    masked-K boundary (symbolic reduce). Before each ``Accum(name, value=V, op)``
    insert ``Init(V_kid, op)`` (the op's neutral element — ``0`` for add, ``-inf``
    for max) and ``Select(V_km, V if pred else V_kid)``, then fold ``V_km``. The
    Accum stays a direct child of the reduce loop (``is_reduce`` + the cross-thread
    combine intact); the Load was already index-clamped for a safe read."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Accum):
            ident = f"{c.value}_kid"
            masked = f"{c.value}_km"
            out.append(Init(name=ident, op=c.op, dtype=c.dtype or F32))
            out.append(
                Select(
                    name=masked,
                    branches=(SelectBranch(value=c.value, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
                )
            )
            out.append(replace(c, value=masked))
        else:
            out.append(c)
    return tuple(out)


def _replace_k_coop(
    stmts: tuple[Stmt, ...], target_names: frozenset[str], k_dim, bk: int, fk: int, br: int, k_c: Axis | None, k_bound: object
) -> tuple[Stmt, ...]:
    """Replace every K loop named in ``target_names`` — the reduce(s) and any
    second-pass map loop — with the cooperative tower, σ-mapping
    ``K → K_o·(br·bk·fk) + K_f·(br·bk) + K_i·br + K_c`` (``K_f`` only when
    ``fk > 1``; ``K_c`` the stride-1 thread lane only when ``br > 1``). Each loop
    gets its own ``K_o``/``K_i``/``K_f`` serial tiles but shares the one ``K_c``
    THREAD axis (laid into the domain by the caller). The reduce's ``Accum.axes``
    propagate ``K_c`` through σ → ``kernel/100`` emits the combine.

    When ``k_bound`` is set (symbolic K), ``K_o`` ceil-divides and the final
    partial tile is masked: the reduce clamps its load index for a safe read and
    folds the carrier identity past ``k_bound`` (``_mask_reduce_accums``); the map
    loop guards its store with ``Cond(decoded_k < k_bound)``."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            k_o_ext = k_dim.ceil_div(br * bk * fk) if k_bound is not None else k_dim // (br * bk * fk)
            k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{kn}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(br * bk * fk, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{kn}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(br * bk, "int")
            expr = expr + Var(k_i.name) * Literal(br, "int")
            if k_c is not None:
                expr = expr + Var(k_c.name)
            sigma_k = Sigma({kn: expr})
            if k_bound is not None:
                decoded_k = sigma_k.apply(Var(kn))
                pred = BinaryExpr("<", decoded_k, k_bound)
                if s.is_reduce:
                    # Clamp the read in-bounds, fold the carrier identity past the bound.
                    clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", k_bound, Literal(1, "int")))
                    body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in s.body)
                    new_body = _mask_reduce_accums(body_c, pred)
                else:
                    body_u = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
                    new_body = (Cond(cond=pred, body=Body(body_u)),)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            if k_f is not None:
                # ``reduce`` follows the loop: the reduce loops accumulate (FK
                # multiple-accumulator), the second-pass map loop only writes.
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def coop_build(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The cooperative-reduce build move: σ-split the free axes (THREAD = the
    pinned ``BN``/``BM``, REGISTER forced to 1) + re-bracket every K loop with the
    cooperative tower (the ``K_c`` THREAD lane, masked-K fill past a symbolic
    bound). The ``K_c`` axis is laid FIRST in ``Block.domain`` so it sits innermost
    in the THREAD tier (fastest threadIdx bits) — matching the legacy
    ``coop_thread`` placement; ``assemble`` reconstructs the tower from
    ``domain`` + ``Schedule.binding`` and the downstream combine rides
    ``Accum.axes``."""
    bk, fk, br = knobs[RED_BK.name], knobs[RED_FK.name], knobs[COOP_BR.name]
    bn = knobs.get(MAP_N_THREAD.name, 1)
    bm = knobs.get(MAP_M_THREAD.name, 1) if dag.outer_m is not None else 1

    inner_n, outer_m, extra_outer = _free_axes(dag)
    free_specs = [(inner_n, bn, 1, True)]
    if outer_m is not None:
        free_specs.append((outer_m, bm, 1, False))

    sigma_map: dict = {}
    bounds: list = []
    domain: list = []
    binding: dict[str, Binding] = {}

    kax = dag.k_node.loop.axis
    src_k = kax.source_axis or kax
    k_c = Axis(f"{kax.name}_c", br, source_axis=src_k) if br > 1 else None
    if k_c is not None:
        # K_c first → innermost in the THREAD tier (matches the legacy coop_thread order).
        domain.append(k_c)
        binding[k_c.name] = Binding.THREAD

    for axis, thread, reg, interleave in free_specs:
        a_b, a_t, a_r, expr, bound = _split_free_axis(axis, thread, reg, interleave_when_masked=interleave)
        sigma_map[axis.name] = expr
        if bound is not None:
            bounds.append((axis.name, bound))
        domain.extend((a_b, a_t, a_r))
        binding[a_b.name] = Binding.GRID
        binding[a_t.name] = Binding.THREAD
        binding[a_r.name] = Binding.REGISTER

    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    block = graph.blocks[0]
    targets = target_names or frozenset({kax.name})
    compute = tuple(s.rewrite(_identity_rename, sigma_outer) for s in block.compute)
    compute = _replace_k_coop(compute, targets, kax.extent, bk, fk, br, k_c, dag.k_bound)
    compute = _apply_masked_guards(compute, bounds, sigma_outer)
    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(compute))
    return replace(
        graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding={**graph.schedule.binding, **binding})
    )


# === Streaming-flash build move (the MONOID streaming schedule, ``plans/tile-ir-block-dag.md`` R6). ===
# The streaming reduce is the simplest reduce build: the free output axes tile like a
# MAP nest, and BOTH contraction axes serial-transform with ``bk=fk=splitk=1`` (each
# output element streams its own reduction; the carrier can't span register cells or
# split-K). For flash attention these axes are the streaming KV reduce + its nested
# QK^T reduce, the free axes q-rows / head-dim, and the coupled ``Monoid`` carrier is
# the online-softmax m/l/O recurrence — it rides through σ untouched
# (``kernel/100_materialize_tile`` + ``kernel/_combine`` synthesize its rescale). This
# is literally ``reduce_decomp`` (knobs forced to 1) then ``free_tile`` (register forced
# to 1), re-expressed as the two existing body moves.


def _mask_streaming_carrier(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask the streaming carrier's score to ``-inf`` past a symbolic-K bound (the
    flash-attention case: a key past the runtime ``seq_len``).
    Before each ``Monoid`` with partial ``(score, value)`` insert
    ``Init(score_kid, maximum)`` (seeds ``-inf``, the score component's identity) +
    ``Select(score_km, score if pred else score_kid)``, then fold ``score_km`` — the
    masked key contributes nothing (``max(m, -inf) = m``, ``exp(-inf) = 0``) while the
    in-place online-softmax state update stays unconditional."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Monoid):
            score = c.partial[0]
            ident, masked = f"{score}_kid", f"{score}_km"
            out.append(Init(name=ident, op=ElementwiseImpl("maximum"), dtype=F32))
            out.append(
                Select(
                    name=masked,
                    branches=(SelectBranch(value=score, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
                )
            )
            out.append(replace(c, partial=(masked, *c.partial[1:])))
        else:
            out.append(c)
    return tuple(out)


def _replace_k_streaming(
    stmts: tuple[Stmt, ...], target_names: frozenset[str], *, coop_axis: str | None = None, k_c: Axis | None = None, br: int = 1
) -> tuple[Stmt, ...]:
    """Serial-transform each streaming contraction loop (``bk=fk=splitk=1``): a static
    axis (e.g. flash attention's nested QK^T reduce) gets a plain ``K_o``/``K_i`` tower;
    a **symbolic** streaming axis (dynamic ``seq_len`` KV) ceil-divides, clamps its load index for a
    safe read, and masks the ``Monoid`` score to ``-inf`` past the runtime bound (the
    `Monoid` identity — fold nothing for an out-of-range key).

    When ``br > 1`` the streaming axis ``coop_axis`` instead σ-splits ``K → K_o·br +
    K_c`` (the ``br`` cooperative THREAD lanes ``k_c``, laid into the domain by the
    caller): each lane streams a strided KV slice into its own ``(m, l, O)`` partial,
    and the ``Monoid.axes`` pick up ``K_c`` through σ so ``kernel/100_materialize_tile``
    emits the cross-thread ``combine_states`` over the lanes."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            symbolic = not s.axis.extent.is_static
            inner = _replace_k_streaming(tuple(s.body), target_names, coop_axis=coop_axis, k_c=k_c, br=br)
            if k_c is not None and kn == coop_axis:
                # Cooperative streaming split: ``br`` THREAD lanes (``K_c``) reduce in
                # parallel, ``K_o`` serial over the strided slice. The carrier's
                # cross-lane combine is synthesized at materialize (rides ``Monoid.axes``).
                k_o = Axis(f"{kn}_o", s.axis.extent // br, source_axis=src)
                sigma_k = Sigma({kn: Var(k_o.name) * Literal(br, "int") + Var(k_c.name)})
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
                out.extend(_wrap_tower([(k_o, Role.SERIAL_OUTER)], new_body))
                continue
            k_o_ext = s.axis.extent.ceil_div(1) if symbolic else Literal(s.axis.extent.as_static(), "int")
            k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{kn}_i", 1, source_axis=src)
            sigma_k = Sigma({kn: Var(k_o.name) + Var(k_i.name)})
            if symbolic:
                decoded_k = sigma_k.apply(Var(kn))
                pred = BinaryExpr("<", decoded_k, s.axis.extent.expr)
                clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", s.axis.extent.expr, Literal(1, "int")))
                body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in inner)
                new_body = _mask_streaming_carrier(body_c, pred)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def streaming_build(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The streaming-flash build move (the MONOID streaming schedule, e.g. flash
    attention): serial-
    transform every contraction axis (``bk=fk=splitk=1``, masked streaming for a
    symbolic KV) then σ-split the free axes (``FM=FN=1``). The caller (``080_streaming``)
    pins those knobs.

    With ``BR > 1`` (cooperative-KV) the **static** streaming axis additionally lays a
    ``K_c`` THREAD lane (``br`` cooperative lanes per row, innermost in the THREAD
    tier): each lane reduces a strided KV slice into its own online-softmax partial,
    merged at materialize via the carrier's ``combine_states`` (``Accum.axes`` analog
    for the ``Monoid`` — ``kernel/100`` emits the warp-shuffle / smem-tree combine)."""
    block = graph.blocks[0]
    br = knobs.get(COOP_BR.name, 1)
    stream = dag.k_node.loop.axis
    k_c, coop_axis = None, None
    if br > 1 and stream.extent.is_static and stream.extent.as_static() % br == 0:
        k_c = Axis(f"{stream.name}_c", br, source_axis=stream.source_axis or stream)
        coop_axis = stream.name
    compute = _replace_k_streaming(tuple(block.compute), target_names, coop_axis=coop_axis, k_c=k_c, br=br)
    g = free_tile(replace(graph, blocks=(replace(block, compute=Body(compute)),)), dag, knobs, target_names=target_names)
    if k_c is not None:
        # Lay K_c FIRST in domain (innermost THREAD bits — matches coop_build, so the
        # segmented cross-lane combine stays an aligned intra-warp segment).
        nb = g.blocks[0]
        g = replace(
            g,
            blocks=(replace(nb, domain=(k_c, *nb.domain)),),
            schedule=replace(g.schedule, binding={**g.schedule.binding, k_c.name: Binding.THREAD}),
        )
    return g


# === Warp-tier (tensor-core ``atomize``) build move (``plans/tile-ir-block-dag.md`` R4). ===
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
    from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import TC_BK, TC_REG_M, TC_REG_N, WARP_M, WARP_N

    atom_m, atom_n, atom_k = atom.shape
    wm, wn = knobs[WARP_M.name], knobs[WARP_N.name]
    fm, fn = knobs[TC_REG_M.name], knobs[TC_REG_N.name]
    bk = knobs[TC_BK.name]

    inner_n, outer_m, extra_outer = _free_axes(dag)
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
