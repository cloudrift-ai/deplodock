"""Realize a complete move choice into the ``TileOp`` tower.

Reproduces the legacy planner's σ-split + tower so the output is accuracy-
equivalent. Each free axis ``A`` splits as ``A → A_b·(T·R) + A_t·R + A_r``
(interleaved ``A_b·(T·R) + A_r·T + A_t`` on a masked *inner* axis), the body is
σ-rewritten, a reduce axis is re-bracketed into a ``K_o`` (serial-outer) /
``K_i`` (stage-inner) tower with an optional ``K_f`` strip-mine, masked axes get
a ``Cond`` store guard, and the layers wrap innermost-first via the shared
:func:`_wrap_tower`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Loop, Monoid, Select, SelectBranch, Stmt
from deplodock.compiler.ir.tile.ir import Atom, RegisterTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    COOP_BR,
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    RED_SPLITK,
    TC_BK,
    TC_REG_M,
    TC_REG_N,
    WARP_M,
    WARP_N,
)

# One free axis to split: (axis, thread factor, register factor, interleave-when-masked).
_FreeSpec = tuple[Axis, int, int, bool]


def _free_axes(dag: IterDag) -> tuple[Axis, Axis | None, tuple[Loop, ...]]:
    """The free (PARALLEL) axes the builders tile, read off the DAG's parallel
    chain: ``(inner_n axis, outer_m axis | None, extra-outer loops)``. The
    skeleton's ``inner_n`` / ``outer_m`` / ``extra_outer`` are projections of
    exactly these nodes, so reading the DAG is byte-identical."""
    parallel = dag.parallel
    inner_n = parallel[-1].loop.axis
    outer_m = parallel[-2].loop.axis if len(parallel) >= 2 else None
    extra_outer = tuple(n.loop for n in parallel[:-2])
    return inner_n, outer_m, extra_outer


def _split_free_axis(axis: Axis, thread: int, reg: int, *, interleave_when_masked: bool):
    """Build the block/thread/register axes + σ entry + optional store-guard
    bound for one free axis."""
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


def _assemble(
    free_specs: list[_FreeSpec],
    inner_body: tuple[Stmt, ...],
    *,
    leading: tuple[Stmt, ...],
    extra_outer: tuple[Loop, ...],
    knobs: dict,
    base_knobs: dict,
    kernel_name: str,
    k_transform: Callable[[tuple[Stmt, ...]], tuple[Stmt, ...]] | None = None,
    coop_thread: tuple[Axis, ...] = (),
    extra_block: tuple[Axis, ...] = (),
) -> TileOp:
    """Shared tower assembly: split free axes, σ-rewrite the body, apply the K
    transform (matmul), wrap masked-axis guards, and build the tile tower.

    ``coop_thread`` are cooperative-K THREAD axes (``K_c``) placed innermost in
    the THREAD tier (fastest threadIdx bits); ``extra_block`` are extra GridTile
    axes (``K_s`` split-K) placed outside the free-axis blocks."""
    sigma_map: dict = {}
    bounds: list[tuple[str, object]] = []
    layers_reg: list[tuple[Axis, Role]] = []
    layers_thread: list[tuple[Axis, Role]] = []
    layers_block: list[tuple[Axis, Role]] = []
    for axis, thread, reg, interleave in free_specs:
        a_b, a_t, a_r, expr, bound = _split_free_axis(axis, thread, reg, interleave_when_masked=interleave)
        sigma_map[axis.name] = expr
        if bound is not None:
            bounds.append((axis.name, bound))
        layers_reg.append((a_r, Role.REGISTER))
        layers_thread.append((a_t, Role.THREAD))
        layers_block.append((a_b, Role.BLOCK))

    sigma_outer = Sigma(sigma_map)
    new_inner: tuple[Stmt, ...] = tuple(s.rewrite(_identity_rename, sigma_outer) for s in inner_body)
    if k_transform is not None:
        new_inner = k_transform(new_inner)
    for name, bound in bounds:
        pred = sigma_outer.reduce(Var(name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", pred, bound), body=Body(new_inner)),)

    layers: list[tuple[Axis, Role | None]] = [
        *layers_reg,
        *[(ax, Role.THREAD) for ax in coop_thread],
        *layers_thread,
        *layers_block,
        *[(ax, Role.BLOCK) for ax in extra_block],
    ]
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(extra_outer))
    chain_body = _wrap_tower(layers, new_inner)

    knobs_full = {**base_knobs, **knobs}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)


def build_pointwise_tile(knobs: dict, *, kernel_name: str, base_knobs: dict, dag: IterDag) -> TileOp:
    inner_n, outer_m, extra_outer = _free_axes(dag)
    free_specs: list[_FreeSpec] = [(inner_n, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True)]
    if outer_m is not None:
        free_specs.append((outer_m, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False))
    return _assemble(
        free_specs,
        dag.inner_body,
        leading=dag.leading,
        extra_outer=extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
    )


def _replace_k_scalar(
    stmts: tuple[Stmt, ...],
    target_names: frozenset[str],
    k_extent: int,
    bk: int,
    fk: int,
    splitk: int,
    k_s: Axis | None,
    k_bounds: dict | None = None,
) -> tuple[Stmt, ...]:
    """Replace every contraction loop named in ``target_names`` with a ``K_o``
    (serial-outer) / ``K_i`` (stage-inner) tower, σ-mapping
    ``K → K_s·(K_o_ext·bk·fk) + K_o·(bk·fk) + K_f·bk + K_i`` (``K_f`` only when
    ``fk > 1``, ``K_s`` only when ``splitk > 1``). For a plain matmul this is the
    one K loop; a multi-accumulator matmul (gated MLP) has several same-K loops,
    each its own ``K_o``/``K_i`` tower (sharing the ``K_s`` grid axis). The
    ``K_s`` grid axis is added to the outer BLOCK layers by the caller.

    ``k_bounds`` (symbolic flash) maps a streaming-reduce axis name to its runtime
    boundary ``Expr``: that axis' ``K_o`` ceil-divides and each step is wrapped in
    ``Cond(decoded_k < bound)`` — the TWISTED_MONOID identity "skip the fold"
    leaves the online-softmax state (m/l/O) unchanged for an out-of-range key."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            kn = s.axis.name
            src = s.axis.source_axis or s.axis
            bound = (k_bounds or {}).get(kn)
            if bound is not None:
                # Symbolic streaming axis: ceil-div over the runtime extent.
                k_o_ext = s.axis.extent.ceil_div(splitk * bk * fk)
            else:
                # Each target loop tiles by its OWN extent (a flash nest's KV and
                # nested QK reduces differ); a plain matmul's K matches k_extent.
                this_extent = s.axis.extent.as_static() if (s.axis is not None and s.axis.extent.is_static) else k_extent
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
            # Recurse first so a nested target loop (flash's QK inside KV) is also
            # split, then σ-rewrite this loop's body.
            inner = _replace_k_scalar(tuple(s.body), target_names, k_extent, bk, fk, splitk, k_s, k_bounds)
            if bound is not None:
                # Symbolic streaming step. A `Cond` around the carrier would
                # block-scope the loop-carried online-softmax state (m/l/O), so
                # instead CLAMP the streaming index in every load (safe read of a
                # duplicate key) and force the masked key's SCORE to the carrier's
                # identity (`-inf`) — `max(m, -inf) = m`, `exp(-inf) = 0`, so the
                # Monoid folds nothing for an out-of-range key (TWISTED_MONOID
                # identity) while the in-place state update stays unconditional.
                decoded_k = sigma_k.apply(Var(kn))
                pred = BinaryExpr("<", decoded_k, bound)
                clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", bound, Literal(1, "int")))
                body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in inner)
                new_body = _mask_flash_monoid(body_c, pred)
            else:
                new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
            if k_f is not None:
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def build_matmul_tile(knobs: dict, *, kernel_name: str, base_knobs: dict, dag: IterDag, target_names: frozenset[str]) -> TileOp:
    inner_n, outer_m, extra_outer = _free_axes(dag)
    free_specs: list[_FreeSpec] = [
        (inner_n, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True),
        (outer_m, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False),
    ]
    bk, fk, splitk = knobs[RED_BK.name], knobs[RED_FK.name], knobs[RED_SPLITK.name]
    kax = dag.k_node.loop.axis
    src_k = kax.source_axis or kax
    k_s = Axis(f"{kax.name}_s", splitk, source_axis=src_k) if splitk > 1 else None
    targets = target_names or frozenset({kax.name})
    # Scalar tier: stamp the warp-tier knobs as explicit OFF sentinels (``MMA="0"``
    # so ``is_warp`` reads False; ``WM=WN=0``) plus the scalar ``BR=1`` (no
    # cooperative-K), so the leaf carries the complete, uniform knob set the tune
    # DB / prior key on (the warp builder stamps ``MMA=<atom>`` + ``WM``/``WN``).
    knobs = {"MMA": "0", "WM": 0, "WN": 0, "BR": 1, **knobs}
    return _assemble(
        free_specs,
        dag.inner_body,
        leading=dag.leading,
        extra_outer=extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_scalar(body, targets, dag.k_extent, bk, fk, splitk, k_s),
        extra_block=(k_s,) if k_s is not None else (),
    )


def _mask_reduce_accums(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask each ``Accum``'s folded value to the carrier's identity past a
    masked-K boundary. Before each ``Accum(name, value=V, op)`` insert
    ``Init(V_kid, op)`` (the op's neutral element — `0` for add, `-inf` for max)
    and ``Select(V_km, V if pred else V_kid)``, then fold ``V_km``. The Accum
    stays a direct child of the reduce loop (`is_reduce` + the cross-thread
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


def _mask_flash_monoid(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask the streaming flash carrier's score to ``-inf`` past a symbolic-K
    boundary. Before each ``Monoid`` with partial ``(score, value)`` insert
    ``Init(score_kid, op=maximum)`` (seeds ``-inf``, the score component's
    identity) and ``Select(score_km, score if pred else score_kid)``, then fold
    ``score_km``. The masked key contributes nothing (``max(m, -inf) = m``,
    ``exp(-inf) = 0``); the in-place state update stays unconditional."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Monoid):
            score = c.partial[0]
            ident = f"{score}_kid"
            masked = f"{score}_km"
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


def _replace_k_coop(
    stmts: tuple[Stmt, ...], target_names: frozenset[str], k_dim, bk: int, fk: int, br: int, k_c: Axis | None, k_bound: object
) -> tuple[Stmt, ...]:
    """Replace every K loop named in ``target_names`` — the reduce(s) and any
    second-pass map loop — with the cooperative tower, σ-mapping
    ``K → K_o·(br·bk·fk) + K_f·(br·bk) + K_i·br + K_c`` (``K_f`` only when
    ``fk > 1``; ``K_c`` the stride-1 thread lane only when ``br > 1``). Each loop
    gets its own ``K_o``/``K_i``/``K_f`` serial tiles but shares the one ``K_c``
    THREAD axis (added by the caller). The reduce's ``Accum.axes`` propagate
    ``K_c`` through σ → kernel/100 emits the combine.

    When ``k_bound`` is set (symbolic K), ``K_o`` ceil-divides and the final
    partial tile is masked: the reduce clamps its load index for a safe read and
    folds the carrier identity past ``k_bound`` (`_mask_reduce_accums`); the map
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


def build_flash_tile(
    knobs: dict, *, kernel_name: str, base_knobs: dict, dag: IterDag, target_names: frozenset[str], k_bounds: dict
) -> TileOp:
    """Fused flash nest: tile the free output axes (q-rows / head-dim), and
    serial-transform the streaming KV reduce + its nested QK^T reduce (bk=fk=1,
    no split-K / cooperative-K — each output element streams KV itself). The
    `FlashCombine` / `Accum` carriers render their own rescale; the Init / divide
    finalize ride the output tile."""
    inner_n, outer_m, extra_outer = _free_axes(dag)
    free_specs: list[_FreeSpec] = [(inner_n, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True)]
    if outer_m is not None:
        free_specs.append((outer_m, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False))
    return _assemble(
        free_specs,
        dag.inner_body,
        leading=dag.leading,
        extra_outer=extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_scalar(body, target_names, 1, 1, 1, 1, None, k_bounds),
    )


def build_coop_reduce_tile(knobs: dict, *, kernel_name: str, base_knobs: dict, dag: IterDag, target_names: frozenset[str]) -> TileOp:
    """Whole-CTA cooperative reduce: free rows → grid (thread/reg forced to 1),
    ``BR`` threads (the ``K_c`` axis) reduce one row's K (masked-K fill past
    ``k_bound`` when symbolic), the combine is emitted downstream from
    ``Accum.axes ∩ ThreadTile``."""
    inner_n, outer_m, extra_outer = _free_axes(dag)
    # Free-axis THREAD tiles default to 1 (whole-CTA: one row per CTA). A pinned
    # ``BN``/``BM`` > 1 thread-binds the free rows alongside ``BR`` — the strided-
    # cooperative form (the segmented-shuffle combine, `cooperative_combine_geometry`).
    bn, bm = knobs.get(MAP_N_THREAD.name, 1), knobs.get(MAP_M_THREAD.name, 1)
    free_specs: list[_FreeSpec] = [(inner_n, bn, 1, True)]
    if outer_m is not None:
        free_specs.append((outer_m, bm, 1, False))
    bk, fk, br = knobs[RED_BK.name], knobs[RED_FK.name], knobs[COOP_BR.name]
    kax = dag.k_node.loop.axis
    src_k = kax.source_axis or kax
    k_c = Axis(f"{kax.name}_c", br, source_axis=src_k) if br > 1 else None
    targets = target_names or frozenset({kax.name})
    k_dim = kax.extent
    return _assemble(
        free_specs,
        dag.inner_body,
        leading=dag.leading,
        extra_outer=extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_coop(body, targets, k_dim, bk, fk, br, k_c, dag.k_bound),
        coop_thread=(k_c,) if k_c is not None else (),
    )


def _warp_axis(axis: Axis, warp: int, reg: int, atom_cell: int):
    """4-level output-axis split for the warp tier:
    ``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom`` (the per-lane ``A_a`` offset
    is owned by ``mma.sync``, so it is NOT in σ). A symbolic / non-divisible axis
    is masked: ``A_b`` ceil-divides and carries ``real_extent`` (the runtime /
    static bound) so ``020_stage_inputs`` / ``005_lower_atom_tile`` clamp the loads
    and guard the per-cell store; the boundary ``Expr`` is returned for the
    output-store ``Cond``."""
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


def _replace_k_warp(stmts: tuple[Stmt, ...], k_name: str, k_extent: int, bk: int, atom_k: int) -> tuple[Stmt, ...]:
    """Replace the K reduce loop with the ``atom_k``-strided ``K_o`` / ``K_i``
    tower: ``σ(K) = K_o·(bk·atom_k) + K_i·atom_k`` (each ``K_i`` step is one
    ``mma.sync`` over ``atom_k`` K-elements). No split-K / strip-mine (v1)."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name == k_name:
            src = s.axis.source_axis or s.axis
            k_o = Axis(f"{k_name}_o", k_extent // (bk * atom_k), source_axis=src)
            k_i = Axis(f"{k_name}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * atom_k, "int") + Var(k_i.name) * Literal(atom_k, "int")
            new_body = tuple(c.rewrite(_identity_rename, Sigma({k_name: expr})) for c in s.body)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def build_warp_matmul_tile(atom: Atom, knobs: dict, *, kernel_name: str, base_knobs: dict, dag: IterDag) -> TileOp:
    """Warp-tier (tensor-core) matmul tower. Emits the canonical AtomTile cell
    ``[Load a, Load b, Assign(multiply), Accum]``; ``011_lower_atom_cell`` folds
    it into an ``Mma``. Clean (divisible) tiles only in this phase."""
    atom_m, atom_n, atom_k = atom.shape
    wm, wn = knobs[WARP_M.name], knobs[WARP_N.name]
    fm, fn = knobs[TC_REG_M.name], knobs[TC_REG_N.name]
    bk = knobs[TC_BK.name]

    inner_n, outer_m, extra_outer = _free_axes(dag)
    n_b, n_w, n_r, n_a, n_expr, n_bound = _warp_axis(inner_n, wn, fn, atom_n)
    m_b, m_w, m_r, m_a, m_expr, m_bound = _warp_axis(outer_m, wm, fm, atom_m)
    sigma_outer = Sigma({inner_n.name: n_expr, outer_m.name: m_expr})

    new_inner: tuple[Stmt, ...] = tuple(s.rewrite(_identity_rename, sigma_outer) for s in dag.inner_body)
    new_inner = _replace_k_warp(new_inner, dag.k_node.loop.axis.name, dag.k_extent, bk, atom_k)

    # Masked output axes (symbolic or static non-divisor): wrap the cell in a
    # boundary ``Cond(σ(axis) < bound)`` — the same store guard the scalar
    # ``_assemble`` emits. ``021_hoist_staged_loads_above_mask`` lifts the
    # K-pipeline above it (and stamps ``gmem_extents`` so ``_stage_expand``
    # clamps the cooperative slab fill); ``005_lower_atom_tile._boundary_guards``
    # classifies the predicate against the Write's M / N coords and stamps the
    # per-element ``RegStore`` row / col guard (a tile straddling the bound passes
    # the Cond but its trailing rows / cols are out of range). N (inner) is
    # wrapped first so it nests inside M, matching the scalar bound order.
    overhang: tuple[str, ...] = ()
    for name, bound in ((inner_n.name, n_bound), (outer_m.name, m_bound)):
        if bound is not None:
            overhang = (*overhang, name)
            pred = sigma_outer.reduce(Var(name), SimplifyCtx({}))
            new_inner = (Cond(cond=BinaryExpr("<", pred, bound), body=Body(new_inner)),)

    layers: list[tuple[Axis, Role | None]] = [
        (n_a, Role.ATOM),
        (m_a, Role.ATOM),
        (n_r, Role.REGISTER),
        (m_r, Role.REGISTER),
        (n_w, Role.WARP),
        (m_w, Role.WARP),
        (n_b, Role.BLOCK),
        (m_b, Role.BLOCK),
    ]
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(extra_outer))
    chain_body = _wrap_tower(layers, new_inner, atom=atom)

    # Bridge to the warp-tier downstream contract: ``020_stage_inputs`` /
    # ``005_lower_atom_tile`` / ``is_warp`` discriminate the tensor-core tier via
    # the legacy ``MMA`` knob (``mma_atom`` reads ``knobs["MMA"]``), not the
    # AtomTile structure. Stamp it alongside the greenfield ``TC_*`` search
    # vocabulary so 020 takes the atom staging path (force all-staged operands +
    # block-stamped affine slab) instead of the scalar path. Dropped when those
    # passes are greenfielded in Phase 4.
    knobs_full = {**base_knobs, **knobs, "MMA": atom.name}
    if overhang:
        knobs_full["OVERHANG"] = overhang
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in dag.leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)
