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
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Loop, Select, SelectBranch, Stmt
from deplodock.compiler.ir.tile.ir import Atom, RegisterTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _identity_rename, _wrap_tower
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
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import CoopReduceSkeleton, MatmulSkeleton, PointwiseSkeleton

# One free axis to split: (axis, thread factor, register factor, interleave-when-masked).
_FreeSpec = tuple[Axis, int, int, bool]


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


def build_pointwise_tile(skel: PointwiseSkeleton, knobs: dict, *, kernel_name: str, base_knobs: dict) -> TileOp:
    free_specs: list[_FreeSpec] = [(skel.inner_n.loop.axis, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True)]
    if skel.outer_m is not None:
        free_specs.append((skel.outer_m.loop.axis, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False))
    return _assemble(
        free_specs,
        skel.inner_body,
        leading=skel.leading,
        extra_outer=skel.extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
    )


def _replace_k_scalar(
    stmts: tuple[Stmt, ...], k_name: str, k_extent: int, bk: int, fk: int, splitk: int, k_s: Axis | None
) -> tuple[Stmt, ...]:
    """Replace the ``K`` reduce loop with a ``K_o`` (serial-outer) / ``K_i``
    (stage-inner) tower, σ-mapping
    ``K → K_s·(K_o_ext·bk·fk) + K_o·(bk·fk) + K_f·bk + K_i`` (``K_f`` only when
    ``fk > 1``, ``K_s`` only when ``splitk > 1``). The ``K_s`` grid axis is added
    to the outer BLOCK layers by the caller; here it only enters the σ. The
    canonical matmul body holds the loop at top level, so a flat scan suffices."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name == k_name:
            src = s.axis.source_axis or s.axis
            k_o_ext = k_extent // (splitk * bk * fk)
            k_o = Axis(f"{k_name}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{k_name}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * fk, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{k_name}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(bk, "int")
            expr = expr + Var(k_i.name)
            if k_s is not None:
                expr = Var(k_s.name) * Literal(k_o_ext * bk * fk, "int") + expr
            sigma_k = Sigma({k_name: expr})
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            if k_f is not None:
                new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=True),)
            out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
        else:
            out.append(s)
    return tuple(out)


def build_matmul_tile(skel: MatmulSkeleton, knobs: dict, *, kernel_name: str, base_knobs: dict) -> TileOp:
    free_specs: list[_FreeSpec] = [
        (skel.inner_n.loop.axis, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True),
        (skel.outer_m.loop.axis, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False),
    ]
    bk, fk, splitk = knobs[RED_BK.name], knobs[RED_FK.name], knobs[RED_SPLITK.name]
    src_k = skel.k_loop.axis.source_axis or skel.k_loop.axis
    k_s = Axis(f"{skel.k_name}_s", splitk, source_axis=src_k) if splitk > 1 else None
    return _assemble(
        free_specs,
        skel.inner_body,
        leading=skel.leading,
        extra_outer=skel.extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_scalar(body, skel.k_name, skel.k_extent, bk, fk, splitk, k_s),
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


def build_coop_reduce_tile(skel: CoopReduceSkeleton, knobs: dict, *, kernel_name: str, base_knobs: dict) -> TileOp:
    """Whole-CTA cooperative reduce: free rows → grid (thread/reg forced to 1),
    ``BR`` threads (the ``K_c`` axis) reduce one row's K (masked-K fill past
    ``k_bound`` when symbolic), the combine is emitted downstream from
    ``Accum.axes ∩ ThreadTile``."""
    free_specs: list[_FreeSpec] = [(skel.inner_n.loop.axis, 1, 1, True)]
    if skel.outer_m is not None:
        free_specs.append((skel.outer_m.loop.axis, 1, 1, False))
    bk, fk, br = knobs[RED_BK.name], knobs[RED_FK.name], knobs[COOP_BR.name]
    src_k = skel.k_loop.axis.source_axis or skel.k_loop.axis
    k_c = Axis(f"{skel.k_name}_c", br, source_axis=src_k) if br > 1 else None
    targets = skel.target_names or frozenset({skel.k_name})
    k_dim = skel.k_loop.axis.extent
    return _assemble(
        free_specs,
        skel.inner_body,
        leading=skel.leading,
        extra_outer=skel.extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_coop(body, targets, k_dim, bk, fk, br, k_c, skel.k_bound),
        coop_thread=(k_c,) if k_c is not None else (),
    )


def _warp_axis(axis: Axis, warp: int, reg: int, atom_cell: int):
    """4-level output-axis split for the warp tier:
    ``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom`` (the per-lane ``A_a`` offset
    is owned by ``mma.sync``, so it is NOT in σ). Clean (divisible) only."""
    src = axis.source_axis or axis
    per_block = warp * reg * atom_cell
    a_b = Axis(f"{axis.name}_b", axis.extent // per_block, source_axis=src)
    a_w = Axis(f"{axis.name}_w", warp, source_axis=src)
    a_r = Axis(f"{axis.name}_r", reg, source_axis=src)
    a_a = Axis(f"{axis.name}_a", atom_cell, source_axis=src)
    expr = (
        Var(a_b.name) * Literal(per_block, "int")
        + Var(a_w.name) * Literal(reg * atom_cell, "int")
        + Var(a_r.name) * Literal(atom_cell, "int")
    )
    return a_b, a_w, a_r, a_a, expr


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


def build_warp_matmul_tile(skel: MatmulSkeleton, atom: Atom, knobs: dict, *, kernel_name: str, base_knobs: dict) -> TileOp:
    """Warp-tier (tensor-core) matmul tower. Emits the canonical AtomTile cell
    ``[Load a, Load b, Assign(multiply), Accum]``; ``011_lower_atom_cell`` folds
    it into an ``Mma``. Clean (divisible) tiles only in this phase."""
    atom_m, atom_n, atom_k = atom.shape
    wm, wn = knobs[WARP_M.name], knobs[WARP_N.name]
    fm, fn = knobs[TC_REG_M.name], knobs[TC_REG_N.name]
    bk = knobs[TC_BK.name]

    n_b, n_w, n_r, n_a, n_expr = _warp_axis(skel.inner_n.loop.axis, wn, fn, atom_n)
    m_b, m_w, m_r, m_a, m_expr = _warp_axis(skel.outer_m.loop.axis, wm, fm, atom_m)
    sigma_outer = Sigma({skel.inner_n.loop.axis.name: n_expr, skel.outer_m.loop.axis.name: m_expr})

    new_inner: tuple[Stmt, ...] = tuple(s.rewrite(_identity_rename, sigma_outer) for s in skel.inner_body)
    new_inner = _replace_k_warp(new_inner, skel.k_name, skel.k_extent, bk, atom_k)

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
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(skel.extra_outer))
    chain_body = _wrap_tower(layers, new_inner, atom=atom)

    # Bridge to the warp-tier downstream contract: ``020_stage_inputs`` /
    # ``005_lower_atom_tile`` / ``is_warp`` discriminate the tensor-core tier via
    # the legacy ``MMA`` knob (``mma_atom`` reads ``knobs["MMA"]``), not the
    # AtomTile structure. Stamp it alongside the greenfield ``TC_*`` search
    # vocabulary so 020 takes the atom staging path (force all-staged operands +
    # block-stamped affine slab) instead of the scalar path. Dropped when those
    # passes are greenfielded in Phase 4.
    knobs_full = {**base_knobs, **knobs, "MMA": atom.name}
    inner_defs = {n for s in Body.coerce(chain_body).iter() for n in s.defines()}
    kept_leading = tuple(s for s in skel.leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=kept_leading + chain_body, name=kernel_name, knobs=knobs_full)
