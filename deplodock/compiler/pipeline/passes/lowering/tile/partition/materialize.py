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

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Loop, Stmt
from deplodock.compiler.ir.tile.ir import RegisterTile, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import MatmulSkeleton, PointwiseSkeleton

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
) -> TileOp:
    """Shared tower assembly: split free axes, σ-rewrite the body, apply the K
    transform (matmul), wrap masked-axis guards, and build the tile tower."""
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

    layers: list[tuple[Axis, Role | None]] = [*layers_reg, *layers_thread, *layers_block]
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


def _replace_k_scalar(stmts: tuple[Stmt, ...], k_name: str, k_extent: int, bk: int, fk: int) -> tuple[Stmt, ...]:
    """Replace the ``K`` reduce loop with a ``K_o`` (serial-outer) / ``K_i``
    (stage-inner) tower, σ-mapping ``K → K_o·(bk·fk) + K_f·bk + K_i`` (the
    ``K_f`` term only when ``fk > 1``). Scalar tier: no split-K, no
    cooperative-K. The canonical matmul body holds the loop at top level, so a
    flat scan suffices."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name == k_name:
            src = s.axis.source_axis or s.axis
            k_o_ext = k_extent // (bk * fk)
            k_o = Axis(f"{k_name}_o", k_o_ext, source_axis=src)
            k_i = Axis(f"{k_name}_i", bk, source_axis=src)
            expr = Var(k_o.name) * Literal(bk * fk, "int")
            k_f = None
            if fk > 1:
                k_f = Axis(f"{k_name}_f", fk, source_axis=src)
                expr = expr + Var(k_f.name) * Literal(bk, "int")
            expr = expr + Var(k_i.name)
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
    bk, fk = knobs[RED_BK.name], knobs[RED_FK.name]
    return _assemble(
        free_specs,
        skel.inner_body,
        leading=skel.leading,
        extra_outer=skel.extra_outer,
        knobs=knobs,
        base_knobs=base_knobs,
        kernel_name=kernel_name,
        k_transform=lambda body: _replace_k_scalar(body, skel.k_name, skel.k_extent, bk, fk),
    )
