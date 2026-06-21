"""Realize a complete pointwise move choice into the ``TileOp`` tower.

Reproduces the legacy planner's pointwise σ-split + tower so the output is
accuracy-equivalent: each free axis ``A`` splits as ``A → A_b·(T·R) + A_t·R +
A_r`` (interleaved ``A_b·(T·R) + A_r·T + A_t`` on a masked inner axis), the body
is σ-rewritten, masked axes get a ``Cond`` store guard, and the layers are
wrapped innermost-first via the shared :func:`_wrap_tower`.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Stmt
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import PointwiseSkeleton


def _block_axis(axis: Axis, tr: int, masked: bool) -> Axis:
    src = axis.source_axis or axis
    extent = axis.extent.ceil_div(tr) if masked else axis.extent // tr
    real = axis.extent.as_static() if masked and axis.extent.is_static else None
    return Axis(f"{axis.name}_b", extent, source_axis=src, real_extent=real)


def build_pointwise_tile(skel: PointwiseSkeleton, knobs: dict, *, kernel_name: str, base_knobs: dict) -> TileOp:
    sigma_map: dict = {}
    bounds: list[tuple[str, object]] = []  # (axis_name, bound_expr) in N-then-M order
    layers_reg: list[tuple[Axis, Role]] = []
    layers_thread: list[tuple[Axis, Role]] = []
    layers_block: list[tuple[Axis, Role]] = []

    # --- N (innermost free axis): interleaved layout when masked. ---
    n_axis = skel.inner_n.loop.axis
    t_n, r_n = knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name]
    n_tr = t_n * r_n
    n_masked = skel.inner_n.symbolic or not (n_axis.extent.is_static and n_axis.extent.as_static() % n_tr == 0)
    src_n = n_axis.source_axis or n_axis
    n_b = _block_axis(n_axis, n_tr, n_masked)
    n_t = Axis(f"{n_axis.name}_t", t_n, source_axis=src_n)
    n_r = Axis(f"{n_axis.name}_r", r_n, source_axis=src_n)
    n_block = Var(n_b.name) * Literal(n_tr, "int")
    if n_masked:
        sigma_map[n_axis.name] = n_block + Var(n_r.name) * Literal(t_n, "int") + Var(n_t.name)
        bounds.append((n_axis.name, n_axis.extent.expr))
    else:
        sigma_map[n_axis.name] = n_block + Var(n_t.name) * Literal(r_n, "int") + Var(n_r.name)
    layers_reg.append((n_r, Role.REGISTER))
    layers_thread.append((n_t, Role.THREAD))
    layers_block.append((n_b, Role.BLOCK))

    # --- M (outer free axis, optional): always blocked layout. ---
    if skel.outer_m is not None:
        m_axis = skel.outer_m.loop.axis
        t_m, r_m = knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name]
        m_tr = t_m * r_m
        m_masked = skel.outer_m.symbolic or not (m_axis.extent.is_static and m_axis.extent.as_static() % m_tr == 0)
        src_m = m_axis.source_axis or m_axis
        m_b = _block_axis(m_axis, m_tr, m_masked)
        m_t = Axis(f"{m_axis.name}_t", t_m, source_axis=src_m)
        m_r = Axis(f"{m_axis.name}_r", r_m, source_axis=src_m)
        sigma_map[m_axis.name] = Var(m_b.name) * Literal(m_tr, "int") + Var(m_t.name) * Literal(r_m, "int") + Var(m_r.name)
        if m_masked:
            bounds.append((m_axis.name, m_axis.extent.expr))
        layers_reg.append((m_r, Role.REGISTER))
        layers_thread.append((m_t, Role.THREAD))
        layers_block.append((m_b, Role.BLOCK))

    sigma_outer = Sigma(sigma_map)
    new_inner: tuple[Stmt, ...] = tuple(s.rewrite(_identity_rename, sigma_outer) for s in skel.inner_body)

    # Boundary store guards (N first, then M — so the M guard is outermost).
    for name, bound in bounds:
        pred = sigma_outer.reduce(Var(name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", pred, bound), body=Body(new_inner)),)

    # Tower layers, innermost-first: register → thread → block, plus extra
    # outer free loops as outer BLOCK axes.
    layers: list[tuple[Axis, Role | None]] = [*layers_reg, *layers_thread, *layers_block]
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(skel.extra_outer))

    chain_body = _wrap_tower(layers, new_inner)

    knobs_full = {**base_knobs, **knobs}
    inner_defs = {name for s in Body.coerce(chain_body).iter() for name in s.defines()}
    leading = tuple(s for s in skel.leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=leading + chain_body, name=kernel_name, knobs=knobs_full)
