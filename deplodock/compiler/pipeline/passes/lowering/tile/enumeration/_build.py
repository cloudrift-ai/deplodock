"""Build the block-DAG ``TileGraph`` from the iteration DAG + a move choice.

This is the composer's new front half (``plans/tile-ir-block-dag.md`` step 1):
instead of materializing the ``TileOp`` tower directly, it emits the invariant
algorithm (one or more :class:`Block`\\ s) plus a reference :class:`Schedule`,
which :func:`assemble.assemble_pointwise` (etc.) lowers to today's tower
byte-identically.

**Covered so far: pointwise.** The free-axis σ-split is the ``tile_axis`` body
move — done here, reusing ``materialize._split_free_axis`` verbatim so the body
is identical — and the GRID/THREAD/REGISTER tiers become ``Schedule.binding``
entries on the split axes.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Loop, Stmt
from deplodock.compiler.ir.tile.ir import Binding, Block, RegisterTile, Schedule, TileGraph
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
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


def build_pointwise_dag(dag: IterDag, knobs: dict, *, kernel_name: str) -> TileGraph:
    """Build a single-block pointwise ``TileGraph`` from the iteration DAG.

    Mirrors ``materialize.build_pointwise_tile``'s free-axis split (so the σ-
    rewritten body is byte-identical) but records the tiers as ``Schedule.binding``
    instead of wrapping the tower. ``assemble_pointwise`` then rebuilds the tower.

    The split axes are laid into ``Block.domain`` per-free-axis (``N_b, N_t, N_r,
    M_b, M_t, M_r, …``) so each binding tier reads back in inner→outer order, with
    extra-outer GRID axes (reversed) trailing — the order ``_assemble`` emits."""
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
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    compute = tuple(s.rewrite(_identity_rename, sigma_outer) for s in dag.inner_body)
    compute = _apply_masked_guards(compute, bounds, sigma_outer)
    block = Block(name=kernel_name, domain=tuple(domain), compute=Body(compute))
    return TileGraph(name=kernel_name, buffers={}, blocks=(block,), schedule=Schedule(binding=binding))


def build_matmul_dag(dag: IterDag, knobs: dict, *, kernel_name: str, target_names: frozenset[str]) -> TileGraph:
    """Build a single-block scalar-matmul ``TileGraph``.

    Mirrors ``materialize.build_matmul_tile``: free axes M / N split into
    GRID/THREAD/REGISTER (the ``tile_axis`` map move), and the contraction axis
    re-bracketed into the ``K_o`` (serial-outer) / ``K_i`` (stage-inner) tower with
    an optional ``K_f`` strip-mine (the ``tile_axis`` reduce move, embedded in
    ``compute``). Split-K's ``K_s`` is one GRID binding placed after the free block
    axes. Masked (non-divisible) M / N / K not yet ported."""
    inner_n, outer_m, extra_outer = _free_axes(dag)
    free_specs = [
        (inner_n, knobs[MAP_N_THREAD.name], knobs[MAP_N_REG.name], True),
        (outer_m, knobs[MAP_M_THREAD.name], knobs[MAP_M_REG.name], False),
    ]
    bk, fk, splitk = knobs[RED_BK.name], knobs[RED_FK.name], knobs[RED_SPLITK.name]
    kax = dag.k_node.loop.axis
    src_k = kax.source_axis or kax
    k_s = Axis(f"{kax.name}_s", splitk, source_axis=src_k) if splitk > 1 else None
    targets = target_names or frozenset({kax.name})

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
    if k_s is not None:
        domain.append(k_s)
        binding[k_s.name] = Binding.GRID
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    inner = tuple(s.rewrite(_identity_rename, sigma_outer) for s in dag.inner_body)
    compute = _replace_k_scalar(inner, targets, dag.k_extent, bk, fk, splitk, k_s)
    compute = _apply_masked_guards(compute, bounds, sigma_outer)
    block = Block(name=kernel_name, domain=tuple(domain), compute=Body(compute))
    return TileGraph(name=kernel_name, buffers={}, blocks=(block,), schedule=Schedule(binding=binding))


# ---------------------------------------------------------------------------
# Routing helpers — build the DAG + assemble. The Fork-tree leaves call these;
# the new block-DAG path is the SOLE pointwise / scalar-matmul lowering (the
# legacy materialize.build_*_tile builders are deleted).
# ---------------------------------------------------------------------------


def lower_pointwise(dag: IterDag, knobs: dict, *, kernel_name: str, base_knobs: dict):
    """Lower a pointwise leaf via the block-DAG path."""
    tg = build_pointwise_dag(dag, knobs, kernel_name=kernel_name)
    return assemble_block(tg, knobs=knobs, base_knobs=base_knobs, kernel_name=kernel_name, leading=dag.leading)


def lower_matmul(dag: IterDag, knobs: dict, *, kernel_name: str, base_knobs: dict, target_names: frozenset[str]):
    """Lower a scalar-matmul leaf via the block-DAG path."""
    tg = build_matmul_dag(dag, knobs, kernel_name=kernel_name, target_names=target_names)
    # Scalar tier carries the warp-tier OFF sentinels.
    stamped = {"MMA": "0", "WM": 0, "WN": 0, "BR": 1, **knobs}
    return assemble_block(tg, knobs=stamped, base_knobs=base_knobs, kernel_name=kernel_name, leading=dag.leading)
