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
from deplodock.compiler.ir.expr import BinaryExpr, SimplifyCtx, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond
from deplodock.compiler.ir.tile.blockdag import Binding, Block, Schedule, TileGraph
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import _identity_rename
from deplodock.compiler.pipeline.passes.lowering.tile.partition.assemble import assemble_block
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
    RED_BK,
    RED_FK,
    RED_SPLITK,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import (
    _free_axes,
    _replace_k_scalar,
    _split_free_axis,
)


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
