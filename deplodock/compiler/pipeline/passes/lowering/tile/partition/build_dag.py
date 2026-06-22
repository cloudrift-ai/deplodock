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

from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.blockdag import Binding, Block, Schedule, TileGraph
from deplodock.compiler.pipeline.passes.lowering.tile.partition._tower import _identity_rename
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import (
    MAP_M_REG,
    MAP_M_THREAD,
    MAP_N_REG,
    MAP_N_THREAD,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import _free_axes, _split_free_axis


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
    domain: list = []
    binding: dict[str, Binding] = {}
    for axis, thread, reg, interleave in free_specs:
        a_b, a_t, a_r, expr, bound = _split_free_axis(axis, thread, reg, interleave_when_masked=interleave)
        if bound is not None:
            raise NotImplementedError("build_pointwise_dag: masked axes not yet ported")
        sigma_map[axis.name] = expr
        domain.extend((a_b, a_t, a_r))
        binding[a_b.name] = Binding.GRID
        binding[a_t.name] = Binding.THREAD
        binding[a_r.name] = Binding.REGISTER
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    compute = tuple(s.rewrite(_identity_rename, sigma_outer) for s in dag.inner_body)
    block = Block(name=kernel_name, domain=tuple(domain), compute=Body(compute))
    return TileGraph(name=kernel_name, buffers={}, blocks=(block,), schedule=Schedule(binding=binding))
