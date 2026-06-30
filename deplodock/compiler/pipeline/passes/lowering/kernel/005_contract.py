"""Construct the high-level :class:`Contraction` node for a ``Semiring`` contraction — **before**
materialize.

ONE binding-driven node for both tiers. 005 reads the operand→role binding — already on
``sched.bind`` for the warp tier (stamped by ``020_schedule``), computed here via
:func:`semiring_binding` for the scalar register-tile tier — resolves the projection epilogue, and
stamps the per-CTA **UNIT** grid (warps for mma, threads for scalar) + the per-unit **REGISTER**
sub-tile + the leaf ``atom``. ``010_materialize`` expands it; the contraction itself is *synthesized*
per atom (``mma.sync`` / a scalar register-tile loop), so the node stores only the operands + the
epilogue.

A contraction whose operands aren't plain ``Load``\\ s (a computed-cone / demoted matmul) or whose
output is 1-D is **not bindable** — it's skipped here (``RuleSkipped``) and the per-cell scalar
fallback in ``010_materialize`` lowers it. A non-contraction / non-tiled ``TileOp`` is skipped too."""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import Contraction
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile import SemiringKernel, TileOp, TilePlan
from deplodock.compiler.ir.tile.atom import SCALAR_ATOM
from deplodock.compiler.ir.tile.schedule import WarpTile
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.passes.lowering.tile._atomize import semiring_binding
from deplodock.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    """Build the :class:`Contraction` for a warp / register-tiled ``Semiring`` kernel; skip every
    other tier (it materializes in ``010``), and skip an unbindable contraction (the per-cell
    fallback in ``010`` lowers it)."""
    tile: TileOp = root.op
    kernel = tile.kernel
    sched = kernel.schedule if kernel is not None else None
    if not isinstance(kernel, SemiringKernel):
        raise RuleSkipped("not a contraction")
    tier = sched.tier
    is_warp = isinstance(tier, WarpTile)
    is_tiled = isinstance(tier, TilePlan) and tier.is_tiled
    if not (is_warp or is_tiled):
        raise RuleSkipped("non-tiled contraction — per-cell fallback in 010")

    node = tile.op
    grid = list(sched.place.grid)
    # The operand→role binding: stamped on the warp schedule by 020_schedule; computed here for the
    # scalar tier. An unbindable contraction (a non-Load operand, or a 1-D output grid) raises — it
    # falls through to the per-cell fallback in 010.
    try:
        bind = sched.bind if sched.bind is not None else semiring_binding(node, grid)
    except LoweringError:
        raise RuleSkipped("contraction not bindable — per-cell fallback in 010") from None

    m_axis, n_axis = grid[-2], grid[-1]
    lead = tuple(grid[:-2])
    k_axis = node.reduce_node.reduce_axis
    # The projection epilogue: the binding's body, or — for a bare contraction — a synthesized store
    # of the accumulator (``with_store`` needs ``node.out`` / the grid, so it stays here).
    tail = list(bind.epilogue)
    if not has_write(tail):
        tail = with_store(tail, root.output.name, grid, node)

    # TODO(warp-spec): emit the producer/consumer warp split from sched.workers (the WSPEC role
    # allocation) — reserved this cut; materialization stays uniform SIMT.
    if is_warp:
        wt = tier
        atom, units_m, units_n, reg_m, reg_n = wt.atom, wt.warps[0], wt.warps[1], wt.reg[0], wt.reg[1]
    else:
        plan = tier  # scalar: the parallel thread-tile IS the UNIT grid (lanes 1×1)
        atom, units_m, units_n, reg_m, reg_n = SCALAR_ATOM, plan.par_m, plan.par_n, plan.reg_m, plan.reg_n

    contraction = Contraction(
        m_axis=m_axis,
        n_axis=n_axis,
        k_axis=k_axis,
        units_m=units_m,
        units_n=units_n,
        reg_m=reg_m,
        reg_n=reg_n,
        a_load=bind.a.load,
        b_load=bind.b.load,
        acc=bind.acc,
        atom=atom,
        lead_axes=lead,
        epilogue=Body(tail),
    )
    return KernelOp(body=Body((contraction,)), name=tile.name)
