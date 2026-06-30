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
from deplodock.compiler.ir.tile import SemiringKernel, TileOp
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
    if tier is None or not tier.is_tiled:
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
    # allocation) — reserved this cut; materialization stays uniform SIMT. The tier's ``atom``
    # selects the codegen (a tensor-core mma cell / the scalar fma cell); the unit / register widths
    # read off it in normalized (m, n) order — for scalar the UNIT grid IS the parallel thread-tile.
    contraction = Contraction(
        axes=(m_axis, n_axis),
        k_axis=k_axis,
        units=(tier.units_m, tier.units_n),
        regs=(tier.reg_m, tier.reg_n),
        a_load=bind.a.load,
        b_load=bind.b.load,
        acc=bind.acc,
        atom=tier.atom,
        lead_axes=lead,
        epilogue=Body(tail),
    )
    return KernelOp(body=Body((contraction,)), name=tile.name)
