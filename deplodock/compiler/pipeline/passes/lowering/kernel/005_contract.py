"""Construct the high-level :data:`Contraction` node for a ``Semiring`` contraction — **before**
materialize.

A contraction kernel is captured here as a single :data:`Contraction` ``KernelOp`` that
``010_materialize`` then expands. Two arms, one per atom (see ``ir/tile/atom``):

- **mma arm** (:class:`MmaContraction`) — a warp-tier ``SemiringKernel`` (its schedule carries a
  ``WarpTile``): the op-tree-dependent part only — capture the m/n/k axes, read the ``020_schedule``
  operand→role binding, resolve the projection epilogue. Tensor-core ``AtomKind`` leaf.
- **scalar arm** (:class:`ScalarContraction`) — a register-tiled ``SemiringKernel`` (its ``TILE``
  plan tiles the output): lower the per-cell body (``lower(op)`` + output-store glue) and capture it
  with the register / parallel widths + tiled axes. Scalar ``1×1`` fma leaf.

The expansion (the four-way GRID/UNIT/REGISTER/ATOM tiling, via the shared ``_tiling`` layer) is
folded into ``010_materialize`` (which runs next). Splitting the construction out keeps the
contraction a first-class node that exists *before* thread-binding; the per-cell scalar fallback and
the cooperative reduce tier stay in ``010_materialize`` (a non-contraction / non-tiled ``TileOp`` is
skipped here and passes through). The node IS ``structural_key``-ed as an intermediate ``KernelOp``."""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import MmaContraction, ScalarContraction
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile import SemiringKernel, TileOp
from deplodock.compiler.ir.tile.ops import lower
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", TileOp)]


def _warp_contraction(tile: TileOp, sched, root: Node) -> MmaContraction:
    """The mma arm: read the ``020_schedule`` binding + warp geometry and resolve the projection
    epilogue (``with_store`` needs the op's ``out`` + grid). The exact atom factorization is
    expanded from the node by ``010_materialize`` (:func:`_warp_factor.factorize_mma`)."""
    node = tile.op
    grid = list(sched.place.grid)
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    m_axis, n_axis = grid[-2], grid[-1]
    k_axis = node.reduce_node.reduce_axis
    bind = sched.bind
    assert bind is not None, "warp tier: 020_schedule did not stamp a binding"
    # TODO(warp-spec): emit the producer/consumer warp split from sched.workers (the WSPEC
    # role allocation) — dedicate producer warps to the Stage load half, compute warps to the mma.
    # Reserved this cut: the codec + schedule field land, but materialization stays uniform SIMT.
    # The projection epilogue: the binding's body, or — for a bare contraction — a synthesized
    # store of the accumulator (``with_store`` needs ``node.out`` / the grid, so it stays here).
    tail = list(bind.epilogue)
    if not has_write(tail):
        tail = with_store(tail, root.output.name, grid, node)
    return MmaContraction(
        a_load=bind.a.load,
        b_load=bind.b.load,
        b_trans=bind.b_trans,
        acc=bind.acc,
        epilogue=Body(tail),
        warp_tile=sched.warp_tile,
        stage=sched.stage,
        m_axis=m_axis,
        n_axis=n_axis,
        k_axis=k_axis,
        output=root.output.name,
    )


def _scalar_contraction(tile: TileOp, sched, root: Node) -> ScalarContraction:
    """The scalar arm: lower the per-cell body (``lower(op)`` + output-store glue) and capture it
    with the tiled output axes + the register / parallel widths. Cell tiling is expanded from the
    node by ``010_materialize`` (:func:`_scalar_factor.factorize_scalar`)."""
    node = tile.op
    grid = list(sched.place.grid)
    n_axis = grid[-1]
    m_axis = grid[-2] if len(grid) >= 2 else None
    lead = tuple(grid[:-2]) if m_axis is not None else tuple(grid[:-1])
    k_axis = node.reduce_node.reduce_axis
    plan = sched.tile
    return ScalarContraction(
        body=Body(with_store(lower(node), root.output.name, grid, node)),
        n_axis=n_axis,
        k_axis=k_axis,
        reg_m=plan.reg_m if m_axis is not None else 1,
        reg_n=plan.reg_n,
        par_m=plan.par_m if m_axis is not None else 1,
        par_n=plan.par_n,
        output=root.output.name,
        m_axis=m_axis,
        lead_axes=lead,
    )


def rewrite(match: Match, root: Node) -> KernelOp | None:
    """Build the :data:`Contraction` node for a warp / register-tiled ``Semiring`` kernel; skip
    every other tier (it materializes in ``010``)."""
    tile: TileOp = root.op
    kernel = tile.kernel
    sched = kernel.schedule if kernel is not None else None
    if not isinstance(kernel, SemiringKernel):
        raise RuleSkipped("not a contraction")
    if getattr(sched, "warp_tile", None) is not None:
        node = _warp_contraction(tile, sched, root)
    elif getattr(sched, "tile", None) is not None and sched.tile.is_tiled:
        node = _scalar_contraction(tile, sched, root)
    else:
        raise RuleSkipped("non-tiled contraction — per-cell fallback in 010")  # scalar fallback
    return KernelOp(body=Body((node,)), name=tile.name)
