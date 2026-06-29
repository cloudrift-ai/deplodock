"""Construct the high-level ``MmaContraction`` for a tensor-core contraction — **before**
materialize.

A warp-tier ``SemiringKernel`` (a ``Semiring`` whose schedule carries a ``WarpTile``) is turned
here into a single :class:`MmaContraction` ``KernelOp``, capturing everything the atom
factorization needs: the operand ``Load``\\ s + roles (read off the ``020_schedule`` binding),
the accumulator, the resolved projection epilogue, the ``WarpTile`` / ``Stage`` geometry, and the
m/n/k axes. The expansion into the ``RegFragment`` / ``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore``
fragment soup is folded into ``010_materialize`` (which runs next), via ``_warp_factor.factorize_mma``.

Splitting the construction out of the materializer keeps the contraction a first-class node that
exists *before* thread-binding — every other tier (scalar / reduce / register-tile) is left to
``010_materialize``. A non-warp ``TileOp`` is skipped here (it passes through to materialize); the
node IS ``structural_key``-ed as an intermediate ``KernelOp``, so the final ``KernelOp`` / ``CudaOp``
keys stay byte-identical to the old single-pass materialize."""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.kernel import KernelOp
from deplodock.compiler.ir.kernel.ir import MmaContraction
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile import SemiringKernel, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", TileOp)]


def rewrite(match: Match, root: Node) -> KernelOp | None:
    """Emit the high-level :class:`MmaContraction` for a warp-tier contraction. Does the
    op-tree-dependent part only — capture the m/n/k axes, read the ``020_schedule`` binding, and
    resolve the projection epilogue (``with_store`` needs the op's ``out`` + grid). The exact atom
    factorization (the four-way split, operand staging, fragments, mma, store) is expanded from the
    node by ``010_materialize`` (:func:`_warp_factor.factorize_mma`)."""
    tile: TileOp = root.op
    kernel = tile.kernel
    sched = kernel.schedule if kernel is not None else None
    if not (isinstance(kernel, SemiringKernel) and getattr(sched, "warp_tile", None) is not None):
        raise RuleSkipped("not a warp-tier contraction")  # every other tier materializes in 010

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
    mma = MmaContraction(
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
    return KernelOp(body=Body((mma,)), name=tile.name)
