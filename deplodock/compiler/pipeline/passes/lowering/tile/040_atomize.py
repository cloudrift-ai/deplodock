"""Atomize — resolve the algebra→hardware-atom binding structurally, on the schedule.

The warp matmul materializer used to ``lower()`` the ``Semiring`` to flat loop-IR and then
re-recognize which operand is the mma ``a`` vs ``b`` (by axis-in-index), whether ``b`` is
transposed, the fold accumulator, and the projection epilogue. Every one of those facts is
already first-class on the ``Semiring`` node (``operands`` / ``fold`` / ``reduce_axis`` /
``out``) and the grid. This pass reads them **structurally** — off each operand's own leaf
``Load`` index, never a flattened loop — and stamps an :class:`AtomBinding` onto the
``SemiringSchedule`` (a sibling of the ``WarpTile`` geometry decision).

The binding rides the **schedule**, not the op tree: ``op_cache_key`` digests
``lower(op.op)`` (not the schedule), so the perf / prior cache stays byte-identical, and the
``Semiring`` combine remains the single source of truth. Runs at ``040`` — after
``030_split`` rewrites operand indices for cross-CTA slices, and after ``020_schedule`` has
chosen the ``WarpTile`` (split partials drop ``warp_tile``, so they fall through here).

Phase 1 handles the ``Semiring`` (matmul) arm; the ``Monoid`` arm + flash recursion land in
Phase 2."""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile import AtomBinding, Operand, SemiringKernel, TileOp
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped
from deplodock.compiler.pipeline.pipeline import LoweringError

PATTERN = [Pattern("root", TileOp)]


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs (the materializer's helper)."""
    return {v for e in index for v in e.free_vars()}


def _operand_leaf(operand) -> Load:
    """The buffer ``Load`` of a one-``Load`` operand ``Map`` — Phase-1's gmem-direct
    contraction operand. A non-``Load`` leaf (a nested reduction / staged-fill prologue) is
    out of scope here (it bails, matching the materializer's gmem-direct guard)."""
    leaf = operand.body[-1] if isinstance(operand, Map) and operand.body else None
    if not isinstance(leaf, Load):
        raise LoweringError("warp tier: a contraction compute prologue isn't supported (gmem-direct, no staging)")
    return leaf


def _atomize_semiring(tile: TileOp) -> AtomBinding:
    """Resolve the operand→role binding off the ``Semiring`` carrier + the output grid."""
    kernel = tile.kernel
    sched = kernel.schedule
    node = tile.op  # a Semiring, or a Map(source=Semiring) projection
    semi = node.reduce_node  # the Semiring (the contraction)
    grid = sched.place.grid
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    m_name, n_name = grid[-2].name, grid[-1].name
    k_name = semi.reduce_axis.name

    # Bind A/B by which grid output axis each operand's OWN index carries — read off the
    # operand's leaf Load, NOT a flattened loop. (Phase 1: each operand is a one-Load Map.)
    leaves = [_operand_leaf(o) for o in semi.operands]
    a_leaf = next((ld for ld in leaves if m_name in _idx_vars(ld.index)), None)
    b_leaf = next((ld for ld in leaves if n_name in _idx_vars(ld.index)), None)
    if a_leaf is None or b_leaf is None:
        raise LoweringError("warp tier: could not bind A/B operands by grid (m, n) axis")
    b_trans = k_name in b_leaf.index[-1].free_vars()  # B[n,k] (K last) vs canonical B[k,n]

    # The projection epilogue is the Map body verbatim (scale/bias/relu/residual + the output
    # Write); a bare Semiring root has none (the materializer synthesizes the store).
    epilogue = node.body if isinstance(node, Map) else Body(())
    return AtomBinding(a=Operand(a_leaf, "a"), b=Operand(b_leaf, "b"), b_trans=b_trans, acc=semi.out, epilogue=epilogue)


def rewrite(match: Match, root: Node) -> TileOp | None:
    tile: TileOp = root.op
    kernel = tile.kernel
    sched = kernel.schedule if kernel is not None else None
    # Only a warp-tiled Semiring contraction atomizes (split partials drop warp_tile).
    if not isinstance(kernel, SemiringKernel) or getattr(sched, "warp_tile", None) is None:
        raise RuleSkipped("not a warp-tiled contraction — nothing to atomize")
    if sched.bind is not None:
        raise RuleSkipped("already atomized")  # idempotent / fixpoint-safe
    bind = _atomize_semiring(tile)
    return replace(tile, kernel=replace(kernel, schedule=replace(sched, bind=bind)))
