"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the schedule's grid axes to GPU threads and realizes the reduce partition. The step
dispatches on the kernel kind / its reduce plan:

- **Scalar tier** (``MapKernel``, or a reduction with a trivial ``ReducePlan``) — one
  thread per output cell. ``lower(op)`` emits the per-cell body (a serial reduce ``Loop``
  sits inside it, run by that one thread); the body is wrapped in a single :class:`Tile`
  (the linear-thread decode + bounds guard + per-axis index decode). A bare reduction's
  output ``Write`` is glue generated here from the schedule grid.

- **Cooperative tier** (a ``MonoidKernel`` whose ``ReducePlan`` carries a BLOCK stage) —
  one CTA per output cell, ``coop`` threads cooperatively folding the reduce axis. The
  serial reduce ``Loop`` becomes a :class:`StridedLoop` (each lane strides the reduce axis
  by ``coop`` from its lane index — the loop bound ``< extent`` masks the tail when ``coop``
  doesn't divide the extent), then the cross-thread combine
  (``_combine.emit_combine`` — derived ``WarpShuffle`` / ``TreeHalve``) merges the per-lane
  partials in place, then the projection / output ``Write`` runs guarded to lane 0. The
  ``Tile`` gains the cooperative lane axis (innermost) and ``block_threads = coop`` so the
  cuda lowering derives ``blockDim = coop`` / ``gridDim = output cells``.

The op tree + ``lower`` are shared across kinds; only the schedule's partition changes —
the article's "schedule separate from combine" thesis.
"""

from __future__ import annotations

from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.stmt import Body, Cond, Loop, StridedLoop, Write
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import MonoidKernel, TileOp, reduce_node
from deplodock.compiler.ir.tile.ops import lower
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import emit_combine

PATTERN = [Pattern("root", TileOp)]


def _has_write(stmts: list[Stmt]) -> bool:
    """Any ``Write`` reachable in ``stmts`` (deep — a projection's output sweep nests
    its ``Write`` inside a per-cell ``Loop``)."""
    for s in stmts:
        if isinstance(s, Write):
            return True
        if any(_has_write(list(b)) for b in s.nested()):
            return True
    return False


def _with_store(stmts: list[Stmt], output: str, grid, op) -> list[Stmt]:
    """Append the output-store glue when the body has none — a bare reduction (``op`` a
    ``Monoid`` / ``Semiring``) produces its finalized value as an SSA name (``op.out``) that
    must be written to the output buffer at the grid cell (index = the grid axes). A body
    that already carries a ``Write`` (a pointwise ``Map`` or a projection whose body writes
    through its own output sweep) needs no glue — and ``op.out`` is left unread (a projection
    ``Map`` ending in a ``Write`` has no bound value)."""
    if _has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=op.out)]


def _cooperative(tile: TileOp, root: Node) -> KernelOp:
    """Materialize a whole-CTA cooperative reduce (see module docstring)."""
    kernel = tile.kernel
    op = kernel.op
    carrier = reduce_node(op)
    coop = kernel.schedule.reduce.coop
    grid = kernel.schedule.place.grid
    stmts = lower(op)

    # Locate the serial reduce ``Loop`` over the carrier's axis (``lower`` emits exactly one
    # for the carrier; the projection — if any — follows it).
    ridx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.axis.name == carrier.axis.name)
    rloop = stmts[ridx]

    # The cooperative lane axis: a synthetic innermost ``Tile`` axis of ``coop`` threads.
    # Each lane strides the reduce axis from its lane index by ``coop`` (the ``< extent``
    # bound masks the tail when ``coop`` ∤ extent — no explicit mask needed).
    lane = Axis(name=f"{carrier.axis.name}_co", extent=coop)
    strided = StridedLoop(axis=rloop.axis, start=Var(lane.name), step=Literal(coop, "int"), body=rloop.body, unroll=rloop.unroll)
    combine = emit_combine(carrier, t=lane.name, n_threads=coop)

    # The post-reduce projection / output store, guarded to lane 0 (every lane holds the
    # full reduction after the combine, so one writer suffices for the scalar output).
    tail = _with_store(list(stmts[ridx + 1 :]), root.output.name, grid, op)
    guarded = Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(tail))

    body = [*stmts[:ridx], strided, *combine, guarded]
    bound = Tile(axes=(*grid, lane), body=Body(tuple(body)), block_threads=coop)
    return KernelOp(body=Body((bound,)), name=tile.name)


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    kernel = tile.kernel
    # Cooperative tier: a Monoid reduction whose plan carries a BLOCK (coop) stage.
    if isinstance(kernel, MonoidKernel) and getattr(kernel.schedule, "reduce", None) is not None and kernel.schedule.reduce.coop > 1:
        return _cooperative(tile, root)

    # Scalar tier: one thread per output cell. ``lower`` emits the per-cell body (any serial
    # reduce ``Loop`` sits inside it); add the output-store glue if the body has none.
    op = tile.op
    stmts = _with_store(lower(op), root.output.name, tile.schedule.place.grid, op)
    bound = Tile(axes=tuple(tile.schedule.place.grid), body=Body(tuple(stmts)))
    return KernelOp(body=Body((bound,)), name=tile.name)
