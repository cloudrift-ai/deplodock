"""Materialize a ``TileOp``'s schedule into a ``KernelOp``.

Binds the schedule's grid axes to GPU threads and realizes the reduce partition. The step
dispatches on the kernel kind / its reduce plan:

- **Scalar tier** (``MapKernel``, or a reduction with a trivial ``ReducePlan``) — one
  thread per output cell. ``lower(op)`` emits the per-cell body (a serial reduce ``Loop``
  sits inside it, run by that one thread); the body is wrapped in a single :class:`Tile`.

- **Reduce tier** (a ``MonoidKernel`` whose ``ReducePlan`` carries a BLOCK ``coop`` and/or a
  REG ``reg`` stage) — :func:`_reduce`. The reduce axis is partitioned ``coop`` ways across
  the CTA's threads (cooperation) and ``reg`` ways across per-thread **register
  accumulators** (ILP — independent chains that hide the fold/load latency). The serial
  reduce ``Loop`` becomes a :class:`StridedLoop` of step ``coop·reg``; for ``reg > 1`` its
  body is replicated ``reg`` times (each copy offset by ``r·coop`` and folding its own
  accumulator). After the loop: the **REG tree** folds the ``reg`` accumulators into one
  (carrier-generic, via ``as_state_merge``), then — if ``coop > 1`` — the cross-thread
  combine (``_combine.emit_combine``), then the projection (full-row sweep distributed
  across the coop lanes, or a scalar output guarded to lane 0). ILP works standalone
  (``coop = 1``: register accumulators on a single thread, no cross-thread combine).

The op tree + ``lower`` are shared across kinds; only the schedule's partition changes —
the article's "schedule separate from combine" thesis.

A symbolic / non-divisible tail is **clamp-to-identity**: a replicated read whose offset
runs past the extent wraps in-bounds (``% extent``) and its value is masked to the carrier
fold's identity (so the overhang folds a no-op). ``reg = 1`` keeps the simple single-body
strided loop whose ``< extent`` bound masks naturally.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Init, Load, Loop, Select, SelectBranch, StridedLoop, Write
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
    must be written to the output buffer at the grid cell. A body that already carries a
    ``Write`` needs no glue (and ``op.out`` is left unread)."""
    if _has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=op.out)]


def _extent_expr(axis: Axis):
    """The reduce axis's extent as an ``Expr`` — a literal int (static) or the symbolic
    ``Dim`` expr (dynamic ``seq_len``)."""
    return Literal(axis.extent.as_static(), "int") if axis.extent.is_static else axis.extent.expr


def _consumer_identity(body: list[Stmt], name: str) -> float:
    """The identity that makes ``name``'s downstream fold a no-op — the op of the first body
    stmt that reads ``name`` (a ``sum`` Accum → 0, a ``max`` → −inf, online-softmax's score →
    the ``maximum`` it feeds → −inf). The masked overhang binds ``name`` to this so the fold
    contributes nothing. Defaults to 0.0 if no op-bearing consumer is found."""
    for s in body:
        op = getattr(s, "op", None)
        if name in s.deps() and op is not None and getattr(op, "identity", None) is not None:
            return op.identity
    return 0.0


def _mask_streamed(body: list[Stmt], axis: str, offset: int, extent) -> list[Stmt]:
    """Clamp-to-identity the replicated reads of a masked tail copy. Each ``Load`` whose
    index references ``axis`` (a streamed read) is split: the index already wraps in-bounds
    (``% extent`` via the caller's σ), the raw value is bound to ``<name>__raw``, and the
    output ``<name>`` becomes a ``Select`` of the raw value when ``axis + offset < extent``
    else the carrier-fold identity — so an out-of-range lane folds a no-op."""
    cond = BinaryExpr("<", BinaryExpr("+", Var(axis), Literal(offset, "int")), extent)
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Load) and s.is_scalar and any(axis in e.free_vars() for e in s.index):
            nm = s.name
            raw, ident = f"{nm}__raw", f"{nm}__id"
            out.append(replace(s, names=(raw,)))
            out.append(Init(name=ident, identity=_consumer_identity(body, nm), dtype=F32))
            out.append(Select(name=nm, branches=(SelectBranch(raw, cond), SelectBranch(ident, Literal(1, "int")))))
        else:
            out.append(s)
    return out


def _replicate(body: Body, r: int, coop: int, axis: Axis, masked: bool, protected: frozenset[str]) -> list[Stmt]:
    """Copy ``r`` of the reduce body for the REG (ILP) fold. Copy 0 is the body verbatim.
    Copy ``r > 0`` suffixes every per-copy SSA name with ``__r{r}`` (its accumulator + temps
    are an independent chain) — EXCEPT the shared iteration coordinates in ``protected`` (the
    grid / reduce / lane axis vars, common to all copies) — and offsets its streamed reads by
    ``r·coop`` (σ on the reduce axis). A ``masked`` copy wraps the read in-bounds (``% extent``)
    and clamps the value to the fold identity past the extent (:func:`_mask_streamed`)."""
    if r == 0:
        return list(body)
    offset = r * coop
    shifted = BinaryExpr("+", Var(axis.name), Literal(offset, "int"))
    index_expr = BinaryExpr("%", shifted, _extent_expr(axis)) if masked else shifted
    sigma = Sigma({axis.name: index_expr})
    rename = lambda n: n if n in protected else f"{n}__r{r}"  # noqa: E731
    out = [s.rewrite(rename, sigma) for s in body]
    return _mask_streamed(out, axis.name, offset, _extent_expr(axis)) if masked else out


def _reduce(tile: TileOp, root: Node) -> KernelOp:
    """Materialize a cooperative / ILP reduce (see module docstring)."""
    kernel = tile.kernel
    op = kernel.op
    carrier = reduce_node(op)
    plan = kernel.schedule.reduce
    coop, reg = plan.coop, plan.reg
    grid = kernel.schedule.place.grid
    stmts = lower(op)

    ridx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.axis.name == carrier.axis.name)
    rloop = stmts[ridx]
    axis = rloop.axis
    stride = coop * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)

    # The cooperative lane axis (Tile-decoded, innermost) — present only when threads
    # cooperate; standalone ILP (coop == 1) runs one thread per cell, lane fixed at 0.
    lane = Axis(name=f"{axis.name}_co", extent=coop) if coop > 1 else None
    start = Var(lane.name) if lane is not None else Literal(0, "int")

    # The reduce loop: ``reg`` interleaved accumulator chains (ILP), striding the axis by
    # ``coop·reg`` from the lane's start. The dissolved fold ``Accum``\\ s seed each copy's
    # accumulator (``StridedLoop.render``).
    # The shared iteration coordinates (grid + reduce + lane axis vars) and the symbolic
    # extent's runtime arg(s) (e.g. ``seq_len``) are common to every register copy — exclude
    # them from the per-copy SSA rename.
    protected = frozenset(
        {axis.name, *(ax.name for ax in grid), *_extent_expr(axis).free_vars()} | ({lane.name} if lane is not None else set())
    )
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, coop, axis, masked, protected))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)

    # REG tree: fold each register copy into the survivor (copy 0's names), carrier-generic —
    # ``as_state_merge`` is the one-shot ``Monoid`` whose ``render`` reassigns the survivor
    # state in place from the copy's renamed state (the same state-merge the cross-partition
    # combine uses; emitted as a stmt so ``render_merge_program`` handles the reassignment, not
    # a shadowing ``float`` redeclare).
    reg_fold: list[Stmt] = []
    for r in range(1, reg):
        other = tuple(f"{n}__r{r}" for n in carrier.state.names)
        merge = carrier.as_state_merge(other)
        # A twisted carrier's state-merge carries internal temps with fixed names; uniquify
        # them per fold (the survivor state stays put, so ``Monoid.rewrite``'s own temp
        # uniquify — keyed on a moving state — doesn't fire). A degenerate fold has no temps.
        carried = set(merge.state.names) | set(merge.twist.state_b)
        temps = {a.name for a in (*merge.twist.merge, *merge.twist.combine_states)} - carried
        if temps:
            sub = {t: f"{t}__rf{r}" for t in temps}
            merge = merge.rewrite(lambda n, sub=sub: sub.get(n, n))
        reg_fold.append(merge)

    combine = emit_combine(carrier, t=lane.name, n_threads=coop) if lane is not None else []

    # Post-reduce projection. A full-row output (softmax / RMSNorm) distributes its sweep
    # across the coop lanes; a scalar output is written once, guarded to lane 0. With no
    # cooperation (coop == 1) the single thread runs the projection as-is.
    tail = list(stmts[ridx + 1 :])
    if lane is None:
        body_tail = _with_store(tail, root.output.name, grid, op)
    elif any(isinstance(s, Loop) for s in tail):
        body_tail = [
            StridedLoop(axis=s.axis, start=Var(lane.name), step=Literal(coop, "int"), body=s.body, unroll=s.unroll)
            if isinstance(s, Loop)
            else s
            for s in tail
        ]
    else:
        stored = _with_store(tail, root.output.name, grid, op)
        body_tail = [Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(stored))]

    body = [*stmts[:ridx], strided, *reg_fold, *combine, *body_tail]
    axes = (*grid, lane) if lane is not None else tuple(grid)
    bound = Tile(axes=axes, body=Body(tuple(body)), block_threads=coop if lane is not None else None)
    return KernelOp(body=Body((bound,)), name=tile.name)


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    kernel = tile.kernel
    plan = getattr(kernel.schedule, "reduce", None) if isinstance(kernel, MonoidKernel) else None
    # Reduce tier: a Monoid reduction whose plan cooperates (BLOCK) and/or register-folds (REG).
    if plan is not None and (plan.coop > 1 or plan.reg > 1):
        return _reduce(tile, root)

    # Scalar tier: one thread per output cell. ``lower`` emits the per-cell body (any serial
    # reduce ``Loop`` sits inside it); add the output-store glue if the body has none.
    op = tile.op
    stmts = _with_store(lower(op), root.output.name, tile.schedule.place.grid, op)
    bound = Tile(axes=tuple(tile.schedule.place.grid), body=Body(tuple(stmts)))
    return KernelOp(body=Body((bound,)), name=tile.name)
