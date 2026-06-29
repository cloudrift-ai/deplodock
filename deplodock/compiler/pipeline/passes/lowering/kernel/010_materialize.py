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
from deplodock.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel import KernelOp, Tile
from deplodock.compiler.ir.kernel.ir import EpilogueLoad, LdmatrixLoad, MmaSyncPtx, RegEpilogue, RegFragment, RegStore, Smem, Sync
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, SelectBranch, StridedLoop, Write
from deplodock.compiler.ir.stmt.base import Stmt
from deplodock.compiler.ir.tile import MonoidKernel, SemiringKernel, TileOp
from deplodock.compiler.ir.tile.ops import lower
from deplodock.compiler.pipeline import Match, Pattern
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import emit_combine
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    TMA_SLAB_ALIGN,
    CtaTile,
    cp_async_barrier,
    cp_async_fill,
    slab_smem,
    tma_descriptor,
    tma_fill,
    tma_mbar_prologue,
)
from deplodock.compiler.pipeline.pipeline import LoweringError

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


def _mask_streamed(body: list[Stmt], axis: str, offset: int, extent) -> list[Stmt]:
    """Clamp-to-identity the FOLD contribution of a masked tail copy. Each ``Accum``'s folded
    ``value`` becomes a ``Select`` of the value when ``axis + offset < extent`` else the fold's
    own identity (``op.identity`` — ``sum`` → 0, ``max`` → −inf), so an out-of-range copy folds a
    no-op. The streamed ``Load`` index is already wrapped in-bounds (``% extent`` via the caller's
    σ), so the read is safe; masking the FOLD (not the load) is what makes a **prologue** correct
    — ``sum(x·x)`` past the extent needs the *additive* identity 0, which masking the load to the
    *multiply* identity (1) would not give. A twisted carrier masks each component Accum to its
    own identity (score → −inf keeps the running max + rescale a no-op; the exp/value sums → 0)."""
    cond = BinaryExpr("<", BinaryExpr("+", Var(axis), Literal(offset, "int")), extent)
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, Accum):
            ident, masked = f"{s.value}__id", f"{s.value}__m"
            out.append(Init(name=ident, identity=s.op.identity, dtype=F32))
            out.append(Select(name=masked, branches=(SelectBranch(s.value, cond), SelectBranch(ident, Literal(1, "int")))))
            out.append(replace(s, value=masked))
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


def _shrink_axis(axis: Axis, reg: int) -> Axis:
    """The grid (cell) axis for a register-tiled free axis: ``ceil(E / reg)`` cells, each a
    per-thread ``reg``-wide register sub-tile. ``Dim.ceil_div`` keeps a symbolic extent
    symbolic (``(seq_len+reg-1)//reg``) so the launch grid sizes from the runtime extent."""
    if reg <= 1:
        return axis
    return Axis(name=axis.name, extent=axis.extent.ceil_div(reg), source_axis=axis.source_axis or axis)


def _needs_mask(axis: Axis | None, reg: int) -> bool:
    """A register-tiled axis masks its tail iff it's symbolic or its extent isn't a clean
    multiple of ``reg`` (the last cell overhangs the real extent)."""
    if axis is None or reg <= 1:
        return False
    return not (axis.extent.is_static and axis.extent.as_static() % reg == 0)


def _cell_offset(axis: Axis, reg: int, k: int):
    """The real coordinate of register cell ``k`` along ``axis``: ``cell·reg + k``."""
    return BinaryExpr("+", BinaryExpr("*", Var(axis.name), Literal(reg, "int")), Literal(k, "int"))


def _cell_sigma(m_axis, reg_m, i, mask_m, n_axis, reg_n, j, mask_n) -> Sigma:
    """σ for register cell ``(i, j)``: each tiled free index becomes its real coordinate
    ``cell·reg + k``; a **masked** axis wraps it in-bounds (``% extent``) so an overhang cell
    clamp-reads rather than runs off the buffer (its guarded write is dropped)."""
    smap: dict = {}
    if reg_m > 1 and m_axis is not None:
        off = _cell_offset(m_axis, reg_m, i)
        smap[m_axis.name] = BinaryExpr("%", off, _extent_expr(m_axis)) if mask_m else off
    if reg_n > 1:
        off = _cell_offset(n_axis, reg_n, j)
        smap[n_axis.name] = BinaryExpr("%", off, _extent_expr(n_axis)) if mask_n else off
    return Sigma(smap) if smap else Sigma.IDENTITY


def _cell_bound(m_axis, reg_m, i, mask_m, n_axis, reg_n, j, mask_n):
    """The in-bounds predicate for register cell ``(i, j)`` — ``cell·reg + k < extent`` for each
    masked axis (anded), or ``None`` when nothing overhangs."""
    conds = []
    if mask_m and m_axis is not None:
        conds.append(BinaryExpr("<", _cell_offset(m_axis, reg_m, i), _extent_expr(m_axis)))
    if mask_n:
        conds.append(BinaryExpr("<", _cell_offset(n_axis, reg_n, j), _extent_expr(n_axis)))
    if not conds:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = BinaryExpr("&&", cond, c)
    return cond


def _dedup_loads(stmts: list[Stmt]) -> list[Stmt]:
    """Collapse syntactically-identical scalar ``Load``s (same buffer + index) to one binding,
    rewriting the dropped names to the survivor — the operand reuse a register tile exists for
    (a load not referencing the ``m`` cell axis is shared across the ``n`` cells, and vice
    versa)."""
    seen: dict = {}
    rename: dict[str, str] = {}
    kept: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            sig = (s.input, tuple(e.pretty() for e in s.index))
            if sig in seen:
                rename[s.names[0]] = seen[sig]
                continue
            seen[sig] = s.names[0]
        kept.append(s)
    if rename:
        kept = [s.rewrite(lambda nm: rename.get(nm, nm)) for s in kept]
    return kept


def _guard_writes(stmts: list[Stmt], cond) -> list[Stmt]:
    """Wrap each output ``Write`` in ``Cond(cond, …)`` — the masked tail cell computes (with
    clamp-read operands) but only stores when in bounds. Non-``Write`` stmts pass through."""
    if cond is None:
        return stmts
    return [Cond(cond=cond, body=(s,)) if isinstance(s, Write) else s for s in stmts]


def _replicate_cells(
    region: list[Stmt],
    reg_m: int,
    reg_n: int,
    m_axis: Axis | None,
    n_axis: Axis,
    mask_m: bool,
    mask_n: bool,
    protected: frozenset[str],
    *,
    guard: bool,
) -> list[Stmt]:
    """Replicate ``region`` over the ``reg_m × reg_n`` register sub-tile: each cell ``(i, j)``
    σ-offsets the free indices (clamp-reading a masked axis) and suffixes its per-cell SSA names
    (``__c{i}_{j}``); the shared iteration coordinates in ``protected`` (the grid / reduce axis
    vars) stay common. With ``guard`` the cell's output ``Write`` is gated to its in-bounds
    predicate (the masked tail). Shared operand loads then collapse via :func:`_dedup_loads`."""
    copies: list[Stmt] = []
    for i in range(reg_m):
        for j in range(reg_n):
            sigma = _cell_sigma(m_axis, reg_m, i, mask_m, n_axis, reg_n, j, mask_n)
            rename = lambda nm, i=i, j=j: nm if nm in protected else f"{nm}__c{i}_{j}"  # noqa: E731
            cell = [s.rewrite(rename, sigma) for s in region]
            if guard:
                cell = _guard_writes(cell, _cell_bound(m_axis, reg_m, i, mask_m, n_axis, reg_n, j, mask_n))
            copies.extend(cell)
    return _dedup_loads(copies)


def _unroll_inner(axis: Axis) -> bool:
    """Mark the inner contraction loop for ``#pragma unroll`` when it's a small static reduce
    (≤ 64 trips) — register-resident operand reuse + ILP, the scalar-SGEMM lever."""
    return axis.extent.is_static and axis.extent.as_static() <= 64


def _reg_tile(tile: TileOp, root: Node) -> KernelOp:
    """Materialize a scalar register-tiled contraction (the ``TILE`` codec): each thread owns a
    ``reg_m × reg_n`` block of output cells. The reduce-loop body is replicated per cell with
    its operand loads deduped (the reuse), the small inner reduce is unrolled, and each cell
    seeds its own accumulator (``Loop.render``) and writes its own output cell. The parallel
    thread-tile (``par_n · par_m``) sets ``block_threads`` so ``par·reg == extent`` is one CTA."""
    kernel = tile.kernel
    op = tile.op
    plan = kernel.schedule.tile
    grid = list(kernel.schedule.place.grid)
    carrier = op.reduce_node
    raxis = carrier.reduce_axis

    n_axis = grid[-1]
    m_axis = grid[-2] if len(grid) >= 2 else None
    reg_n = plan.reg_n
    reg_m = plan.reg_m if m_axis is not None else 1
    mask_n = _needs_mask(n_axis, reg_n)
    mask_m = _needs_mask(m_axis, reg_m)

    full = _with_store(lower(op), root.output.name, grid, op)
    ridx = next(i for i, s in enumerate(full) if isinstance(s, Loop) and s.axis.name == raxis.name)
    pre, rloop, tail = full[:ridx], full[ridx], full[ridx + 1 :]

    new_grid = list(grid)
    new_grid[-1] = _shrink_axis(n_axis, reg_n)
    if m_axis is not None:
        new_grid[-2] = _shrink_axis(m_axis, reg_m)

    # The grid / reduce axis vars AND any symbolic-extent runtime arg (e.g. ``seq_len`` in a
    # ``% extent`` clamp or a ceil-div grid stride) are shared across cells — exclude them from
    # the per-cell SSA rename.
    ext_vars = {v for a in (*grid, raxis) for v in _extent_expr(a).free_vars()}
    protected = frozenset({a.name for a in grid} | {raxis.name} | ext_vars)
    cells = lambda region, guard: _replicate_cells(  # noqa: E731
        list(region), reg_m, reg_n, m_axis, n_axis, mask_m, mask_n, protected, guard=guard
    )
    pre_cells = cells(pre, False) if pre else []
    loop_body = cells(rloop.body, False)
    tail_cells = cells(tail, True)

    new_loop = Loop(axis=raxis, body=Body(tuple(loop_body)), unroll=rloop.unroll or _unroll_inner(raxis))
    block_threads = plan.par_n * plan.par_m if (plan.par_n > 1 or plan.par_m > 1) else None
    bound = Tile(axes=tuple(new_grid), body=Body((*pre_cells, new_loop, *tail_cells)), block_threads=block_threads)
    return KernelOp(body=Body((bound,)), name=tile.name)


def _scalar_loads(stmts: list[Stmt]) -> list[Load]:
    """Every scalar ``Load`` reachable in ``stmts`` (deep)."""
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            out.append(s)
        for b in s.nested():
            out.extend(_scalar_loads(list(b)))
    return out


def _has_accum(stmts: list[Stmt]) -> bool:
    return any(isinstance(s, Accum) or any(_has_accum(list(b)) for b in s.nested()) for s in stmts)


def _has_contraction_tail(stmts: list[Stmt]) -> bool:
    """The post-reduce tail contracts over a NEW free axis — a ``Loop`` (the free output
    axis) whose body holds an inner reduce ``Loop`` (an ``Accum``). This is the fused
    norm→linear shape (``for n: for k: acc += …``), and it distinguishes it from a plain
    softmax tail (a single ``for k`` sum over the SAME reduce axis, no nested contraction).
    Only the former benefits from staging the shared input row — and only it is rewritten."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and _has_accum(list(c.body)) for c in s.body):
            return True
        if any(_has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def _shared_row_buf(carrier_body, tail: list[Stmt], grid_vars: tuple, raxis: Axis, inputs: dict) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — an
    input read in the carrier reduce at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``,
    whose trailing dim is the (static) reduce extent. That row (e.g. RMSNorm's ``x[m, :]``,
    folded by the mean reduce then re-read per output column of the fused linear) is the one
    operand worth staging into smem. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not _has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input
        for s in _scalar_loads(list(carrier_body))
        if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in _scalar_loads(tail):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _restage_loads(stmts: list[Stmt], buf: str, smem: str, n_grid: int, grid_vars: tuple) -> list[Stmt]:
    """Rewrite every ``(grid…, k)`` scalar ``Load`` of ``buf`` to read ``smem[k]`` (the staged
    row), recursing into nested bodies. Other loads (and ``buf`` loads with a different index
    shape) pass through untouched."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar and s.input == buf and len(s.index) == n_grid + 1 and tuple(s.index[:n_grid]) == grid_vars:
            out.append(Load(name=s.name, input=smem, index=(s.index[-1],)))
            continue
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(tuple(_restage_loads(list(b), buf, smem, n_grid, grid_vars))) for b in bodies))
        out.append(s)
    return out


def _shared_row_fill(buf: str, smem: str, extent: int, grid_vars: tuple, n_threads: int, start, dtype_c: str) -> list[Stmt]:
    """Cooperatively copy the CTA-shared ``buf`` row ``[grid…, 0:extent]`` into ``smem`` (the
    ``n_threads`` lanes stripe it, ``for k = lane; k < extent; k += n_threads``), then a CTA
    barrier so every lane sees the filled row before the reduce + tail read it."""
    fe = Axis(name=f"_{smem}_f", extent=extent)
    load = Load(name=f"_{smem}_v", input=buf, index=(*grid_vars, Var(fe.name)))
    write = Write(output=smem, index=(Var(fe.name),), value=f"_{smem}_v")
    loop = StridedLoop(axis=fe, start=start, step=Literal(n_threads, "int"), body=Body((load, write)), unroll=False)
    return [Smem(name=smem, extents=(extent,), dtype=dtype_c), loop, Sync()]


def _reduce(tile: TileOp, root: Node) -> KernelOp:
    """Materialize a cooperative / ILP reduce (see module docstring)."""
    kernel = tile.kernel
    op = kernel.op
    carrier = op.reduce_node
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

    # Shared-row staging (the fused norm→linear prologue): when an input row is folded by the
    # cooperative reduce AND re-read per output column of a contraction tail, stage it into smem
    # once (cooperatively) and rewrite both readers to the slab — one ``__shared__`` row shared
    # by the prologue + the matmul body. Only the cooperative tier (coop > 1) stages.
    pre = list(stmts[:ridx])
    tail_src = list(stmts[ridx + 1 :])
    fill_stmts: list[Stmt] = []
    if lane is not None:
        grid_vars = tuple(Var(a.name) for a in grid)
        staged = _shared_row_buf(rloop.body, tail_src, grid_vars, axis, tile.inputs)
        if staged is not None:
            from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

            smem_name = f"{staged}_smem"
            fill_stmts = _shared_row_fill(
                staged, smem_name, axis.extent.as_static(), grid_vars, coop, start, cuda_name(tile.inputs[staged].dtype)
            )
            n_grid = len(grid)
            rloop = replace(rloop, body=Body(tuple(_restage_loads(list(rloop.body), staged, smem_name, n_grid, grid_vars))))
            pre = _restage_loads(pre, staged, smem_name, n_grid, grid_vars)
            tail_src = _restage_loads(tail_src, staged, smem_name, n_grid, grid_vars)

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
        # ``as_state_merge`` regenerates the finalize with its temps keyed on ``other[0]`` (or has
        # none, for a degenerate fold), so each fold's internal temps are already unique — no
        # per-fold uniquify needed.
        reg_fold.append(carrier.as_state_merge(other))

    combine = emit_combine(carrier, t=lane.name, n_threads=coop) if lane is not None else []

    # Post-reduce projection. A full-row output (softmax / RMSNorm) distributes its sweep
    # across the coop lanes; a scalar output is written once, guarded to lane 0. With no
    # cooperation (coop == 1) the single thread runs the projection as-is.
    tail = tail_src
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

    body = [*fill_stmts, *pre, strided, *reg_fold, *combine, *body_tail]
    axes = (*grid, lane) if lane is not None else tuple(grid)
    bound = Tile(axes=axes, body=Body(tuple(body)), block_threads=coop if lane is not None else None)
    return KernelOp(body=Body((bound,)), name=tile.name)


# --------------------------------------------------------------------------- #
# The warp (tensor-core mma) tier — a ``Semiring`` contraction's ``WarpTile`` fragment.
# --------------------------------------------------------------------------- #


def _idx_vars(index) -> set[str]:
    """Every free Var name across an index tuple's exprs."""
    return {v for e in index for v in e.free_vars()}


def _axis_base(block: str, warp: str, n_warp: int, n_reg: int, atom_dim: int, r: int):
    """The base coordinate (NO atom/lane offset) of register cell ``r`` along one free axis:
    ``block·(n_warp·n_reg·atom_dim) + warp·(n_reg·atom_dim) + r·atom_dim``. The atom-lane
    offset is added at render (inside ``dpl_mma_load_*`` / ``RegStore``), so it stays out of σ
    — the four-way GRID/WARP/REGISTER/ATOM split with the ATOM term omitted."""
    tile = n_warp * n_reg * atom_dim
    e = BinaryExpr("*", Var(block), Literal(tile, "int"))
    e = BinaryExpr("+", e, BinaryExpr("*", Var(warp), Literal(n_reg * atom_dim, "int")))
    return BinaryExpr("+", e, Literal(r * atom_dim, "int"))


def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for a dim varying with the output row /
    col axis, else ``"fixed"`` (batch / grid literal — uniform across the fragment cell)."""
    roles = []
    for e in index:
        fv = e.free_vars()
        roles.append("m" if m_name in fv else "n" if n_name in fv else "fixed")
    return tuple(roles)


def _warp_epilogue(pre: list[Stmt], tail: list[Stmt], acc: str, m_name: str, n_name: str, sigma: Sigma) -> RegEpilogue | None:
    """Fold the projection ``Map`` into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
    there is no projection (a bare ``Write`` of the accumulator).

    The projection is the ``lower`` stmts straddling the K reduce loop: ``pre`` — the
    loop-invariant scalar leaf ``Load``s the lift parks above the loop (a fused matmul's scale /
    mask constants) — plus ``tail`` — the post-reduce leaf ``Load``s + pointwise ``Assign``s +
    an optional causal ``Select``. Each leaf ``Load`` becomes an :class:`EpilogueLoad` at the
    cell-base coordinate (σ-applied; the render adds the per-element row/col motion on the
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args)`` op; a coord-predicated
    ``Select`` (causal mask) rewrites its ``m``/``n`` coordinate vars to the ``__M__`` / ``__N__``
    placeholders the store substitutes with the element's own (row, col)."""
    loads, ops, selects = [], [], []
    write = None
    ph = {m_name: Var("__M__"), n_name: Var("__N__")}
    for s in (*pre, *tail):
        if isinstance(s, Load):
            loads.append(
                EpilogueLoad(
                    name=s.names[0],
                    buffer=s.input,
                    index=tuple(sigma.apply(e) for e in s.index),
                    roles=_warp_roles(s.index, m_name, n_name),
                )
            )
        elif isinstance(s, Assign):
            ops.append((s.name, s.op.name, tuple(s.args)))
        elif isinstance(s, Select):
            selects.append((s.name, tuple((br.select.substitute(ph), br.value) for br in s.branches)))
        elif isinstance(s, Write):
            write = s
    if write is None or (not ops and not selects):
        return None
    return RegEpilogue(acc=acc, loads=tuple(loads), ops=tuple(ops), result=write.value, selects=tuple(selects))


def _can_stage_warp(stage, k_axis: Axis, tile_m: int, tile_n: int, bk: int, atom_k: int, mask_m: bool, mask_n: bool, b_trans: bool) -> bool:
    """Staging eligibility for the warp tier (cp.async): a ``cp.async`` stage over a
    contraction with a STATIC, tile-divisible K axis and a canonical (non-transposed) B
    operand. A masked / symbolic **M** (output rows) is fine — the A-slab fill clamp-reads
    the overhanging rows in-bounds and the ``RegStore`` guards their store. A masked **N**
    (the B-slab inner dim) and a symbolic / non-divisible **K** stay gmem-direct for now
    (K zero-fill is the symbolic-K follow-up). Staging only ever *adds* a faster lowering,
    so an ineligible kernel silently falls back to gmem-direct."""
    if stage is None or stage.transport != "cp.async" or b_trans or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    k = k_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    # cp.async needs a ≥4-byte contiguous chunk; the 16-bit mma operands give 2 B/elem,
    # so the inner slab dim must be even (A's BK, B's tile_n). Odd ⇒ fall back.
    return (bk_elems % 2 == 0) and (tile_n % 2 == 0)


def _staged_inner_atom_loop(*, a_slab, b_slab, m_w, n_w, fm, fn, atom, bk_elems, tile_n, ki) -> StridedLoop:
    """The inner atom-K loop shared by the cp.async and TMA staged paths: read the
    A/B slabs via ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. Slab-local indices —
    A[tile_m][bk_elems] (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent
    of which producer (cp.async / TMA) filled the (plain row-major, NONE-swizzle) slab."""
    atom_m, atom_n, atom_k = atom.shape
    inner: list[Stmt] = []
    for i in range(fm):
        a_row = BinaryExpr("+", BinaryExpr("*", Var(m_w), Literal(fm * atom_m, "int")), Literal(i * atom_m, "int"))
        inner.append(LdmatrixLoad(frag=f"_a{i}", src_buffer=a_slab, src_index=(a_row, Var(ki)), role="a", staged=True, ldm=bk_elems))
    for j in range(fn):
        b_col = BinaryExpr("+", BinaryExpr("*", Var(n_w), Literal(fn * atom_n, "int")), Literal(j * atom_n, "int"))
        inner.append(LdmatrixLoad(frag=f"_b{j}", src_buffer=b_slab, src_index=(Var(ki), b_col), role="b", staged=True, ldm=tile_n))
    for i in range(fm):
        for j in range(fn):
            inner.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
    return StridedLoop(
        axis=Axis(name=ki, extent=bk_elems), start=Literal(0, "int"), step=Literal(atom_k, "int"), body=Body(tuple(inner)), unroll=True
    )


def _warp_staged_kloop(
    *, a_load, b_load, m_axis, n_axis, k_axis, m_b, n_b, m_w, n_w, wm, wn, fm, fn, atom, bk, tile_m, tile_n, slab_dtype, elem_bytes, mask_m
) -> tuple[list[Stmt], list[Stmt]]:
    """The warp tier's STAGED K loop (cp.async, single-buffer): allocate the A/B smem
    slabs, then an outer K-slab loop that cooperatively cp.async-fills both slabs, syncs,
    and runs the inner atom loop reading the slabs via ``LdmatrixLoad(staged=True)``.
    Returns ``(slab_decls, [outer_loop])``. depth>1 double-buffering is a follow-up — this
    is the correct single-buffer form (numerically identical, no prefetch overlap).

    ``mask_m`` (symbolic / non-divisible output rows): the A-slab fill clamps the gmem row
    in-bounds (``% M``) so an overhanging row reads a duplicate rather than past the buffer;
    the duplicate's contribution is discarded by the ``RegStore`` ``m_guard``."""
    atom_k = atom.shape[2]
    bk_elems = bk * atom_k  # K elements per slab (the BK chunk)
    a_slab, b_slab = "_a_smem", "_b_smem"
    n_threads = wm * wn * 32
    row_base = BinaryExpr("*", Var(m_b), Literal(tile_m, "int"))  # CTA tile base row
    col_base = BinaryExpr("*", Var(n_b), Literal(tile_n, "int"))  # CTA tile base col
    # intra-CTA linear thread id from the decoded warp / lane axis vars (the innermost
    # block_threads of the [m_b, n_b, m_w, n_w, _lane] tile) — never a raw threadIdx.x.
    linear_tid = BinaryExpr(
        "+", BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(m_w), Literal(wn, "int")), Var(n_w)), Literal(32, "int")), Var("_lane")
    )
    cta = CtaTile(row_base=row_base, col_base=col_base, linear_tid=linear_tid, n_threads=n_threads)

    k0 = "_ks"  # the outer K-slab offset var
    ki = "_ki"  # the inner atom-K offset within the slab

    def a_gmem(row, col):  # slab[row][col] = A[row_base + row][k0 + col]
        m = BinaryExpr("+", row_base, row)
        if mask_m:  # clamp an overhanging row to the last valid one — its store is RegStore-guarded
            mext = _extent_expr(m_axis)
            m = TernaryExpr(cond=BinaryExpr("<", m, mext), if_true=m, if_false=BinaryExpr("-", mext, Literal(1, "int")))
        sig = Sigma({m_axis.name: m, k_axis.name: BinaryExpr("+", Var(k0), col)})
        return tuple(sig.apply(e) for e in a_load.index)

    def b_gmem(row, col):  # slab[row][col] = B[k0 + row][col_base + col]
        sig = Sigma({k_axis.name: BinaryExpr("+", Var(k0), row), n_axis.name: BinaryExpr("+", col_base, col)})
        return tuple(sig.apply(e) for e in b_load.index)

    fills: list[Stmt] = []
    fills += cp_async_fill(
        slab=a_slab, slab_rows=tile_m, slab_cols=bk_elems, src=a_load.input, gmem_index=a_gmem, cta=cta, elem_bytes=elem_bytes, name="a"
    )
    fills += cp_async_fill(
        slab=b_slab, slab_rows=bk_elems, slab_cols=tile_n, src=b_load.input, gmem_index=b_gmem, cta=cta, elem_bytes=elem_bytes, name="b"
    )
    fills += cp_async_barrier()

    # Inner atom loop: read the slabs via staged ldmatrix + mma (shared with the TMA path).
    inner_loop = _staged_inner_atom_loop(
        a_slab=a_slab, b_slab=b_slab, m_w=m_w, n_w=n_w, fm=fm, fn=fn, atom=atom, bk_elems=bk_elems, tile_n=tile_n, ki=ki
    )

    # Single buffer: a trailing Sync so the next slab fill can't clobber the slab while a
    # lagging thread still reads it.
    outer_body = (*fills, inner_loop, Sync())
    outer = StridedLoop(
        axis=Axis(name=k0, extent=k_axis.extent.as_static()),
        start=Literal(0, "int"),
        step=Literal(bk_elems, "int"),
        body=Body(outer_body),
        unroll=False,
    )
    slab_decls = [slab_smem(a_slab, tile_m, bk_elems, slab_dtype), slab_smem(b_slab, bk_elems, tile_n, slab_dtype)]
    return slab_decls, [outer]


def _can_stage_warp_tma(
    stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n: bool, b_trans: bool
) -> bool:
    """Staging eligibility for the warp tier via TMA (``cp.async.bulk.tensor``): a ``tma``
    stage over a contraction with a STATIC, tile-divisible K and a canonical B. A masked /
    symbolic **M** is fine — the descriptor's globalDim is the runtime M and TMA zero-fills
    the box overhang past it (no fill clamp needed). A masked **N** and a symbolic / non-
    divisible **K** stay gmem-direct. The box's inner dim (A's BK, B's tile_n) and the
    source's inner global stride (A's K, B's N) must be 16 B-aligned (the NONE-swizzle TMA
    box-copy rule)."""
    if stage is None or stage.transport != "tma" or b_trans or mask_n:
        return False
    if not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    bk_elems = bk * atom_k
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    # 16 B alignment: box inner dims (BK, tile_n) and source inner strides (K, N).
    return all((x * elem_bytes) % 16 == 0 for x in (bk_elems, tile_n, k, n))


def _warp_tma_staged_kloop(
    *, a_load, b_load, k_axis, m_b, n_b, m_w, n_w, wm, wn, fm, fn, atom, bk, tile_m, tile_n, slab_dtype, elem_bytes
) -> tuple[list[Stmt], list[Stmt]]:
    """The warp tier's STAGED K loop via **TMA** (single-buffer): declare the A/B
    ``CUtensorMap`` descriptors + the destination slabs + one mbarrier, then an outer
    K-slab loop where the issuer thread box-copies both operand tiles (``cp.async.bulk.tensor``)
    and every thread waits on the mbarrier parity before the shared staged-``LdmatrixLoad`` +
    mma drain. ``mask_m`` needs no fill clamp — TMA zero-fills the box overhang past the
    descriptor's runtime globalDim (the RegStore still guards the masked-row stores)."""
    atom_k = atom.shape[2]
    bk_elems = bk * atom_k
    a_slab, b_slab, mbar = "_a_smem", "_b_smem", "_mbar"
    desc_a, desc_b = "_desc_a", "_desc_b"
    row_base = BinaryExpr("*", Var(m_b), Literal(tile_m, "int"))
    col_base = BinaryExpr("*", Var(n_b), Literal(tile_n, "int"))
    linear_tid = BinaryExpr(
        "+", BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(m_w), Literal(wn, "int")), Var(n_w)), Literal(32, "int")), Var("_lane")
    )
    tid0 = BinaryExpr("==", linear_tid, Literal(0, "int"))

    k0, ki = "_ks", "_ki"
    a_bytes = tile_m * bk_elems * elem_bytes
    b_bytes = bk_elems * tile_n * elem_bytes
    # Box origins (C-order) per K chunk: A[row_base][k0], B[k0][col_base].
    a_coords = (row_base, Var(k0))
    b_coords = (Var(k0), col_base)
    phase = BinaryExpr("%", BinaryExpr("/", Var(k0), Literal(bk_elems, "int")), Literal(2, "int"))

    fill = tma_fill(
        loads=[(desc_a, a_slab, a_coords), (desc_b, b_slab, b_coords)], mbar=mbar, tid0=tid0, phase=phase, total_bytes=a_bytes + b_bytes
    )
    inner_loop = _staged_inner_atom_loop(
        a_slab=a_slab, b_slab=b_slab, m_w=m_w, n_w=n_w, fm=fm, fn=fn, atom=atom, bk_elems=bk_elems, tile_n=tile_n, ki=ki
    )
    outer = StridedLoop(
        axis=Axis(name=k0, extent=k_axis.extent.as_static()),
        start=Literal(0, "int"),
        step=Literal(bk_elems, "int"),
        body=Body((*fill, inner_loop, Sync())),
        unroll=False,
    )
    slab_decls = [
        tma_descriptor(desc_a, a_load.input, (tile_m, bk_elems), slab_dtype),
        tma_descriptor(desc_b, b_load.input, (bk_elems, tile_n), slab_dtype),
        slab_smem(a_slab, tile_m, bk_elems, slab_dtype, align=TMA_SLAB_ALIGN),
        slab_smem(b_slab, bk_elems, tile_n, slab_dtype, align=TMA_SLAB_ALIGN),
    ]
    prologue = tma_mbar_prologue(mbar, tid0)
    return slab_decls, [*prologue, outer]


def _warp(tile: TileOp, root: Node) -> KernelOp:
    """Materialize a tensor-core (mma) contraction (the ``WARP`` codec): the CTA runs ``WM·WN``
    warps over a ``tile_m × tile_n`` output block, each warp owning an ``FM × FN`` block of
    ``atom`` cells. Free axes split four ways (GRID block / WARP / REGISTER / ATOM); the block +
    warp + register offsets ride σ, the atom-lane offset is decoded at render inside the
    ``dpl_mma_load_*`` / ``RegStore`` helpers (so a ``Tile`` of ``[m_b, n_b, m_w, n_w, lane32]``
    with ``block_threads = WM·WN·32`` binds threads to lanes). Per register cell: a persistent
    f32 ``RegFragment`` accumulator + a K loop of gmem-direct ``LdmatrixLoad`` a/b + ``MmaSyncPtx``,
    then a ``RegStore`` (fused projection epilogue + masked-tile guards). Gmem-direct, single
    launch, static K (symbolic K needs the staged zero-fill — out of scope, bails)."""
    kernel = tile.kernel
    node = tile.op
    carrier = node.reduce_node
    wt = kernel.schedule.warp_tile
    atom = wt.atom
    atom_m, atom_n, atom_k = atom.shape
    wm, wn = wt.warps
    fm, fn = wt.reg
    grid = list(kernel.schedule.place.grid)
    if len(grid) < 2:
        raise LoweringError("warp tier: contraction output needs an (m, n) grid")
    m_axis, n_axis = grid[-2], grid[-1]
    k_axis = carrier.reduce_axis

    full = _with_store(lower(node), root.output.name, grid, node)
    ridx = next(i for i, s in enumerate(full) if isinstance(s, Loop) and s.axis.name == k_axis.name)
    pre, rloop, tail = list(full[:ridx]), full[ridx], list(full[ridx + 1 :])
    # The lift parks loop-invariant scalar leaf Loads (a fused matmul's scale / mask constants)
    # above the K loop — fold them into the epilogue. A non-Load prologue (a staged slab fill) is
    # out of scope (gmem-direct, no staging).
    if any(not isinstance(s, Load) for s in pre):
        raise LoweringError("warp tier: a contraction compute prologue isn't supported (gmem-direct, no staging)")
    rbody = list(rloop.body)
    loads = [s for s in rbody if isinstance(s, Load)]
    a_load = next((s for s in loads if m_axis.name in _idx_vars(s.index)), None)
    b_load = next((s for s in loads if n_axis.name in _idx_vars(s.index)), None)
    write = next((s for s in tail if isinstance(s, Write)), None)
    acc = next((s.name for s in rbody if isinstance(s, Accum)), None)
    if a_load is None or b_load is None or write is None or acc is None:
        raise LoweringError("warp tier: unrecognised contraction body (expected A/B loads + Accum + Write)")
    b_trans = k_axis.name in b_load.index[-1].free_vars()  # B[n,k] (K last) vs canonical B[k,n]

    tile_m, tile_n = wt.tile_m, wt.tile_n
    mask_m = not (m_axis.extent.is_static and m_axis.extent.as_static() % tile_m == 0)
    mask_n = not (n_axis.extent.is_static and n_axis.extent.as_static() % tile_n == 0)
    m_b, n_b = f"{m_axis.name}_wb", f"{n_axis.name}_wb"
    m_w, n_w = f"{m_axis.name}_ww", f"{n_axis.name}_ww"
    base_m = lambda r: _axis_base(m_b, m_w, wm, fm, atom_m, r)  # noqa: E731
    base_n = lambda r: _axis_base(n_b, n_w, wn, fn, atom_n, r)  # noqa: E731

    # Per register cell (i, j): operand fragments dedup across the OTHER axis (A on m only,
    # B on n only — the arithmetic-intensity reuse), the C accumulator is per (i, j).
    decls: list[Stmt] = []
    for i in range(fm):
        decls.append(RegFragment(name=f"_a{i}", role="a", shape=atom.shape, dtype=atom.operand_dtype("a")))
    for j in range(fn):
        decls.append(RegFragment(name=f"_b{j}", role="b", shape=atom.shape, dtype=atom.operand_dtype("b")))
    for i in range(fm):
        for j in range(fn):
            decls.append(RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c")))

    stage = kernel.schedule.stage
    slab_decls: list[Stmt] = []
    if _can_stage_warp_tma(stage, k_axis, n_axis, tile_n, wt.bk, atom_k, atom.operand_dtype("a").nbytes, mask_n, b_trans):
        from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

        slab_dtype = cuda_name(atom.operand_dtype("a"))
        slab_decls, kstmts = _warp_tma_staged_kloop(
            a_load=a_load,
            b_load=b_load,
            k_axis=k_axis,
            m_b=m_b,
            n_b=n_b,
            m_w=m_w,
            n_w=n_w,
            wm=wm,
            wn=wn,
            fm=fm,
            fn=fn,
            atom=atom,
            bk=wt.bk,
            tile_m=tile_m,
            tile_n=tile_n,
            slab_dtype=slab_dtype,
            elem_bytes=atom.operand_dtype("a").nbytes,
        )
    elif _can_stage_warp(stage, k_axis, tile_m, tile_n, wt.bk, atom_k, mask_m, mask_n, b_trans):
        from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

        slab_dtype = cuda_name(atom.operand_dtype("a"))
        elem_bytes = atom.operand_dtype("a").nbytes
        slab_decls, kstmts = _warp_staged_kloop(
            a_load=a_load,
            b_load=b_load,
            m_axis=m_axis,
            n_axis=n_axis,
            k_axis=k_axis,
            m_b=m_b,
            n_b=n_b,
            m_w=m_w,
            n_w=n_w,
            wm=wm,
            wn=wn,
            fm=fm,
            fn=fn,
            atom=atom,
            bk=wt.bk,
            tile_m=tile_m,
            tile_n=tile_n,
            slab_dtype=slab_dtype,
            elem_bytes=elem_bytes,
            mask_m=mask_m,
        )
    else:
        # Gmem-direct. A symbolic / non-divisible K zero-fills the masked-K tail via the
        # ``k_zero`` helper variants (``dpl_mma_load_*_kzero``) — a duplicate read would
        # corrupt the reduction (unlike a masked M/N row, whose store is just guarded), so
        # the overhang must contribute ZERO, not a clamped duplicate. Transposed-B has no
        # gmem-direct K zero-fill helper, so a transposed-B symbolic K still bails.
        k_static = k_axis.extent.is_static
        if not k_static and b_trans:
            raise LoweringError("warp tier: transposed-B symbolic-K mma not supported (no gmem-direct K zero-fill)")
        k_zero = None if k_static else (Var(k_axis.name), _extent_expr(k_axis))
        chain: list[Stmt] = []
        for i in range(fm):
            idx = tuple(Sigma({m_axis.name: base_m(i)}).apply(e) for e in a_load.index)
            guard = (base_m(i), _extent_expr(m_axis)) if mask_m else None
            chain.append(
                LdmatrixLoad(frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero)
            )
        for j in range(fn):
            idx = tuple(Sigma({n_axis.name: base_n(j)}).apply(e) for e in b_load.index)
            guard = (base_n(j), _extent_expr(n_axis)) if mask_n else None
            chain.append(
                LdmatrixLoad(
                    frag=f"_b{j}",
                    src_buffer=b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=guard,
                    k_zero=k_zero,
                )
            )
        for i in range(fm):
            for j in range(fn):
                chain.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
        kstmts = [StridedLoop(axis=k_axis, start=Literal(0, "int"), step=Literal(atom_k, "int"), body=Body(tuple(chain)), unroll=k_static)]

    stores: list[Stmt] = []
    for i in range(fm):
        for j in range(fn):
            sigma = Sigma({m_axis.name: base_m(i), n_axis.name: base_n(j)})
            stores.append(
                RegStore(
                    dst_buffer=write.output,
                    dst_index=tuple(sigma.apply(e) for e in write.index),
                    frag=f"_c{i}_{j}",
                    shape=atom.shape,
                    epilogue=_warp_epilogue(pre, tail, acc, m_axis.name, n_axis.name, sigma),
                    m_guard=(base_m(i), _extent_expr(m_axis)) if mask_m else None,
                    n_guard=(base_n(j), _extent_expr(n_axis)) if mask_n else None,
                )
            )

    axes = (
        _shrink_axis(Axis(name=m_b, extent=m_axis.extent, source_axis=m_axis), tile_m),
        _shrink_axis(Axis(name=n_b, extent=n_axis.extent, source_axis=n_axis), tile_n),
        Axis(name=m_w, extent=wm),
        Axis(name=n_w, extent=wn),
        Axis(name="_lane", extent=32),
    )
    bound = Tile(axes=axes, body=Body((*decls, *slab_decls, *kstmts, *stores)), block_threads=wt.block_threads)
    return KernelOp(body=Body((bound,)), name=tile.name)


def rewrite(match: Match, root: Node) -> KernelOp | None:
    tile: TileOp = root.op
    kernel = tile.kernel
    # By the kernel pass, ``030_split`` has consumed every cross-CTA ``GRID`` stage (the
    # partial's plan is stripped, the finalize is a fresh ``ReducePlan``). A surviving split
    # request is a bug — the materializer only lowers single-launch kernels.
    rplan = getattr(kernel.schedule, "reduce", None) if kernel is not None else None
    assert rplan is None or not rplan.needs_split, "materialize: a GRID split stage survived 030_split"
    # Warp (tensor-core mma) tier: a Semiring contraction carrying a WarpTile fragment.
    sched = kernel.schedule
    if isinstance(kernel, SemiringKernel) and getattr(sched, "warp_tile", None) is not None:
        return _warp(tile, root)
    # Register-tile tier: a Semiring contraction whose ``TILE`` plan tiles the output (each
    # thread owns a reg_m×reg_n register block of cells, operands reused across them).
    tplan = getattr(sched, "tile", None) if isinstance(kernel, SemiringKernel) else None
    if tplan is not None and tplan.is_tiled:
        return _reg_tile(tile, root)
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
