"""The factorizer — the ``TileOp``-root node-kind dispatcher, the contraction-tiling glue, and the
cooperative / ILP reduce tier. The per-atom codegen **strategies** it drives live in ``_atom.py``.

:func:`factorize` is the node-kind dispatcher ``010_materialize`` calls once per kernel: it reads the
structural node off ``tile.op`` (its kind + role + reduce plan) and routes to one of three tiers, all
here — :func:`_factorize_contraction` (a tiled :class:`~...ir.Contraction`), :func:`_factorize_reduce`
(a cooperative / ILP ``PLANAR`` / ``TWISTED`` reduce), or the inline **scalar tier** (a pointwise
``Map`` / trivial-plan reduction: ``lower(op)`` + :func:`with_store`, one thread per output cell).

:func:`_factorize_contraction` reads the tiling **geometry straight off the** ``Contraction`` **node**
(``tile_m`` / ``mask_m`` / ``m_b`` / ``block_threads`` / …, derived there from the ``tile`` schedule +
the output axes), expands both atoms through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``, in ``_tiling.py``), and splices in two codegen halves from
the per-atom strategies in **``_atom.py``**: :func:`~...kernel._atom.reduce_codegen` — the reusable,
**sink-agnostic** ``(state_decls, reduce_region)`` (accumulator/operand decls + the contraction
K-loop) — and a per-cell **sink** ``store(i, j, offset, mn)`` (default
:func:`~...kernel._atom.store_sink`, the matmul sink; ``_factorize_contraction(c, store=…)`` swaps it —
the flash inner QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist,
reusing the same ``reduce_codegen``).

The cooperative / ILP reduce tier (:func:`_factorize_reduce` + the shared-row staging helpers) folds
the reduce axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (``_combine.emit_combine``), and the projection — carrier-
generic (a contraction is the degenerate carrier of its additive fold).

The smem operand-staging pipeline lives in ``_stage.py`` (the :class:`~...kernel._stage.Transport`
strategy + the one :func:`~...kernel._stage.staged_kloop`); the per-tier builders (``_atom._mma_staged``
/ ``_atom._scalar_staged``) build the transport + drain leaf. It is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``sync``/``cp``/``tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct; the fused
norm→linear **shared-row** prologue (:func:`_factorize_reduce`) is a *distinct* reduce-tier smem row,
not this operand slab (a full unification is a follow-up). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.schedule import Stage
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Contraction
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_plan
from deplodock.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import emit_combine
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import copy_cell
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    sync_row_fill,
)
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile


def factorize(tile, root, store=None) -> Tile:
    """The single node-kind dispatcher — expand a ``TileOp``'s ``op`` into its bound ``Tile``.

    Two routes, split only on whether the OUTPUT is tiled into a register/warp tile:

    - a :class:`Contraction` (warp / register tile) → :func:`_factorize_contraction`, the atom-generic
      four-level pipeline. The bare grid-``Write`` is synthesized here (it needs ``root.output``, so it
      can't ride the node) into the projection ``epilogue`` before the tiling.
    - everything else → :func:`_factorize_reduce`: a ``PLANAR`` / ``TWISTED`` reduce, a non-output-tiled
      ``CONTRACTION``, or a pure pointwise ``Map``. That one path partitions the reduce axis when the
      :class:`ReducePlan` cooperates (BLOCK ``coop`` / REG ``reg``) and otherwise folds serially,
      one thread per output cell (the degenerate ``lower(op)`` + store) — the same fold either way.

    There is **no** flash / attention special case, and **no** separate "scalar tier": flash is the
    two-``Contraction`` ``TWISTED`` reduce tree, so its Q@K / P@V contractions and its streaming reduce
    factorize through these same two routes (scalar block=1 today). A third, bespoke emitter would be a
    divergent codegen path — forbidden by the mandate."""
    op = tile.op
    if isinstance(op, Contraction):
        tail = list(op.epilogue)
        if not has_write(tail):
            op = replace(op, epilogue=with_store(tail, root.output.name, tile.place.grid, op))
        return _factorize_contraction(op, tile.stage, store, tile.inputs)
    # Everything else is ONE path — a PLANAR / TWISTED reduce, a non-output-tiled CONTRACTION, or a
    # pure pointwise Map. They differ only in whether the reduce axis is partitioned (cooperative /
    # ILP) or folded serially one-thread-per-cell; `_factorize_reduce` owns both. There is no separate
    # "scalar tier" branch here — the degenerate no-partition case is that path's trivial arm.
    return _factorize_reduce(tile, root)


def _factorize_contraction(c: Contraction, stage: Stage | None = None, store=None, inputs=None) -> Tile:
    """Expand a :class:`Contraction` into its tiled ``Tile`` — the one pipeline for both atoms. The
    node supplies the per-level geometry + its operand ``stage`` (the smem pipeline both tiers lower);
    :func:`reduce_codegen` synthesizes the operand load + K-loop (``inputs`` supplies the scalar slab
    dtype) and ``store`` is the **per-cell sink** (default: the matmul :func:`store_sink`; the flash
    inner QK/PV pass a sink that bridges the accumulator into the softmax twist); the layer owns the
    offset, the axes, and the splice."""
    state_decls, reduce_region = reduce_codegen(c, stage, inputs)
    if store is None:
        store = store_sink(c)
    mn = c.mn
    t = atomize(c.atom.shape[:2])
    t = register_tile(t, mn)
    t = unit_tile(t, mn)
    return grid_tile(
        t,
        mn=mn,
        lead_axes=c.lead_axes,
        block_threads=c.block_threads,
        lanes=c.atom.lanes,
        state_decls=state_decls,
        reduce_region=reduce_region,
        store=store,
    )


# ---- cooperative / ILP reduce tier ------------------------------------------------------------- #
# A PLANAR / TWISTED monoid reduce (sum / max / mean / RMSNorm / softmax / the coop-KV TWISTED flash
# reduce) partitions the reduce axis ``coop`` ways across the CTA's threads (cooperation) and ``reg``
# ways across per-thread register accumulators (ILP). The serial reduce ``Loop`` becomes a
# :class:`StridedLoop` of step ``coop·reg``; for ``reg > 1`` its body is replicated ``reg`` times
# (each copy offset by ``r·coop`` and folding its own accumulator). After the loop: the REG tree
# folds the ``reg`` accumulators into one (``as_state_merge``), then — if ``coop > 1`` — the
# cross-thread combine (:func:`emit_combine`), then the projection. The op tree + ``lower`` are
# shared with the other tiers; only the partition changes.


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
    out = copy_cell(body, sigma, f"__r{r}", protected)
    return _mask_streamed(out, axis.name, offset, _extent_expr(axis)) if masked else out


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


def _factorize_reduce(tile, root) -> Tile:
    """Materialize the non-output-tiled path into its bound ``Tile`` (see the section header): a
    ``PLANAR`` / ``TWISTED`` reduce, a non-tiled ``CONTRACTION``, or a pointwise ``Map``.

    **Degenerate arm (no partition).** A pointwise ``Map`` (no reduce axis), or any reduce / contraction
    whose :class:`ReducePlan` is trivial (``coop == reg == 1``), is one thread per output cell: ``lower``
    emits the per-cell body (a serial reduce ``Loop`` sits inside it) and the store glue is appended.
    **Partitioned arm.** Otherwise partition the reduce axis ``coop`` ways across threads and/or ``reg``
    ways across register accumulators, as the section header describes."""
    op = tile.op
    grid = tile.place.grid
    # Eligible to partition only when the reduce axis has a cooperating plan: a PLANAR / TWISTED reduce,
    # or a non-output-tiled CONTRACTION — read structurally, not by kernel kind.
    role = axis_role(op) if op is not None else AxisRole.FREE
    tier = tile.tier
    coop_eligible = role in (AxisRole.PLANAR, AxisRole.TWISTED) or (role is AxisRole.CONTRACTION and (tier is None or not tier.is_tiled))
    plan = reduce_plan(tile) if coop_eligible else None
    if plan is None or (plan.coop <= 1 and plan.reg <= 1):
        # One thread per output cell — the degenerate fold, no cross-thread / ILP partition.
        stmts = with_store(lower(op), root.output.name, grid, op)
        return Tile(axes=tuple(grid), body=Body(tuple(stmts)))
    coop, reg = plan.coop, plan.reg
    stmts = lower(op)

    # The cooperative / cross-thread combine reads its :class:`Carrier` off the annotated reduce
    # loop (``loop.carrier``, stamped by ``lower``), NOT an op-tree node — a contraction's K loop
    # and a monoid's reduce loop both carry their carrier here. (A contraction is a monoid with a
    # ⊗ lift, so the same carrier-generic machinery — ``state`` / ``as_state_merge`` /
    # ``combine_states`` — folds it; the ⊗ lift already sits in the loop body.)
    ridx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.carrier is not None)
    rloop = stmts[ridx]
    carrier = rloop.carrier
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
            fill_stmts = sync_row_fill(
                slab=smem_name,
                src=staged,
                extent=axis.extent.as_static(),
                grid_vars=grid_vars,
                linear_tid=start,
                n_threads=coop,
                dtype=cuda_name(tile.inputs[staged].dtype),
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
    # ``as_state_merge`` is the one-shot ``StateMerge`` whose ``render`` reassigns the survivor
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
        body_tail = with_store(tail, root.output.name, grid, op)
    elif any(isinstance(s, Loop) for s in tail):
        body_tail = [
            StridedLoop(axis=s.axis, start=Var(lane.name), step=Literal(coop, "int"), body=s.body, unroll=s.unroll)
            if isinstance(s, Loop)
            else s
            for s in tail
        ]
    else:
        stored = with_store(tail, root.output.name, grid, op)
        body_tail = [Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(stored))]

    body = [*fill_stmts, *pre, strided, *reg_fold, *combine, *body_tail]
    axes = (*grid, lane) if lane is not None else tuple(grid)
    return Tile(axes=axes, body=Body(tuple(body)), block_threads=coop if lane is not None else None)
