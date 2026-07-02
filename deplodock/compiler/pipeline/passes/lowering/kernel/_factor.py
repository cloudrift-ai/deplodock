"""The factorizer — the recursive ``TileOp``-root emitter and the ONE root binder every kernel
seals through. The per-atom codegen **strategies** it drives live in ``_atom.py``.

:func:`factorize` is the entry ``010_materialize`` calls once per kernel: it builds the ambient
:class:`Ctx` and dispatches ``tile.op`` through the recursion :func:`_factorize`, which walks the
``Map`` / ``Reduction`` / ``Contraction`` node tree. A :class:`~...ir.Map` with a ``source``
**recurses** (its projection walked into the ``tail``); the leaf binds to the grid via the single
:func:`_bind` pipeline, whose form is read off the node's SCHEDULE — which axes are tiled — never a
kernel kind: a tiled :class:`~...ir.Contraction` tiles its OUTPUT ``(m, n)`` axes (register / warp
cells), a cooperating :class:`~...ir.Reduction` tiles its REDUCE axis (:func:`_tile_reduce_axis` —
``coop`` lanes + ``reg`` ILP chains), and everything else tiles nothing (the degenerate
one-thread-per-cell fold). All three seal through the one :func:`grid_tile` finalizer; the per-cell
body is built by the shared recursion :func:`_emit` (which walks ``source`` AND ``partial``,
reaching flash's Q@K / P@V as nodes).

The output tiling reads its **geometry straight off the** ``Contraction`` **node** (``tile_m`` /
``mask_m`` / ``m_b`` / ``block_threads`` / …, derived there from the ``tile`` schedule + the output
axes), expands both atoms through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``, in this module), and splices in two codegen halves from
the per-atom strategies in **``_atom.py``**: :func:`~...kernel._atom.reduce_codegen` — the reusable,
**sink-agnostic** ``(state_decls, reduce_region)`` (accumulator/operand decls + the contraction
K-loop) — and a per-cell **sink** ``store(i, j, offset, mn)`` (default
:func:`~...kernel._atom.store_sink`, the matmul sink; ``factorize(tile, root, store=…)`` swaps it —
the flash inner QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist,
reusing the same ``reduce_codegen``).

The reduce-axis tiling (:func:`_tile_reduce_axis` + the shared-row staging apply) folds the reduce
axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (:func:`emit_combine`), and the projection — carrier-
generic (a contraction is the degenerate carrier of its additive fold).

The smem operand-staging pipeline lives in ``_stage.py`` (the :class:`~...kernel._stage.Transport`
strategy + the one :func:`~...kernel._stage.staged_kloop`); the ONE atom-agnostic driver
(``_atom._staged``) builds the transport, the atom strategy supplying only the slab drain leaf.
It is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``sync``/``cp``/``tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct. The fused
norm→linear **shared-row** prologue is Stage-driven too: ``020_schedule`` detects the reused input row
and stamps a ``sync`` :class:`~...schedule.Stage` whose ``smem`` names it; :func:`_tile_reduce_axis` only
applies it (the 1-D ``sync_row_fill`` + the load rewrite). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import Smem, Sync, TreeHalve, WarpShuffle
from deplodock.compiler.ir.schedule import Stage
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile import Fold, Level, ReduceStage
from deplodock.compiler.ir.tile.ir import Contraction, Map, Reduction, Side
from deplodock.compiler.pipeline.passes.lowering.kernel._atom import copy_cell, reduce_codegen, shrink_axis, store_sink
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import sync_row_fill
from deplodock.compiler.pipeline.passes.lowering.kernel._twist import realize_warp_twist, warp_source


# ---- generic tiling layer: atomize → register_tile → unit_tile → grid_tile ---------------------- #
# A contraction is lowered by tiling a leaf atom four ways: GRID block / UNIT / REGISTER / ATOM. The
# UNIT is the atom's parallel thread footprint (a warp for mma, a thread for scalar). Each level zips
# the per-axis :class:`AxisOffset` pair (``Tiling.offset``) with the ``(m, n)`` :class:`Side` pair, so
# the two axes never split into ``*_m`` / ``*_n`` locals; :func:`grid_tile` (the finalizer) splices the
# atom's ``state`` / ``reduce_region`` / ``store`` callables (from :func:`reduce_codegen` / the sink) in.
@dataclass(frozen=True)
class AxisOffset:
    """One output axis's per-register-cell coordinate, accumulated across the tiling levels (atom →
    register → unit → grid). :meth:`base` reproduces ``block·(units·reg·atom) + unit·(reg·atom) +
    r·atom`` once the UNIT level is present (the mma warp tile AND the scalar thread tile both go
    through :func:`unit_tile`), else the bare ``Var(block)·reg + r``."""

    atom_dim: int  # the atom step along this axis
    reg: int = 1  # register sub-cells per unit
    block_var: str | None = None  # the grid-block axis var (set at grid_tile)
    unit_var: str | None = None  # the UNIT-level var — a warp for mma, a thread for scalar
    unit_count: int = 1

    def base(self, r: int) -> Expr:
        """The offset of register cell index ``r`` along this axis."""
        reg_term = Literal(r * self.atom_dim, "int")
        if self.unit_var is not None:  # block·(units·reg·atom) + unit·(reg·atom) + r·atom
            tile = self.unit_count * self.reg * self.atom_dim
            e = BinaryExpr("*", Var(self.block_var), Literal(tile, "int"))
            e = BinaryExpr("+", e, BinaryExpr("*", Var(self.unit_var), Literal(self.reg * self.atom_dim, "int")))
            return BinaryExpr("+", e, reg_term)
        return BinaryExpr("+", BinaryExpr("*", Var(self.block_var), Literal(self.reg, "int")), reg_term)  # no unit level


@dataclass(frozen=True)
class Tiling:
    """The accumulating tiling state threaded through ``atomize → register_tile → unit_tile →
    grid_tile`` — the per-axis ``(m, n)`` :class:`AxisOffset` tuple ``offset`` + the bound ``Tile``
    axes (unit → grid) + ``block_threads``. Each level ``zip``\\ s ``offset`` with the ``(m, n)``
    :class:`Side` pair, so the two axes never split into ``*_m`` / ``*_n`` locals."""

    offset: tuple[AxisOffset, AxisOffset]
    axes: tuple[Axis, ...] = ()
    block_threads: int | None = None


def atomize(atoms: tuple[int, int]) -> Tiling:
    """The leaf: a single ``(atom_m, atom_n)`` atom (1×1 for a scalar cell). Seeds the per-axis
    offset with the atom step; the atom-lane offset stays OUT of σ (added at render)."""
    return Tiling(offset=tuple(AxisOffset(atom_dim=a) for a in atoms))


def register_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The REGISTER level: ``m.reg × n.reg`` atoms per thread/warp. Records the cell counts; the
    per-cell ``r·atom_dim`` term is applied at :meth:`AxisOffset.base`."""
    return replace(t, offset=tuple(replace(o, reg=s.reg) for o, s in zip(t.offset, mn, strict=True)))


def unit_tile(t: Tiling, mn: tuple[Side, Side]) -> Tiling:
    """The UNIT level: ``m.units × n.units`` parallel units per CTA, where a *unit* is the atom's
    thread footprint — a warp (32 lanes) for an mma atom, a single thread for a scalar atom (so the
    tensor-core warp tile and the scalar parallel thread-tile are the same level, differing only in
    the atom's ``lanes``). Adds the unit term ``unit·(reg·atom)`` to each axis offset + the per-axis
    unit axes."""
    offset = tuple(replace(o, unit_var=s.unit, unit_count=s.units) for o, s in zip(t.offset, mn, strict=True))
    axes = (*t.axes, *(Axis(name=s.unit, extent=s.units) for s in mn))
    return replace(t, offset=offset, axes=axes)


def grid_tile(
    t: Tiling,
    *,
    mn: tuple[Side | None, Side | None],
    lead_axes: tuple[Axis, ...] = (),
    block_threads: int | None,
    lanes: int = 1,
    state_decls: Callable[[list[tuple[int, int]]], list[Stmt]],
    reduce_region: Callable[..., tuple[list[Stmt], list[Stmt]]],
    store: Callable[..., list[Stmt]],
) -> Tile:
    """The GRID level + finalize — the ONE seal every kernel binds through: bind the block axes (the
    shrunk grid), set the per-axis grid term ``block·tile``, append any leading (untiled) grid axes
    verbatim and — when the atom is warp-cooperative (``lanes > 1``) — the atom ``_lane`` axis, then
    splice the codegen callables' state + reduce-region + per-cell stores into the ``Tile``. The
    three callables (atom-specific for a contraction, from :func:`reduce_codegen` + the ``store``
    sink; the reduce tier's fill / partitioned fold / projection close) are the only variation; the
    splice is shared. They take the per-cell ``offset`` (the ``(m, n)`` :class:`AxisOffset` tuple) +
    the ``mn`` :class:`Side` pair.

    ``mn[0] is None`` is a 1-D output grid (only ``n`` tiled) — no ``m`` block axis is bound.
    ``mn == (None, None)`` is the fully-untiled output (the reduce tier / degenerate fold): one cell
    per thread, no block axis at all — the whole grid rides ``lead_axes``, and a tiled REDUCE axis
    contributes its lane through ``t.axes``. ``lanes == 1`` (scalar) emits no ``_lane`` axis."""
    offset = tuple(replace(o, block_var=s.block) if s is not None else o for o, s in zip(t.offset, mn, strict=True))
    block_axes = tuple(shrink_axis(Axis(name=s.block, extent=s.axis.extent, source_axis=s.axis), s.tile) for s in mn if s is not None)
    lane_axes = (Axis(name="_lane", extent=lanes),) if lanes > 1 else ()
    axes = (*lead_axes, *block_axes, *t.axes, *lane_axes)

    cells = [(i, j) for i in range(offset[0].reg) for j in range(offset[1].reg)]
    state = state_decls(cells)
    top_decls, kstmts = reduce_region(cells, offset, mn)
    stores = [s for (i, j) in cells for s in store(i, j, offset, mn)]
    return Tile(axes=axes, body=Body((*state, *top_decls, *kstmts, *stores)), block_threads=block_threads)


# ---- the recursive node walk: Ctx down, Frag up ------------------------------------------------ #
# The hierarchical emitter (the tile-IR-rebuild mandate: ONE recursion over the node tree, no
# divergent codegen path). :func:`_emit` walks a ``Map`` / ``Reduction`` / ``Contraction`` tree —
# through ``source`` AND ``partial`` — threading a :class:`Ctx` down (the ambient cell environment)
# and returning a :class:`Frag` up (the per-cell loop-IR body + the produced :class:`Handle` wire +
# the reduce ``carrier`` when the node folds one). The ONE root binder (:func:`_bind`) consumes the
# recursion: the output-tiled ``Contraction`` arm splices the atom's codegen through ``grid_tile``,
# and the reduce partitioner (:func:`_tile_reduce_axis`) builds its per-cell reduce loop via
# :func:`_emit`, so a nested ``Contraction`` (flash's Q@K / P@V) is reached AS A NODE — scalar-nested
# at block=1, while a WARP-TILED tree realizes at fragment residence through ``_twist`` (the
# ``warp_source`` read in :func:`_bind`), the per-node warp tiles stamped by the scheduler.


@dataclass(frozen=True)
class Handle:
    """A produced tensor a parent wires up — "a value that needs wiring." ``name`` is the SSA name
    holding it; ``residence`` is HOW a consumer reads it (``reg`` = a scalar register value today;
    the tensor-core rebuild adds ``reg_frag`` — an mma fragment — plus a fragment descriptor
    ``(mma_role, shape, dtype)`` and the accumulator→operand recast at a node boundary)."""

    name: str
    residence: str = "reg"


@dataclass(frozen=True)
class Frag:
    """What a node contributes UP the recursion: the per-cell loop-IR ``body`` it emits (the reduce /
    contraction loop nest / the projection sweep), the produced :class:`Handle` ``out`` (the wire a
    parent connects to), and the reduce ``carrier`` — set iff this node folds a reduce whose
    cross-partition combine a root binder must emit (``None`` for a pure pointwise map / a scalar
    per-cell contraction)."""

    body: list[Stmt]
    out: Handle
    carrier: object | None = None


@dataclass(frozen=True)
class Ctx:
    """The ambient cell environment threaded DOWN the recursion — established once for the whole
    kernel and passed unchanged so every node reads/writes at the same output cell. ``grid`` is the
    kernel's grid axes; ``inputs`` the operand buffer table (dtype/shape); ``stage`` the operand smem
    pipeline; ``output`` the root output buffer name. (The tensor-core rebuild adds the warp
    ``bind``/``cell`` register tile — owned per-node by a ``Contraction``'s ``tile`` — and the
    inbound ``wires`` handles, e.g. flash's score fragment feeding P@V's A operand.)"""

    grid: tuple
    inputs: dict | None = None
    stage: Stage | None = None
    output: str = ""


def _emit(op, ctx: Ctx) -> Frag:
    """Recurse over a structural node, returning its :class:`Frag` (per-cell body + wire + carrier).
    The single node-kind dispatch every kernel's compute flows through — walking ``source`` AND
    ``partial`` so flash's Q@K / P@V contractions are reached as nodes. Scalar-nested: a node's body
    is its lowered loop-IR (byte-identical to ``ops.lower``); a WARP-TILED tree does not reach this
    walk — ``_bind`` realizes it at fragment residence through ``_twist`` instead."""
    if isinstance(op, Map):
        src = _emit(op.source, ctx) if op.source is not None else None
        prefix = list(src.body) if src is not None else []
        return Frag(body=[*prefix, *_emit_body(op.body, ctx)], out=_map_wire(op), carrier=src.carrier if src is not None else None)
    if isinstance(op, Reduction):
        prefix = list(_emit(op.source, ctx).body) if op.source is not None else []
        loop = Loop(axis=op.axis, body=Body((*prefix, *_emit_body(op.partial, ctx))), unroll=op.unroll, role=op.role, carrier=op.carrier)
        return Frag(body=[loop], out=Handle(op.out), carrier=op.carrier)
    if isinstance(op, Contraction):
        # Scalar / block=1: the synthesized ``CONTRACTION`` loop nest + fused epilogue — byte-identical
        # to ``op.lower()``. A warp-tiled nested contraction never reaches here — the warp-tiled tree
        # realizes wholesale at fragment residence (``_twist``, keyed in ``_bind``).
        return Frag(body=list(op.lower()), out=Handle(op.acc))
    raise TypeError(f"_emit: expected a Map / Reduction / Contraction node, got {type(op).__name__}")


def _map_wire(op: Map) -> Handle:
    """The :class:`Handle` a parent wires to for a ``Map`` node — mirrors ``Map.out``'s cases but
    stays robust where ``Map.out`` would raise. An empty body surfaces the ``source``'s wire; a
    ``Write``-terminated body is a ROOT sink (stored to gmem, never wired) so surfaces the written
    value at ``gmem`` residence; a body ending in an annotated reduce / contraction ``Loop`` surfaces
    its **carrier** state (``carrier.out`` — the acc / carried value, NOT the loop's empty ``defines``);
    otherwise the last defining stmt (a pointwise lift / projection), or ``""`` for a sink whose store
    rides inside a projection sweep ``Loop`` (a don't-care — nothing consumes it)."""
    if len(op.body) == 0:
        return _emit_wire(op.source) if op.source is not None else Handle("")
    last = op.body[-1]
    if isinstance(last, Write):
        return Handle(last.values[-1], residence="gmem")
    carrier = getattr(last, "carrier", None)
    if carrier is not None:
        return Handle(carrier.out)
    defs = last.defines()
    return Handle(defs[-1] if defs else "")


def _emit_wire(op) -> Handle:
    """The produced-value :class:`Handle` of any node — a ``Reduction`` / ``Contraction`` names its
    carrier / accumulator; a ``Map`` scans for its last defining stmt (:func:`_map_wire`)."""
    if isinstance(op, Map):
        return _map_wire(op)
    return Handle(op.out)  # Reduction.out (carrier state) / Contraction.out (acc) — always safe


def _emit_body(body, ctx: Ctx) -> list[Stmt]:
    """Walk a ``Body`` of loop-IR stmts, recursing into any nested structural node (a
    :class:`Contraction` / :class:`Reduction` / :class:`Map`) via :func:`_emit` and passing plain
    stmts through — the codegen-layer node-walk (the dispatch seam ``ir._flatten_nodes`` cannot host,
    since a warp-tiled nested contraction lowers to mma, not a scalar loop)."""
    out: list[Stmt] = []
    for s in body:
        if isinstance(s, (Contraction, Reduction, Map)):
            out.extend(_emit(s, ctx).body)
        else:
            out.append(s)
    return out


def factorize(tile, root, store=None) -> Tile:
    """The entry to the recursive emitter — build the ambient :class:`Ctx` from the ``TileOp`` and its
    root graph node, then dispatch its ``op`` into a bound ``Tile`` via :func:`_factorize`. ``out_val``
    (the kernel's finalized output SSA name — the root node's produced :class:`Handle`) is resolved
    once here and threaded down for the store glue."""
    op = tile.op
    ctx = Ctx(grid=tuple(tile.place.grid), inputs=tile.inputs, stage=tile.stage, output=(root.output.name if root is not None else ""))
    out_val = _emit_wire(op).name if op is not None else ""
    return _factorize(op, ctx, tail=(), out_val=out_val, store=store)


def _factorize(op, ctx: Ctx, tail: tuple, out_val: str, store=None) -> Tile:
    """The recursive root walk — peel the projecting ``Map``\\ s, then bind the leaf to the grid via
    the ONE binder. A :class:`Map` with a ``source`` **recurses**: its ``body`` (the projection /
    epilogue) is walked (:func:`_emit_body`, reaching any nested node) and prepended to ``tail``;
    everything else is a leaf, bound by :func:`_bind` — the single pipeline, whose form is read off
    the node's SCHEDULE (which axes are tiled), never a kernel kind. There is **no** flash /
    attention special case: flash is the two-``Contraction`` ``TWISTED`` reduce tree, so its Q@K /
    P@V contractions and its streaming reduce factorize through this one walk (scalar block=1
    today; a nested warp-tiled contraction routes through the ``_emit`` ``Contraction`` seam). A
    bespoke emitter would be a divergent codegen path the mandate forbids."""
    if isinstance(op, Map) and op.source is not None:
        return _factorize(op.source, ctx, tail=(*_emit_body(op.body, ctx), *tail), out_val=out_val, store=store)
    return _bind(op, ctx, tail, out_val, store)


def has_write(stmts: list[Stmt]) -> bool:
    """Any ``Write`` reachable in ``stmts`` (deep — a projection's output sweep nests its
    ``Write`` inside a per-cell ``Loop``)."""
    for s in stmts:
        if isinstance(s, Write):
            return True
        if any(has_write(list(b)) for b in s.nested()):
            return True
    return False


def with_store(stmts: list[Stmt], output: str, grid, value: str) -> list[Stmt]:
    """Append the output-store glue when the body has none — a bare reduction / contraction produces
    its finalized value as the SSA name ``value`` (the carrier state / accumulator, or a projection's
    last def) that must be written to the output buffer at the grid cell. A body that already carries
    a ``Write`` needs no glue (``value`` is left unread). The caller resolves ``value`` off the node
    (``Contraction.out`` / the recursion's produced ``Handle``) so this helper stays node-agnostic."""
    if has_write(stmts):
        return stmts
    index = tuple(Var(ax.name) for ax in grid)
    return [*stmts, Write(output=output, index=index, value=value)]


def _bind(op, ctx: Ctx, tail: tuple, out_val: str, store=None) -> Tile:
    """The ONE root binder — every kernel binds through the same pipeline: read WHICH AXES the
    schedule tiles off the node, build the fold region, and seal through the one :func:`grid_tile`
    finalizer. The cases are points of one ``(output-tiling) × (reduce-folding)`` space, selected by
    the schedule — never separate emitters:

    - a :class:`Contraction` tiles its OUTPUT ``(m, n)`` axes — register / warp cells through
      ``atomize → register_tile → unit_tile``, the reduce (K) serial per cell from the atom's
      :func:`reduce_codegen`, ``store`` the per-cell sink (default :func:`store_sink`; the flash
      inner QK/PV pass a sink that bridges the accumulator into the softmax twist). Its projection
      ``epilogue`` (+ any ``tail``) rides the node; the bare grid-``Write`` glue is synthesized here
      (it needs ``ctx.output``, so it can't ride the node).
    - a :class:`Reduction` whose :class:`ReducePlan` cooperates tiles its REDUCE axis instead
      (:func:`_tile_reduce_axis` — ``coop`` lanes at the unit level, ``reg`` ILP chains at the
      register level, the carrier merge closing the fold). The output stays one cell per thread:
      the 1×1 ``atomize`` with the whole grid riding ``lead_axes`` untiled.
    - anything else (a pure pointwise ``Map``, a trivial plan) tiles NOTHING — the degenerate
      one-thread-per-cell fold: the per-cell body (:func:`_emit`; a serial reduce ``Loop`` sits
      inside it) + ``tail`` + the ``out_val`` store glue is the whole fold region."""
    grid = tuple(ctx.grid)
    if isinstance(op, Contraction):
        epi = [*op.epilogue, *tail]
        if not has_write(epi):
            epi = with_store(epi, ctx.output, grid, op.out)
        c = replace(op, epilogue=Body(tuple(epi)))
        state_decls, reduce_region = reduce_codegen(c, ctx.stage, ctx.inputs)
        sink = store if store is not None else store_sink(c)
        t = unit_tile(register_tile(atomize(c.atom.shape[:2]), c.mn), c.mn)
        mn, lead, bt, lanes = c.mn, c.lead_axes, c.block_threads, c.atom.lanes
    else:
        # The reduce partition rides the :class:`Reduction` node; ``None`` for a pure pointwise /
        # scalar per-cell ``Map`` (no partition). Every partitioned reduce — monoid, flash, coop-K /
        # split contraction — is a ``Reduction`` node after ``ops.nodify_reduce`` (a projecting
        # ``Map`` was already peeled off by :func:`_factorize`).
        plan = op.reduce if isinstance(op, Reduction) else None
        t, mn, lead, lanes = atomize((1, 1)), (None, None), grid, 1
        wsrc = warp_source(op)
        csrc = chain_source(op) if wsrc is None else None
        if wsrc is not None:
            # A warp-tiled TWISTED tree (the schedule stamped mma TilePlans on its contractions):
            # the per-step values live in mma C-fragments, so the whole reduce realizes at fragment
            # residence (``_twist``) and the kernel is warp-collective — the same ``lanes`` seam the
            # output-tiled contraction arm uses.
            state, fold, close = realize_warp_twist(op, ctx, tail)
            lanes = wsrc.tile.atom.lanes
            bt = lanes
        elif csrc is not None:
            # The chain schedule — the expect column axis rides a per-thread register vector (the
            # FA-2 shared-score form); one thread per (grid) cell, the column index a literal.
            state, fold, close = _realize_chain(op, ctx, tail, csrc)
            bt = None
        elif plan is None or (plan.coop <= 1 and plan.reg <= 1):
            state, fold, close, bt = [], with_store([*_emit(op, ctx).body, *tail], ctx.output, grid, out_val), [], None
        else:
            state, fold, close, lane = _tile_reduce_axis(op, plan, ctx, tail, out_val)
            t = replace(t, axes=(lane,)) if lane is not None else t
            bt = plan.coop if lane is not None else None

        def state_decls(_cells):
            return state

        def reduce_region(_cells, _offset, _mn):
            return [], fold

        def sink(_i, _j, _offset, _mn):
            return close

    return grid_tile(
        t, mn=mn, lead_axes=lead, block_threads=bt, lanes=lanes, state_decls=state_decls, reduce_region=reduce_region, store=sink
    )


# ---- the tiled REDUCE axis (cooperative / ILP) -------------------------------------------------- #
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
    index_expr = BinaryExpr("%", shifted, axis.extent_expr()) if masked else shifted
    sigma = Sigma({axis.name: index_expr})
    out = copy_cell(body, sigma, f"__r{r}", protected)
    return _mask_streamed(out, axis.name, offset, axis.extent_expr()) if masked else out


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


def emit_combine(carrier, t: str, n_threads: int, *, warp_size: int = 32, segmented: bool = False) -> list[Stmt]:
    """Build the cross-thread combine of a cooperative reduce ``carrier`` (a :class:`Carrier`)
    over ``n_threads`` cooperating threads, reassigning the carried state in place.

    The mechanism per level is derived by :meth:`ReduceStage.combine`:

    - a ``SHFL`` fold → one ``WarpShuffle`` register butterfly. The XOR butterfly never
      crosses an aligned ``width``-lane group, so a lone ``SHFL`` is also the SEGMENTED
      per-row combine for strided-cooperative rows (caller passes ``segmented=True``).
    - a ``SMEM`` fold **after** a ``SHFL`` → the hierarchical cross-warp slab: lane-0 of each
      warp stages its broadcast state to a ``smem[n_warps]`` slab per component; one ``Sync``
      + ``TreeHalve(tid_var="warp")`` collapses across warps and broadcasts.
    - a standalone ``SMEM`` → the block slab: every thread stages its partial, one ``Sync``,
      a single ``TreeHalve`` reduces + broadcasts in place.

    The carrier's combine surface (``state.names`` / ``twist.state_b`` /
    ``twist.combine_states``) drives the nodes; the combine renders at the accumulator dtype
    (fp32 for a reduction, with the carrier's own dtype honored when set)."""
    state = carrier.state.names
    state_b = carrier.state_b
    prog = carrier.combine_states
    dtype = next((a.dtype for a in prog if a.dtype is not None), None) or F32
    folds = ReduceStage(Level.BLOCK, n_threads).combine(warp_size=warp_size, segmented=segmented)

    from deplodock.compiler.backend.cuda.dtype import cuda_name as _cuda_name  # noqa: PLC0415

    smem_c = _cuda_name(dtype)
    bufs = tuple(f"{st}_smem" for st in state)
    out: list[Stmt] = []
    for i, fold in enumerate(folds):
        if fold is Fold.SHFL:
            # The lane-level butterfly: warp-wide when followed by a cross-warp SMEM stage
            # (hierarchical), else the full ``n_threads`` (one warp / segment).
            width = warp_size if (len(folds) == 2 and folds[1] is Fold.SMEM) else n_threads
            out.append(WarpShuffle(state=state, state_b=state_b, combine_states=prog, length=width, dtype=dtype))
        elif fold is Fold.SMEM:
            hierarchical = i > 0 and folds[i - 1] is Fold.SHFL
            width = n_threads // warp_size if hierarchical else n_threads
            tid_var = "warp" if hierarchical else t
            out += [Smem(name=b, extents=(width,), dtype=smem_c) for b in bufs]
            if hierarchical:
                # Lane-0 of each warp stages that warp's broadcast state, indexed by ``warp``.
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var("lane"), Literal(0, "int")),
                        body=tuple(Write(output=b, index=(Var("warp"),), value=st) for b, st in zip(bufs, state, strict=True)),
                    )
                )
            else:
                out += [Write(output=b, index=(Var(tid_var),), value=st) for b, st in zip(bufs, state, strict=True)]
            out.append(Sync())
            out.append(TreeHalve(bufs=bufs, state=state, state_b=state_b, combine_states=prog, length=width, tid_var=tid_var, dtype=dtype))
        else:  # Fold.ATOMIC / Fold.REG — cross-CTA / register tiers, not emitted by the intra-CTA walk.
            raise NotImplementedError(f"intra-CTA combine cannot emit {fold} (cta/reg tiers are future work)")
    return out


def combine_tail(carrier, *, reg: int, coop: int, lane) -> list[Stmt]:
    """The carrier-driven **partial merge** that follows a partitioned reduce loop — the one place the
    two partial-fold geometries are assembled: the REG-tree fold of the ``reg`` ILP register copies
    into copy 0 (``as_state_merge``), then — when threads cooperate (``lane`` is a lane :class:`Axis`,
    not ``None``) — the cross-thread :func:`emit_combine`. Both reassign the carried state **in place**
    (the survivor SSA names hold the full reduction), so the post-reduce projection reads them directly.

    Carrier-generic: a monoid reduce and a contraction's degenerate additive carrier fold identically,
    so a cooperative reduce and a (future) cooperative-K contraction share this tail. ``as_state_merge``
    keys its finalize temps on the copy name, so each fold's internals are already unique."""
    merge: list[Stmt] = [carrier.as_state_merge(tuple(f"{n}__r{r}" for n in carrier.state.names)) for r in range(1, reg)]
    if lane is not None:
        merge += emit_combine(carrier, t=lane.name, n_threads=coop)
    return merge


def chain_source(op) -> Contraction | None:
    """The expect :class:`Contraction` of a TWISTED tree carrying a SCALAR register tile over its
    output column axis (the chain schedule — the column axis rides a per-thread register vector),
    or ``None``. The structural schedule read the one binder keys the chain realization on."""
    red = op.source if isinstance(op, Map) else op
    if not isinstance(red, Reduction):
        return None
    pv = next((s for s in list(red.partial)[1:] if isinstance(s, Contraction)), None)
    if pv is not None and not pv.tile.is_warp and pv.tile.regs != (1, 1):
        return pv
    return None


def _flat_stmts(stmts):
    for s in stmts:
        yield s
        for b in s.nested():
            yield from _flat_stmts(list(b))


def _stmt_axis_hit(s: Stmt, axis: str) -> bool:
    idx = getattr(s, "index", None)
    if idx and any(axis in e.free_vars() for e in idx):
        return True
    return isinstance(s, Select) and any(axis in br.select.free_vars() for br in s.branches)


def _stmt_reads(s: Stmt) -> set[str]:
    if isinstance(s, Accum):
        return {s.name, s.value}
    if isinstance(s, Select):
        return {br.value for br in s.branches}
    deps = getattr(s, "deps", None)
    return set(deps()) if callable(deps) else set(getattr(s, "args", ()) or ())


def _taint(stmts: list[Stmt], axis: str) -> frozenset[str]:
    """The SSA names transitively dependent on the free ``axis`` — the register-vector slice of the
    per-cell body (everything else is shared across the vector's columns)."""
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for s in _flat_stmts(stmts):
            d = set(s.defines())
            if d and not (d <= tainted) and (_stmt_axis_hit(s, axis) or (_stmt_reads(s) & tainted)):
                tainted |= d
                changed = True
    return frozenset(tainted)


def _vector_carrier(carrier, tainted: frozenset[str], count: int):
    """The carrier with each column-dependent state component fanned out per register column —
    one expectation channel per column, so the loop render seeds every replica (the chain IS the
    same LSE carrier with ``count`` expect channels)."""
    if carrier is None or not (set(carrier.state.names) & tainted):
        return carrier
    names: list[str] = []
    channels: list = []
    for nm, ch in zip(carrier.state.names, carrier.twist.channels, strict=True):
        if nm in tainted:
            names += [f"{nm}_{j}" for j in range(count)]
            channels += [ch] * count
        else:
            names.append(nm)
            channels.append(ch)
    return replace(carrier, state=replace(carrier.state, names=tuple(names)), twist=replace(carrier.twist, channels=tuple(channels)))


def _vectorize_axis(stmts: list[Stmt], axis: str, count: int, tainted: frozenset[str], protected: frozenset[str]) -> list[Stmt]:
    """Replicate every column-dependent stmt per register column (σ ``axis → j``, names suffixed
    ``_{j}``), keeping shared stmts single — the FA-2 shared-score restructuring: the score /
    softmax stats compute once per streamed key, the per-column slice fans out. Recurses into loop
    bodies (a loop stays single; a column-touched carrier fans out per :func:`_vector_carrier`)."""
    out: list[Stmt] = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            s = s.with_bodies(tuple(Body(tuple(_vectorize_axis(list(b), axis, count, tainted, protected))) for b in bodies))
            if getattr(s, "carrier", None) is not None:
                s = replace(s, carrier=_vector_carrier(s.carrier, tainted, count))
            out.append(s)
            continue
        if (set(s.defines()) & tainted) or _stmt_axis_hit(s, axis) or (_stmt_reads(s) & tainted):
            for j in range(count):
                out += copy_cell([s], Sigma({axis: Literal(j, "int")}), f"_{j}", protected)
        else:
            out.append(s)
    return out


def _realize_chain(op, ctx: Ctx, tail: tuple, pv: Contraction) -> tuple[list[Stmt], list[Stmt], list[Stmt]]:
    """Realize a chain-scheduled TWISTED tree — the ``(state, fold, close)`` triple: the per-cell
    body with the expect column axis register-vectorized (the score shared), and the projection +
    store replicated per column (the column index a literal — the axis left the grid)."""
    axis = pv.n_axis.name
    count = pv.tile.regs[0]  # scalar reg order (reg_n, reg_m) — the column (n) register vector
    (rloop,) = _emit(op, ctx).body
    all_stmts = [*rloop.body, *tail]
    tainted = _taint(all_stmts, axis)
    protected = frozenset({nm for s in _flat_stmts(all_stmts) for nm in (*s.defines(), *_stmt_reads(s))} - tainted)
    body = _vectorize_axis(list(rloop.body), axis, count, tainted, protected)
    fold = [replace(rloop, body=Body(tuple(body)), carrier=_vector_carrier(rloop.carrier, tainted, count))]
    close = _vectorize_axis(list(tail), axis, count, tainted, protected)
    grid_vars = tuple(Var(a.name) for a in ctx.grid)
    out_val = tail[-1].defines()[-1] if tail else pv.acc
    for j in range(count):
        val = f"{out_val}_{j}" if out_val in tainted else out_val
        close.append(Write(output=ctx.output, index=(*grid_vars, Literal(j, "int")), value=val))
    return [], fold, close


def _tile_reduce_axis(op: Reduction, plan, ctx: Ctx, tail: tuple, out_val: str) -> tuple[list[Stmt], list[Stmt], list[Stmt], Axis | None]:
    """Tile the REDUCE axis per the node's cooperating :class:`ReducePlan` — the reduce counterpart
    of the output ``unit_tile`` / ``register_tile`` levels: ``coop`` lanes across threads (the
    ``_co`` lane axis, the axis's UNIT level) and ``reg`` ILP chains across per-thread accumulators
    (its REGISTER level — cyclic, copy ``r`` offset by ``r·coop``, the loop striding ``coop·reg``).
    It drives the recursion (:func:`_emit`) for the per-cell reduce loop and returns the pieces the
    one pipeline (:func:`_bind` → :func:`grid_tile`) seals: ``(state, fold, close, lane)`` — the
    shared-row fill decls, the strided fold loop + the carrier merge (the REG tree + the
    cross-thread combine), the distributed projection close, and the lane :class:`Axis` (``None``
    for standalone ILP — one thread per cell, lane fixed at 0)."""
    grid = ctx.grid
    coop, reg = plan.coop, plan.reg

    # Build the per-cell reduce loop via the recursion (:func:`_emit`) off the :class:`Reduction`
    # **node** — the walk reaches any nested contraction (flash Q@K / P@V) as a node. The synthesized
    # ``loop`` carries the :class:`Carrier` (a contraction's K loop and a monoid's reduce loop both
    # carry it here — the ⊗ lift already sits in the loop body, so the carrier-generic ``state`` /
    # ``as_state_merge`` / ``combine_states`` machinery folds either). A ``Reduction`` has no prologue
    # ahead of its loop; the enclosing ``Map``'s projection is ``tail`` (already walked).
    (rloop,) = _emit(op, ctx).body
    carrier = rloop.carrier
    axis = rloop.axis
    stride = coop * reg
    masked = reg > 1 and not (axis.extent.is_static and axis.extent.as_static() % stride == 0)

    # The cooperative lane axis (Tile-decoded, innermost) — present only when threads
    # cooperate; standalone ILP (coop == 1) runs one thread per cell, lane fixed at 0.
    lane = Axis(name=f"{axis.name}_co", extent=coop) if coop > 1 else None
    start = Var(lane.name) if lane is not None else Literal(0, "int")

    # Shared-row staging (the fused norm→linear prologue): an input row folded by the cooperative
    # reduce AND re-read per output column of a contraction tail rides a first-class ``sync``
    # :class:`Stage` whose ``smem`` names the row — DETECTED scheduler-side (``_schedule._row_stage``)
    # and only APPLIED here: fill the row into smem once (cooperatively) and rewrite both readers to
    # the slab. Only the cooperative tier (coop > 1) is ever stamped; a contraction operand ``Stage``
    # (the coop-K matmul's pinned pipeline) never sets ``smem``, so it passes through untouched.
    tail_src = list(tail)
    fill_stmts: list[Stmt] = []
    if lane is not None and ctx.stage is not None and ctx.stage.smem:
        from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

        (staged,) = ctx.stage.smem
        grid_vars = tuple(Var(a.name) for a in grid)
        smem_name = f"{staged}_smem"
        fill_stmts = sync_row_fill(
            slab=smem_name,
            src=staged,
            extent=axis.extent.as_static(),
            grid_vars=grid_vars,
            linear_tid=start,
            n_threads=coop,
            dtype=cuda_name(ctx.inputs[staged].dtype),
        )
        n_grid = len(grid)
        rloop = replace(rloop, body=Body(tuple(_restage_loads(list(rloop.body), staged, smem_name, n_grid, grid_vars))))
        tail_src = _restage_loads(tail_src, staged, smem_name, n_grid, grid_vars)

    # The reduce loop: ``reg`` interleaved accumulator chains (ILP), striding the axis by
    # ``coop·reg`` from the lane's start. The dissolved fold ``Accum``\\ s seed each copy's
    # accumulator (``StridedLoop.render``).
    # The shared iteration coordinates (grid + reduce + lane axis vars) and the symbolic
    # extent's runtime arg(s) (e.g. ``seq_len``) are common to every register copy — exclude
    # them from the per-copy SSA rename.
    protected = frozenset(
        {axis.name, *(ax.name for ax in grid), *axis.extent_expr().free_vars()} | ({lane.name} if lane is not None else set())
    )
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, coop, axis, masked, protected))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)

    # The carrier-driven partial merge: the REG-tree fold of the ``reg`` ILP copies into the survivor
    # (copy 0's names) + (when threads cooperate) the cross-thread combine, reassigning the carried
    # state in place. The one shared tail a cooperative reduce and a future cooperative-K contraction
    # both emit (``combine_tail``).
    merge = combine_tail(carrier, reg=reg, coop=coop, lane=lane)

    # Post-reduce projection. A full-row output (softmax / RMSNorm) distributes its sweep
    # across the coop lanes; a scalar output is written once, guarded to lane 0. With no
    # cooperation (coop == 1) the single thread runs the projection as-is.
    tail = tail_src
    if lane is None:
        body_tail = with_store(tail, ctx.output, grid, out_val)
    elif any(isinstance(s, Loop) for s in tail):
        body_tail = [
            StridedLoop(axis=s.axis, start=Var(lane.name), step=Literal(coop, "int"), body=s.body, unroll=s.unroll)
            if isinstance(s, Loop)
            else s
            for s in tail
        ]
    else:
        stored = with_store(tail, ctx.output, grid, out_val)
        body_tail = [Cond(cond=BinaryExpr("==", Var(lane.name), Literal(0, "int")), body=tuple(stored))]

    return fill_stmts, [strided, *merge], body_tail, lane
