"""The factorizer — the recursive ``TileOp``-root emitter, the contraction-tiling glue, and the
cooperative / ILP reduce binder. The per-atom codegen **strategies** it drives live in ``_atom.py``.

:func:`factorize` is the entry ``010_materialize`` calls once per kernel: it builds the ambient
:class:`Ctx` and dispatches ``tile.op`` through the recursion :func:`_factorize`, which walks the
``Map`` / ``Reduction`` / ``Contraction`` node tree. A :class:`~...ir.Map` with a ``source``
**recurses** (its projection walked into the ``tail``); a leaf binds to the grid via one of the two
ROOT binders — :func:`_bind_contraction` → :func:`_factorize_contraction` (a tiled
:class:`~...ir.Contraction`, register / warp OUTPUT tile) or :func:`_bind_reduce` (a ``PLANAR`` /
``TWISTED`` reduce — cooperative / ILP, or the degenerate one-thread-per-cell fold). The per-cell body
under either binder is built by the shared recursion :func:`_emit` (which walks ``source`` AND
``partial``, reaching flash's Q@K / P@V as nodes).

:func:`_factorize_contraction` reads the tiling **geometry straight off the** ``Contraction`` **node**
(``tile_m`` / ``mask_m`` / ``m_b`` / ``block_threads`` / …, derived there from the ``tile`` schedule +
the output axes), expands both atoms through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``, in this module), and splices in two codegen halves from
the per-atom strategies in **``_atom.py``**: :func:`~...kernel._atom.reduce_codegen` — the reusable,
**sink-agnostic** ``(state_decls, reduce_region)`` (accumulator/operand decls + the contraction
K-loop) — and a per-cell **sink** ``store(i, j, offset, mn)`` (default
:func:`~...kernel._atom.store_sink`, the matmul sink; ``_factorize_contraction(c, store=…)`` swaps it —
the flash inner QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist,
reusing the same ``reduce_codegen``).

The cooperative / ILP reduce tier (:func:`_bind_reduce` + the shared-row staging helpers) folds
the reduce axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (``_combine.emit_combine``), and the projection — carrier-
generic (a contraction is the degenerate carrier of its additive fold).

The smem operand-staging pipeline lives in ``_stage.py`` (the :class:`~...kernel._stage.Transport`
strategy + the one :func:`~...kernel._stage.staged_kloop`); the ONE atom-agnostic driver
(``_atom._staged``) builds the transport, the atom strategy supplying only the slab drain leaf.
It is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``sync``/``cp``/``tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct. The fused
norm→linear **shared-row** prologue is Stage-driven too: ``020_schedule`` detects the reused input row
and stamps a ``sync`` :class:`~...schedule.Stage` whose ``smem`` names it; :func:`_bind_reduce` only
applies it (the 1-D ``sync_row_fill`` + the load rewrite). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.schedule import Stage
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Contraction, Map, Reduction, Side
from deplodock.compiler.pipeline.passes.lowering.kernel._atom import reduce_codegen, store_sink
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import combine_tail
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import copy_cell
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import shrink_axis as _shrink_axis
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import sync_row_fill
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store


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
    mn: tuple[Side | None, Side],
    lead_axes: tuple[Axis, ...] = (),
    block_threads: int | None,
    lanes: int = 1,
    state_decls: Callable[[list[tuple[int, int]]], list[Stmt]],
    reduce_region: Callable[..., tuple[list[Stmt], list[Stmt]]],
    store: Callable[..., list[Stmt]],
) -> Tile:
    """The GRID level + finalize: bind the block axes (the shrunk grid), set the per-axis grid term
    ``block·tile``, append any leading (batch) grid axes verbatim and — when the atom is warp-cooperative
    (``lanes > 1``) — the atom ``_lane`` axis, then splice the codegen callables' state + reduce-region +
    per-cell stores into the ``Tile``. The three callables (atom-specific, from :func:`reduce_codegen` +
    the ``store`` sink) are the only per-atom variation; the splice is shared. They take the per-cell
    ``offset`` (the ``(m, n)`` :class:`AxisOffset` tuple) + the ``mn`` :class:`Side` pair.

    ``mn[0] is None`` is a 1-D output grid (only ``n`` tiled) — no ``m`` block axis is bound.
    ``lead_axes`` are extra outer grid axes carried through untiled; ``lanes == 1`` (scalar) emits no
    ``_lane`` axis."""
    offset = tuple(replace(o, block_var=s.block) if s is not None else o for o, s in zip(t.offset, mn, strict=True))
    block_axes = tuple(_shrink_axis(Axis(name=s.block, extent=s.axis.extent, source_axis=s.axis), s.tile) for s in mn if s is not None)
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
# the reduce ``carrier`` when the node folds one). The two ROOT binders consume the recursion:
# :func:`_factorize_contraction` (the ``grid_tile`` output-tiling pipeline) for an output-tiled
# ``Contraction`` root, and :func:`_bind_reduce` (the reduce partitioner) for a cooperative /
# scalar reduce — the latter builds its per-cell reduce loop via :func:`_emit`, so a nested
# ``Contraction`` (flash's Q@K / P@V) is reached AS A NODE. Scalar-nested today (block=1); the
# per-node warp tile that routes a nested contraction through the mma path is the tensor-core seam
# marked in :func:`_emit` (a ``Contraction`` case).


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
    is its lowered loop-IR (byte-identical to ``ops.lower``); the tensor-core tier adds the per-node
    warp-tile branch in the ``Contraction`` case."""
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
        # to ``op.lower()``. Tensor-core seam: an output-warp-tiled ``Contraction`` (an mma ``TilePlan``)
        # emits through the register-tile pipeline + the fragment recast here instead.
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
    """The recursive root-binder dispatch — bind ``op`` to the grid, projecting ``tail`` (an enclosing
    ``Map``'s post-source sweep) after it. Three cases, mirroring the node kinds:

    - a :class:`Map` with a ``source`` **recurses**: its ``body`` (the projection / epilogue) is walked
      (:func:`_emit_body`, reaching any nested node) and prepended to ``tail``, then ``source`` is bound.
    - a :class:`Contraction` (warp / register OUTPUT tile) → :func:`_factorize_contraction`, the
      atom-generic four-level ``grid_tile`` pipeline. Its projection ``epilogue`` (+ any ``tail``) rides
      the node; the bare grid-``Write`` is synthesized here (it needs ``ctx.output``, so it can't ride
      the node).
    - everything else (a :class:`Reduction`, or a pure pointwise ``Map``) → :func:`_bind_reduce`, the
      reduce partitioner: one thread per output cell, partitioning the reduce axis when the
      :class:`ReducePlan` cooperates (BLOCK ``coop`` / REG ``reg``).

    The two binders are the two OUTPUT-binding strategies (register-tile the output vs. one-cell-
    per-thread + partition the reduce) — a kernel makes that choice once at its root; the body inside
    is built by the recursion (:func:`_emit`). There is **no** flash / attention special case: flash is
    the two-``Contraction`` ``TWISTED`` reduce tree, so its Q@K / P@V contractions and its streaming
    reduce factorize through this one dispatch (scalar block=1 today; a nested warp-tiled contraction
    routes through the ``_emit`` ``Contraction`` seam). A bespoke emitter would be a divergent codegen
    path the mandate forbids."""
    if isinstance(op, Map) and op.source is not None:
        return _factorize(op.source, ctx, tail=(*_emit_body(op.body, ctx), *tail), out_val=out_val, store=store)
    if isinstance(op, Contraction):
        return _bind_contraction(op, ctx, tail, store)
    return _bind_reduce(op, ctx, tail, out_val)


def _bind_contraction(c: Contraction, ctx: Ctx, tail: tuple, store=None) -> Tile:
    """The output-tiling binder case — append the enclosing ``Map``'s ``tail`` (empty for a bare
    matmul) to the contraction's projection ``epilogue``, synthesize the grid-``Write`` glue when the
    epilogue carries none (it needs ``ctx.output``), and expand through :func:`_factorize_contraction`."""
    epi = [*c.epilogue, *tail]
    if not has_write(epi):
        epi = with_store(epi, ctx.output, ctx.grid, c.out)
    return _factorize_contraction(replace(c, epilogue=Body(tuple(epi))), ctx.stage, store, ctx.inputs)


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


def _bind_reduce(op, ctx: Ctx, tail: tuple, out_val: str) -> Tile:
    """The reduce-partitioner binder case — bind ``op`` (a :class:`Reduction`, or a pointwise / scalar
    per-cell ``Map``) one output cell per thread, projecting the enclosing ``Map``'s ``tail`` after the
    reduce. It drives the recursion (:func:`_emit`) for the per-cell body.

    **Degenerate arm (no partition).** A pointwise ``Map``, or a reduce whose :class:`ReducePlan` is
    trivial (``coop == reg == 1``), is one thread per output cell: :func:`_emit` emits the per-cell
    body (a serial reduce ``Loop`` sits inside it), ``tail`` is appended, and the ``out_val`` store glue
    closes it. **Partitioned arm.** Otherwise partition the reduce axis ``coop`` ways across threads
    and/or ``reg`` ways across register accumulators, as the section header describes."""
    grid = ctx.grid
    # The reduce partition rides the :class:`Reduction` node; ``None`` for a pure pointwise / scalar
    # per-cell ``Map`` (no partition). Every partitioned reduce — monoid, flash, coop-K / split
    # contraction — is a ``Reduction`` node after ``ops.nodify_reduce`` (a projecting ``Map`` was
    # already peeled off by :func:`_factorize`, so a partition here always means ``op`` is a ``Reduction``).
    plan = op.reduce if isinstance(op, Reduction) else None
    if plan is None or (plan.coop <= 1 and plan.reg <= 1):
        # One thread per output cell — the degenerate fold, no cross-thread / ILP partition.
        stmts = with_store([*_emit(op, ctx).body, *tail], ctx.output, grid, out_val)
        return Tile(axes=tuple(grid), body=Body(tuple(stmts)))
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
        {axis.name, *(ax.name for ax in grid), *_extent_expr(axis).free_vars()} | ({lane.name} if lane is not None else set())
    )
    copies: list[Stmt] = []
    for r in range(reg):
        copies.extend(_replicate(rloop.body, r, coop, axis, masked, protected))
    strided = StridedLoop(axis=axis, start=start, step=Literal(stride, "int"), body=Body(tuple(copies)), unroll=rloop.unroll)

    # The carrier-driven partial merge: the REG-tree fold of the ``reg`` ILP copies into the survivor
    # (copy 0's names) + (when threads cooperate) the cross-thread combine, reassigning the carried
    # state in place. The one shared tail a cooperative reduce and a future cooperative-K contraction
    # both emit (``_combine.combine_tail``).
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

    body = [*fill_stmts, strided, *merge, *body_tail]
    axes = (*grid, lane) if lane is not None else tuple(grid)
    return Tile(axes=axes, body=Body(tuple(body)), block_threads=coop if lane is not None else None)
