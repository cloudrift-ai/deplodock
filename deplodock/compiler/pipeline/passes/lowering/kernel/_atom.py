r"""The per-atom codegen strategies — the one seam every tiled contraction dispatches through.

``_factor._bind`` (the output-tiled contraction arm) reads the tiling geometry off the :class:`Contraction`
node and asks this module for the two codegen halves: :func:`reduce_codegen` (the sink-agnostic
``(state_decls, reduce_region)`` — the accumulator/operand decls + the shared :func:`_contract_kloop`
K-loop) and :func:`store_sink` (the per-cell matmul sink). Both resolve through :func:`_atom_ops` to
one of the two concrete strategies — :class:`_MmaOps` (tensor-core ``ldmatrix`` + ``mma.sync``, a
``RegStore`` sink) or :class:`_ScalarOps` (plain ``Load``\ s + an ``fma`` cell, the replicated-
``epilogue`` sink). The K-loop itself is ONE driver on the base strategy (:meth:`_AtomOps.reduce`),
deciding nothing: the **scheduler-resolved** ``stage`` (eligibility + sizing ran once in
``020_schedule`` — ``_resolve_warp_stage`` / ``_resolve_scalar_stage``; ``None`` = gmem-direct)
picks its form — gmem-direct through the shared :func:`_contract_kloop` spine, or staged through
the shared :func:`_staged` fill→drain skeleton (over the one ``_stage.staged_kloop``) — and the
atom supplies only descriptor reads: the four gmem leaf constructors (:meth:`gmem_leaves`), the
slab drain leaf (:meth:`staged_drain`), and the slab element dtype. This IS the "atom as
descriptor" seam: one factory, one loop over atoms, no scattered ``isinstance``.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it."""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.atom import AtomKind
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel.ir import (
    EpilogueLoad,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
)
from deplodock.compiler.ir.schedule import Stage
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Contraction, Side
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    CpAsyncTransport,
    CtaTile,
    Operand,
    SyncOperand,
    SyncTransport,
    TmaTransport,
    staged_kloop,
)

#: The contraction semiring — multiply ⊗ then accumulate ⊕ (add). The same multiply-add ``mma.sync``
#: realizes; in the scalar tier it is a plain scalar fma loop.
_MUL = ElementwiseImpl("multiply")
_ADD = ElementwiseImpl("add")


# Shared axis-geometry helpers, used across this module (the atom-generic mma/scalar codegen) AND
# ``_factor.py`` (the tiling layer + the cooperative / ILP reduce tier).
def shrink_axis(axis: Axis, reg: int) -> Axis:
    """The grid (cell) axis for a register-tiled free axis: ``ceil(E / reg)`` cells, each a
    per-thread ``reg``-wide register sub-tile. ``Dim.ceil_div`` keeps a symbolic extent
    symbolic (``(seq_len+reg-1)//reg``) so the launch grid sizes from the runtime extent."""
    if reg <= 1:
        return axis
    return Axis(name=axis.name, extent=axis.extent.ceil_div(reg), source_axis=axis.source_axis or axis)


def copy_cell(body, sigma, suffix: str, protected) -> list:
    """One copy of a tiled reduce ``body``: σ-substitute its indices (``sigma``) and suffix every
    per-copy SSA name (the shared grid / reduce / lane coordinates in ``protected`` pass through
    unrenamed). This is the **one** replication mechanic shared by the register tile (this module,
    one copy per output cell ``(i, j)`` → ``__c{i}_{j}``) and the ILP register fold (``_factor``
    ``_tile_reduce_axis``, one copy per accumulator chain ``r`` → ``__r{r}``); the caller supplies the per-copy
    ``sigma`` (the coordinate offset) and ``suffix`` (the SSA tag)."""
    rename = lambda n: n if n in protected else f"{n}{suffix}"  # noqa: E731
    return [s.rewrite(rename, sigma) for s in body]


# The per-axis masked-overhang helpers — one :class:`Side` in, so the codegen never re-derives a
# ``mask_m`` / ``m_ext`` scalar pair: ``_guard`` predicates a store, ``_wrap`` clamp-reads an operand.
def _guard(side: Side, coord: Expr):
    """The overhang guard for cell ``coord`` on ``side`` — ``(coord, extent)`` when the axis is masked
    (its store is predicated on it), else ``None``."""
    return (coord, side.ext) if side.mask else None


def _wrap(side: Side, coord: Expr) -> Expr:
    """Wrap ``coord`` in-bounds (``coord % extent``) on a masked ``side`` so an overhanging cell
    clamp-reads, else ``coord`` unchanged."""
    return BinaryExpr("%", coord, side.ext) if side.mask else coord


def _cells(mn: tuple, offset, i: int, j: int):
    """Yield ``(side, cell-base coord)`` for each present output axis of register cell ``(i, j)`` —
    ``(m, offset[0].base(i))`` then ``(n, offset[1].base(j))`` (``m`` skipped for a 1-D output)."""
    for side, off, r in ((mn[0], offset[0], i), (mn[1], offset[1], j)):
        if side is not None:
            yield side, off.base(r)


# ---- warp/mma tier ----------------------------------------------------------------------------- #
def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for a dim varying with the output row /
    col axis, else ``"fixed"`` (batch / grid literal — uniform across the fragment cell)."""
    roles = []
    for e in index:
        fv = e.free_vars()
        roles.append("m" if m_name in fv else "n" if n_name in fv else "fixed")
    return tuple(roles)


def _warp_epilogue(tail: list[Stmt], acc: str, m_name: str, n_name: str, sigma: Sigma) -> RegEpilogue | None:
    """Fold the projection ``Map`` into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
    there is no projection (a bare ``Write`` of the accumulator).

    The projection is the post-reduce ``tail`` stmts: the leaf ``Load``s + pointwise ``Assign``s +
    an optional causal ``Select``. Each leaf ``Load`` becomes an :class:`EpilogueLoad` at the
    cell-base coordinate (σ-applied; the render adds the per-element row/col motion on the
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args)`` op; a coord-predicated
    ``Select`` (causal mask) rewrites its ``m``/``n`` coordinate vars to the ``__M__`` / ``__N__``
    placeholders the store substitutes with the element's own (row, col)."""
    loads, ops, selects = [], [], []
    write = None
    ph = {m_name: Var("__M__"), n_name: Var("__N__")}
    for s in tail:
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


# ---- operand staging (smem slab + ldmatrix drain) ---------------------------------------------- #
# The warp tier's smem operand pipeline, driven off the node's :class:`Stage`. cp.async and TMA
# share the (plain row-major, NONE-swizzle) slab + the staged-``LdmatrixLoad`` drain
# (:func:`_staged_inner_atom_loop`); only the producer differs. Staging is a **pure perf
# transform**: an ineligible kernel silently falls back to gmem-direct, and a staged kernel is
# bit-identical to its gmem-direct baseline. The transport primitives (the fill loops + the
# commit/wait / mbarrier handshakes) live in ``_stage.py``; these functions schedule them onto the
# K-loop off the :class:`Contraction` geometry.
def _staged_inner_atom_loop(
    *, slabs: tuple[str, str], mn: tuple[Side, Side], atom, bk_elems, ki, reg_depth: int = 1, offs=(None, None)
) -> list[Stmt]:
    """The inner atom-K drain shared by the cp.async and TMA staged paths: read the A/B ``slabs`` via
    ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. Slab-local indices — A[tile_m][bk_elems]
    (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent of which producer filled the (plain
    row-major, NONE-swizzle) slab; ``mn`` is the ``(m, n)`` :class:`Side` pair.

    ``reg_depth == 1`` (default): one ``StridedLoop`` over the ``bk`` atom-K steps, ldmatrix-then-mma
    inline (the operand fragments ``_a{i}``/``_b{j}`` reused every step). ``reg_depth >= 2`` (the
    ``STAGE`` ``/p<n>`` smem→register double-buffer): the loop is **fully unrolled** into a software
    pipeline that ldmatrixes the next atom-K step into an alternate fragment slot (``_a{i}_s{slot}``)
    ``reg_depth-1`` steps ahead while the mma consumes the current slot — breaking the per-step WAR
    hazard on the operand fragments. Numerically identical to the inline form.

    ``offs`` (the gmem→smem ring, ``STAGE`` depth>1): the ``(a, b)`` read SLOT row offsets — added to
    each slab's ROW (A's tile row / B's K row) so the drain reads the ring slot the producer already
    filled, while a later chunk prefetches into another slot."""
    (a_slab, b_slab), (m, n) = slabs, mn
    atom_m, atom_n, atom_k = atom.shape
    n_steps = bk_elems // atom_k
    # Per-operand drain spec: (tag, slab, ldm, tile-is-slab-row, reg count, warp-unit var, atom dim, slot row off).
    # A stacks the tile axis on the slab row (K the col); B swaps (K the row, tile the col); the slot
    # offset always lands on the ROW. The two share ONE emission loop.
    specs = (
        ("a", a_slab, bk_elems, True, m.reg, m.unit, atom_m, offs[0]),
        ("b", b_slab, n.tile, False, n.reg, n.unit, atom_n, offs[1]),
    )

    def ldms(kexpr, suffix):  # both operands' ldmatrix reads at K position `kexpr`, into fragment slot `suffix`
        reads: list[Stmt] = []
        for tag, slab, ldm, is_row, reg, unit, adim, off in specs:
            for x in range(reg):  # within-tile coord for register cell x: warp·(reg·adim) + x·adim
                prim = BinaryExpr("+", BinaryExpr("*", Var(unit), Literal(reg * adim, "int")), Literal(x * adim, "int"))
                row, col = (prim, kexpr) if is_row else (kexpr, prim)
                if off is not None:
                    row = BinaryExpr("+", off, row)
                reads.append(LdmatrixLoad(frag=f"_{tag}{x}{suffix}", src_buffer=slab, src_index=(row, col), role=tag, staged=True, ldm=ldm))
        return reads

    def mmas(suffix):  # every (i, j) cell's mma.sync over the `suffix`-slotted operand fragments
        return [
            MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}{suffix}", b_frag=f"_b{j}{suffix}", shape=atom.shape, ab_dtype=atom.ab_dtype)
            for i in range(m.reg)
            for j in range(n.reg)
        ]

    if reg_depth < 2 or n_steps < 2:  # single-buffer: the inline ldmatrix→mma loop
        body = ldms(Var(ki), "") + mmas("")
        return [
            StridedLoop(
                axis=Axis(name=ki, extent=bk_elems),
                start=Literal(0, "int"),
                step=Literal(atom_k, "int"),
                body=Body(tuple(body)),
                unroll=True,
            )
        ]

    # reg_depth ≥ 2: the unrolled register double-buffer. ``slot = step % depth`` cycles the fragment
    # buffers; prefetch runs ``depth-1`` steps ahead of the consuming mma.
    depth = min(reg_depth, n_steps)
    kcol = lambda step: Literal(step * atom_k, "int")  # slab-local K col of atom-K step `step`  # noqa: E731
    stmts: list[Stmt] = []
    for s in range(depth - 1):  # prologue: prime the first depth-1 steps
        stmts += ldms(kcol(s), f"_s{s % depth}")
    for step in range(n_steps):
        nxt = step + depth - 1
        if nxt < n_steps:  # prefetch depth-1 ahead, into the slot the mma below frees
            stmts += ldms(kcol(nxt), f"_s{nxt % depth}")
        stmts += mmas(f"_s{step % depth}")
    return stmts


def _clamp_last(idx: Expr, ext: Expr) -> Expr:
    """Clamp an overhanging gmem coordinate to the last valid index — the overhanging cell still
    reads an in-bounds (duplicate) operand, and its store is discarded by the guard (``RegStore`` /
    ``Cond``)."""
    return TernaryExpr(cond=BinaryExpr("<", idx, ext), if_true=idx, if_false=BinaryExpr("-", ext, Literal(1, "int")))


def _slab_index(operand_index, *, tile: Side, tile_base, k_axis, tile_is_row: bool):
    """The **one** cp.async slab gmem-index factory, for either operand and either tier. The slab's
    inner (contiguous) dim maps to the contraction ``k_axis``, its outer dim to the stationary ``tile``
    axis (``m`` for A, ``n`` for B). For A the tile axis is the slab ROW (K the col); for B they swap
    (``slot[row][col] = A[row_base + row][k0 + col]`` / ``B[k0 + row][col_base + col]``). A masked tile
    coordinate is clamped in-bounds — the overhanging cell reads a duplicate and its store is guarded.
    Returns a ``k0 -> ((row, col) -> gmem index)`` map — one K-chunk offset per :func:`staged_kloop`
    fill."""

    def at(k0):
        def gmem(row, col):
            tc, kc = (row, col) if tile_is_row else (col, row)
            t = BinaryExpr("+", tile_base, tc)
            sig = Sigma({tile.axis.name: _clamp_last(t, tile.ext) if tile.mask else t, k_axis.name: BinaryExpr("+", k0, kc)})
            return tuple(sig.apply(e) for e in operand_index)

        return gmem

    return at


def _tile_base(mn: tuple[Side, Side]) -> tuple[Expr, Expr]:
    """The CTA tile's ``(row_base, col_base)`` top-left origin — ``(m_b·tile_m, n_b·tile_n)``."""
    return tuple(BinaryExpr("*", Var(s.block), Literal(s.tile, "int")) for s in mn)


def _box_coords(tile_base: Expr, is_row: bool):
    """The TMA box origin at K-chunk ``k0`` — ``(tile_base, k0)`` when the tile axis is the slab ROW (A),
    ``(k0, tile_base)`` when it is the COL (B)."""
    return (lambda k0: (tile_base, k0)) if is_row else (lambda k0: (k0, tile_base))


def _slab_operands(*, index_srcs: tuple, bufs: tuple[str, str], mn: tuple[Side, Side], k_axis, bk_elems: int, base: tuple[Expr, Expr]):
    """The staged ``(A, B)`` :class:`Operand` pair — the one operand-geometry factory both tiers build,
    looped over the two operands. A is ``(tile_m × bk)`` indexed by the M tile axis (the slab ROW); B is
    ``(bk × tile_n)`` by the N tile axis (the slab COL) — ``is_row`` flips the slot shape + the TMA box
    origin. ``base`` is the ``(row_base, col_base)`` CTA tile origin; ``index_srcs`` are the operands'
    gmem index expressions (``load.index``)."""
    ops: list[Operand] = []
    for i, (tag, is_row) in enumerate((("a", True), ("b", False))):
        tile, tile_base = mn[i], base[i]
        shape = (tile.tile, bk_elems) if is_row else (bk_elems, tile.tile)
        ops.append(
            Operand(
                tag=tag,
                buf=bufs[i],
                shape=shape,
                coords=_box_coords(tile_base, is_row),
                index=_slab_index(index_srcs[i], tile=tile, tile_base=tile_base, k_axis=k_axis, tile_is_row=is_row),
            )
        )
    return tuple(ops)


def _cta(mn: tuple[Side, Side], lanes: int, n_threads: int) -> CtaTile:
    """The staging :class:`CtaTile` for either atom — the intra-CTA linear thread id from the
    decoded unit axis vars (never a raw threadIdx.x): unit-major ``m_unit·units_n + n_unit``,
    ``·lanes + _lane`` when the unit is a warp (``lanes > 1``; a scalar unit IS one thread)."""
    m, n = mn
    tid = BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(n.units, "int")), Var(n.unit))
    if lanes > 1:
        tid = BinaryExpr("+", BinaryExpr("*", tid, Literal(lanes, "int")), Var("_lane"))
    return CtaTile(linear_tid=tid, n_threads=n_threads)


def _sync_operands(c: Contraction, bk_elems: int, mn: tuple[Side, Side]) -> tuple[SyncOperand, SyncOperand]:
    """The ``sync``-transport (fused-edge) operand pair: A **compute-filled** from the node's
    producer cone (``a_operand`` is a ``Body`` — each thread evaluates the cone at the slab cell's
    absolute ``(m, k)`` coords and writes the result), B copy-filled from its gmem ``Load``. The
    schedule's eligibility guarantees exact-cover geometry (no masked overhang), so the σ needs no
    clamps."""
    m_name, n_name, k_name = c.m_axis.name, c.n_axis.name, c.k_axis.name
    row_base, col_base = _tile_base(mn)

    def a_value(k0, row, col):
        sigma = Sigma({m_name: BinaryExpr("+", row_base, row), k_name: BinaryExpr("+", k0, col)})
        return [s.rewrite(lambda nm: nm, sigma) for s in c.a_body], c.a_name

    def b_value(k0, row, col):
        sigma = Sigma({k_name: BinaryExpr("+", k0, row), n_name: BinaryExpr("+", col_base, col)})
        name = f"{c.b_load.names[0]}__f"
        return [Load(name=name, input=c.b_load.input, index=tuple(sigma.apply(e) for e in c.b_load.index))], name

    return (
        SyncOperand(tag="a", shape=(mn[0].tile, bk_elems), value=a_value),
        SyncOperand(tag="b", shape=(bk_elems, mn[1].tile), value=b_value),
    )


def _staged(ops: _AtomOps, cells, offset, mn: tuple[Side, Side]):
    """The **one** STAGED K-loop driver, atom-agnostic — build the ``(A, B)`` operand pair, the
    :class:`Transport` (a cp.async prefetch ring or the TMA box-copy producer) and run the one
    :func:`staged_kloop`; the atom supplies only the slab drain leaf (:meth:`_AtomOps.staged_drain`
    — ``ldmatrix`` + ``mma.sync`` vs plain-``Load`` fma) and the slab element dtype. ``ops.stage``
    is the **scheduler-RESOLVED** stage (``_schedule._resolve_warp_stage`` / ``_resolve_scalar_stage``
    ran eligibility + sizing once) — its ``transport`` / ``bk_elems`` / ``depth`` / ``reg_depth``
    are applied verbatim, no decision here. A pure perf transform, numerically identical to
    gmem-direct (mma: bit-identical). ``depth == 1`` is the single-buffer degenerate; ``depth >= 2``
    the gmem→smem ring; ``reg_depth`` composes the mma inner smem→register double-buffer. A masked
    M / N overhang is fill-side clamped (cp.async) or box zero-filled (TMA); the discard stays with
    the store guard."""
    c, stage = ops.c, ops.stage
    k_axis = c.k_axis
    K = k_axis.extent.as_static()  # static K (the resolution eligibility rule)
    elem = ops.slab_elem()
    cta = _cta(mn, c.atom.lanes, c.block_threads)
    if stage.transport == "sync":
        # The fused-edge compute-fill: the A tile is COMPUTED into its slab (the producer cone),
        # B plain-copied — the mma tier's ``sync`` transport; single-buffer, one CTA barrier.
        operands = _sync_operands(c, stage.bk_elems, mn)
        transport = SyncTransport(operands=operands, slab_dtype=cuda_name(elem), cta=cta)
    else:
        operands = _slab_operands(
            index_srcs=(c.a_operand.index, c.b_load.index),
            bufs=(c.a_operand.input, c.b_load.input),
            mn=mn,
            k_axis=k_axis,
            bk_elems=stage.bk_elems,
            base=_tile_base(mn),
        )
        common = dict(
            operands=operands,
            slab_dtype=cuda_name(elem),
            elem_bytes=elem.nbytes,
            cta=cta,
        )
        transport = TmaTransport(**common) if stage.transport == "tma" else CpAsyncTransport(**common)

    def drain(slot):  # the atom's slab-reading leaf, over ring `slot`
        return ops.staged_drain(operands, slot, cells, offset, mn)

    return staged_kloop(
        transport=transport, drain=drain, depth=stage.depth, bk_elems=stage.bk_elems, n_chunks=K // stage.bk_elems, k_extent=K
    )


def _contract_kloop(c, cells, *, read_row, read_col, contract, wrap):
    """The shared contraction K-loop skeleton — the ``read → ⊗ → fold`` spine both atoms lower through.

    Read each register ROW's A operand once and each register COL's B operand once (register-tile
    operand reuse — an A read is shared across the row's columns and vice versa), contract every
    ``(row, col)`` pair into its accumulator, then wrap the whole body in the reduce loop. The only
    per-atom variation is the four leaf constructors (the "atom factory"): ``read_row`` / ``read_col``
    build the operand read (``LdmatrixLoad`` fragment vs scalar ``Load``), ``contract`` the ⊗+accumulate
    (``MmaSyncPtx`` vs ``Assign``+``Accum``), ``wrap`` the K-loop (``StridedLoop`` step ``atom_k`` vs a
    unit ``Loop``). Returns ``(pre_decls, kloop_stmts)`` — no pre-decls here (accumulators ride
    ``state``)."""
    rows = sorted({i for i, _ in cells})
    cols = sorted({j for _, j in cells})
    body: list[Stmt] = []
    for i in rows:
        body += read_row(i)
    for j in cols:
        body += read_col(j)
    for i, j in cells:
        body += contract(i, j)
    return [], wrap(body)


# ---- scalar (register-tile) tier --------------------------------------------------------------- #
def _unroll_inner(axis) -> bool:
    """Mark the inner contraction loop for ``#pragma unroll`` when it's a small static reduce
    (≤ 64 trips) — register-resident operand reuse + ILP, the scalar-SGEMM lever."""
    return axis.extent.is_static and axis.extent.as_static() <= 64


def _dedup_loads(stmts: list[Stmt]) -> list[Stmt]:
    """Collapse syntactically-identical scalar ``Load``s (same buffer + index) to one binding,
    rewriting the dropped names to the survivor — the operand reuse a register tile exists for (a
    load not referencing the ``m`` cell axis is shared across the ``n`` cells, and vice versa)."""
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


def _scalar_sigma(mn, offset, i: int, j: int) -> Sigma:
    """σ mapping each present output axis to register cell ``(i, j)``'s real coordinate (the offset's
    block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
    return Sigma({side.name: _wrap(side, cell) for side, cell in _cells(mn, offset, i, j)})


def _scalar_bound(mn, offset, i: int, j: int):
    """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` anded over the masked axes,
    or ``None`` when nothing overhangs."""
    conds = [BinaryExpr("<", cell, side.ext) for side, cell in _cells(mn, offset, i, j) if side.mask]
    if not conds:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = BinaryExpr("&&", cond, c)
    return cond


def _scalar_protected(c: Contraction) -> frozenset[str]:
    """The shared iteration coordinates — the block / unit / loop / extent vars excluded from the
    per-cell SSA rename (everything else is suffixed ``__c{i}_{j}`` so each cell owns its names)."""
    m, n, k_axis = c.m, c.n, c.k_axis
    prot = {k_axis.name}
    for s in (m, n):
        prot |= {s.block, s.unit}
    for a in c.lead_axes:
        prot.add(a.name)
    for a in (m.axis, n.axis, k_axis, *c.lead_axes):
        prot |= set(a.extent_expr().free_vars())
    return frozenset(prot)


def _scalar_drain(c: Contraction, cells, offset, slabs: tuple[str, str], ki: str, bk_elems: int, base: tuple[Expr, Expr]) -> Loop:
    """The inner slab-drain reduce loop ``for ki: b = b_slab[ki, n_local]; a = a_slab[m_local, ki];
    v = a·b; acc += v`` — the scalar counterpart of the mma ``ldmatrix`` drain. Built per-cell directly
    (NOT via the masked gmem-direct σ, whose ``% extent`` wrap would corrupt the slab index for an
    overhanging cell): the slab is indexed by the **local** tile coordinate ``offset[{0,1}].base(...) −
    base`` (``m_uvar·reg_m + i`` ∈ [0, tile_m), always in-slab), so an overhanging cell reads a
    clamped / zero-filled slab row and its store is discarded by the guard. ``_dedup_loads`` still
    shares A across the n-cells and B across the m-cells exactly as gmem-direct does. **Carrier-less**
    (no ``Loop.carrier``): the accumulators are pre-seeded once by :meth:`_ScalarOps.state` outside the
    outer slab loop, so the drain folds into them without re-seeding."""
    (a_slab, b_slab), (row_base, col_base) = slabs, base
    b_name, a_name = c.b_load.names[0], c.a_name
    body: list[Stmt] = []
    for i, j in cells:
        sfx = f"__c{i}_{j}"
        bn, an, vn, cn = f"{b_name}{sfx}", f"{a_name}{sfx}", f"{c.acc}__v{sfx}", f"{c.acc}{sfx}"
        m_local = BinaryExpr("-", offset[0].base(i), row_base)
        n_local = BinaryExpr("-", offset[1].base(j), col_base)
        body.append(Load(names=(bn,), input=b_slab, index=(Var(ki), n_local)))
        body.append(Load(names=(an,), input=a_slab, index=(m_local, Var(ki))))
        body.append(Assign(name=vn, op=_MUL, args=(bn, an)))
        body.append(Accum(name=cn, value=vn, op=_ADD, axes=(ki,)))
    body = _dedup_loads(body)
    # seed=False: the accumulators are pre-seeded once by _ScalarOps.state outside the outer slab loop, so
    # this inner drain must NOT re-declare (re-zero) them each slab iteration.
    return Loop(axis=Axis(name=ki, extent=bk_elems), body=Body(tuple(body)), unroll=bk_elems <= 64, seed=False)


@dataclass(frozen=True)
class _AtomOps:
    """The per-atom codegen **strategy** — the one seam every tiled contraction dispatches through.
    Bound to the contraction ``c`` + its **scheduler-resolved** operand ``stage`` (``None`` =
    gmem-direct; eligibility + sizing already ran in ``020_schedule``) and ``inputs``, it supplies
    the three ``grid_tile`` callables — ``state(cells)`` (accumulator decls), :meth:`reduce` (the
    K-loop — **shared on this base**, one loop over atoms), ``store(i, j, offset, mn)`` (the
    per-cell sink; ``mn`` is the contraction's ``(m, n)`` :class:`Side` pair). The two concrete
    atoms (:class:`_MmaOps` / :class:`_ScalarOps`) supply only descriptor reads:
    :meth:`gmem_leaves` (the four gmem-direct leaf constructors), :meth:`staged_drain` (the
    slab-reading leaf) and :meth:`slab_elem` (the slab element dtype). This IS the "atom as
    descriptor" seam: one factory (:func:`_atom_ops`), no scattered ``isinstance``."""

    c: Contraction
    stage: Stage | None = None
    inputs: object = None

    def reduce(self, cells, offset, mn):
        """The contraction K-loop — the ONE driver both atoms flow through, deciding nothing: a
        resolved ``stage`` means staged (an smem operand slab over the one :func:`_staged`
        fill→drain skeleton), ``None`` means gmem-direct (the shared :func:`_contract_kloop`
        ``read → ⊗ → fold`` spine). Either way the atom contributes only leaves, never a loop."""
        if self.stage is not None:
            return _staged(self, cells, offset, mn)
        return _contract_kloop(self.c, cells, **self.gmem_leaves(offset, mn))


class _MmaOps(_AtomOps):
    """Tensor-core atom — ``ldmatrix`` fragment reads + ``mma.sync``, a ``RegStore`` sink."""

    def slab_elem(self):
        """The slab element dtype — the mma A/B operand dtype (f16/bf16 fragments)."""
        return self.c.atom.operand_dtype("a")

    def staged_drain(self, operands, slot, cells, offset, mn):
        """The mma slab drain — the ``ldmatrix`` + ``mma.sync`` leaf reading ring ``slot``
        (:func:`_staged_inner_atom_loop`; the cells ride ``mn``'s reg counts, so ``cells`` /
        ``offset`` are unused here)."""
        return _staged_inner_atom_loop(
            slabs=tuple(op.slab for op in operands),
            offs=tuple(op.slot_row(slot) for op in operands),
            mn=mn,
            atom=self.c.atom,
            bk_elems=self.stage.bk_elems,
            ki="_ki",
            reg_depth=self.stage.reg_depth,
        )

    def state(self, cells):
        """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
        one ``_c`` accumulator per cell (held across the K-loop). A staged ``reg_depth >= 2`` slots the
        operand fragments (``_a{i}_s{slot}``) for the smem→register double-buffer's ping-pong."""
        c = self.c
        atom, m, n = c.atom, c.m, c.n
        reg_depth = self.stage.reg_depth if self.stage is not None else 1

        def frags(tag, reg):  # reg-tile operand fragment names (slotted ``_s{s}`` when double-buffered)
            return (
                [f"_{tag}{i}_s{s}" for i in range(reg) for s in range(reg_depth)] if reg_depth >= 2 else [f"_{tag}{i}" for i in range(reg)]
            )

        decls: list[Stmt] = [
            RegFragment(name=nm, role=tag, shape=atom.shape, dtype=atom.operand_dtype(tag))
            for tag, reg in (("a", m.reg), ("b", n.reg))
            for nm in frags(tag, reg)
        ]
        decls += [
            RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c"))
            for i in range(m.reg)
            for j in range(n.reg)
        ]
        return decls

    def gmem_leaves(self, offset, mn):
        """The gmem-direct mma leaf constructors: ``ldmatrix`` each operand fragment straight from
        gmem, ``mma.sync`` every cell, the K-loop a ``StridedLoop`` of step ``atom_k`` (a symbolic /
        non-divisible K zero-fills the masked-K tail via the ``k_zero`` helper variants — canonical
        and transposed-B both have gmem-direct K zero-fill helpers)."""
        c = self.c
        atom, (m, n) = c.atom, mn
        k_axis = c.k_axis
        assert not c.a_computed, (
            "mma matmul arm: a register-resident (computed) A operand lowers through the fragment realizer (_twist), not here"
        )
        a_load, b_load, b_trans = c.a_operand, c.b_load, c.b_trans
        k_static = k_axis.extent.is_static
        k_zero = None if k_static else (Var(k_axis.name), k_axis.extent_expr())

        def read_row(i):
            cell = offset[0].base(i)
            idx = tuple(Sigma({m.axis.name: cell}).apply(e) for e in a_load.index)
            return [
                LdmatrixLoad(
                    frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=_guard(m, cell), k_zero=k_zero
                )
            ]

        def read_col(j):
            cell = offset[1].base(j)
            idx = tuple(Sigma({n.axis.name: cell}).apply(e) for e in b_load.index)
            return [
                LdmatrixLoad(
                    frag=f"_b{j}",
                    src_buffer=b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=_guard(n, cell),
                    k_zero=k_zero,
                )
            ]

        def contract(i, j):
            return [MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype)]

        def wrap(body):
            return [
                StridedLoop(axis=k_axis, start=Literal(0, "int"), step=Literal(atom.atom_k, "int"), body=Body(tuple(body)), unroll=k_static)
            ]

        return dict(read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

    def store(self, i, j, offset, mn):
        """Store cell ``(i, j)``'s ``_c`` fragment to the output, folding the projection ``tail`` into a
        :class:`RegEpilogue` and guarding overhanging M/N rows."""
        c = self.c
        atom = c.atom
        m, n = mn
        mcell, ncell = offset[0].base(i), offset[1].base(j)
        tail = list(c.epilogue)
        write = next(s for s in tail if isinstance(s, Write))
        sigma = Sigma({m.axis.name: mcell, n.axis.name: ncell})
        return [
            RegStore(
                dst_buffer=write.output,
                dst_index=tuple(sigma.apply(e) for e in write.index),
                frag=f"_c{i}_{j}",
                shape=atom.shape,
                epilogue=_warp_epilogue(tail, c.acc, m.axis.name, n.axis.name, sigma),
                m_guard=_guard(m, mcell),
                n_guard=_guard(n, ncell),
            )
        ]


class _ScalarOps(_AtomOps):
    """Scalar fma atom — plain ``Load``\\ s + an ``fma`` cell, the replicated-``epilogue`` sink."""

    def slab_elem(self):
        """The slab element dtype — the gmem operand's own dtype (fp32 SGEMM stages fp32)."""
        return self.inputs[self.c.a_operand.input].dtype

    def staged_drain(self, operands, slot, cells, offset, mn):  # noqa: ARG002 — slot: single-buffer (no ring)
        """The scalar slab drain — the plain-``Load`` fma leaf (:func:`_scalar_drain`), reading by
        LOCAL tile coords; ``slot`` is unused (the scalar tier is single-buffer, ``depth == 1``)."""
        a_op, b_op = operands
        return [_scalar_drain(self.c, cells, offset, (a_op.slab, b_op.slab), "_ki", self.stage.bk_elems, _tile_base(mn))]

    def state(self, cells):
        """The scalar accumulator seeds. Gmem-direct (unstaged): none — the accumulators are seeded
        inside the reduce ``Loop`` (the dissolved fold ``Accum``\\ s + ``Loop.render``). **Staged**: a
        per-cell ``Init(acc__c{i}_{j} = 0)`` emitted here, **outside** the outer slab loop, so the
        carrier-less :func:`_scalar_drain` folds across every slab without re-seeding (the nested-loop
        accumulator-lifetime split — the scalar analogue of :meth:`_MmaOps.state` declaring the ``_c``
        fragments outside the mma K-loop)."""
        c = self.c
        if self.stage is None:
            return []
        return [Init(name=f"{c.acc}__c{i}_{j}", identity=_ADD.identity, dtype=F32) for i, j in cells]

    def gmem_leaves(self, offset, mn):
        """The gmem-direct scalar leaf constructors: each register ROW reads its A operand once (a
        gmem ``Load`` — or the computed register-resident body, e.g. flash PV's ``P``), each COL its
        B ``Load`` once, each ``(i, j)`` cell folds ``acc__c{i}_{j} += b·a``, and the K-loop is a unit
        ``Loop`` (``Loop.render`` seeds the accumulators; the store reads them). A masked axis wraps
        its read in-bounds (``% extent``) and the overhanging store is guarded (:meth:`store`)."""
        c = self.c
        k_axis = c.k_axis
        m, n = mn
        prot = _scalar_protected(c)
        b_name, a_name = c.b_load.names[0], c.a_name

        def read_row(i):
            if m is None:
                return copy_cell(c.a_body, Sigma({}), f"__ar{i}", prot)
            return copy_cell(c.a_body, Sigma({m.name: _wrap(m, offset[0].base(i))}), f"__ar{i}", prot)

        def read_col(j):
            return copy_cell([c.b_load], Sigma({n.name: _wrap(n, offset[1].base(j))}), f"__bc{j}", prot)

        def contract(i, j):
            v = f"{c.acc}__v__c{i}_{j}"
            return [
                Assign(name=v, op=_MUL, args=(f"{b_name}__bc{j}", f"{a_name}__ar{i}")),
                Accum(name=f"{c.acc}__c{i}_{j}", value=v, op=_ADD, axes=(k_axis.name,)),
            ]

        def wrap(body):
            return [Loop(axis=k_axis, body=Body(tuple(body)), unroll=_unroll_inner(k_axis))]

        return dict(read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

    def store(self, i, j, offset, mn):
        """Replicate the projection ``tail`` for cell ``(i, j)`` — σ-offset, suffix the SSA names, guard
        the (overhanging) write, dedup shared operand loads."""
        c = self.c
        sigma = _scalar_sigma(mn, offset, i, j)
        cell = copy_cell(c.epilogue, sigma, f"__c{i}_{j}", _scalar_protected(c))
        cell = _guard_writes(cell, _scalar_bound(mn, offset, i, j))
        return _dedup_loads(cell)


def _atom_ops(c: Contraction, stage: Stage | None = None, inputs=None) -> _AtomOps:
    """The **one** atom dispatch — select the codegen strategy off the atom kind."""
    return _MmaOps(c, stage, inputs) if isinstance(c.atom, AtomKind) else _ScalarOps(c, stage, inputs)


def reduce_codegen(c: Contraction, stage: Stage | None = None, inputs=None):
    """The reusable, **sink-agnostic** ``(state_decls, reduce_region)`` from the atom strategy — the
    accumulator decls + the contraction K-loop (the ONE :meth:`_AtomOps.reduce` driver: the shared
    :func:`_contract_kloop` spine gmem-direct, the shared :func:`_staged` fill→drain skeleton staged).
    ``stage`` / ``inputs`` bind operand staging (both atoms stage the same smem slab off it, differing
    only in the drain leaf — ``ldmatrix`` vs plain ``Load``)."""
    ops = _atom_ops(c, stage, inputs)
    return ops.state, ops.reduce


def store_sink(c: Contraction):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, mn)`` from the atom strategy
    (an mma ``RegStore`` / the replicated scalar ``epilogue`` tail). ``factorize(c, store=…)`` swaps it
    (a flash sink that bridges the accumulator into the streaming-softmax twist), reusing the shared
    ``reduce`` emission."""
    return _atom_ops(c).store
