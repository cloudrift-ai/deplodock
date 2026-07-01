r"""The per-atom codegen strategies — the one seam every tiled contraction dispatches through.

``_factorize_contraction`` (in ``_factor.py``) reads the tiling geometry off the :class:`Contraction`
node and asks this module for the two codegen halves: :func:`reduce_codegen` (the sink-agnostic
``(state_decls, reduce_region)`` — the accumulator/operand decls + the shared :func:`_contract_kloop`
K-loop) and :func:`store_sink` (the per-cell matmul sink). Both resolve through :func:`_atom_ops` to
one of the two concrete strategies — :class:`_MmaOps` (tensor-core ``ldmatrix`` + ``mma.sync``, a
``RegStore`` sink) or :class:`_ScalarOps` (plain ``Load``\ s + an ``fma`` cell, the replicated-
``epilogue`` sink) — which share the ``_contract_kloop`` skeleton and the operand-staging helpers
(:func:`_mma_staged` / :func:`_scalar_staged` over the one ``_stage.staged_kloop``). This IS the
"atom as descriptor" seam: one factory, no scattered ``isinstance``.

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
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import copy_cell
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    CpAsyncTransport,
    CtaTile,
    Operand,
    TmaTransport,
    staged_kloop,
)

#: The contraction semiring — multiply ⊗ then accumulate ⊕ (add). The same multiply-add ``mma.sync``
#: realizes; in the scalar tier it is a plain scalar fma loop.
_MUL = ElementwiseImpl("multiply")
_ADD = ElementwiseImpl("add")


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
def _can_stage_warp(stage, k_axis: Axis, tile_m: int, tile_n: int, bk: int, atom_k: int, mask_m: bool, mask_n: bool, b_trans: bool) -> bool:
    """cp.async staging eligibility: a ``cp.async`` stage over a contraction with a STATIC,
    tile-divisible K axis and a canonical (non-transposed) B operand. A masked / symbolic **M**
    (output rows) is fine — the A-slab fill clamp-reads the overhanging rows in-bounds and the
    ``RegStore`` guards their store. A masked **N** (the B-slab inner dim) and a symbolic /
    non-divisible **K** stay gmem-direct (K zero-fill is a follow-up). Staging only ever *adds* a
    faster lowering, so an ineligible kernel silently falls back to gmem-direct."""
    if stage is None or stage.transport != "cp.async" or b_trans or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    if k_axis.extent.as_static() % bk_elems != 0:
        return False
    # cp.async needs a ≥4-byte contiguous chunk; the 16-bit mma operands give 2 B/elem, so the
    # inner slab dim must be even (A's BK, B's tile_n). Odd ⇒ fall back.
    return (bk_elems % 2 == 0) and (tile_n % 2 == 0)


def _can_stage_warp_tma(
    stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n: bool, b_trans: bool
) -> bool:
    """TMA (``cp.async.bulk.tensor``) staging eligibility: a ``tma`` stage over a contraction with a
    STATIC, tile-divisible K and a canonical B. A masked / symbolic **M** is fine — the descriptor's
    globalDim is the runtime M and TMA zero-fills the box overhang past it (no fill clamp needed). A
    masked **N** and a symbolic / non-divisible **K** stay gmem-direct. The box's inner dim (A's BK,
    B's tile_n) and the source's inner global stride (A's K, B's N) must be 16 B-aligned (the
    NONE-swizzle TMA box-copy rule)."""
    if stage is None or stage.transport != "tma" or b_trans or mask_n:
        return False
    if not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    bk_elems = bk * atom_k
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    return all((x * elem_bytes) % 16 == 0 for x in (bk_elems, tile_n, k, n))


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


def _mma_staged(*, loads: tuple, mn: tuple[Side, Side], k_axis, atom, bk, slab_dtype, elem_bytes, reg_depth: int, depth: int, mode: str):
    """The warp tier's STAGED K-loop — build the operand :class:`Transport` (a cp.async prefetch ring
    or the TMA box-copy producer) from the ``(a_load, b_load)`` operand pair + the shared ``ldmatrix``
    drain leaf, then run the one :func:`staged_kloop`. A pure perf transform, bit-identical to
    gmem-direct. ``depth == 1`` is the single-buffer degenerate; ``depth >= 2`` (cp.async ``d<depth>``)
    is the gmem→smem ring; ``reg_depth`` composes the inner smem→register double-buffer. ``mask_m``: the
    cp.async A fill clamps the overhanging gmem row in-bounds; TMA zero-fills the box overhang instead
    (both leave the discard to the ``RegStore`` ``m_guard``)."""
    m, n = mn
    bk_elems = bk * atom.shape[2]
    K = k_axis.extent.as_static()  # static K (the staging eligibility rule)
    # intra-CTA linear thread id from the decoded warp / lane axis vars (never a raw threadIdx.x).
    linear_tid = BinaryExpr(
        "+",
        BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(n.units, "int")), Var(n.unit)), Literal(32, "int")),
        Var("_lane"),
    )
    operands = _slab_operands(
        index_srcs=tuple(ld.index for ld in loads),
        bufs=tuple(ld.input for ld in loads),
        mn=mn,
        k_axis=k_axis,
        bk_elems=bk_elems,
        base=_tile_base(mn),
    )
    common = dict(
        operands=operands,
        slab_dtype=slab_dtype,
        elem_bytes=elem_bytes,
        cta=CtaTile(linear_tid=linear_tid, n_threads=m.units * n.units * 32),
    )
    transport = TmaTransport(**common) if mode == "tma" else CpAsyncTransport(**common)

    def drain(slot):  # the ldmatrix + mma.sync leaf, reading ring `slot`
        return _staged_inner_atom_loop(
            slabs=tuple(op.slab for op in operands),
            offs=tuple(op.slot_row(slot) for op in operands),
            mn=mn,
            atom=atom,
            bk_elems=bk_elems,
            ki="_ki",
            reg_depth=reg_depth,
        )

    return staged_kloop(transport=transport, drain=drain, depth=depth, bk_elems=bk_elems, n_chunks=K // bk_elems, k_extent=K)


def _mma_stage_plan(c: Contraction, stage: Stage | None) -> tuple[str, int, int]:
    """The operand-staging decision for the mma contraction ``c`` under ``stage`` — read once and
    shared by :meth:`_MmaOps.state` (which slots the fragments) and :meth:`_MmaOps.reduce` (which emits the
    K-loop). Returns ``(mode, gmem_depth, reg_depth)`` where ``mode`` is ``"tma"`` / ``"cp"`` /
    ``"gmem"`` (TMA > cp.async > gmem-direct). ``gmem`` forces both depths to 1 (no slab). Both
    transports share the ``gmem_depth`` gmem→smem ring — the collapse's one buffering knob (TMA's ring
    rides the per-slot mbarrier, cp.async's the commit group), capped so the ``depth``-slot slab fits in
    a 48 KiB smem budget."""
    if stage is None or c.a_computed:
        return "gmem", 1, 1
    atom = c.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = c.tile.bk
    m, n = c.m, c.n
    tma_ok = _can_stage_warp_tma(stage, c.k_axis, n.axis, n.tile, bk, atom.atom_k, a_nbytes, n.mask, c.b_trans)
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.k_axis, m.tile, n.tile, bk, atom.atom_k, m.mask, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return "gmem", 1, 1
    reg_depth = min(stage.reg_depth, bk)
    slot_bytes = (m.tile * bk * atom.atom_k + bk * atom.atom_k * n.tile) * a_nbytes
    gmem_depth = min(stage.depth, max(1, (48 * 1024) // slot_bytes))
    return ("tma" if tma_ok else "cp"), gmem_depth, reg_depth


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
    """σ mapping the output axes to register cell ``(i, j)``'s real coordinate (the offset's
    block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
    m, n = mn
    smap: dict = {}
    if m is not None:
        bm = offset.m.base(i)
        smap[m.name] = BinaryExpr("%", bm, m.ext) if m.mask else bm
    bn = offset.n.base(j)
    smap[n.name] = BinaryExpr("%", bn, n.ext) if n.mask else bn
    return Sigma(smap)


def _scalar_bound(mn, offset, i: int, j: int):
    """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` for each masked axis (anded),
    or ``None`` when nothing overhangs."""
    m, n = mn
    conds = []
    if m is not None and m.mask:
        conds.append(BinaryExpr("<", offset.m.base(i), m.ext))
    if n.mask:
        conds.append(BinaryExpr("<", offset.n.base(j), n.ext))
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
        prot |= set(_extent_expr(a).free_vars())
    return frozenset(prot)


def _scalar_stage_plan(c: Contraction, stage: Stage | None, inputs) -> tuple[str, int]:
    """The scalar-tier operand-staging decision — ``(mode, bk_elems)``, ``mode`` one of ``"tma"`` /
    ``"cp"`` / ``"gmem"`` (gmem = no slab). Staging is **opt-in behind a ``STAGE`` pin**: a scalar
    contraction with no ``stage`` (every default scalar matmul) is ``"gmem"``, byte-identical. Eligible
    when ``stage`` is set with a ``tma`` / ``cp.async`` transport, K is static, and the operands are
    plain gmem ``Load``\\ s (not a computed-A flash body). A masked (overhanging) M / N is fine — the
    drain reads the slab by LOCAL tile coords and the overhanging store is guarded, so TMA zero-fills
    the box overhang and cp.async clamps the gmem read. The slab K-chunk ``bk_elems`` is **derived** to
    fit a single ``tile_m×bk + bk×tile_n`` operand slab in 48 KiB (largest power-of-two dividing K) —
    not spelled by a codec, so no schema change."""
    if stage is None or c.a_computed or not c.k_axis.extent.is_static:
        return "gmem", 0
    if inputs is None or c.a_operand.input not in inputs or stage.transport not in ("tma", "cp.async"):
        return "gmem", 0
    K = c.k_axis.extent.as_static()
    elem_bytes = inputs[c.a_operand.input].dtype.nbytes
    cap = (48 * 1024) // (max(1, c.m.tile + c.n.tile) * elem_bytes)
    bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and K % v == 0), 0)
    if bk_elems < 4:
        return "gmem", 0
    return ("tma" if stage.transport == "tma" else "cp"), bk_elems


def _scalar_drain(c: Contraction, cells, offset, slabs: tuple[str, str], ki: str, bk_elems: int, base: tuple[Expr, Expr]) -> Loop:
    """The inner slab-drain reduce loop ``for ki: b = b_slab[ki, n_local]; a = a_slab[m_local, ki];
    v = a·b; acc += v`` — the scalar counterpart of the mma ``ldmatrix`` drain. Built per-cell directly
    (NOT via the masked gmem-direct σ, whose ``% extent`` wrap would corrupt the slab index for an
    overhanging cell): the slab is indexed by the **local** tile coordinate ``offset.{m,n}.base(...) −
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
        m_local = BinaryExpr("-", offset.m.base(i), row_base)
        n_local = BinaryExpr("-", offset.n.base(j), col_base)
        body.append(Load(names=(bn,), input=b_slab, index=(Var(ki), n_local)))
        body.append(Load(names=(an,), input=a_slab, index=(m_local, Var(ki))))
        body.append(Assign(name=vn, op=_MUL, args=(bn, an)))
        body.append(Accum(name=cn, value=vn, op=_ADD, axes=(ki,)))
    body = _dedup_loads(body)
    # seed=False: the accumulators are pre-seeded once by _ScalarOps.state outside the outer slab loop, so
    # this inner drain must NOT re-declare (re-zero) them each slab iteration.
    return Loop(axis=Axis(name=ki, extent=bk_elems), body=Body(tuple(body)), unroll=bk_elems <= 64, seed=False)


def _scalar_staged(c: Contraction, inputs, cells, offset, mode: str, bk_elems: int) -> tuple[list[Stmt], list[Stmt]]:
    """The scalar tier's STAGED K-loop — a scalar operand :class:`Transport` (cp.async gmem-index clamps
    / TMA box-copy zero-fill) + the plain-``Load`` :func:`_scalar_drain`, over the one
    :func:`staged_kloop` (single-buffer, ``depth == 1``). The scalar :class:`CtaTile` (``block_threads``
    threads, the ``m.unit·n.units + n.unit`` linear id) is the only per-tier seam; a pure perf transform,
    numerically identical to gmem-direct."""
    mn, k_axis = c.mn, c.k_axis
    m, n = mn
    K = k_axis.extent.as_static()
    bufs = (c.a_operand.input, c.b_load.input)
    slab_dtype, elem_bytes = cuda_name(inputs[bufs[0]].dtype), inputs[bufs[0]].dtype.nbytes
    base = _tile_base(mn)
    linear_tid = BinaryExpr("+", BinaryExpr("*", Var(m.unit), Literal(n.units, "int")), Var(n.unit))
    operands = _slab_operands(index_srcs=(c.a_operand.index, c.b_load.index), bufs=bufs, mn=mn, k_axis=k_axis, bk_elems=bk_elems, base=base)
    common = dict(
        operands=operands, slab_dtype=slab_dtype, elem_bytes=elem_bytes, cta=CtaTile(linear_tid=linear_tid, n_threads=c.block_threads)
    )
    transport = TmaTransport(**common) if mode == "tma" else CpAsyncTransport(**common)

    def drain(_slot):  # single-buffer: the scalar slab drain reads by local tile coords (no ring slot)
        a_op, b_op = operands
        return [_scalar_drain(c, cells, offset, (a_op.slab, b_op.slab), "_ki", bk_elems, base)]

    return staged_kloop(transport=transport, drain=drain, depth=1, bk_elems=bk_elems, n_chunks=K // bk_elems, k_extent=K)


@dataclass(frozen=True)
class _AtomOps:
    """The per-atom codegen **strategy** — the one seam every tiled contraction dispatches through.
    Bound to the contraction ``c`` + its operand ``stage`` / ``inputs``, it supplies the three
    ``grid_tile`` callables — ``state(cells)`` (accumulator decls), ``reduce(cells, offset, mn)``
    (the K-loop, via the shared :func:`_contract_kloop` skeleton), ``store(i, j, offset, mn)`` (the
    per-cell sink; ``mn`` is the contraction's ``(m, n)`` :class:`Side` pair). The two concrete atoms
    (:class:`_MmaOps` / :class:`_ScalarOps`) differ only in the
    leaf codegen they delegate to. This IS the "atom as descriptor" seam: one factory
    (:func:`_atom_ops`), no scattered ``isinstance``."""

    c: Contraction
    stage: Stage | None = None
    inputs: object = None


class _MmaOps(_AtomOps):
    """Tensor-core atom — ``ldmatrix`` fragment reads + ``mma.sync``, a ``RegStore`` sink."""

    def state(self, cells):
        """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
        one ``_c`` accumulator per cell (held across the K-loop). A staged ``reg_depth >= 2`` slots the
        operand fragments (``_a{i}_s{slot}``) for the smem→register double-buffer's ping-pong."""
        c, stage = self.c, self.stage
        atom, m, n = c.atom, c.m, c.n
        _mode, _gmem_depth, reg_depth = _mma_stage_plan(c, stage)

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

    def reduce(self, cells, offset, mn):
        """The mma K-loop, dispatched on the staging decision (TMA > cp.async > gmem-direct). Staged:
        cooperatively fill an smem slab then ``ldmatrix``-drain it (:func:`_mma_staged`, the one
        :func:`staged_kloop`) — a pure perf transform, bit-identical to gmem-direct. Gmem-direct:
        ``ldmatrix`` each operand fragment straight from gmem, then ``mma.sync`` every cell (a symbolic /
        non-divisible K zero-fills the masked-K tail via the ``k_zero`` helper variants — canonical and
        transposed-B both have gmem-direct K zero-fill helpers)."""
        c, stage = self.c, self.stage
        atom, (m, n) = c.atom, mn
        m_axis, n_axis, k_axis = m.axis, n.axis, c.k_axis
        assert not c.a_computed, (
            "mma tier: register-resident A operand (a computed flash-PV fragment feed) is a scalar-tier-only capability"
        )
        a_load, b_load, b_trans = c.a_operand, c.b_load, c.b_trans
        mode, gmem_depth, reg_depth = _mma_stage_plan(c, stage)
        if mode != "gmem":
            return _mma_staged(
                loads=(a_load, b_load),
                mn=mn,
                k_axis=k_axis,
                atom=atom,
                bk=c.tile.bk,
                slab_dtype=cuda_name(atom.operand_dtype("a")),
                elem_bytes=atom.operand_dtype("a").nbytes,
                reg_depth=reg_depth,
                depth=gmem_depth,
                mode=mode,
            )
        mask_m, mask_n, m_ext, n_ext = m.mask, n.mask, m.ext, n.ext
        k_static = k_axis.extent.is_static
        k_zero = None if k_static else (Var(k_axis.name), _extent_expr(k_axis))

        def read_row(i):
            idx = tuple(Sigma({m_axis.name: offset.m.base(i)}).apply(e) for e in a_load.index)
            guard = (offset.m.base(i), m_ext) if mask_m else None
            return [
                LdmatrixLoad(frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero)
            ]

        def read_col(j):
            idx = tuple(Sigma({n_axis.name: offset.n.base(j)}).apply(e) for e in b_load.index)
            guard = (offset.n.base(j), n_ext) if mask_n else None
            return [
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
            ]

        def contract(i, j):
            return [MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype)]

        def wrap(body):
            return [
                StridedLoop(axis=k_axis, start=Literal(0, "int"), step=Literal(atom.atom_k, "int"), body=Body(tuple(body)), unroll=k_static)
            ]

        return _contract_kloop(c, cells, read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

    def store(self, i, j, offset, mn):
        """Store cell ``(i, j)``'s ``_c`` fragment to the output, folding the projection ``tail`` into a
        :class:`RegEpilogue` and guarding overhanging M/N rows."""
        c = self.c
        atom = c.atom
        m, n = mn
        m_axis, n_axis = m.axis, n.axis
        mask_m, mask_n, m_ext, n_ext = m.mask, n.mask, m.ext, n.ext
        tail = list(c.epilogue)
        write = next(s for s in tail if isinstance(s, Write))
        sigma = Sigma({m_axis.name: offset.m.base(i), n_axis.name: offset.n.base(j)})
        return [
            RegStore(
                dst_buffer=write.output,
                dst_index=tuple(sigma.apply(e) for e in write.index),
                frag=f"_c{i}_{j}",
                shape=atom.shape,
                epilogue=_warp_epilogue(tail, c.acc, m_axis.name, n_axis.name, sigma),
                m_guard=(offset.m.base(i), m_ext) if mask_m else None,
                n_guard=(offset.n.base(j), n_ext) if mask_n else None,
            )
        ]


class _ScalarOps(_AtomOps):
    """Scalar fma atom — plain ``Load``\\ s + an ``fma`` cell, the replicated-``epilogue`` sink."""

    def state(self, cells):
        """The scalar accumulator seeds. Gmem-direct (unstaged): none — the accumulators are seeded
        inside the reduce ``Loop`` (the dissolved fold ``Accum``\\ s + ``Loop.render``). **Staged**: a
        per-cell ``Init(acc__c{i}_{j} = 0)`` emitted here, **outside** the outer slab loop, so the
        carrier-less :func:`_scalar_drain` folds across every slab without re-seeding (the nested-loop
        accumulator-lifetime split — the scalar analogue of :meth:`_MmaOps.state` declaring the ``_c``
        fragments outside the mma K-loop)."""
        c, stage, inputs = self.c, self.stage, self.inputs
        mode, _bk = _scalar_stage_plan(c, stage, inputs)
        if mode == "gmem":
            return []
        return [Init(name=f"{c.acc}__c{i}_{j}", identity=_ADD.identity, dtype=F32) for i, j in cells]

    def reduce(self, cells, offset, mn):
        """The scalar contraction K-loop, through the shared :func:`_contract_kloop` skeleton with scalar
        leaf constructors: each register ROW reads its A operand once (a gmem ``Load`` — or the computed
        register-resident body, e.g. flash PV's ``P``), each COL its B ``Load`` once, and each ``(i, j)``
        cell folds ``acc__c{i}_{j} += b·a`` in a unit ``Loop`` (``Loop.render`` seeds the accumulators; the
        store reads them). A masked axis wraps its read in-bounds (``% extent``) and the overhanging store
        is guarded (:meth:`_ScalarOps.store`). Under a ``STAGE`` pin the eligible contraction routes to
        :func:`_scalar_staged` (smem operand slab, the one :func:`staged_kloop`); unstaged is this path."""
        c, stage, inputs = self.c, self.stage, self.inputs
        mode, bk_elems = _scalar_stage_plan(c, stage, inputs)
        if mode != "gmem":
            return _scalar_staged(c, inputs, cells, offset, mode, bk_elems)
        k_axis = c.k_axis
        m, n = mn
        mask_m, mask_n, m_ext, n_ext = m.mask, n.mask, m.ext, n.ext
        prot = _scalar_protected(c)
        m_name = m.name if m is not None else None
        n_name = n.name
        b_name, a_name = c.b_load.names[0], c.a_name

        def read_row(i):
            if m_name is None:
                return copy_cell(c.a_body, Sigma({}), f"__ar{i}", prot)
            bm = offset.m.base(i)
            return copy_cell(c.a_body, Sigma({m_name: BinaryExpr("%", bm, m_ext) if mask_m else bm}), f"__ar{i}", prot)

        def read_col(j):
            bn = offset.n.base(j)
            return copy_cell([c.b_load], Sigma({n_name: BinaryExpr("%", bn, n_ext) if mask_n else bn}), f"__bc{j}", prot)

        def contract(i, j):
            v = f"{c.acc}__v__c{i}_{j}"
            return [
                Assign(name=v, op=_MUL, args=(f"{b_name}__bc{j}", f"{a_name}__ar{i}")),
                Accum(name=f"{c.acc}__c{i}_{j}", value=v, op=_ADD, axes=(k_axis.name,)),
            ]

        def wrap(body):
            return [Loop(axis=k_axis, body=Body(tuple(body)), unroll=_unroll_inner(k_axis))]

        return _contract_kloop(c, cells, read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)

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
    accumulator decls + the contraction K-loop (the shared :func:`_contract_kloop` skeleton with the
    atom's leaf constructors). ``stage`` / ``inputs`` bind operand staging (both tiers stage an smem
    slab off it — the mma tier ``ldmatrix``-drains, the scalar tier plain-``Load``-drains)."""
    ops = _atom_ops(c, stage, inputs)
    return ops.state, ops.reduce


def store_sink(c: Contraction):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, mn)`` from the atom strategy
    (an mma ``RegStore`` / the replicated scalar ``epilogue`` tail). ``factorize(c, store=…)`` swaps it
    (a flash sink that bridges the accumulator into the streaming-softmax twist), reusing the shared
    ``reduce`` emission."""
    return _atom_ops(c).store
