"""Warp/mma tier — the exact tensor-core atom factorization (the expansion ``010_materialize``
folds in).

:func:`factorize_mma` expands an :class:`MmaContraction` into the ``Tile`` of ``RegFragment`` /
``LdmatrixLoad`` / ``MmaSyncPtx`` / ``RegStore`` — the four-way GRID/WARP/REGISTER/ATOM split,
the operand-staging decision (cp.async / TMA / gmem-direct), and the per-cell projection
epilogue. All atom geometry lives here, out of the node constructor (``005_contract`` emits only
the high-level ``MmaContraction``). Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import (
    EpilogueLoad,
    LdmatrixLoad,
    MmaContraction,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
    Sync,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Assign, Body, Load, Select, Stmt, StridedLoop, Write
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    TMA_SLAB_ALIGN,
    CtaTile,
    cp_async_barrier,
    cp_async_commit,
    cp_async_fill,
    cp_async_wait,
    slab_smem,
    tma_descriptor,
    tma_fill,
    tma_mbar_prologue,
)
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import Operand, Unit, atomize, grid_tile, register_tile, unit_tile
from deplodock.compiler.pipeline.pipeline import LoweringError


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


def _staged_inner_atom_loop(
    *, a_slab, b_slab, m_w, n_w, fm, fn, atom, bk_elems, tile_n, ki, reg_depth: int = 1, a_slab_off=None, b_slab_off=None
) -> list[Stmt]:
    """The inner atom-K drain shared by the cp.async and TMA staged paths: read the A/B slabs
    via ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. Slab-local indices — A[tile_m][bk_elems]
    (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent of which producer (cp.async /
    TMA) filled the (plain row-major, NONE-swizzle) slab.

    ``reg_depth == 1`` (default): one ``StridedLoop`` over the ``bk`` atom-K steps, ldmatrix-then-
    mma inline (the operand fragments ``_a{i}``/``_b{j}`` are reused every step). ``reg_depth >= 2``
    (the ``STAGE`` ``/p<n>`` smem→register double-buffer): the loop is **fully unrolled** into a
    software pipeline that ldmatrixes the next atom-K step into an alternate fragment slot
    (``_a{i}_s{slot}``) ``reg_depth-1`` steps ahead while the mma consumes the current slot —
    breaking the per-step WAR hazard on the operand fragments. Numerically identical to the
    inline form (same loads, same mmas, only reordered onto distinct registers).

    ``a_slab_off`` / ``b_slab_off`` (the gmem→smem ring, ``STAGE`` depth>1): the read SLOT row
    offset into a multi-slot slab — added to the A row / the B (K) row so the drain reads the
    ring slot the producer already filled, while a later chunk prefetches into another slot."""
    atom_m, atom_n, atom_k = atom.shape
    n_steps = bk_elems // atom_k

    def a_row(i):  # within-tile A row for register cell i (warp m_w · FM·atom_m + i·atom_m) + ring slot
        r = BinaryExpr("+", BinaryExpr("*", Var(m_w), Literal(fm * atom_m, "int")), Literal(i * atom_m, "int"))
        return BinaryExpr("+", a_slab_off, r) if a_slab_off is not None else r

    def b_col(j):  # within-tile B col for register cell j
        return BinaryExpr("+", BinaryExpr("*", Var(n_w), Literal(fn * atom_n, "int")), Literal(j * atom_n, "int"))

    def b_krow(kc):  # the B slab (K) row for atom-K col `kc` + ring slot
        return BinaryExpr("+", b_slab_off, kc) if b_slab_off is not None else kc

    if reg_depth < 2 or n_steps < 2:  # single-buffer: the original inline ldmatrix→mma loop
        inner: list[Stmt] = []
        for i in range(fm):  # A: row = a_slab_off + within-tile row (a_row), col = ki (the K position)
            inner.append(LdmatrixLoad(frag=f"_a{i}", src_buffer=a_slab, src_index=(a_row(i), Var(ki)), role="a", staged=True, ldm=bk_elems))
        for j in range(fn):  # B: row = b_slab_off + ki (the K position), col = within-tile col (b_col)
            inner.append(
                LdmatrixLoad(frag=f"_b{j}", src_buffer=b_slab, src_index=(b_krow(Var(ki)), b_col(j)), role="b", staged=True, ldm=tile_n)
            )
        for i in range(fm):
            for j in range(fn):
                inner.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
        return [
            StridedLoop(
                axis=Axis(name=ki, extent=bk_elems),
                start=Literal(0, "int"),
                step=Literal(atom_k, "int"),
                body=Body(tuple(inner)),
                unroll=True,
            )
        ]

    # reg_depth ≥ 2: the unrolled register double-buffer. ``slot = step % depth`` cycles the
    # fragment buffers; prefetch runs ``depth-1`` steps ahead of the consuming mma.
    depth = min(reg_depth, n_steps)
    kcol = lambda step: Literal(step * atom_k, "int")  # slab-local K col of atom-K step `step`  # noqa: E731

    def load_step(step: int) -> list[Stmt]:
        slot = step % depth
        out: list[Stmt] = []
        for i in range(fm):
            out.append(
                LdmatrixLoad(
                    frag=f"_a{i}_s{slot}", src_buffer=a_slab, src_index=(a_row(i), kcol(step)), role="a", staged=True, ldm=bk_elems
                )
            )
        for j in range(fn):
            out.append(
                LdmatrixLoad(
                    frag=f"_b{j}_s{slot}", src_buffer=b_slab, src_index=(b_krow(kcol(step)), b_col(j)), role="b", staged=True, ldm=tile_n
                )
            )
        return out

    def mma_step(step: int) -> list[Stmt]:
        slot = step % depth
        return [
            MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}_s{slot}", b_frag=f"_b{j}_s{slot}", shape=atom.shape, ab_dtype=atom.ab_dtype)
            for i in range(fm)
            for j in range(fn)
        ]

    stmts: list[Stmt] = []
    for s in range(depth - 1):  # prologue: prime the first depth-1 steps
        stmts += load_step(s)
    for step in range(n_steps):
        nxt = step + depth - 1
        if nxt < n_steps:  # prefetch depth-1 ahead, into the slot the mma below frees
            stmts += load_step(nxt)
        stmts += mma_step(step)
    return stmts


def _warp_staged_kloop(
    *,
    a_load,
    b_load,
    m_axis,
    n_axis,
    k_axis,
    m_b,
    n_b,
    m_w,
    n_w,
    wm,
    wn,
    fm,
    fn,
    atom,
    bk,
    tile_m,
    tile_n,
    slab_dtype,
    elem_bytes,
    mask_m,
    reg_depth=1,
    depth=1,
) -> tuple[list[Stmt], list[Stmt]]:
    """The warp tier's STAGED K loop (cp.async). ``depth == 1`` (single-buffer): one outer
    K-slab loop that cooperatively cp.async-fills both slabs, waits, and drains them. ``depth >= 2``
    (the ``STAGE`` ``d<depth>`` gmem→smem ring): the slab carries ``depth`` slots; a prologue
    fills the first ``depth-1`` chunks, then each loop step prefetches the chunk ``depth-1`` ahead
    into a free slot while computing the current one — so the cp.async copy overlaps the mma.
    ``CpAsyncWait(group=depth-1)`` keeps the ``depth-1`` prefetches in flight; the tail prefetch
    is clamped to the last chunk (in-bounds re-read into an unread slot) so the commit/wait stays
    uniform across all CTA threads (the barrier-under-mask invariant). Returns
    ``(slab_decls, [stmts])``. Composes with ``reg_depth`` (the inner smem→register double-buffer).

    ``mask_m`` (symbolic / non-divisible output rows): the A-slab fill clamps the gmem row
    in-bounds (``% M``) so an overhanging row reads a duplicate rather than past the buffer;
    the duplicate's contribution is discarded by the ``RegStore`` ``m_guard``."""
    atom_k = atom.shape[2]
    bk_elems = bk * atom_k  # K elements per slab (the BK chunk)
    a_slab, b_slab = "_a_smem", "_b_smem"
    n_threads = wm * wn * 32
    n_chunks = k_axis.extent.as_static() // bk_elems  # static K (the cp.async eligibility rule)
    ring = min(depth, n_chunks) if n_chunks >= 2 else 1  # ring slots; <2 chunks ⇒ nothing to prefetch
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

    def fill_chunk(k0_expr, slot_off_a, slot_off_b) -> list[Stmt]:
        """The cooperative A+B cp.async fill for the K-chunk at ``k0_expr``, written into the ring
        slot at row offsets ``slot_off_a`` (A) / ``slot_off_b`` (B). One commit closes the batch."""

        def a_gmem(row, col):  # slot[row][col] = A[row_base + row][k0 + col]
            m = BinaryExpr("+", row_base, row)
            if mask_m:  # clamp an overhanging row to the last valid one — its store is RegStore-guarded
                mext = _extent_expr(m_axis)
                m = TernaryExpr(cond=BinaryExpr("<", m, mext), if_true=m, if_false=BinaryExpr("-", mext, Literal(1, "int")))
            sig = Sigma({m_axis.name: m, k_axis.name: BinaryExpr("+", k0_expr, col)})
            return tuple(sig.apply(e) for e in a_load.index)

        def b_gmem(row, col):  # slot[row][col] = B[k0 + row][col_base + col]
            sig = Sigma({k_axis.name: BinaryExpr("+", k0_expr, row), n_axis.name: BinaryExpr("+", col_base, col)})
            return tuple(sig.apply(e) for e in b_load.index)

        out = cp_async_fill(
            slab=a_slab,
            slab_rows=tile_m,
            slab_cols=bk_elems,
            src=a_load.input,
            gmem_index=a_gmem,
            cta=cta,
            elem_bytes=elem_bytes,
            name="a",
            row_offset=slot_off_a,
        )
        out += cp_async_fill(
            slab=b_slab,
            slab_rows=bk_elems,
            slab_cols=tile_n,
            src=b_load.input,
            gmem_index=b_gmem,
            cta=cta,
            elem_bytes=elem_bytes,
            name="b",
            row_offset=slot_off_b,
        )
        return out

    if ring == 1:
        # Single buffer: fill → wait(0) → drain → trailing Sync (so the next fill can't clobber
        # the slab while a lagging thread still reads it).
        fills = fill_chunk(Var(k0), None, None) + cp_async_barrier()
        inner = _staged_inner_atom_loop(
            a_slab=a_slab,
            b_slab=b_slab,
            m_w=m_w,
            n_w=n_w,
            fm=fm,
            fn=fn,
            atom=atom,
            bk_elems=bk_elems,
            tile_n=tile_n,
            ki=ki,
            reg_depth=reg_depth,
        )
        outer = StridedLoop(
            axis=Axis(name=k0, extent=k_axis.extent.as_static()),
            start=Literal(0, "int"),
            step=Literal(bk_elems, "int"),
            body=Body((*fills, *inner, Sync())),
            unroll=False,
        )
        slab_decls = [slab_smem(a_slab, tile_m, bk_elems, slab_dtype), slab_smem(b_slab, bk_elems, tile_n, slab_dtype)]
        return slab_decls, [outer]

    # depth ≥ 2 gmem→smem ring. Slot row offsets: A by tile_m rows / slot, B by bk_elems rows.
    K = k_axis.extent.as_static()
    last_k0 = (n_chunks - 1) * bk_elems
    a_slot = lambda s: BinaryExpr("*", s, Literal(tile_m, "int"))  # noqa: E731
    b_slot = lambda s: BinaryExpr("*", s, Literal(bk_elems, "int"))  # noqa: E731

    prologue: list[Stmt] = []
    for s in range(ring - 1):  # prime the first ring-1 chunks into slots 0..ring-2
        prologue += fill_chunk(Literal(s * bk_elems, "int"), a_slot(Literal(s, "int")), b_slot(Literal(s, "int")))
        prologue += cp_async_commit()

    # Main loop over k0 = 0, bk, …, (n_chunks-1)·bk. i = k0/bk_elems.
    i_expr = BinaryExpr("/", Var(k0), Literal(bk_elems, "int"))
    pref_slot = BinaryExpr("%", BinaryExpr("+", i_expr, Literal(ring - 1, "int")), Literal(ring, "int"))
    read_slot = BinaryExpr("%", i_expr, Literal(ring, "int"))
    k0_next = BinaryExpr("+", Var(k0), Literal((ring - 1) * bk_elems, "int"))
    k0_pref = TernaryExpr(cond=BinaryExpr("<", k0_next, Literal(K, "int")), if_true=k0_next, if_false=Literal(last_k0, "int"))

    body: list[Stmt] = []
    body += fill_chunk(k0_pref, a_slot(pref_slot), b_slot(pref_slot))  # prefetch depth-1 ahead (clamped tail)
    body += cp_async_commit()
    body += cp_async_wait(ring - 1)  # drain chunk i (keep the ring-1 prefetches in flight) + Sync
    body += _staged_inner_atom_loop(
        a_slab=a_slab,
        b_slab=b_slab,
        m_w=m_w,
        n_w=n_w,
        fm=fm,
        fn=fn,
        atom=atom,
        bk_elems=bk_elems,
        tile_n=tile_n,
        ki=ki,
        reg_depth=reg_depth,
        a_slab_off=a_slot(read_slot),
        b_slab_off=b_slot(read_slot),
    )
    body.append(Sync())  # done reading slot i before a later chunk prefetches into it
    outer = StridedLoop(
        axis=Axis(name=k0, extent=K), start=Literal(0, "int"), step=Literal(bk_elems, "int"), body=Body(tuple(body)), unroll=False
    )
    slab_decls = [slab_smem(a_slab, ring * tile_m, bk_elems, slab_dtype), slab_smem(b_slab, ring * bk_elems, tile_n, slab_dtype)]
    return slab_decls, [*prologue, outer]


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
    *, a_load, b_load, k_axis, m_b, n_b, m_w, n_w, wm, wn, fm, fn, atom, bk, tile_m, tile_n, slab_dtype, elem_bytes, reg_depth=1
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
    inner = _staged_inner_atom_loop(
        a_slab=a_slab,
        b_slab=b_slab,
        m_w=m_w,
        n_w=n_w,
        fm=fm,
        fn=fn,
        atom=atom,
        bk_elems=bk_elems,
        tile_n=tile_n,
        ki=ki,
        reg_depth=reg_depth,
    )
    outer = StridedLoop(
        axis=Axis(name=k0, extent=k_axis.extent.as_static()),
        start=Literal(0, "int"),
        step=Literal(bk_elems, "int"),
        body=Body((*fill, *inner, Sync())),
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


class AtomUnit(Unit):
    """The mma-atom leaf for the warp tier — wraps the tensor-core fragment / staging logic so
    the generic tiling layer assembles it. The staging decision (gmem-direct / cp.async / TMA)
    + fragment naming are computed up front: they drive BOTH the state decls and the K-loop."""

    def __init__(self, mma: MmaContraction):
        wt = mma.warp_tile
        atom = wt.atom
        atom_m, atom_n, atom_k = atom.shape
        self.mma, self.wt, self.atom = mma, wt, atom
        self.atom_m, self.atom_n, self.atom_k = atom_m, atom_n, atom_k
        self.wm, self.wn = wt.warps
        self.fm, self.fn = wt.reg
        self.m_axis, self.n_axis, self.k_axis = mma.m_axis, mma.n_axis, mma.k_axis
        self.a_load, self.b_load, self.b_trans, self.acc = mma.a_load, mma.b_load, mma.b_trans, mma.acc
        self.stage = mma.stage
        self.pre: list[Stmt] = []
        self.tail = list(mma.epilogue)
        self.write = next(s for s in self.tail if isinstance(s, Write))
        self.tile_m, self.tile_n = wt.tile_m, wt.tile_n
        self.mask_m = not (self.m_axis.extent.is_static and self.m_axis.extent.as_static() % self.tile_m == 0)
        self.mask_n = not (self.n_axis.extent.is_static and self.n_axis.extent.as_static() % self.tile_n == 0)
        self.m_b, self.n_b = f"{self.m_axis.name}_wb", f"{self.n_axis.name}_wb"
        self.m_w, self.n_w = f"{self.m_axis.name}_ww", f"{self.n_axis.name}_ww"
        # The operand-staging decision (TMA > cp.async > gmem-direct) drives the operand-fragment
        # naming + the K-loop, so it is computed once here and read by state_decls + reduce_region.
        a_nbytes = atom.operand_dtype("a").nbytes
        self.a_nbytes = a_nbytes
        self.tma_ok = _can_stage_warp_tma(
            self.stage, self.k_axis, self.n_axis, self.tile_n, wt.bk, atom_k, a_nbytes, self.mask_n, self.b_trans
        )
        self.cp_ok = (not self.tma_ok) and _can_stage_warp(
            self.stage, self.k_axis, self.tile_m, self.tile_n, wt.bk, atom_k, self.mask_m, self.mask_n, self.b_trans
        )
        self.reg_depth = min(self.stage.reg_depth, wt.bk) if (self.stage is not None and (self.tma_ok or self.cp_ok)) else 1
        _slot_bytes = (self.tile_m * wt.bk * atom_k + wt.bk * atom_k * self.tile_n) * a_nbytes
        self.gmem_depth = min(self.stage.depth, max(1, (48 * 1024) // _slot_bytes)) if (self.stage is not None and self.cp_ok) else 1
        rd = self.reg_depth
        self.a_frags = [f"_a{i}_s{s}" for i in range(self.fm) for s in range(rd)] if rd >= 2 else [f"_a{i}" for i in range(self.fm)]
        self.b_frags = [f"_b{j}_s{s}" for j in range(self.fn) for s in range(rd)] if rd >= 2 else [f"_b{j}" for j in range(self.fn)]

    def state_decls(self, cells) -> list[Stmt]:
        atom = self.atom
        decls: list[Stmt] = []
        for name in self.a_frags:
            decls.append(RegFragment(name=name, role="a", shape=atom.shape, dtype=atom.operand_dtype("a")))
        for name in self.b_frags:
            decls.append(RegFragment(name=name, role="b", shape=atom.shape, dtype=atom.operand_dtype("b")))
        for i in range(self.fm):
            for j in range(self.fn):
                decls.append(RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c")))
        return decls

    def operands(self) -> list[Operand]:
        return [Operand(self.a_load, "a", frozenset({self.m_axis.name})), Operand(self.b_load, "b", frozenset({self.n_axis.name}))]

    def reduce_region(self, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
        atom, wt, atom_k = self.atom, self.wt, self.atom_k
        if self.tma_ok:
            from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

            slab_dtype = cuda_name(atom.operand_dtype("a"))
            return _warp_tma_staged_kloop(
                a_load=self.a_load,
                b_load=self.b_load,
                k_axis=self.k_axis,
                m_b=self.m_b,
                n_b=self.n_b,
                m_w=self.m_w,
                n_w=self.n_w,
                wm=self.wm,
                wn=self.wn,
                fm=self.fm,
                fn=self.fn,
                atom=atom,
                bk=wt.bk,
                tile_m=self.tile_m,
                tile_n=self.tile_n,
                slab_dtype=slab_dtype,
                elem_bytes=self.a_nbytes,
                reg_depth=self.reg_depth,
            )
        if self.cp_ok:
            from deplodock.compiler.backend.cuda.dtype import cuda_name  # noqa: PLC0415

            slab_dtype = cuda_name(atom.operand_dtype("a"))
            return _warp_staged_kloop(
                a_load=self.a_load,
                b_load=self.b_load,
                m_axis=self.m_axis,
                n_axis=self.n_axis,
                k_axis=self.k_axis,
                m_b=self.m_b,
                n_b=self.n_b,
                m_w=self.m_w,
                n_w=self.n_w,
                wm=self.wm,
                wn=self.wn,
                fm=self.fm,
                fn=self.fn,
                atom=atom,
                bk=wt.bk,
                tile_m=self.tile_m,
                tile_n=self.tile_n,
                slab_dtype=slab_dtype,
                elem_bytes=self.a_nbytes,
                mask_m=self.mask_m,
                reg_depth=self.reg_depth,
                depth=self.gmem_depth,
            )
        # Gmem-direct. A symbolic / non-divisible K zero-fills the masked-K tail via the ``k_zero``
        # helper variants — a duplicate read would corrupt the reduction (unlike a masked M/N row,
        # whose store is just guarded). Transposed-B has no gmem-direct K zero-fill helper, so it bails.
        mask_m, mask_n, m_ext, n_ext = masks
        k_static = self.k_axis.extent.is_static
        if not k_static and self.b_trans:
            raise LoweringError("warp tier: transposed-B symbolic-K mma not supported (no gmem-direct K zero-fill)")
        k_zero = None if k_static else (Var(self.k_axis.name), _extent_expr(self.k_axis))
        chain: list[Stmt] = []
        for i in range(self.fm):
            idx = tuple(Sigma({self.m_axis.name: offset.base("m", i)}).apply(e) for e in self.a_load.index)
            guard = (offset.base("m", i), m_ext) if mask_m else None
            chain.append(
                LdmatrixLoad(
                    frag=f"_a{i}", src_buffer=self.a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero
                )
            )
        for j in range(self.fn):
            idx = tuple(Sigma({self.n_axis.name: offset.base("n", j)}).apply(e) for e in self.b_load.index)
            guard = (offset.base("n", j), n_ext) if mask_n else None
            chain.append(
                LdmatrixLoad(
                    frag=f"_b{j}",
                    src_buffer=self.b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=self.b_trans,
                    gmem_guard=guard,
                    k_zero=k_zero,
                )
            )
        for i in range(self.fm):
            for j in range(self.fn):
                chain.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
        kstmts = [
            StridedLoop(axis=self.k_axis, start=Literal(0, "int"), step=Literal(atom_k, "int"), body=Body(tuple(chain)), unroll=k_static)
        ]
        return [], kstmts

    def store(self, i, j, offset, masks) -> list[Stmt]:
        mask_m, mask_n, m_ext, n_ext = masks
        sigma = Sigma({self.m_axis.name: offset.base("m", i), self.n_axis.name: offset.base("n", j)})
        return [
            RegStore(
                dst_buffer=self.write.output,
                dst_index=tuple(sigma.apply(e) for e in self.write.index),
                frag=f"_c{i}_{j}",
                shape=self.atom.shape,
                epilogue=_warp_epilogue(self.pre, self.tail, self.acc, self.m_axis.name, self.n_axis.name, sigma),
                m_guard=(offset.base("m", i), m_ext) if mask_m else None,
                n_guard=(offset.base("n", j), n_ext) if mask_n else None,
            )
        ]


def factorize_mma(mma: MmaContraction) -> Tile:
    """Expand a high-level :class:`MmaContraction` into the warp-tier ``Tile`` via the composable
    tiling layer: the mma :class:`AtomUnit` leaf, tiled four ways
    ``grid_tile(unit_tile(register_tile(atomize(...))))`` (GRID block / UNIT=warp / REGISTER / ATOM;
    the atom-lane offset decoded at render). The unit owns the K-loop + operand staging
    (gmem-direct / cp.async / TMA); the layer owns the offset, the axes, and the splice."""
    unit = AtomUnit(mma)
    masks = (unit.mask_m, unit.mask_n, _extent_expr(unit.m_axis), _extent_expr(unit.n_axis))
    t = atomize(unit, unit.atom_m, unit.atom_n)
    t = register_tile(t, unit.fm, unit.fn)
    t = unit_tile(t, unit.wm, unit.wn, unit.m_w, unit.n_w)
    return grid_tile(
        t,
        masks,
        m_axis=unit.m_axis,
        n_axis=unit.n_axis,
        m_b=unit.m_b,
        n_b=unit.n_b,
        tile_m=unit.tile_m,
        tile_n=unit.tile_n,
        block_threads=unit.wt.block_threads,
        lanes=unit.atom.lanes,
    )
