"""The one factorizer — the single ``TileOp``-root emitter, all tiers in one place.

:func:`factorize` is the node-kind dispatcher ``010_materialize`` calls once per kernel: it reads the
structural node off ``tile.op`` (its kind + role + reduce plan) and routes to one of three tiers, all
here — :func:`_factorize_contraction` (a tiled :class:`~...ir.Contraction`), :func:`_factorize_reduce`
(a cooperative / ILP ``PLANAR`` / ``TWISTED`` reduce), or the inline **scalar tier** (a pointwise
``Map`` / trivial-plan reduction: ``lower(op)`` + :func:`with_store`, one thread per output cell).

Both atoms of a :class:`~...ir.Contraction` (a tensor-core :class:`AtomKind` or the scalar
:class:`ScalarAtom`) expand through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``). :func:`_factorize_contraction` reads the tiling **geometry
straight off the** ``Contraction`` **node** (``tile_m`` / ``mask_m`` / ``m_b`` / ``m_uvar`` /
``units_m`` / ``block_threads`` / …, derived there from the ``tile`` schedule + the output axes) and
splices two codegen halves into ``grid_tile``:

- :func:`reduce_codegen` — the reusable, **sink-agnostic** ``(state_decls, reduce_region)``: the
  accumulator/operand declarations + the contraction K-loop. The K-loop shares **one skeleton**,
  :func:`_contract_kloop` (read each register row's A + each col's B once, contract every ``(row,
  col)`` pair, wrap in the reduce loop); the atom supplies only the leaf constructors — the mma pair
  (:func:`_mma_state` / :func:`_mma_reduce`: ``ldmatrix`` fragment reads + ``mma.sync``) vs the scalar
  pair (:func:`_scalar_state` / :func:`_scalar_reduce`: plain ``Load``\\ s + an ``fma`` cell). Under a
  :class:`Stage` the operands instead ride an smem slab (cp.async / TMA fill + drain —
  :func:`_warp_staged_kloop` / :func:`_warp_tma_staged_kloop` / :func:`_scalar_staged_kloop`, a pure
  bit-identical perf transform). Both leave the accumulator (mma ``_c{i}_{j}`` fragments / scalar
  ``acc__c{i}_{j}``) for the sink.
- the **sink** ``store(i, j, offset, masks)`` — the per-cell consumer of that accumulator.
  :func:`store_sink` is the default **matmul** sink (an mma ``RegStore`` / the replicated scalar
  ``epilogue`` tail, projecting to the output). ``_factorize_contraction(c, store=…)`` swaps it — the
  flash inner QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist, reusing
  the same :func:`reduce_codegen`.

The cooperative / ILP reduce tier (:func:`_factorize_reduce` + the shared-row staging helpers) folds
the reduce axis ``coop`` ways across threads and ``reg`` ways across per-thread accumulators, then the
REG-tree fold, the cross-thread combine (``_combine.emit_combine``), and the projection — carrier-
generic (a contraction is the degenerate carrier of its additive fold).

The warp-tier smem operand-staging pipeline lives in ``_stage.py`` (the transport primitives) + the
``_warp[_tma]_staged_kloop`` helpers here; it is driven off the node's ``STAGE`` codec →
:class:`~...schedule.Stage` (``d<depth>`` gmem→smem ring · ``sync``/``cp``/``tma`` transport ·
``p<n>`` smem→register double-buffer). The **scalar** contraction tier stays gmem-direct; the fused
norm→linear **shared-row** prologue (:func:`_factorize_reduce`) is a *distinct* reduce-tier smem row,
not this operand slab (a full unification is a follow-up). Leading ``_`` so the pass loader skips this
module."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from deplodock.compiler.backend.cuda.dtype import cuda_name
from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.atom import AtomKind
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import (
    EpilogueLoad,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
    Sync,
)
from deplodock.compiler.ir.schedule import Stage
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Select, SelectBranch, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Contraction
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_plan
from deplodock.compiler.pipeline.passes.lowering.kernel._combine import emit_combine
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import copy_cell
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._stage import (
    TMA_SLAB_ALIGN,
    CtaTile,
    cp_async_barrier,
    cp_async_commit,
    cp_async_fill,
    cp_async_wait,
    slab_smem,
    sync_row_fill,
    tma_descriptor,
    tma_fill,
    tma_mbar_prologue,
)
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile

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
    *, a_slab, b_slab, m_w, n_w, fm, fn, atom, bk_elems, tile_n, ki, reg_depth: int = 1, a_slab_off=None, b_slab_off=None
) -> list[Stmt]:
    """The inner atom-K drain shared by the cp.async and TMA staged paths: read the A/B slabs via
    ``LdmatrixLoad(staged=True)`` + ``MmaSyncPtx``. Slab-local indices — A[tile_m][bk_elems]
    (ldm=bk_elems), B[bk_elems][tile_n] (ldm=tile_n) — independent of which producer filled the
    (plain row-major, NONE-swizzle) slab.

    ``reg_depth == 1`` (default): one ``StridedLoop`` over the ``bk`` atom-K steps, ldmatrix-then-mma
    inline (the operand fragments ``_a{i}``/``_b{j}`` reused every step). ``reg_depth >= 2`` (the
    ``STAGE`` ``/p<n>`` smem→register double-buffer): the loop is **fully unrolled** into a software
    pipeline that ldmatrixes the next atom-K step into an alternate fragment slot (``_a{i}_s{slot}``)
    ``reg_depth-1`` steps ahead while the mma consumes the current slot — breaking the per-step WAR
    hazard on the operand fragments. Numerically identical to the inline form.

    ``a_slab_off`` / ``b_slab_off`` (the gmem→smem ring, ``STAGE`` depth>1): the read SLOT row offset
    into a multi-slot slab — added to the A row / the B (K) row so the drain reads the ring slot the
    producer already filled, while a later chunk prefetches into another slot."""
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
    """The warp tier's STAGED K loop (cp.async). ``depth == 1`` (single-buffer): one outer K-slab
    loop that cooperatively cp.async-fills both slabs, waits, and drains them. ``depth >= 2`` (the
    ``STAGE`` ``d<depth>`` gmem→smem ring): the slab carries ``depth`` slots; a prologue fills the
    first ``depth-1`` chunks, then each loop step prefetches the chunk ``depth-1`` ahead into a free
    slot while computing the current one — so the cp.async copy overlaps the mma.
    ``CpAsyncWait(group=depth-1)`` keeps the ``depth-1`` prefetches in flight; the tail prefetch is
    clamped to the last chunk (in-bounds re-read into an unread slot) so the commit/wait stays
    uniform across all CTA threads (the barrier-under-mask invariant). Returns ``(slab_decls,
    [stmts])``. Composes with ``reg_depth`` (the inner smem→register double-buffer).

    ``mask_m`` (symbolic / non-divisible output rows): the A-slab fill clamps the gmem row in-bounds
    (bounded by the runtime extent) so an overhanging row reads a duplicate rather than past the
    buffer; the duplicate's contribution is discarded by the ``RegStore`` ``m_guard``."""
    atom_k = atom.shape[2]
    bk_elems = bk * atom_k  # K elements per slab (the BK chunk)
    a_slab, b_slab = "_a_smem", "_b_smem"
    n_threads = wm * wn * 32
    n_chunks = k_axis.extent.as_static() // bk_elems  # static K (the cp.async eligibility rule)
    ring = min(depth, n_chunks) if n_chunks >= 2 else 1  # ring slots; <2 chunks ⇒ nothing to prefetch
    row_base = BinaryExpr("*", Var(m_b), Literal(tile_m, "int"))  # CTA tile base row
    col_base = BinaryExpr("*", Var(n_b), Literal(tile_n, "int"))  # CTA tile base col
    # intra-CTA linear thread id from the decoded warp / lane axis vars (the innermost block_threads
    # of the [m_b, n_b, m_w, n_w, _lane] tile) — never a raw threadIdx.x.
    linear_tid = BinaryExpr(
        "+", BinaryExpr("*", BinaryExpr("+", BinaryExpr("*", Var(m_w), Literal(wn, "int")), Var(n_w)), Literal(32, "int")), Var("_lane")
    )
    cta = CtaTile(row_base=row_base, col_base=col_base, linear_tid=linear_tid, n_threads=n_threads)

    k0 = "_ks"  # the outer K-slab offset var
    ki = "_ki"  # the inner atom-K offset within the slab

    def fill_chunk(k0_expr, slot_off_a, slot_off_b) -> list[Stmt]:
        """The cooperative A+B cp.async fill for the K-chunk at ``k0_expr``, written into the ring
        slot at row offsets ``slot_off_a`` (A) / ``slot_off_b`` (B)."""

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
        # Single buffer: fill → wait(0) → drain → trailing Sync (so the next fill can't clobber the
        # slab while a lagging thread still reads it).
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


def _warp_tma_staged_kloop(
    *, a_load, b_load, k_axis, m_b, n_b, m_w, n_w, wm, wn, fm, fn, atom, bk, tile_m, tile_n, slab_dtype, elem_bytes, reg_depth=1
) -> tuple[list[Stmt], list[Stmt]]:
    """The warp tier's STAGED K loop via **TMA** (single-buffer): declare the A/B ``CUtensorMap``
    descriptors + the destination slabs + one mbarrier, then an outer K-slab loop where the issuer
    thread box-copies both operand tiles (``cp.async.bulk.tensor``) and every thread waits on the
    mbarrier parity before the shared staged-``LdmatrixLoad`` + mma drain. A masked M needs no fill
    clamp — TMA zero-fills the box overhang past the descriptor's runtime globalDim (the RegStore
    still guards the masked-row stores)."""
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


def _mma_stage_plan(c: Contraction, stage: Stage | None) -> tuple[str, int, int]:
    """The operand-staging decision for the mma contraction ``c`` under ``stage`` — read once and
    shared by :func:`_mma_state` (which slots the fragments) and :func:`_mma_reduce` (which emits the
    K-loop). Returns ``(mode, gmem_depth, reg_depth)`` where ``mode`` is ``"tma"`` / ``"cp"`` /
    ``"gmem"`` (TMA > cp.async > gmem-direct). ``gmem`` forces both depths to 1 (no slab)."""
    if stage is None or c.a_computed:
        return "gmem", 1, 1
    atom = c.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = c.tile.bk
    tma_ok = _can_stage_warp_tma(stage, c.k_axis, c.n_axis, c.tile_n, bk, atom.atom_k, a_nbytes, c.mask_n, c.b_trans)
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.k_axis, c.tile_m, c.tile_n, bk, atom.atom_k, c.mask_m, c.mask_n, c.b_trans)
    if not (tma_ok or cp_ok):
        return "gmem", 1, 1
    reg_depth = min(stage.reg_depth, bk)
    if tma_ok:
        return "tma", 1, reg_depth
    slot_bytes = (c.tile_m * bk * atom.atom_k + bk * atom.atom_k * c.tile_n) * a_nbytes
    gmem_depth = min(stage.depth, max(1, (48 * 1024) // slot_bytes))
    return "cp", gmem_depth, reg_depth


def _contract_kloop(c, cells, offset, masks, *, read_row, read_col, contract, wrap):
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


def _mma_state(c: Contraction, stage: Stage | None, cells) -> list[Stmt]:
    """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
    one ``_c`` accumulator per cell (held across the K-loop). A staged ``reg_depth >= 2`` slots the
    operand fragments (``_a{i}_s{slot}``) for the smem→register double-buffer's ping-pong."""
    atom = c.atom
    _mode, _gmem_depth, reg_depth = _mma_stage_plan(c, stage)
    a_frags = [f"_a{i}_s{s}" for i in range(c.reg_m) for s in range(reg_depth)] if reg_depth >= 2 else [f"_a{i}" for i in range(c.reg_m)]
    b_frags = [f"_b{j}_s{s}" for j in range(c.reg_n) for s in range(reg_depth)] if reg_depth >= 2 else [f"_b{j}" for j in range(c.reg_n)]
    decls: list[Stmt] = []
    for name in a_frags:
        decls.append(RegFragment(name=name, role="a", shape=atom.shape, dtype=atom.operand_dtype("a")))
    for name in b_frags:
        decls.append(RegFragment(name=name, role="b", shape=atom.shape, dtype=atom.operand_dtype("b")))
    for i in range(c.reg_m):
        for j in range(c.reg_n):
            decls.append(RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c")))
    return decls


def _mma_reduce(c: Contraction, stage: Stage | None, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
    """The mma K-loop, dispatched on the staging decision (TMA > cp.async > gmem-direct). Staged:
    cooperatively fill an smem slab then ``ldmatrix``-drain it (:func:`_warp_tma_staged_kloop` /
    :func:`_warp_staged_kloop`) — a pure perf transform, bit-identical to gmem-direct. Gmem-direct:
    ``ldmatrix`` each operand fragment straight from gmem, then ``mma.sync`` every cell (a symbolic /
    non-divisible K zero-fills the masked-K tail via the ``k_zero`` helper variants — canonical and
    transposed-B both have gmem-direct K zero-fill helpers)."""
    atom = c.atom
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    assert not c.a_computed, "mma tier: register-resident A operand (a computed flash-PV fragment feed) is a scalar-tier-only capability"
    a_load, b_load, b_trans = c.a_operand, c.b_load, c.b_trans
    mode, gmem_depth, reg_depth = _mma_stage_plan(c, stage)
    if mode != "gmem":
        slab_dtype = cuda_name(atom.operand_dtype("a"))
        elem_bytes = atom.operand_dtype("a").nbytes
        common = dict(
            a_load=a_load,
            b_load=b_load,
            k_axis=k_axis,
            m_b=c.m_b,
            n_b=c.n_b,
            m_w=c.m_uvar,
            n_w=c.n_uvar,
            wm=c.units_m,
            wn=c.units_n,
            fm=c.reg_m,
            fn=c.reg_n,
            atom=atom,
            bk=c.tile.bk,
            tile_m=c.tile_m,
            tile_n=c.tile_n,
            slab_dtype=slab_dtype,
            elem_bytes=elem_bytes,
            reg_depth=reg_depth,
        )
        if mode == "tma":
            return _warp_tma_staged_kloop(**common)
        return _warp_staged_kloop(m_axis=m_axis, n_axis=n_axis, mask_m=c.mask_m, depth=gmem_depth, **common)
    mask_m, mask_n, m_ext, n_ext = masks
    k_static = k_axis.extent.is_static
    k_zero = None if k_static else (Var(k_axis.name), _extent_expr(k_axis))

    def read_row(i):
        idx = tuple(Sigma({m_axis.name: offset.base("m", i)}).apply(e) for e in a_load.index)
        guard = (offset.base("m", i), m_ext) if mask_m else None
        return [
            LdmatrixLoad(frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero)
        ]

    def read_col(j):
        idx = tuple(Sigma({n_axis.name: offset.base("n", j)}).apply(e) for e in b_load.index)
        guard = (offset.base("n", j), n_ext) if mask_n else None
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

    return _contract_kloop(c, cells, offset, masks, read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)


def _mma_store(c: Contraction, i: int, j: int, offset, masks) -> list[Stmt]:
    """Store cell ``(i, j)``'s ``_c`` fragment to the output, folding the projection ``tail`` into a
    :class:`RegEpilogue` and guarding overhanging M/N rows."""
    atom = c.atom
    m_axis, n_axis = c.m_axis, c.n_axis
    mask_m, mask_n, m_ext, n_ext = masks
    tail = list(c.epilogue)
    write = next(s for s in tail if isinstance(s, Write))
    sigma = Sigma({m_axis.name: offset.base("m", i), n_axis.name: offset.base("n", j)})
    return [
        RegStore(
            dst_buffer=write.output,
            dst_index=tuple(sigma.apply(e) for e in write.index),
            frag=f"_c{i}_{j}",
            shape=atom.shape,
            epilogue=_warp_epilogue(tail, c.acc, m_axis.name, n_axis.name, sigma),
            m_guard=(offset.base("m", i), m_ext) if mask_m else None,
            n_guard=(offset.base("n", j), n_ext) if mask_n else None,
        )
    ]


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


def _scalar_sigma(m_axis, n_axis, offset, i: int, j: int, masks) -> Sigma:
    """σ mapping the output axes to register cell ``(i, j)``'s real coordinate (the offset's
    block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
    mask_m, mask_n, m_ext, n_ext = masks
    smap: dict = {}
    if m_axis is not None:
        bm = offset.base("m", i)
        smap[m_axis.name] = BinaryExpr("%", bm, m_ext) if mask_m else bm
    bn = offset.base("n", j)
    smap[n_axis.name] = BinaryExpr("%", bn, n_ext) if mask_n else bn
    return Sigma(smap)


def _scalar_bound(m_axis, n_axis, offset, i: int, j: int, masks):
    """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` for each masked axis (anded),
    or ``None`` when nothing overhangs."""
    mask_m, mask_n, m_ext, n_ext = masks
    conds = []
    if mask_m and m_axis is not None:
        conds.append(BinaryExpr("<", offset.base("m", i), m_ext))
    if mask_n:
        conds.append(BinaryExpr("<", offset.base("n", j), n_ext))
    if not conds:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = BinaryExpr("&&", cond, c)
    return cond


def _scalar_protected(c: Contraction) -> frozenset[str]:
    """The shared iteration coordinates — the block / unit / loop / extent vars excluded from the
    per-cell SSA rename (everything else is suffixed ``__c{i}_{j}`` so each cell owns its names)."""
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    prot = {c.n_b, c.n_uvar, k_axis.name}
    if m_axis is not None:
        prot |= {c.m_b, c.m_uvar}
    axes_for_ext = [n_axis, k_axis, *c.lead_axes]
    if m_axis is not None:
        axes_for_ext.append(m_axis)
    for a in c.lead_axes:
        prot.add(a.name)
    for a in axes_for_ext:
        prot |= set(_extent_expr(a).free_vars())
    return frozenset(prot)


def _scalar_state(c: Contraction, stage: Stage | None, inputs, cells) -> list[Stmt]:
    """The scalar accumulator seeds. Gmem-direct (unstaged): none — the accumulators are seeded
    inside the reduce ``Loop`` (the dissolved fold ``Accum``\\ s + ``Loop.render``). **Staged**: a
    per-cell ``Init(acc__c{i}_{j} = 0)`` emitted here, **outside** the outer slab loop, so the
    carrier-less :func:`_scalar_drain` folds across every slab without re-seeding (the nested-loop
    accumulator-lifetime split — the scalar analogue of ``_mma_state`` declaring the ``_c`` fragments
    outside the mma K-loop)."""
    mode, _bk = _scalar_stage_plan(c, stage, inputs)
    if mode == "gmem":
        return []
    return [Init(name=f"{c.acc}__c{i}_{j}", identity=_ADD.identity, dtype=F32) for i, j in cells]


def _scalar_reduce(c: Contraction, stage: Stage | None, inputs, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
    """The scalar contraction K-loop, through the shared :func:`_contract_kloop` skeleton with scalar
    leaf constructors: each register ROW reads its A operand once (a gmem ``Load`` — or the computed
    register-resident body, e.g. flash PV's ``P``), each COL its B ``Load`` once, and each ``(i, j)``
    cell folds ``acc__c{i}_{j} += b·a`` in a unit ``Loop`` (``Loop.render`` seeds the accumulators; the
    store reads them). A masked axis wraps its read in-bounds (``% extent``) and the overhanging store
    is guarded (:func:`_scalar_store`). Under a ``STAGE`` pin the eligible contraction routes to
    :func:`_scalar_staged_kloop` (smem operand slab); unstaged is this path."""
    mode, bk_elems = _scalar_stage_plan(c, stage, inputs)
    if mode != "gmem":
        return _scalar_staged_kloop(c, inputs, cells, offset, masks, mode, bk_elems)
    k_axis = c.k_axis
    mask_m, mask_n, m_ext, n_ext = masks
    prot = _scalar_protected(c)
    m_name = c.m_axis.name if c.m_axis is not None else None
    n_name = c.n_axis.name
    b_name, a_name = c.b_load.names[0], c.a_name

    def read_row(i):
        if m_name is None:
            return copy_cell(c.a_body, Sigma({}), f"__ar{i}", prot)
        bm = offset.base("m", i)
        return copy_cell(c.a_body, Sigma({m_name: BinaryExpr("%", bm, m_ext) if mask_m else bm}), f"__ar{i}", prot)

    def read_col(j):
        bn = offset.base("n", j)
        return copy_cell([c.b_load], Sigma({n_name: BinaryExpr("%", bn, n_ext) if mask_n else bn}), f"__bc{j}", prot)

    def contract(i, j):
        v = f"{c.acc}__v__c{i}_{j}"
        return [
            Assign(name=v, op=_MUL, args=(f"{b_name}__bc{j}", f"{a_name}__ar{i}")),
            Accum(name=f"{c.acc}__c{i}_{j}", value=v, op=_ADD, axes=(k_axis.name,)),
        ]

    def wrap(body):
        return [Loop(axis=k_axis, body=Body(tuple(body)), unroll=_unroll_inner(k_axis))]

    return _contract_kloop(c, cells, offset, masks, read_row=read_row, read_col=read_col, contract=contract, wrap=wrap)


def _scalar_store(c: Contraction, i: int, j: int, offset, masks) -> list[Stmt]:
    """Replicate the projection ``tail`` for cell ``(i, j)`` — σ-offset, suffix the SSA names, guard
    the (overhanging) write, dedup shared operand loads."""
    m_axis, n_axis = c.m_axis, c.n_axis
    sigma = _scalar_sigma(m_axis, n_axis, offset, i, j, masks)
    cell = copy_cell(c.epilogue, sigma, f"__c{i}_{j}", _scalar_protected(c))
    cell = _guard_writes(cell, _scalar_bound(m_axis, n_axis, offset, i, j, masks))
    return _dedup_loads(cell)


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
    cap = (48 * 1024) // (max(1, c.tile_m + c.tile_n) * elem_bytes)
    bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and K % v == 0), 0)
    if bk_elems < 4:
        return "gmem", 0
    return ("tma" if stage.transport == "tma" else "cp"), bk_elems


def _scalar_drain(c: Contraction, cells, offset, a_slab: str, b_slab: str, ki: str, bk_elems: int, row_base, col_base) -> Loop:
    """The inner slab-drain reduce loop ``for ki: b = b_slab[ki, n_local]; a = a_slab[m_local, ki];
    v = a·b; acc += v`` — the scalar counterpart of the mma ``ldmatrix`` drain. Built per-cell directly
    (NOT via the masked gmem-direct σ, whose ``% extent`` wrap would corrupt the slab index for an
    overhanging cell): the slab is indexed by the **local** tile coordinate ``offset.base(...) −
    base`` (``m_uvar·reg_m + i`` ∈ [0, tile_m), always in-slab), so an overhanging cell reads a
    clamped / zero-filled slab row and its store is discarded by the guard. ``_dedup_loads`` still
    shares A across the n-cells and B across the m-cells exactly as gmem-direct does. **Carrier-less**
    (no ``Loop.carrier``): the accumulators are pre-seeded once by :func:`_scalar_state` outside the
    outer slab loop, so the drain folds into them without re-seeding."""
    b_name, a_name = c.b_load.names[0], c.a_name
    body: list[Stmt] = []
    for i, j in cells:
        sfx = f"__c{i}_{j}"
        bn, an, vn, cn = f"{b_name}{sfx}", f"{a_name}{sfx}", f"{c.acc}__v{sfx}", f"{c.acc}{sfx}"
        m_local = BinaryExpr("-", offset.base("m", i), row_base)
        n_local = BinaryExpr("-", offset.base("n", j), col_base)
        body.append(Load(names=(bn,), input=b_slab, index=(Var(ki), n_local)))
        body.append(Load(names=(an,), input=a_slab, index=(m_local, Var(ki))))
        body.append(Assign(name=vn, op=_MUL, args=(bn, an)))
        body.append(Accum(name=cn, value=vn, op=_ADD, axes=(ki,)))
    body = _dedup_loads(body)
    # seed=False: the accumulators are pre-seeded once by _scalar_state outside the outer slab loop, so
    # this inner drain must NOT re-declare (re-zero) them each slab iteration.
    return Loop(axis=Axis(name=ki, extent=bk_elems), body=Body(tuple(body)), unroll=bk_elems <= 64, seed=False)


def _scalar_staged_kloop(c: Contraction, inputs, cells, offset, masks, mode: str, bk_elems: int) -> tuple[list[Stmt], list[Stmt]]:
    """The scalar tier's STAGED K loop (single-buffer): an outer K-slab loop that cooperatively fills
    the A ``[tile_m×bk_elems]`` + B ``[bk_elems×tile_n]`` smem slabs (``tma`` box copy or ``cp.async``),
    barriers, then drains them with :func:`_scalar_drain`. Reuses the tier-agnostic ``_stage.py`` fill
    primitives via a scalar :class:`CtaTile` (``block_threads`` threads, the ``m_uvar·units_n + n_uvar``
    linear id). A pure perf transform — the K order ``k0+ki`` = 0..K matches gmem-direct, so it is
    numerically identical. Returns ``(pre_decls, [loop_stmts])``; the per-cell accumulator seeds ride
    :func:`_scalar_state`."""
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    tile_m, tile_n, K = c.tile_m, c.tile_n, k_axis.extent.as_static()
    a_buf, b_buf = c.a_operand.input, c.b_load.input
    slab_dtype = cuda_name(inputs[a_buf].dtype)
    elem_bytes = inputs[a_buf].dtype.nbytes
    a_slab, b_slab, k0, ki = "_a_smem", "_b_smem", "_ks", "_ki"
    row_base = BinaryExpr("*", Var(c.m_b), Literal(tile_m, "int"))
    col_base = BinaryExpr("*", Var(c.n_b), Literal(tile_n, "int"))
    linear_tid = BinaryExpr("+", BinaryExpr("*", Var(c.m_uvar), Literal(c.units_n, "int")), Var(c.n_uvar))
    cta = CtaTile(row_base=row_base, col_base=col_base, linear_tid=linear_tid, n_threads=c.block_threads)
    drain = _scalar_drain(c, cells, offset, a_slab, b_slab, ki, bk_elems, row_base, col_base)

    if mode == "tma":
        desc_a, desc_b, mbar = "_desc_a", "_desc_b", "_mbar"
        tid0 = BinaryExpr("==", linear_tid, Literal(0, "int"))
        total_bytes = (tile_m * bk_elems + bk_elems * tile_n) * elem_bytes
        phase = BinaryExpr("%", BinaryExpr("/", Var(k0), Literal(bk_elems, "int")), Literal(2, "int"))
        fill = tma_fill(
            loads=[(desc_a, a_slab, (row_base, Var(k0))), (desc_b, b_slab, (Var(k0), col_base))],
            mbar=mbar,
            tid0=tid0,
            phase=phase,
            total_bytes=total_bytes,
        )
        outer = StridedLoop(
            axis=Axis(name=k0, extent=K),
            start=Literal(0, "int"),
            step=Literal(bk_elems, "int"),
            body=Body((*fill, drain, Sync())),
            unroll=False,
        )
        slab_decls = [
            tma_descriptor(desc_a, a_buf, (tile_m, bk_elems), slab_dtype),
            tma_descriptor(desc_b, b_buf, (bk_elems, tile_n), slab_dtype),
            slab_smem(a_slab, tile_m, bk_elems, slab_dtype, align=TMA_SLAB_ALIGN),
            slab_smem(b_slab, bk_elems, tile_n, slab_dtype, align=TMA_SLAB_ALIGN),
        ]
        return slab_decls, [*tma_mbar_prologue(mbar, tid0), outer]

    mask_m, mask_n, m_ext, n_ext = masks

    def _clamp(idx, ext):  # clamp an overhanging gmem coord to the last valid one (the store is guarded)
        return TernaryExpr(cond=BinaryExpr("<", idx, ext), if_true=idx, if_false=BinaryExpr("-", ext, Literal(1, "int")))

    def a_gmem(row, col):  # slab[row][col] = A[row_base + row][k0 + col]
        m = BinaryExpr("+", row_base, row)
        sig = Sigma({m_axis.name: _clamp(m, m_ext) if mask_m else m, k_axis.name: BinaryExpr("+", Var(k0), col)})
        return tuple(sig.apply(e) for e in c.a_operand.index)

    def b_gmem(row, col):  # slab[row][col] = B[k0 + row][col_base + col]
        n = BinaryExpr("+", col_base, col)
        sig = Sigma({k_axis.name: BinaryExpr("+", Var(k0), row), n_axis.name: _clamp(n, n_ext) if mask_n else n})
        return tuple(sig.apply(e) for e in c.b_load.index)

    fills = cp_async_fill(
        slab=a_slab, slab_rows=tile_m, slab_cols=bk_elems, src=a_buf, gmem_index=a_gmem, cta=cta, elem_bytes=elem_bytes, name="a"
    )
    fills += cp_async_fill(
        slab=b_slab, slab_rows=bk_elems, slab_cols=tile_n, src=b_buf, gmem_index=b_gmem, cta=cta, elem_bytes=elem_bytes, name="b"
    )
    fills += cp_async_barrier()
    outer = StridedLoop(
        axis=Axis(name=k0, extent=K),
        start=Literal(0, "int"),
        step=Literal(bk_elems, "int"),
        body=Body((*fills, drain, Sync())),
        unroll=False,
    )
    slab_decls = [slab_smem(a_slab, tile_m, bk_elems, slab_dtype), slab_smem(b_slab, bk_elems, tile_n, slab_dtype)]
    return slab_decls, [outer]


#: The reusable ``(state_decls, reduce_region)`` pair, keyed by atom kind — the operand fragments +
#: the K-loop, **sink-agnostic**: both leave the accumulator a sink consumes (mma ``_c{i}_{j}``
#: register fragments / scalar ``acc__c{i}_{j}`` per cell). :func:`reduce_codegen` binds the node.
_MMA_REDUCE = (_mma_state, _mma_reduce)
_SCALAR_REDUCE = (_scalar_state, _scalar_reduce)


def reduce_codegen(c: Contraction, stage: Stage | None = None, inputs=None):
    """The reusable ``(state_decls, reduce_region)`` — operand fragments + the contraction K-loop
    (``ldmatrix`` + ``mma.sync`` / the synthesized scalar fma), dispatched off the atom and bound to
    ``c`` + its operand ``stage`` (both tiers stage an smem slab off it: the mma tier via ``ldmatrix``,
    the scalar tier via plain-``Load`` slab reads — ``inputs`` supplies the operand slab dtype).
    **Sink-agnostic**: it leaves the accumulator the :func:`store_sink` (or a flash sink) then
    consumes, so the same K-loop emission is reused wherever a contraction is tiled."""
    if isinstance(c.atom, AtomKind):
        return partial(_mma_state, c, stage), partial(_mma_reduce, c, stage)
    return partial(_scalar_state, c, stage, inputs), partial(_scalar_reduce, c, stage, inputs)


def store_sink(c: Contraction):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, masks)`` callable that writes
    each accumulator cell to the output through the projection ``epilogue`` (an mma ``RegStore`` /
    the replicated scalar epilogue tail), dispatched off the atom. The flash branch swaps a sink that
    instead feeds the accumulator fragments into the streaming-softmax twist, reusing
    :func:`reduce_codegen`."""
    store = _mma_store if isinstance(c.atom, AtomKind) else _scalar_store
    return partial(store, c)


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
    masks = (c.mask_m, c.mask_n, c.m_ext, c.n_ext)
    t = atomize(c.atom.atom_m, c.atom.atom_n)
    t = register_tile(t, c.reg_m, c.reg_n)
    t = unit_tile(t, c.units_m, c.units_n, c.m_uvar, c.n_uvar)
    return grid_tile(
        t,
        masks,
        n_axis=c.n_axis,
        n_b=c.n_b,
        tile_n=c.tile_n,
        m_axis=c.m_axis,
        m_b=c.m_b,
        tile_m=c.tile_m,
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
