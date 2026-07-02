"""Operand-staging assembly for the kernel emitters (``_factor.factorize``).

The single home for every operand-staging *transport*: the warp tier's cooperative gmem→smem
2-D slab fills (``cp.async`` / TMA, off a :class:`~deplodock.compiler.ir.schedule.Stage`) and
the reduce tier's ``sync`` 1-D shared-row fill (:func:`sync_row_fill`, the fused norm→linear
prologue) both live here, indexed off the same linear-tid / thread-count seam. Assembles the
surviving kernel-IR transport leaf nodes (``Smem`` / ``CpAsyncCopy`` / ``CpAsyncCommit`` /
``CpAsyncWait`` / ``Sync`` — and the TMA quartet ``TmaDescriptor`` / ``TmaLoad`` /
``Mbarrier*``).

The fill is written against a small :class:`CtaTile` thread-striping seam (a linear intra-CTA
thread id + the thread count), NOT a materializer's internal warp/register geometry — so one fill
helper drives any tier that stages; the per-operand gmem tile-base rides the :class:`Operand`. The
A/B operands themselves ride as an ``(a, b)`` :class:`Operand` pair, so a transport loops over the
pair instead of spelling A then B. The staged K-loop itself is ONE
skeleton, :func:`staged_kloop` (``fill → commit → wait → drain → Sync``, ``depth`` the sole
buffering knob), driven by a :class:`Transport` strategy (:class:`CpAsyncTransport` /
:class:`TmaTransport`) — the two producers put behind one ``fill``/``commit``/``wait`` seam. The
(plain row-major, NONE-swizzle) slab feeds the same staged ``LdmatrixLoad`` / scalar ``Load`` drain
regardless of which producer (cp.async / TMA) filled it; ``_atom._staged`` — the one atom-agnostic
driver — builds the transport, asks the atom strategy for the drain leaf, and calls
:func:`staged_kloop`.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, TernaryExpr, Var
from deplodock.compiler.ir.kernel.ir import (
    CpAsyncCommit,
    CpAsyncCopy,
    CpAsyncWait,
    MbarrierArriveExpectTx,
    MbarrierInit,
    MbarrierWait,
    Smem,
    Sync,
    TmaDescriptor,
    TmaLoad,
)
from deplodock.compiler.ir.stmt import Body, Cond, Load, Stmt, StridedLoop, Write


def _mul(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("*", a, b)


def _add(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("+", a, b)


def _lit(n: int) -> Expr:
    return Literal(int(n), "int")


@dataclass(frozen=True)
class CtaTile:
    """The tile-agnostic thread-striping seam a cooperative fill indexes off — a linear intra-CTA
    thread id + the CTA thread count. Built from a materializer's decoded grid vars (the warp tier's
    ``(m_w·WN + n_w)·32 + lane`` linear id), so the tier's geometry never leaks into the fill. The
    per-operand gmem tile-base rides the :class:`Operand` instead."""

    linear_tid: Expr  # intra-CTA linear thread id (0 .. n_threads-1)
    n_threads: int


def _cp_async_width(slab_cols: int, elem_bytes: int) -> int:
    """Elements per ``cp.async`` — the widest contiguous run whose byte size is a
    legal cp.async width (4 / 8 / 16) and that divides the inner (contiguous) slab
    extent (a chunk never straddles a slab row). The slab's inner dim maps stride-1
    to the gmem inner dim (canonical A[m,k] / B[k,n]), so a V-run is contiguous in
    both."""
    for nbytes in (16, 8, 4):
        v = nbytes // elem_bytes
        if v >= 1 and slab_cols % v == 0:
            return v
    return 1  # elem_bytes > 16 — never (fp16/bf16/fp32 only)


def cp_async_fill(
    *, slab: str, shape: tuple[int, int], src: str, gmem_index, cta: CtaTile, elem_bytes: int, name: str, row_offset: Expr | None = None
) -> list[Stmt]:
    """Cooperatively ``cp.async``-copy a ``rows × cols`` (= ``shape``) row-major smem
    ``slab`` from gmem ``src``. ``gmem_index(row_expr, col_expr)`` returns the gmem
    index tuple for slab cell ``(row, col)``. The CTA's ``n_threads`` lanes stripe
    ``rows·cols / V``-element chunks (``V`` = :func:`_cp_async_width`); each
    lane runs ``for e = tid; e < n_chunks; e += n_threads``. Emits the fill loop only
    — the caller appends one ``CpAsyncCommit`` + ``CpAsyncWait`` + ``Sync`` after the
    A and B fills together. The loop bound (not a predicate) masks the tail, so every
    lane still reaches the shared barrier (the barrier-under-mask invariant).

    ``row_offset`` (the gmem→smem ring): when staging through a depth>1 slab, it picks
    the ring SLOT — the write row becomes ``row_offset + row`` (each slot is a contiguous
    ``rows``-row block), so the fill targets one slot while the drain reads another."""
    slab_rows, slab_cols = shape
    v = _cp_async_width(slab_cols, elem_bytes)
    n_chunks = (slab_rows * slab_cols) // v
    fe = Axis(name=f"_f{name}", extent=n_chunks)
    base = _mul(Var(fe.name), _lit(v))  # flat element offset of this chunk
    row = BinaryExpr("/", base, _lit(slab_cols))
    col = BinaryExpr("%", base, _lit(slab_cols))
    smem_row = _add(row_offset, row) if row_offset is not None else row
    copy = CpAsyncCopy(
        smem=slab,
        smem_index=(smem_row, col),
        src=src,
        src_index=tuple(gmem_index(row, col)),
        nbytes=v * elem_bytes,
    )
    loop = StridedLoop(axis=fe, start=cta.linear_tid, step=_lit(cta.n_threads), body=Body((copy,)), unroll=False)
    return [loop]


def cp_async_commit() -> list[Stmt]:
    """Close the current cp.async batch into a commit-group (the depth-``D`` ring commits one
    group per filled slot, so ``CpAsyncWait(group=D-1)`` can drain exactly the slot it needs)."""
    return [CpAsyncCommit()]


def cp_async_wait(group: int) -> list[Stmt]:
    """Wait until at most ``group`` cp.async commit-groups remain in flight, then a CTA barrier
    so every lane sees the drained slot (the depth-``D`` ring's per-chunk handshake — the commit
    happened in the prefetch fill above)."""
    return [CpAsyncWait(group=group), Sync()]


def slab_smem(name: str, rows: int, cols: int, dtype: str, *, align: int = 0) -> Smem:
    """A row-major ``rows × cols`` operand slab; ``ldm`` (the staged-load row stride)
    is ``cols``. ``align`` stamps an explicit byte alignment — TMA destination slabs
    need 128 B (``cp.async.bulk.tensor`` requires an aligned smem base)."""
    return Smem(name=name, extents=(rows, cols), dtype=dtype, align=align)


def sync_row_fill(*, slab: str, src: str, extent: int, grid_vars: tuple, linear_tid: Expr, n_threads: int, dtype: str) -> list[Stmt]:
    """The ``sync``-transport 1-D operand fill: cooperatively copy the CTA-shared row
    ``src[grid…, 0:extent]`` into a length-``extent`` smem ``slab``, then a CTA barrier so
    every lane sees the filled row before the reader drains it. The ``n_threads`` cooperating
    lanes stripe it (``for k = linear_tid; k < extent; k += n_threads``), the same
    linear-tid / thread-count seam :func:`cp_async_fill` indexes off — so every transport's
    fill lives here. This is the scalar reduce tier's shared-row prologue (the fused
    norm→linear input row), the single-buffer ``sync`` counterpart of the async 2-D slab
    fills above."""
    fe = Axis(name=f"_{slab}_f", extent=extent)
    val = f"_{slab}_v"
    load = Load(name=val, input=src, index=(*grid_vars, Var(fe.name)))
    write = Write(output=slab, index=(Var(fe.name),), value=val)
    loop = StridedLoop(axis=fe, start=linear_tid, step=_lit(n_threads), body=Body((load, write)), unroll=False)
    return [Smem(name=slab, extents=(extent,), dtype=dtype), loop, Sync()]


# --------------------------------------------------------------------------- #
# TMA (``cp.async.bulk.tensor``) descriptor — the host-built ``CUtensorMap`` the
# box copies index off (the copy + mbarrier handshake live in :class:`TmaTransport`
# below). Shares the slab + staged-``LdmatrixLoad`` drain with the cp.async path.
# --------------------------------------------------------------------------- #

# TMA destination smem must be aligned (``cp.async.bulk.tensor`` faults otherwise);
# 128 B satisfies the NONE-swizzle box copy.
TMA_SLAB_ALIGN = 128


def tma_descriptor(name: str, src: str, box: tuple[int, int], dtype: str) -> TmaDescriptor:
    """A host-encoded ``CUtensorMap`` for the operand ``src`` with a ``box`` tile
    (C-order). The source globalDim is resolved from the bound array at launch, so a
    symbolic (masked-M) extent rides the runtime shape and TMA zero-fills the box
    overhang. NONE swizzle: the plain row-major slab feeds the staged ``LdmatrixLoad``
    drain directly."""
    return TmaDescriptor(name=name, src_buf=src, src_shape=(), box_extents=box, swizzle="NONE", dtype=dtype)


# --------------------------------------------------------------------------- #
# The Transport strategy — the one interface the staged K-loop drives, and the
# two producers behind it (cp.async / TMA). A :class:`Transport` owns the operand
# slab layout + the fill/commit/wait handshake; :func:`staged_kloop` owns the
# depth-parametrized control flow. The two are structurally different primitives
# (cp.async is fill → commit → wait-group; TMA is an arrive/expect-tx + box copy
# gated by an mbarrier phase) put behind ONE seam so ``depth`` becomes the sole
# buffering knob: ``depth == 1`` is the degenerate single-buffer loop, ``depth >= 2``
# a gmem→smem prefetch ring.
# --------------------------------------------------------------------------- #


def _slot_row(slot: Expr, rows_per_slot: int) -> Expr | None:
    """The row offset of ring ``slot`` — ``slot·rows_per_slot``, or ``None`` for a literal
    slot 0 (the single-buffer / ring-slot-0 case). ``None`` keeps the emitted index free of a
    dead ``+ 0·rows`` term, so single-buffer staging stays bit-identical to its gmem baseline."""
    if isinstance(slot, Literal) and slot.value == 0:
        return None
    return _mul(slot, _lit(rows_per_slot))


@dataclass(frozen=True)
class Operand:
    """One staged operand — A or B — the per-operand slab geometry both transports index off. The two
    ride as an ``(a, b)`` pair so :meth:`CpAsyncTransport.fill` / ``slab_decls`` loop over them instead
    of spelling A then B. ``shape`` is ``(rows, cols)`` of one ring slot (A ``(tile_m, bk)`` / B
    ``(bk, tile_n)``); each slab is ``ring·rows`` rows (slot ``s`` at row ``s·rows``), plain row-major /
    NONE-swizzle, feeding the same staged ``LdmatrixLoad`` (mma) / scalar ``Load`` drain. ``index`` maps
    a K-chunk offset ``k0`` to the cp.async ``(row, col) → gmem-index`` closure; ``coords`` maps ``k0`` to
    the TMA box origin. Each transport reads only the one it needs."""

    tag: str  # "a" / "b" — the cp.async fill name + the smem-slab / descriptor suffix
    buf: str  # gmem source buffer
    shape: tuple[int, int]  # (rows, cols) of one ring slot
    index: Callable[[Expr], Callable]  # k0 -> ((row, col) -> gmem index)   (cp.async)
    coords: Callable[[Expr], tuple]  # k0 -> gmem box origin                (TMA)

    @property
    def slab(self) -> str:
        return f"_{self.tag}_smem"

    @property
    def desc(self) -> str:
        return f"_desc_{self.tag}"

    def slot_row(self, slot: Expr) -> Expr | None:
        """The ring-slot row offset into this operand's multi-slot slab (``None`` for slot 0)."""
        return _slot_row(slot, self.shape[0])


@dataclass(frozen=True)
class CpAsyncTransport:
    """The cp.async producer: cooperative gmem→smem fills committed into groups, drained by
    ``CpAsyncWait(group=in_flight)``. ``operands`` is the ``(A, B)`` :class:`Operand` pair; each
    ``operand.index(k0)`` is the per-cell ``(row, col) → gmem-index`` closure (its tier bakes in the
    masked-axis clamp)."""

    operands: tuple[Operand, Operand]
    slab_dtype: str
    elem_bytes: int
    cta: CtaTile

    def slab_decls(self, ring: int) -> list[Stmt]:
        return [slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype) for op in self.operands]

    def prologue(self, ring: int) -> list[Stmt]:
        return []

    def fill(self, *, k0: Expr, slot: Expr) -> list[Stmt]:
        out: list[Stmt] = []
        for op in self.operands:
            out += cp_async_fill(
                slab=op.slab,
                shape=op.shape,
                src=op.buf,
                gmem_index=op.index(k0),
                cta=self.cta,
                elem_bytes=self.elem_bytes,
                name=op.tag,
                row_offset=op.slot_row(slot),
            )
        return out

    def commit(self) -> list[Stmt]:
        return cp_async_commit()

    def wait(self, *, in_flight: int, slot: Expr, phase: Expr) -> list[Stmt]:
        return cp_async_wait(in_flight)  # keep ``in_flight`` prefetch groups outstanding + a CTA barrier


@dataclass(frozen=True)
class TmaTransport:
    """The TMA (``cp.async.bulk.tensor``) producer: one thread issues an ``arrive.expect_tx`` + a box
    copy per :class:`Operand` onto a **per-slot mbarrier array**; every thread waits on the slot's
    parity. The multi-slot mbarrier is what makes ``depth`` a free knob for TMA — ``wait(slot, phase)``
    gates the ring slot the same way ``CpAsyncWait(in_flight)`` gates a commit group."""

    operands: tuple[Operand, Operand]
    slab_dtype: str
    elem_bytes: int
    cta: CtaTile
    mbar: str = "_mbar"

    @property
    def _tid0(self) -> Expr:
        return BinaryExpr("==", self.cta.linear_tid, _lit(0))

    @property
    def _total_bytes(self) -> int:
        return sum(math.prod(op.shape) for op in self.operands) * self.elem_bytes

    def slab_decls(self, ring: int) -> list[Stmt]:
        # TMA destination smem must be 128 B-aligned; one mbarrier per ring slot.
        decls: list[Stmt] = [tma_descriptor(op.desc, op.buf, op.shape, self.slab_dtype) for op in self.operands]
        decls += [slab_smem(op.slab, ring * op.shape[0], op.shape[1], self.slab_dtype, align=TMA_SLAB_ALIGN) for op in self.operands]
        decls.append(Smem(name=self.mbar, extents=(ring,), dtype="unsigned long long"))
        return decls

    def prologue(self, ring: int) -> list[Stmt]:
        # Init every ring slot's mbarrier (one producer ``arrive`` per phase), then a CTA barrier so
        # every consumer sees the init before its first wait.
        inits = tuple(MbarrierInit(mbar=self.mbar, count=1, slot=_lit(s)) for s in range(ring))
        return [Cond(cond=self._tid0, body=inits), Sync()]

    def fill(self, *, k0: Expr, slot: Expr) -> list[Stmt]:
        body: list[Stmt] = [MbarrierArriveExpectTx(mbar=self.mbar, bytes_=self._total_bytes, slot=slot)]
        for op in self.operands:
            body.append(
                TmaLoad(
                    smem=op.slab,
                    smem_index=(op.slot_row(slot) or _lit(0), _lit(0)),
                    desc=op.desc,
                    coords=op.coords(k0),
                    mbar=self.mbar,
                    mbar_slot=slot,
                )
            )
        return [Cond(cond=self._tid0, body=tuple(body))]

    def commit(self) -> list[Stmt]:
        return []  # TMA has no commit-group; the arrive.expect_tx above already armed the barrier

    def wait(self, *, in_flight: int, slot: Expr, phase: Expr) -> list[Stmt]:
        return [MbarrierWait(mbar=self.mbar, phase=phase, slot=slot)]


def staged_kloop(
    *, transport, drain: Callable[[Expr], list[Stmt]], depth: int, bk_elems: int, n_chunks: int, k_extent: int
) -> tuple[list[Stmt], list[Stmt]]:
    """The **one** staged K-loop skeleton — ``fill → commit → wait → drain → Sync`` over the K-chunks,
    with ``depth`` the sole buffering knob and ``transport`` the sole producer seam. Returns
    ``(slab_decls, [prologue…, outer_loop])``.

    ``ring = min(depth, n_chunks)`` slots (``<2`` chunks ⇒ nothing to prefetch, ``ring == 1``):

    - ``ring == 1`` (single buffer): fill chunk ``i`` into slot 0, wait everything, ``drain`` slot 0.
    - ``ring >= 2`` (gmem→smem prefetch ring): a prologue primes chunks ``0..ring-2`` into slots
      ``0..ring-2``; each loop step prefetches chunk ``i+ring-1`` (clamped to the last chunk so the
      commit/wait stays uniform across all CTA threads — the barrier-under-mask invariant) into slot
      ``(i+ring-1) % ring``, then waits ``ring-1`` chunks in flight and ``drain``\\ s slot ``i % ring``.

    ``transport`` supplies fill/commit/wait + the slab layout; ``drain(slot)`` is the atom leaf reading
    ring ``slot`` (``ldmatrix`` fragments / scalar slab ``Load``\\ s). For TMA the wait phase toggles per
    slot generation (``chunk // ring``); cp.async ignores it (it gates on the commit group instead)."""
    ring = min(depth, n_chunks) if n_chunks >= 2 else 1
    k0, K = "_ks", k_extent
    decls = transport.slab_decls(ring)
    pre = transport.prologue(ring)
    for s in range(ring - 1):  # prime chunks 0..ring-2 into slots 0..ring-2 (phase 0)
        pre += transport.fill(k0=_lit(s * bk_elems), slot=_lit(s))
        pre += transport.commit()

    i_expr = BinaryExpr("/", Var(k0), _lit(bk_elems))  # chunk index of the current step
    body: list[Stmt] = []
    if ring == 1:
        phase = BinaryExpr("%", i_expr, _lit(2))
        body += transport.fill(k0=Var(k0), slot=_lit(0))
        body += transport.commit()
        body += transport.wait(in_flight=0, slot=_lit(0), phase=phase)
        body += drain(_lit(0))
        body.append(Sync())
    else:
        pref_chunk = BinaryExpr("+", i_expr, _lit(ring - 1))  # logical index of the prefetched chunk
        pref_slot = BinaryExpr("%", pref_chunk, _lit(ring))
        read_slot = BinaryExpr("%", i_expr, _lit(ring))
        read_phase = BinaryExpr("%", BinaryExpr("/", i_expr, _lit(ring)), _lit(2))
        last_k0 = (n_chunks - 1) * bk_elems
        k0_next = BinaryExpr("+", Var(k0), _lit((ring - 1) * bk_elems))
        k0_pref = TernaryExpr(cond=BinaryExpr("<", k0_next, _lit(K)), if_true=k0_next, if_false=_lit(last_k0))
        body += transport.fill(k0=k0_pref, slot=pref_slot)
        body += transport.commit()
        body += transport.wait(in_flight=ring - 1, slot=read_slot, phase=read_phase)
        body += drain(read_slot)
        body.append(Sync())  # done reading this slot before a later chunk prefetches into it

    outer = StridedLoop(axis=Axis(name=k0, extent=K), start=_lit(0), step=_lit(bk_elems), body=Body(tuple(body)), unroll=False)
    return decls, [*pre, outer]
