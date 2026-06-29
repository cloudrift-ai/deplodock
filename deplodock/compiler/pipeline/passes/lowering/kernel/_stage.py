"""Operand-staging assembly for the tile materializers (``010_materialize``).

Builds the cooperative gmem→smem slab fill + the staged-load drain off a
:class:`~deplodock.compiler.ir.tile.schedule.Stage`, assembling the surviving
kernel-IR transport leaf nodes (``Smem`` / ``CpAsyncCopy`` / ``CpAsyncCommit`` /
``CpAsyncWait`` / ``Sync``) — **not** the demolished ``StageBundle`` /
``StagePolicy`` orchestration.

The fill is written against a small :class:`CtaTile` seam (the CTA tile-base + a
linear intra-CTA thread id + the thread count), NOT a materializer's internal
warp/register geometry — so the same helper drives both the warp
(``_warp_factor.factorize_mma``) and the future scalar (``_reg_tile``) tiers.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
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
from deplodock.compiler.ir.stmt import Body, Cond, StridedLoop
from deplodock.compiler.ir.stmt.base import Stmt


def _mul(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("*", a, b)


def _add(a: Expr, b: Expr) -> Expr:
    return BinaryExpr("+", a, b)


def _lit(n: int) -> Expr:
    return Literal(int(n), "int")


@dataclass(frozen=True)
class CtaTile:
    """The tile-agnostic seam a cooperative fill indexes off — the CTA's tile-base
    coordinates, a linear intra-CTA thread id, and the CTA thread count. Built from
    a materializer's decoded grid vars (the warp tier's ``m_b``/``n_b`` block axes +
    ``(m_w·WN + n_w)·32 + lane`` linear id; the scalar tier's block axes + thread id),
    so neither tier's geometry leaks into the fill."""

    row_base: Expr  # global row of the CTA tile's top-left cell
    col_base: Expr  # global col of the CTA tile's top-left cell
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
    *,
    slab: str,
    slab_rows: int,
    slab_cols: int,
    src: str,
    gmem_index,
    cta: CtaTile,
    elem_bytes: int,
    name: str,
    row_offset: Expr | None = None,
) -> list[Stmt]:
    """Cooperatively ``cp.async``-copy a ``slab_rows × slab_cols`` row-major smem
    ``slab`` from gmem ``src``. ``gmem_index(row_expr, col_expr)`` returns the gmem
    index tuple for slab cell ``(row, col)``. The CTA's ``n_threads`` lanes stripe
    ``slab_rows·slab_cols / V``-element chunks (``V`` = :func:`_cp_async_width`); each
    lane runs ``for e = tid; e < n_chunks; e += n_threads``. Emits the fill loop only
    — the caller appends one ``CpAsyncCommit`` + ``CpAsyncWait`` + ``Sync`` after the
    A and B fills together. The loop bound (not a predicate) masks the tail, so every
    lane still reaches the shared barrier (the barrier-under-mask invariant).

    ``row_offset`` (the gmem→smem ring): when staging through a depth>1 slab, it picks
    the ring SLOT — the write row becomes ``row_offset + row`` (each slot is a contiguous
    ``slab_rows``-row block), so the fill targets one slot while the drain reads another."""
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


def cp_async_barrier(group: int = 0) -> list[Stmt]:
    """The handshake after the cooperative cp.async fills: commit the group, wait until
    at most ``group`` groups remain in flight, then a CTA barrier so every lane sees the
    drained slab. ``group=0`` (single-buffer) drains everything; a depth-``D`` ring keeps
    ``D-1`` prefetch groups outstanding (``group=D-1``)."""
    return [CpAsyncCommit(), CpAsyncWait(group=group), Sync()]


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


# --------------------------------------------------------------------------- #
# TMA (``cp.async.bulk.tensor``) fill — single-thread descriptor box copy +
# mbarrier handshake. Shares the slab + staged-``LdmatrixLoad`` drain with the
# cp.async path; only the producer differs (one thread issues the TMA, every
# thread waits on the mbarrier parity).
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


def tma_mbar_prologue(mbar: str, tid0: Expr, count: int = 1) -> list[Stmt]:
    """One single-slot mbarrier, initialized by the issuer thread, then a CTA barrier so
    every consumer sees the init before the first wait. ``count`` = number of producer
    ``arrive``\\ s per phase (one — the issuer's single ``arrive.expect_tx`` covers both
    operand box copies via the summed transaction-byte count)."""
    init = Cond(cond=tid0, body=(MbarrierInit(mbar=mbar, count=count, slot=_lit(0)),))
    return [Smem(name=mbar, extents=(1,), dtype="unsigned long long"), init, Sync()]


def tma_fill(*, loads: list[tuple[str, str, tuple]], mbar: str, tid0: Expr, phase: Expr, total_bytes: int) -> list[Stmt]:
    """One pipeline stage of the TMA fill: the issuer thread declares the expected
    transaction bytes and issues every operand's box ``TmaLoad`` onto ``mbar``; every
    thread then waits on the barrier parity for ``phase``. ``loads`` is a list of
    ``(desc_name, slab, coords)`` (coords C-order, the box origin in the source). The
    barrier (not a predicate) gates the readers — every CTA thread reaches the wait."""
    body: list[Stmt] = [MbarrierArriveExpectTx(mbar=mbar, bytes_=total_bytes, slot=_lit(0))]
    for desc_name, slab, coords in loads:
        body.append(TmaLoad(smem=slab, smem_index=(_lit(0), _lit(0)), desc=desc_name, coords=tuple(coords), mbar=mbar, mbar_slot=_lit(0)))
    return [Cond(cond=tid0, body=tuple(body)), MbarrierWait(mbar=mbar, phase=phase, slot=_lit(0))]
