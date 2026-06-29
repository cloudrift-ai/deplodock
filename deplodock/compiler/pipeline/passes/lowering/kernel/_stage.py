"""Operand-staging assembly for the tile materializers (``010_materialize``).

Builds the cooperative gmem→smem slab fill + the staged-load drain off a
:class:`~deplodock.compiler.ir.tile.schedule.Stage`, assembling the surviving
kernel-IR transport leaf nodes (``Smem`` / ``CpAsyncCopy`` / ``CpAsyncCommit`` /
``CpAsyncWait`` / ``Sync``) — **not** the demolished ``StageBundle`` /
``StagePolicy`` orchestration.

The fill is written against a small :class:`CtaTile` seam (the CTA tile-base + a
linear intra-CTA thread id + the thread count), NOT a materializer's internal
warp/register geometry — so the same helper drives both the warp (``_warp``) and
the future scalar (``_reg_tile``) tiers.

Leading ``_`` so the pass loader (globs ``*.py``, skips ``_``-prefixed) skips it.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from deplodock.compiler.ir.kernel.ir import CpAsyncCommit, CpAsyncCopy, CpAsyncWait, Smem, Sync
from deplodock.compiler.ir.stmt import Body, StridedLoop
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
) -> list[Stmt]:
    """Cooperatively ``cp.async``-copy a ``slab_rows × slab_cols`` row-major smem
    ``slab`` from gmem ``src``. ``gmem_index(row_expr, col_expr)`` returns the gmem
    index tuple for slab cell ``(row, col)``. The CTA's ``n_threads`` lanes stripe
    ``slab_rows·slab_cols / V``-element chunks (``V`` = :func:`_cp_async_width`); each
    lane runs ``for e = tid; e < n_chunks; e += n_threads``. Emits the fill loop only
    — the caller appends one ``CpAsyncCommit`` + ``CpAsyncWait`` + ``Sync`` after the
    A and B fills together. The loop bound (not a predicate) masks the tail, so every
    lane still reaches the shared barrier (the barrier-under-mask invariant)."""
    v = _cp_async_width(slab_cols, elem_bytes)
    n_chunks = (slab_rows * slab_cols) // v
    fe = Axis(name=f"_f{name}", extent=n_chunks)
    base = _mul(Var(fe.name), _lit(v))  # flat element offset of this chunk
    row = BinaryExpr("/", base, _lit(slab_cols))
    col = BinaryExpr("%", base, _lit(slab_cols))
    copy = CpAsyncCopy(
        smem=slab,
        smem_index=(row, col),
        src=src,
        src_index=tuple(gmem_index(row, col)),
        nbytes=v * elem_bytes,
    )
    loop = StridedLoop(axis=fe, start=cta.linear_tid, step=_lit(cta.n_threads), body=Body((copy,)), unroll=False)
    return [loop]


def cp_async_barrier() -> list[Stmt]:
    """The handshake after the cooperative cp.async fills: commit the group, wait for
    it to drain, then a CTA barrier so every lane sees the filled slab."""
    return [CpAsyncCommit(), CpAsyncWait(group=0), Sync()]


def slab_smem(name: str, rows: int, cols: int, dtype: str) -> Smem:
    """A row-major ``rows × cols`` operand slab; ``ldm`` (the staged-load row stride)
    is ``cols``."""
    return Smem(name=name, extents=(rows, cols), dtype=dtype)
