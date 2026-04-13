"""Shared memory K-loop expansion for SmemPipelineKLoop.

Backend-agnostic: builds LoopIR ops from a Schedule, using standard
GPU primitives (shared memory, barriers, tiled loops).  Works for any
backend with shared memory (CUDA, HIP, SYCL).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deplodock.compiler.backend.ir.expr import Builtin, Literal, Ternary, Var
from deplodock.compiler.backend.ir.loop_ir import (
    Accum,
    Alloc,
    Barrier,
    Let,
    Load,
    LoopNest,
    Store,
)

if TYPE_CHECKING:
    from deplodock.compiler.backend.cuda.generators.analysis import TileAnalysis
    from deplodock.compiler.backend.cuda.schedule import Schedule


def _safe(name: str) -> str:
    return name.replace("-", "_").replace(".", "_").replace(" ", "_")


def emit_smem_k_loop(schedule: Schedule, analysis: TileAnalysis) -> list:
    """Expand a smem K-tile loop into primitive LoopIR ops.

    Shared memory for A (bank-conflict padded), scalar global loads for B.
    Produces: smem Alloc + k-range + outer tile loop (load A→smem, barrier,
    inner FMA loop, barrier).
    """
    thread_m = schedule.thread_m or 4
    thread_n = schedule.thread_n or 4
    bk = schedule.block_k
    tile_m = schedule.tile_m or 16
    smem_stride = bk + 1
    I = "int"  # noqa: E741
    M, N, K = Var("M"), Var("N"), Var("K")  # noqa: N806
    A = Var(_safe(analysis.contraction_a))  # noqa: N806
    B = Var(_safe(analysis.contraction_b))  # noqa: N806

    ops: list = []

    # Shared memory for A tile (bank-conflict padded)
    ops.append(Alloc("As", "float", (tile_m * smem_stride,), "smem"))

    # K-range for k_splits
    if schedule.k_splits > 1:
        bidz = Builtin("blockIdx.z")
        k_per, k_start = Var("k_per"), Var("k_start")
        ops.append(Let("k_per", K / bk / Var("k_splits") * bk, dtype=I))
        ops.append(Let("k_start", bidz * k_per, dtype=I))
        ops.append(Let("k_end", Ternary(bidz.eq(Literal(schedule.k_splits - 1, I)), K, k_start + k_per), dtype=I))
    else:
        ops.append(Let("k_start", Literal(0, I), dtype=I))
        ops.append(Let("k_end", K, dtype=I))

    # Batch pointer aliases already emitted by _emit_accumulators
    a_src = Var("Ab") if schedule.is_batched else A
    b_src = Var("Bb") if schedule.is_batched else B

    # K-tile loop body
    row_base, col_base = Var("row_base"), Var("col_base")
    sr, tk, kk = Var("sr"), Var("tk"), Var("kk")
    tidx = Builtin("threadIdx.x")

    tk_body: list = []
    for r in range(thread_m):
        row_r = row_base + r
        k_col = tk + tidx
        tk_body.append(Load(f"As_ld_{r}", a_src, row_r * K + k_col, "global", guard=row_r.lt(M).and_(k_col.lt(K))))
        tk_body.append(Store("As", (sr + r) * smem_stride + tidx, Var(f"As_ld_{r}"), "smem"))
    tk_body.append(Barrier())

    kk_body: list = []
    for c in range(thread_n):
        col_c = col_base + c
        kk_body.append(Load(f"b{c}", b_src, (tk + kk) * N + col_c, "global", guard=col_c.lt(N)))
    for r in range(thread_m):
        kk_body.append(Load(f"a{r}", "As", (sr + r) * smem_stride + kk, "smem"))
        for c in range(thread_n):
            kk_body.append(Accum(f"c{r}{c}", "sum", Var(f"a{r}") * Var(f"b{c}")))
    tk_body.append(LoopNest("kk", Literal(0, I), Literal(bk, I), None, kk_body))
    tk_body.append(Barrier())

    ops.append(LoopNest("tk", Var("k_start"), Var("k_end"), Literal(bk, I), tk_body))
    return ops
