"""CUDA-specific LoopIR extensions for TMA (Tensor Memory Accelerator).

These ops live in the CUDA backend (not in the shared ``loop_ir.py``)
because they map to sm_90+ PTX inline asm with no C equivalent.
The CUDA codegen handles them alongside standard LoopOps.

Usage in LoopProgram.body::

    body = [
        ...grid setup...,
        TMAKLoop(
            a_tma_ref="&A_tma", b_tma_ref="&B_tma",
            tile_m=64, tile_n=128, block_k=32,
            a_size=2048, stage=6144,
            thread_m=8, thread_n=4, tx=32,
            k_splits=1, is_batched=False,
        ),
        ...epilogue...,
        ...write...,
    ]
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TMAKLoop:
    """TMA double-buffered K-loop with mbarrier pipeline.

    Encapsulates the entire TMA setup + double-buffered tile loop:
    - Dynamic shared memory declaration (extern __shared__)
    - Shared memory address computation (__cvta_generic_to_shared)
    - Mbarrier initialization (PTX mbarrier.init)
    - First tile prefetch (PTX cp.async.bulk.tensor.2d)
    - Double-buffered tile loop with parity-based mbarrier wait/arrive
    - Inner FMA loop reading A/B from shared memory

    The codegen expands this into the full PTX inline asm sequence.
    Register accumulators (c{r}{c}) are referenced by name and must
    be declared before this op (via Alloc with register array).
    """

    # TMA descriptor references (e.g. "&A_tma" or "&A_tma[batch]")
    a_tma_ref: str
    b_tma_ref: str

    # Tile geometry
    tile_m: int
    tile_n: int
    block_k: int
    a_size: int  # tile_m * block_k
    stage: int  # a_size + block_k * tile_n (one double-buffer stage)

    # Thread tile
    thread_m: int
    thread_n: int
    tx: int  # blockDim.x (32)

    # K-splitting
    k_splits: int

    # Batching
    is_batched: bool
