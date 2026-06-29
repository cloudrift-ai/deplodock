# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `010_materialize` — bind the schedule to threads

`010_materialize` dispatches on the kernel kind / its schedule (the article's "schedule separate from combine" thesis —
the op tree + `ir/tile/ops.lower` are shared across kinds; only the partition changes):

- **Scalar tier** — one thread per output cell (`lower(op)` + an output-store glue).
- **Reduce tier** (`_reduce`, a `MonoidKernel` whose `ReducePlan` cooperates / register-folds) — the reduce axis is
  partitioned `coop` ways across the CTA's threads and `reg` ways across per-thread accumulators (ILP), then a REG-tree
  fold, the cross-thread combine (`_combine`), and the projection.
- **Register-tile tier** (`_reg_tile`, a `SemiringKernel` whose `TILE` plan tiles the output) — each thread owns a
  `reg_m × reg_n` block of cells, the reduce body replicated per cell with its operand loads deduped (the
  arithmetic-intensity reuse).
- **Warp / tensor-core tier** (`_warp`, a `SemiringKernel` carrying a `WarpTile`) — `WM·WN` warps over a
  `tile_m × tile_n` block of `mma_m16n8k16` atom cells; gmem-direct `LdmatrixLoad` + `MmaSyncPtx`, then a `RegStore`
  (fused projection epilogue + masked-tile guards).

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

## Operand staging (`_stage.py`) — the `STAGE` codec

When the schedule carries a `Stage` (the `STAGE` codec `d<depth>/sync|cp|tma[/ring]`), the warp tier stages its A/B
operands through a shared-memory slab instead of reading gmem-direct. `_stage.py` assembles the surviving Kernel-IR
transport leaves — it does **not** resurrect the demolished `StageBundle` / `StagePolicy` orchestration.

**The `CtaTile` seam.** The fill is written against a small `CtaTile` (the CTA tile-base coords + an independent linear
intra-CTA thread id + the thread count), NOT a materializer's internal warp/register geometry — so one fill helper drives
any tier. `_warp` builds the seam from its decoded block / warp / lane axis vars (never a raw `threadIdx.x`, whose
`free_vars()` would leak into param collection).

**Two producers, one drain.** `cp_async_fill` (the `sync` / `cp.async` thread-stripe) and `tma_fill`
(`cp.async.bulk.tensor` — one thread issues `MbarrierArriveExpectTx` + the operand box `TmaLoad`s onto a single mbarrier;
every thread waits the parity) share the slab + the staged `LdmatrixLoad(staged=True)` drain (`_staged_inner_atom_loop`),
plain row-major NONE-swizzle slabs. The TMA path declares a `TmaDescriptor` per operand (encoded host-side off the bound
array's device pointer at launch — `backend/cuda/_tma.py`); its slabs are 128 B-aligned. A masked / symbolic **M** rides
the fill: cp.async clamp-reads the overhang row (`% M`), TMA zero-fills the box past the descriptor's runtime globalDim —
either way the `RegStore` `m_guard` discards the masked-row stores. A symbolic / non-divisible **K** zero-fills the tail;
a masked **N** or transposed-B stays gmem-direct.

**Shared-row staging (`_reduce`).** The fused norm→linear prologue is a `MonoidKernel`: an input row folded by the
cooperative reduce AND re-read per output column of a contraction tail (a free-axis `Loop` over an inner reduce).
`_reduce` stages that one row into a single `__shared__` slab (cooperatively filled) and rewrites both readers to it. The
trigger is narrow (`_has_contraction_tail`) so a plain softmax sum or a bare reduction is untouched.

Staging only ever *adds* a faster lowering — an ineligible kernel silently falls back to gmem-direct.

## Kernel-IR peepholes

`030_stamp_types` / `040_demote_to_write_dtype` resolve element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `110_drop_redundant_syncs` collapses the defensive `Sync`s the staging
templates emit (body-level only — the slab `Smem` decls flag `smem_seen`, so a load-bearing prologue `Sync` after a
`Cond(MbarrierInit)` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).
