# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `005_contract` — construct the contraction node (before materialize)

A `Semiring` contraction's high-level node is built one pass **before** the materializer. It's **one flat**
`Contraction` Stmt (`ir/kernel/ir.py`) — binding-driven for both atoms, with **no per-atom subclass** — that cleanly
splits two concerns:

- **algebra params** (what to contract): the m/n output `axes` + the `k_axis`, the leading batch `lead_axes`, the
  structured A/B operand `Load`s (read off the atomize binding resolved in `020_schedule`), the fold accumulator `acc`,
  and the projection `epilogue` (the binding's body, or a synthesized accumulator store via `_store.with_store`).
- **schedule** (how to tile it): one `tile: TilePlan` field carrying the leaf `atom` (a tensor-core `AtomKind`,
  `lanes == 32`, or the scalar `ScalarAtom`, `lanes == 1` — `ir/tile/atom.py`) plus the unit/register widths + K-chunk.
  The atom selects the codegen at materialize; the rest of the per-CTA geometry (`tile_m` / `mask_m` / `block_threads` /
  …) is **derived** on the node from `tile` × `axes` (`@property`, out of the structural-key fields).

Keeping the schedule a single swappable field is what lets the same operand/`acc` params be tiled by a *different*
`TilePlan` — the seam the flash inner QK/PV reuse needs.

A non-tiled contraction (per-cell fallback) and the cooperative reduce tier are left untouched here (`RuleSkipped`) and
bind to threads in `010_materialize`. Homing the construction here keeps the contraction a first-class node that exists
*before* thread-binding; it IS `structural_key`-ed as an intermediate `KernelOp` (the kernel-stmt protocol + a
`_rewrite` handler). It routes through the shared tiling layer with the generic `_b` / `_u` block/unit axis naming.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` dispatches on the kernel kind / its schedule (the article's "schedule separate from combine" thesis —
the op tree + `ir/tile/ops.lower` are shared across kinds; only the partition changes):

- **Scalar tier** — one thread per output cell (`lower(op)` + an output-store glue).
- **Reduce tier** (`_reduce`, a `MonoidKernel` whose `ReducePlan` cooperates / register-folds) — the reduce axis is
  partitioned `coop` ways across the CTA's threads and `reg` ways across per-thread accumulators (ILP), then a REG-tree
  fold, the cross-thread combine (`_combine`), and the projection.

The materializer takes one of the thread-binding tiers above for a `TileOp` root, OR — for a `KernelOp(Contraction)`
root that `005_contract` produced — **expands the contraction** through the one atom-generic `_factor.factorize` over the
shared tiling layer (below); the leaf type selects the codegen (mma / scalar). The pass's `PATTERN` matches `(TileOp,
KernelOp)` so the single rule covers both; at this point in the kernel pass the only `KernelOp`s are the contraction
nodes.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The atom factorization — one factorizer, two atoms (`_factor.py` / `_tiling.py`)

`_factor.factorize` is the **single** contraction factorizer — there is no per-atom variant, and **no per-atom geometry
object**. It expands any `Contraction` by tiling a **leaf atom** four ways through the layer in `_tiling.py`:
`grid_tile(unit_tile(register_tile(atomize(...))))` — **GRID** block / **UNIT** / **REGISTER** / **ATOM**. The tiling
geometry (`tile_m` / `mask_m` / `m_b` / `m_uvar` / `block_threads` / `lanes` / …) is **derived on the `Contraction` node
itself** (`@property`, computed from the leaf widths + skeleton axes); `factorize` reads it straight off `c` and hands
`grid_tile` the atom-specific **codegen callables** (`state_decls` / `reduce_region` / `store`). The single
`_factor.codegen` is a thin seam: it **dispatches that triple off the atom** (`isinstance(c.atom, AtomKind)`) and binds
the `Contraction` — the only per-atom difference:

- **mma** (`_mma_state` / `_mma_reduce` / `_mma_store`) — atom `(16, 8, 16)`, `lanes == 32`. The UNIT is a **warp**; the
  codegen emits `RegFragment` / `LdmatrixLoad` / `MmaSyncPtx` / `RegStore`, owns the K-loop (operands **gmem-direct**),
  and decodes the atom-lane offset at render.
- **scalar** (`_scalar_state` / `_scalar_reduce` / `_scalar_store`) — atom `(1, 1, 1)`, `lanes == 1`. The UNIT is a
  **single thread** (so there is no `_lane` axis); the codegen **synthesizes** the reduce `Loop` from the operands and
  replicates it + a projection `tail` per register cell with its operand loads deduped (the arithmetic-intensity reuse).

Both triples are free functions in `_factor.py`, the new-atom seam.

The **unit** is the atom's parallel thread footprint (`atom.lanes`) — so the tensor-core warp tile and the scalar
parallel thread-tile are the *same* level, differing only in `lanes`; `block_threads = units · lanes`. `grid_tile` also
carries any leading (batch) grid axes and supports a 1-D (m-absent) output. (The store-glue helpers `with_store` /
`has_write`, shared by the constructor and the thread-binding tiers, live in `_store.py`.)

## Operand staging — reserved

The warp tier's smem **operand-staging** pipeline (the `STAGE` codec's cp.async / TMA slab — formerly `_stage.py` +
the warp factorizer's `_warp_staged_kloop` / `_warp_tma_staged_kloop`) was **dropped** so both contraction tiers load
operands gmem-direct (both tiers are symmetric — neither carries a `stage`). The `STAGE` codec +
`schedule.Stage` field still land (`020_schedule` stamps them — the knob/featurizer path is intact), but they are **not
materialized**; a **symmetric** operand-staging mechanism for *both* tiers is the planned follow-up. The
mma-staging structure tests are xfailed in `tests/xfail_registry.py` (the `_STAGE` reason) until it lands.

**Shared-row staging (`_reduce`) — distinct, still present.** The fused norm→linear prologue is a `MonoidKernel`: an
input row folded by the cooperative reduce AND re-read per output column of a contraction tail (a free-axis `Loop` over an
inner reduce). `_reduce` (in `010_materialize`) stages that one row into a single `__shared__` slab (cooperatively filled)
and rewrites both readers to it. The trigger is narrow (`_has_contraction_tail`) so a plain softmax sum or a bare
reduction is untouched. This is the *reduce* tier's shared-row reuse, not the (dropped) warp operand pipeline.

## Kernel-IR peepholes

`030_stamp_types` / `040_demote_to_write_dtype` resolve element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `110_drop_redundant_syncs` collapses the defensive `Sync`s the
cooperative / shared-row templates emit (body-level only — a slab `Smem` decl flags `smem_seen`, so a load-bearing
prologue `Sync` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).
