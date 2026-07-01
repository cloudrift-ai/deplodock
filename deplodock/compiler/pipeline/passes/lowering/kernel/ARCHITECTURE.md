# Kernel lowering — `TileOp` → `KernelOp`

This stage turns a scheduled `TileOp` into a `KernelOp` (a thread-bound CUDA-IR body), then runs a short chain of
Kernel-IR peepholes over it. The CUDA lowering (`lowering/cuda`) renders the `KernelOp` to a `__global__` source string
afterwards.

## `010_materialize` — bind the schedule to threads (and expand the contraction)

`010_materialize` is a thin wrapper: after the split-survivor assert it makes **one** call to
`_factor.factorize(tile, root)`, the single node-kind dispatcher. `factorize` reads the node kind + role + reduce plan
off `tile.op` and picks the tier (the article's "schedule separate from combine" thesis — the op tree + `ir/tile/ops.lower`
are shared across kinds; only the partition changes):

- **Tiled `CONTRACTION`** (warp / register tile) — the high-level `Contraction` Stmt
  (`ir/tile/ir.py`) was already built **recognize-side** at fork-emit
  (`lowering/tile/_schedule._contraction_node`, resolving the operand→role binding via `_atomize.semiring_binding`), so
  `factorize` only **synthesizes its bare grid-`Write`** (needs `root.output`, so it can't ride the node) and
  **expands** it (`_factorize_contraction`) through the shared tiling layer (below); the leaf type selects the codegen
  (mma / scalar). An unbindable contraction (a non-`Load` operand) keeps the `Map` form and falls through to the scalar
  tier here. (This build was a separate `005_contract` pass, then folded into materialize, and now lives recognize-side
  so the node's `tile` / `bind` exist before scheduling — seam #1.)
- **Reduce tier** (`_factorize_reduce`, a `PLANAR` / `TWISTED` reduce — or a non-output-tiled `CONTRACTION` — whose
  `ReducePlan` cooperates / register-folds) — the reduce axis is partitioned `coop` ways across the CTA's threads and
  `reg` ways across per-thread accumulators (ILP), then a REG-tree fold, the cross-thread combine (`_combine`), and the
  projection.
- **Scalar tier** — one thread per output cell (`lower(op)` + an output-store glue).

The `Contraction` node is **one flat** Stmt — binding-driven for both atoms, with **no per-atom subclass** — that cleanly
splits the **algebra params** (what to contract: the m/n output `axes` + the `k_axis`, the leading batch `lead_axes`, the
B operand `Load` + the A `a_operand` — a gmem `Load` **or** a computed register-resident `Body` (flash PV's `P = exp(S −
M)`, produced from an in-register score, not a gmem address) — the fold accumulator `acc`, and the projection `epilogue`)
from the **schedule** (one `tile: TilePlan` field carrying the leaf `atom` — a tensor-core `AtomKind` / the scalar
`ScalarAtom`, `ir/atom.py` — plus the unit/register widths + K-chunk). The per-CTA geometry (`tile_m` / `mask_m` /
`block_threads` / …) is **derived** on the node from `tile` × `axes` (`@property`). Keeping the schedule a single swappable
field is what lets the same operand/`acc` params be tiled by a *different* `TilePlan` — the seam the flash inner QK/PV
reuse needs.

A symbolic / non-divisible tail is **clamp-to-identity** (the masked overhang folds a no-op or guards its store); the
dynamic-grid tier ceil-divides the launch and threads the runtime extent as an `int seq_len` arg.

### The one factorizer — dispatch, atoms, and the reduce tier (`_factor.py` / `_tiling.py`)

`_factor.factorize(tile, root)` is the **single emitter** every `TileOp` root lowers through. It reads the node kind +
role + reduce plan off `tile.op` and routes to `_factorize_contraction` (a tiled `Contraction`), `_factorize_reduce`
(a cooperative / ILP `PLANAR` / `TWISTED` reduce), or the inline **scalar tier** (`lower(op)` + `with_store`, one
thread per output cell). All three tiers, plus the shared-row staging helpers, live in `_factor.py`. The one exception
is the **tensor-core flash** (the mma-flash tree — a `TWISTED` `Reduction` whose `source`/`partial` `Contraction`s carry
a warp `TilePlan`, built by `DEPLODOCK_CHAIN`): `factorize` dispatches it to `_flash_warp.factorize_flash`, the
fragment-resident FA-2 warp chain (score C-fragments → in-register online softmax → the C→A `flash_pv_smem` handoff →
the PV mma → `O/l`). It is a distinct single-warp *orchestration* (the streaming softmax + the register-resident A
handoff — the one genuinely-new primitive) but reuses the **shared** mma / fragment / store kernel-IR nodes, so there is
no divergent tensor-core codegen, only a distinct schedule of the same primitives.

**The contraction factorization — two atoms.** `_factorize_contraction` is the atom-generic path — there is no per-atom
variant, and **no per-atom geometry object**. It expands any `Contraction` by tiling a **leaf atom** four ways through
the layer in `_tiling.py`:
`grid_tile(unit_tile(register_tile(atomize(...))))` — **GRID** block / **UNIT** / **REGISTER** / **ATOM**. The tiling
geometry (`tile_m` / `mask_m` / `m_b` / `m_uvar` / `block_threads` / `lanes` / …) is **derived on the `Contraction` node
itself** (`@property`, from the `tile` schedule × the output axes); `factorize` reads it straight off `c` and hands
`grid_tile` the codegen in two halves: `_factor.reduce_codegen` — the reusable, **sink-agnostic** `(state_decls,
reduce_region)` (operand fragments + the K-loop, dispatched off the atom) — and a per-cell **sink** `store`. The
default is the matmul `_factor.store_sink`; `factorize(c, store=…)` swaps it. Per-atom diff:

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

## Operand staging — the warp-tier smem pipeline (`STAGE` codec → `Stage`)

The warp (mma) tier stages its reused gmem operands through an smem slab, driven off the node's `STAGE` codec →
`schedule.Stage`. The transport primitives (the cooperative gmem→smem fill loops + the cp.async commit/wait and TMA
mbarrier handshakes) live in **`_stage.py`**; the K-loop that schedules them onto the `Contraction` geometry lives in
`_factor.py` (`_warp_staged_kloop` / `_warp_tma_staged_kloop`, plus the shared inner `ldmatrix` drain
`_staged_inner_atom_loop`). `_mma_stage_plan` decodes the `Stage` once (TMA > cp.async > gmem-direct, with the
eligibility rules `_can_stage_warp[_tma]`) and both `_mma_state` (which slots the operand fragments) and `_mma_reduce`
(which emits the loop) read it. The `Stage` spells two buffering levels: `d<depth>` is the gmem→smem ring (cp.async /
TMA prefetch over the K-slab loop), `p<reg_depth>` is the smem→register double-buffer (the `ldmatrix` ping-pong over the
inner atom-K steps). Staging is a **pure perf transform** — an ineligible kernel (transposed-B, masked N, symbolic /
non-divisible K, or a computed-A flash operand) silently falls back to gmem-direct, and a staged kernel is
**bit-identical** to its gmem-direct baseline. It is **pin-only** today (`DEPLODOCK_STAGE`); auto-fork enumeration is a
follow-up. The **scalar** contraction tier is gmem-direct (no operand slab).

**Shared-row staging (`_factorize_reduce`) — a distinct reduce-tier mechanism.** The fused norm→linear prologue is a
cooperative reduce: an input row folded by the cooperative reduce AND re-read per output column of a contraction tail (a
free-axis `Loop` over an inner reduce). `_factorize_reduce` (in `_factor.py`) stages that one row into a single
`__shared__` slab (cooperatively filled, `sync` transport) and rewrites both readers to it. The trigger is narrow
(`_has_contraction_tail`) so a plain softmax sum or a bare reduction is untouched. This is the *reduce* tier's shared-row
reuse, NOT the warp operand pipeline above — folding the two into one `Stage`-driven mechanism (a `Stage` on the reduce's
input) is the remaining Phase-2 purge item.

## Kernel-IR peepholes

`030_stamp_types` / `040_demote_to_write_dtype` resolve element dtypes; `050_vectorize_loads` / `080_vectorize_stores` /
`095_interleave_loads` pack/reorder memory ops; `110_drop_redundant_syncs` collapses the defensive `Sync`s the
cooperative / shared-row templates emit (body-level only — a slab `Smem` decl flags `smem_seen`, so a load-bearing
prologue `Sync` is correctly retained; `with_bodies` preserves the cooperative tile's `block_threads`).
