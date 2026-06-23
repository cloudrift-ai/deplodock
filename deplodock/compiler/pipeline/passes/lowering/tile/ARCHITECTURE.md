# Tile lowering — enumeration + assembly over the block-DAG IR

The tile phase lowers each fused `LoopOp` to a kernel-ready `TileOp` in **two passes** over the block-DAG Tile IR
(`ir/tile/ir.py`), following `plans/tile-ir-block-dag.md`:

- **`enumeration/`** — `LoopOp` → a generative `Fork` tree over a **stored algorithm refined in place by incremental
  body moves** (F3-b): `000_build` seeds a *logical* (un-tiled) `TileGraph`, then each fork rewrites it move by move.
  This is the **search**: every variant is a point in the move/schedule space. It is split into **per-family rule
  passes** (`plans/tile-ir-block-dag.md` RF/F3-b). The **scalar** chain: `010_reduce_tile` applies the
  **reduce-decomposition body move** (re-bracket K), `020_thread_tile` pins the thread knob (no body move),
  `030_register_tile` applies the **free-axis σ-split body move** (after which the algorithm is fully tiled),
  `040_seal_scalar_tier` (deterministic: stamps the reduce regime's scalar-tier OFF sentinels), `050_stage` (the first
  `Schedule`-move fork — annotates `Schedule.staged`). The **warp-tier** chain (R4 `atomize`): `005_tensorize` forks the
  atom-vs-scalar choice, `006_warp_geometry` / `008_warp_reg` pin the warp counts + register cells, `009_warp_build`
  applies the **warp build body move** (four-way GRID/WARP/REGISTER/ATOM σ-split + K re-bracket at `atom_k` granularity +
  fuse the cell into an `Mma`); the scalar passes gate off when an `MMA` atom is pinned, and `050_stage` then stages the
  warp operands too. The algorithm is built up across the passes, never all-at-once.
- **`assembly/`** (`010_assemble`) — the fully-tiled `TileGraphOp` → `TileOp`, in one deterministic step. No build here:
  `assemble_block` only **materializes** the stored algorithm — the register/thread tower (`_wrap_tower`) + slab
  synthesis from `Schedule.staged`. Every scheduling decision already lives on the `TileGraph` / `Schedule`, so there is
  no search here.

```
                          ┌─ scalar: reduce_decomp ─(thread)─ free_tile ─ 040_seal ─┐
LoopOp ─000_build─▶ logical TileGraph ─005_tensorize┤                                         ├─ 050_stage ─▶ tiled TileGraphOp ─010_assemble (materialize)─▶ TileOp
                          └─ warp:   006/008 geom+reg ─ 009_warp_build (atomize) ───┘
```

## The block-DAG Tile IR (`ir/tile/ir.py`)

One file, two comment-block sections:

- **ENUMERATION** — the *invariant algorithm* + the *variant* `Schedule` the composer searches: `Block`
  (`name + domain + compute`), `Buffer`, `Edge`, `Schedule`, `TileGraph`. The derived projections
  (`Block.reads`/`writes`/`carrier`/`atom`, `TileGraph.edges`) are **computed on demand, never stored** — the same
  discipline as `Loop.algebra_kind`, so they can't drift and don't enter `op_cache_key`. `TileGraphOp` wraps a chosen
  `TileGraph` as a graph node (the enumeration-pass output; `op_cache_key` keys it on `TileGraph.structural_key` +
  knobs).
- **MATERIALIZED** — what `assemble` emits: `TileOp` + the typed tile flavors (`GridTile` / `ThreadTile` /
  `RegisterTile` / `WarpTile` / `AtomTile` / `SerialTile`) + `StageBundle` / `Source` / `WarpSpecialize` / `AsyncWait` /
  `Atom`. (Slated for removal once `assemble` emits `KernelOp` directly.)

## INVARIANT: a purely algebraic moveset, no specializations

**The composer never dispatches on a named shape** (matmul / pointwise / attention / RMSNorm / gated-MLP). It
dispatches on the reduce axes' **carrier algebra** (`ir/algebra.py::AlgebraKind`) — `MAP` / `SEMIRING` / `MONOID` /
`TWISTED_MONOID` — read off the body by `_tree.classify`. A "matmul" is just a `SEMIRING`; "pointwise" a `MAP`;
"RMSNorm" / "softmax" a `MONOID`; "flash attention" a `TWISTED_MONOID`. Adding a model architecture is **never** a new
branch — it is the same move set on its algebra.

Every move is gated by a **carrier trait**, not a shape (`_moves.legal_decomps`):

- **`tile_axis`** on a free (`PARALLEL` / `MAP`) axis — split it into GRID / THREAD / REGISTER. Always legal (no
  carrier, nothing to recombine).
- **the reduce decomposition** on a contraction axis — `associative` → split at all; `commutative` → a partition
  factor (split-K across CTAs / cooperative across threads); `has_identity` → mask a non-divisible / symbolic axis
  with the carrier identity. The recombine is derived (`carrier.combine_partials`); the hardware realization
  (atomic / shuffle / tree / mma) is a downstream cost choice keyed off placement.
- **`atomize`** (a body move, R4) — fuse a `SEMIRING` contraction cell `[Load, Load, mul, Accum]` into one `Mma` (the
  tensor-core atom). Legal iff the carrier is `SEMIRING` (`⊗` distributes over `⊕`) and the atom is eligible
  (`_atom.eligible_atoms`: cc ≥ 8.0, operand dtype, K / output extents divisible by the cell, a foldable pointwise
  epilogue). `Block.atom` then *derives* from the `Mma`; `assemble` + `kernel/005` synthesize the
  `RegFragment`/`ldmatrix`/`mma.sync`/`RegStore` chain.
- **`stage`** (a `Schedule`-only move, no body edit) — annotate a reused gmem read `Schedule.staged[edge] = SYNC`;
  legal iff the read is AFFINE and has fan-in reuse (a parallel axis absent from its free axes) over a K-tower.
  `assemble` synthesizes the smem slab + cooperative producer; nothing is stored in the algorithm.

The only algebra-*conditioned* heuristic is a **ranking** cost model, not a code path: the free-axis register menu is
bandwidth-biased for a `MAP` nest (`map_reg_offers`) and compute/ILP-biased for a reduce regime (`reduce_reg_offers`).
This is the tile-phase instance of the global rule in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — *No
shape-specific pattern matching.*

## `enumeration/` — the move composer

| Module | Role |
| ------ | ---- |
| `000_build.py`        | Seed pass: `LoopOp` → `iter_dag` + `classify` → a `TileGraphOp` carrying the **logical** `TileGraph` (`seed_graph`) + dag + regime. |
| `005_tensorize.py`    | Fork (warp tier, R4): atom-vs-scalar — `eligible_atoms` (gate) → `MMA=<kind>` per eligible atom + the scalar fallback (warp ranked first). An explicit scalar `BN`/`BM` pin defers to scalar. |
| `006_warp_geometry.py`| Fork (warp): the per-CTA warp counts — `warp_offers` → `(WM, WN)`. Knob-only. |
| `008_warp_reg.py`     | Fork (warp): the per-warp register cells — `warp_reg_offers` → `(FM, FN)`. Knob-only. |
| `009_warp_build.py`   | Fork (warp): the K chunk — `warp_bk_offers` → `BK`; **applies the `warp_build` body move** (four-way σ-split + K re-bracket + `atomize` the cell → `Mma`). |
| `010_reduce_tile.py`  | Fork (scalar `SEMIRING`): the reduce decomposition — `reduce_offers` → `(bk, fk, splitk)`; **applies the `reduce_decomp` body move**. Skips on a warp variant. |
| `020_thread_tile.py`  | Fork (scalar): the free-axis thread tile — `thread_offers` → `(thread_n, thread_m)`. Pins the thread knob, **no body move**. Skips on a warp variant. |
| `030_register_tile.py`| Fork (scalar): the free-axis register tile — `map_reg_offers` / `reduce_reg_offers` → `(reg_n, reg_m)`; **applies the `free_tile` body move** (the algorithm is fully tiled after). |
| `040_seal_scalar_tier.py`| Deterministic: stamp the reduce regime's scalar-tier OFF sentinels (`MMA=0 WM=0 WN=0 BR=1`). Knob-only; skips on a warp variant (it carries `MMA`). |
| `050_stage.py`        | Fork (`Schedule`-move): `stage_candidates` off the stored tiled `TileGraph` → a `STAGE` bitmask → `Schedule.staged[edge] = SYNC` (scalar **and** warp operands; the transposed-B operand is excluded — gmem-direct). |
| `_iterdag.py`         | `iter_dag` — the derived iteration-DAG view (axes tagged `PARALLEL` / `REDUCE` + carrier). |
| `_classify.py`        | `classify` → `_Regime(algebra=AlgebraKind)`. |
| `_atom.py`            | The warp-tier gate: `eligible_atoms` (per-atom eligibility over the dag + dtypes + cc) + `classify_matmul_operands` (the one A/B layout decision, shared by the gate and the `atomize` move). |
| `_moves.py`           | `Budget` + `legal_decomps` + the offers (`thread_offers`, `map_reg_offers`, `reduce_offers`, `reduce_reg_offers`, `warp_offers` / `warp_reg_offers` / `warp_bk_offers`) + knob deltas. |
| `_stage.py`           | `stage_candidates` — the `stage` move's ranked offer set (AFFINE + fan-in reuse + K-tower) off the derived `Block.reads`; excludes the transposed-B operand. |
| `_knobs.py`           | The knob schema (`BN`/`BM`/`BK`/`FK`/`STAGE`/`MMA`/`WM`/`WN`/… + the composer aliases `MAP_*` / `RED_*` / `TC_*`). |
| `_build.py`           | The F3-b incremental body moves — `seed_graph` (logical block), `reduce_decomp` (K re-bracket), `free_tile` (free-axis σ-split), `warp_build` (the warp-tier four-way split + `atomize`); `build_dag` is the scalar composition (the byte-identity oracle). |

The per-pass moves serve every regime: a `MAP` nest applies only `free_tile` (`reduce_decomp` is a no-op without a
contraction); a `SEMIRING` reduce applies both — `reduce_decomp` (gated on `target_names`) then `free_tile`. The
composition order (reduce then free) reverses the old monolith (free then reduce), but the two σ-rewrites touch disjoint
axis sets (K vs the free N/M) so they commute — `build_dag` stays the byte-identity oracle for the distribution.

## `assembly/` — `assemble`

| Module | Role |
| ------ | ---- |
| `010_assemble.py` | The pass: fully-tiled `TileGraphOp` → `TileOp`. **No build** — `assemble_block` materializes the stored algorithm (tower + slab synthesis). |
| `_assemble.py`    | `assemble_block` — synthesize staging (`_slab`), then reconstruct the binding tower from `Schedule.binding` (ATOM/REGISTER/WARP/THREAD/GRID tiers) over the block's σ-rewritten `compute`; the warp tier's `Mma` cell rides inside the `AtomTile` for `kernel/005` to lower. |
| `_slab.py`        | `synthesize_staging` — materialize `Schedule.staged` into one SYNC `StageBundle` per K-tower (a `Source` per staged buffer, cache axes off the consumer `Load`, GRID + serial-outer K folded to the slab origin). The per-axis `AffineAddressing.block` multiplier is **derived from the σ coefficients** — `()` for a scalar tile, atom-strided for a warp tile (the ATOM cell rides the surviving cache axis); a size-1 REGISTER cell is dropped first so its atom stride migrates to the warp axis. |
| `_tower.py`       | `_wrap_tower` — the shared innermost-first tower-building primitive (`Role` → tile flavor). |

## Coverage

Built today: **`MAP`** + scalar **`SEMIRING`** (including masked / symbolic free axes, split-K, and the `FK`
strip-mine) + the **warp-tier `SEMIRING`** (tensor-core `mma.sync` via the R4 `atomize` move — matmul, residual /
pointwise / causal-mask epilogue fold, transposed-B, symbolic M/N), with **smem staging** (`stage` move) on both the
scalar reduce regimes (R1) and the warp operands (atom-strided slab + `ldmatrix`). Recognized by `classify` but not yet
built (they re-enter the same uniform tree once their decomposition placement + assembly land): `MONOID`
cooperative-reduce, `TWISTED_MONOID` fused flash, the cross-CTA `partition_reduce` split-K, and the R5 transports
(cp.async / TMA). Remaining R4 follow-ups still quarantined: the masked cooperative-load clamp + non-divisor
`real_extent` (need scalar-offer env-pin honoring) and the gmem-direct unstaged atom. See `plans/tile-ir-block-dag.md`
and `plans/algebra-licensed-decomposition-moves.md`.
