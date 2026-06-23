# Tile lowering — enumeration + assembly over the block-DAG IR

The tile phase lowers each fused `LoopOp` to a kernel-ready `TileOp` in **two passes** over the block-DAG Tile IR
(`ir/tile/ir.py`), following `plans/tile-ir-block-dag.md`:

- **`enumeration/`** — `LoopOp` → a generative `Fork` tree over a **stored algorithm refined in place by incremental
  body moves** (F3-b): `000_build` seeds a *logical* (un-tiled) `TileGraph`, then each fork rewrites it move by move.
  This is the **search**: every variant is a point in the move/schedule space. It is split into **per-family rule
  passes** (`plans/tile-ir-block-dag.md` RF/F3-b) — `010_reduce_tile` applies the **reduce-decomposition body move**
  (re-bracket K), `020_thread_tile` pins the thread knob (no body move), `030_register_tile` applies the **free-axis
  σ-split body move** (after which the algorithm is fully tiled), `040_seal_scalar_tier` (deterministic: stamps the
  reduce regime's scalar-tier OFF sentinels), `050_stage` (the first `Schedule`-move fork — annotates
  `Schedule.staged`). The algorithm is built up across the passes, never all-at-once.
- **`assembly/`** (`010_assemble`) — the fully-tiled `TileGraphOp` → `TileOp`, in one deterministic step. No build here:
  `assemble_block` only **materializes** the stored algorithm — the register/thread tower (`_wrap_tower`) + slab
  synthesis from `Schedule.staged`. Every scheduling decision already lives on the `TileGraph` / `Schedule`, so there is
  no search here.

```
LoopOp ─000_build─▶ logical TileGraph ─reduce_decomp ─(thread)─ free_tile body moves─▶ 040_seal ─050_stage (Schedule.staged)─▶ tiled TileGraphOp ─010_assemble (materialize)─▶ TileOp
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
| `010_reduce_tile.py`  | Fork (`SEMIRING`): the reduce decomposition — `reduce_offers` → `(bk, fk, splitk)`; **applies the `reduce_decomp` body move**. |
| `020_thread_tile.py`  | Fork: the free-axis thread tile — `thread_offers` → `(thread_n, thread_m)`. Pins the thread knob, **no body move**. |
| `030_register_tile.py`| Fork: the free-axis register tile — `map_reg_offers` / `reduce_reg_offers` → `(reg_n, reg_m)`; **applies the `free_tile` body move** (the algorithm is fully tiled after). |
| `040_seal_scalar_tier.py`| Deterministic: stamp the reduce regime's scalar-tier OFF sentinels (`MMA=0 WM=0 WN=0 BR=1`). Knob-only. |
| `050_stage.py`        | Fork (`Schedule`-move): `stage_candidates` off the stored tiled `TileGraph` → a `STAGE` bitmask → `Schedule.staged[edge] = SYNC`. |
| `_iterdag.py`         | `iter_dag` — the derived iteration-DAG view (axes tagged `PARALLEL` / `REDUCE` + carrier). |
| `_classify.py`        | `classify` → `_Regime(algebra=AlgebraKind)`. |
| `_moves.py`           | `Budget` + `legal_decomps` + the offers (`thread_offers`, `map_reg_offers`, `reduce_offers`, `reduce_reg_offers`) + knob deltas. |
| `_stage.py`           | `stage_candidates` — the `stage` move's ranked offer set (AFFINE + fan-in reuse + K-tower) off the derived `Block.reads`. |
| `_knobs.py`           | The knob schema (`BN`/`BM`/`BK`/`FK`/`STAGE`/… + the composer aliases `MAP_*` / `RED_*` / `TC_*`). |
| `_build.py`           | The F3-b incremental body moves — `seed_graph` (logical block), `reduce_decomp` (K re-bracket), `free_tile` (free-axis σ-split); `build_dag` is their composition (the byte-identity oracle, for unit / equivalence callers). |

The per-pass moves serve every regime: a `MAP` nest applies only `free_tile` (`reduce_decomp` is a no-op without a
contraction); a `SEMIRING` reduce applies both — `reduce_decomp` (gated on `target_names`) then `free_tile`. The
composition order (reduce then free) reverses the old monolith (free then reduce), but the two σ-rewrites touch disjoint
axis sets (K vs the free N/M) so they commute — `build_dag` stays the byte-identity oracle for the distribution.

## `assembly/` — `assemble`

| Module | Role |
| ------ | ---- |
| `010_assemble.py` | The pass: fully-tiled `TileGraphOp` → `TileOp`. **No build** — `assemble_block` materializes the stored algorithm (tower + slab synthesis). |
| `_assemble.py`    | `assemble_block` — synthesize staging (`_slab`), then reconstruct the binding tower from `Schedule.binding` (ATOM/REGISTER/WARP/THREAD/GRID tiers) over the block's σ-rewritten `compute`. |
| `_slab.py`        | `synthesize_staging` — materialize `Schedule.staged` into one SYNC `StageBundle` per K-tower (a `Source` per staged buffer, cache axes off the consumer `Load`, GRID + serial-outer K folded to the slab origin). |
| `_tower.py`       | `_wrap_tower` — the shared innermost-first tower-building primitive (`Role` → tile flavor). |

## Coverage

Built today: **`MAP`** + scalar **`SEMIRING`** (including masked / symbolic free axes, split-K, and the `FK`
strip-mine), with **smem staging** (`stage` move, scalar tier — R1) on the `SEMIRING` reduce regimes. Recognized by
`classify` but not yet built (they re-enter the same uniform tree once their decomposition placement + assembly land):
`MONOID` cooperative-reduce, `TWISTED_MONOID` fused flash, and the `SEMIRING` warp-tier (tensor-core MMA — which also
brings warp/atom + symbolic-K staging). See `plans/tile-ir-block-dag.md` and
`plans/algebra-licensed-decomposition-moves.md`.
