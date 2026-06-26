# Tile lowering — enumeration + assembly over the block-DAG IR

> **Knob schema — algebra-native `MOVE@element` (`_families.py`).** The composer keys each knob on a *move applied to a
> DAG element*, not on a rank-2 GEMM letter (the grammar + codecs live in `enumeration/_families.py`). `op.knobs` speaks:
> `SPLIT@<free-axis>` = `"par×reg"` (the legacy `BN`/`FN` thread tile **and** `WN`/`FN` warp tile — the tier is read off
> the cell's `ATOM`, not the value; `BM`/`FM`/`WM` are the outer axis); `REDUCE@<reduce-axis>` = `"s/f/c/t"` (legacy
> `BK`/`FK`/`SPLITK`/`BR`, **per reduce axis** — flash's two reduce axes become first-class); `ATOM@<cell>` (legacy
> `MMA`; the matmul's single cell is the structural constant `out`); `PLACE@<edge>` = `place[:xport]` (the legacy
> `STAGE`→`smem`, `TMA`→`:tma`/`:sync`, `CHAIN`→`PLACE@score=inline`, and `CUT`→`PLACE@cone=cut` — a demoted cone
> materialized to a gmem intermediate; `cut` is distinct from an operand `gmem`-direct read so the structural recognizer
> keys on the value). The **implementation
> reads native keys / the IR, never legacy names**; the legacy `DEPLODOCK_BN`/`BK`/`MMA`/`STAGE`/… env pins + legacy
> golden YAMLs resolve through the ingest-only mapper `_knob_legacy.py`. `Schedule.staged`/`binding` stay the codegen
> source of truth, so the rename is byte-identical. The per-pass descriptions below still use the **legacy** names for
> continuity — read them through that table. The **prior featurizer** `knob.knob_features` is likewise schema-agnostic:
> it reads the native `SPLIT@`/`REDUCE@`/`ATOM@` keys via `knob._free_slots` / `_reduce_decomp` (free axes canonicalized
> by `par` — dag-free, so the `n`/`m` role split needs no `IterDag`), falling back to the legacy `BN`/`BM`/`MMA`/… names
> so a legacy-recorded golden YAML still featurizes identically; `knob.tile_signature` is the matching bridge that pairs
> a legacy golden against the native enumeration candidates (`scripts/golden_knob_heuristics.py`,
> `search/analytic.evaluate_golden`). (The learned prior is cold — its `prior.json` is deleted — so the greedy cold pick
> is the `AnalyticPrior` argmin over those native features; before the featurizer spoke native it scored every native row
> identically — a degenerate emission-order pick. The `BN`/`MMA`/… `Knob` descriptors stay registered for the legacy
> ingest/display and the `MMA` atom featurizer.)

The tile phase lowers each fused `LoopOp` to a kernel-ready `TileOp` in **three passes** over the block-DAG Tile IR
(`ir/tile/ir.py`):

- **`split/`** — the **pre-build structural-fork head** (R7 `010_split_demoted`). Runs FIRST, on the still-un-tiled
  `LoopOp`, and may un-fuse a **demoted matmul** (a multiply operand reading a computed / K-folded cone instead of a
  plain `Load` — fusion merged a producer chain into the matmul reduce, killing the warp tier) into a producer/consumer
  kernel set: an `xn` operand-materialization producer beside a clean gemm consumer, returned as a `Graph` fragment the
  engine splices (a kernel-set change → the **outer** two-level tree). The familiar instance is the **score-materializing
  SDPA**: the fused softmax-prologue + P@V `k_sdpa_reduce` un-fuses into a softmax-normalizing `xn` producer + a clean
  (static **or** symbolic-K) gemm consumer that both lower. The decision is a **derived tier query**
  (`enumeration/_cut.py`, R7 edge placement): it **forces** the split (single option) iff the fused body is
  `UNBUILDABLE` — a demoted matmul whose cone operand keeps it below any buildable tier, which materializing the operand
  strictly raises. (`tier(inline) is UNBUILDABLE` ⟺ `classify(fused) is None` — a cone-operand cell is never
  atom-eligible — so this reproduces the legacy force condition through the lattice.) The decision rides the `CUT`
  knob — a width-1 `BINMASK` over `cut_offers`' ranked cuttable edges (`"0"` keep, `"1"` cut). The buildable-fused
  keep(SMEM)-vs-cut(GMEM) *fork* is offered when the cone fuses on-chip (`seed_fused` expressible), greedy default
  keep(SMEM); a non-fusible cone (multi-cone / multi-accum) forces the cut. The cut
  names its products inline and `_assemble_fragment` re-stamps the `S_*` structural features (the cut runs after
  `loop/stamp`, so the fragments don't re-flow through it). The bespoke monolith is dissolved into three pieces: the
  **offer policy** (`enumeration/_cut.py::cut_offers`, the derived `Tier` lattice), the **fission**
  (`split/_extract.py::extract_block`), and the **thin fork shell** (`010_split_demoted.py`, holding no decision logic).
  `split/` survives as a pre-build phase because the demoted matmul never classifies as a buildable seed
  (`010_build` would `RuleSkip` it), so the cut can't yet fold into `enumeration/` as an edge-placement move.
- **`enumeration/`** — `LoopOp` → a generative `Fork` tree over a **stored algorithm refined in place by incremental
  body moves** (F3-b): `010_build` seeds a *logical* (un-tiled) `TileGraph`, then each fork rewrites it move by move.
  This is the **search**: every variant is a point in the move/schedule space. It is split into **per-family rule
  passes**. The **scalar** chain: `060_reduce_tile` applies the
  **reduce-decomposition body move** (re-bracket K), `090_thread_tile` pins the thread knob (no body move),
  `100_register_tile` applies the **free-axis σ-split body move** (after which the algorithm is fully tiled),
  `110_seal_scalar_tier` (deterministic: stamps the reduce regime's scalar-tier OFF sentinels), `120_stage` (the first
  `Schedule`-move fork — annotates `Schedule.staged`). The **warp-tier** chain (R4 `atomize`): `020_tensorize` forks the
  atom-vs-scalar choice, `030_warp_geometry` / `040_warp_reg` pin the warp counts + register cells, `050_warp_build`
  applies the **warp build body move** (four-way GRID/WARP/REGISTER/ATOM σ-split + K re-bracket at `atom_k` granularity +
  fuse the cell into an `Mma`); the scalar passes gate off when an `MMA` atom is pinned, and `120_stage` then stages the
  warp operands too. The **MONOID** chain (R2/R6): `070_coop_reduce` is the **single MONOID pass** — it owns BOTH the
  flat cooperative reduce (softmax / rmsnorm / mean / max) AND the streaming flash (online-softmax over a nested QK^T),
  applying the **`monoid_build` body move** (apply the reduce-decomposition tower to each contraction axis the DAG
  exposes — recursive for the nested stream — with the `K_c` cooperative-THREAD lane on the primary axis + free-axis
  σ-split with the register tile forced to 1). Only the **offer set** differs by regime (the algebra-conditioned ranking
  heuristic, not a code path): a flat reduce searches `(bk, fk, br)`; a streaming flash searches the free THREAD tile with
  `BK=FK=1` and `BR` over the static KV axis. The scalar passes `090`/`100`/`110` + `120_stage` gate off `MONOID` (a
  monoid reduce stays smem-free). The algorithm is built up across the passes, never all-at-once.
- **`assembly/`** (`010_assemble` + `020_peel` + `030_mark_unroll`) — the fully-tiled `TileGraphOp` → `TileOp`, in one
  deterministic step followed by deterministic post-passes. No build here: `assemble_block` only **materializes** the
  stored algorithm — the register/thread tower (`_wrap_tower`) + slab synthesis from `Schedule.staged`; `020_peel` then
  software-pipelines a ring-staged (TMA) K loop; `030_mark_unroll` flags small loop nests (total trips ≤ 64) for
  `#pragma unroll` so the FMA inner reduce gets register-resident operand reuse + ILP. Every scheduling decision already
  lives on the `TileGraph` / `Schedule`, so there is no search here.

```
              ┌─ split? (xn producer + clean gemm)        ┌─ scalar: reduce_decomp ─(thread)─ free_tile ─ 110_seal ─┐
LoopOp ─split─┤                              ─010_build─▶ logical TileGraph ─020_tensorize┤─ monoid: 070_coop_reduce (monoid_build) ─┼─ 120_stage ─ 130_transport ─▶ tiled TileGraphOp ─010_assemble ─ 020_peel ─ 030_mark_unroll ─▶ TileOp
              └─ keep fused ─────────────────▶            (per LoopOp)         └─ warp: 030/040 geom+reg ─ 050_warp_build ───────┘
```

## The block-DAG Tile IR (`ir/tile/ir.py`)

One file, two comment-block sections:

- **ENUMERATION** — the *invariant algorithm* + the *variant* `Schedule` the composer searches: `Block`
  (`name + domain + compute`), `Buffer`, `Edge`, `Schedule`, `TileGraph`. The derived projections
  (`Block.reads`/`writes`/`carrier`/`atom`, `TileGraph.edges`, `TileGraph.placement(edge)`) are **computed on demand,
  never stored** — the same discipline as `Loop.algebra_kind`, so they can't drift and don't enter `op_cache_key`.
  `TileGraph.placement(edge)` is the unifying **edge-placement** view: `INLINE` (default, register/gmem-direct), `SMEM`
  (`Schedule.staged` — today's `stage` move), or `GMEM` (a cross-launch-group cut — the buffer lives in gmem) read off
  `Schedule.staged` + `Schedule.launch`, so `stage` and `split` are two values of one query. `TileGraphOp` wraps a
  chosen `TileGraph` as a graph node (the enumeration-pass output; `op_cache_key` keys it on `TileGraph.structural_key`
  + knobs).
- **MATERIALIZED** — what `assemble` emits: `TileOp` + the typed tile flavors (`GridTile` / `ThreadTile` /
  `RegisterTile` / `WarpTile` / `AtomTile` / `SerialTile`) + `StageBundle` / `Source` / `WarpSpecialize` / `AsyncWait` /
  `Atom`. (Slated for removal once `assemble` emits `KernelOp` directly.)

## INVARIANT: a purely algebraic moveset, no specializations

**The composer never dispatches on a named shape** (matmul / pointwise / attention / RMSNorm / gated-MLP). It
dispatches on the reduce axes' **carrier algebra** (`ir/algebra.py::AlgebraKind`) — `MAP` / `SEMIRING` / `MONOID` —
read off the body by `_tree.classify`. A "matmul" is just a `SEMIRING`; "pointwise" a `MAP`; "RMSNorm" / "softmax" /
"flash attention" all a `MONOID` (a twisted monoid **is** a monoid — transport of structure). Flash differs only in its
*schedule*: a tuple `Monoid` carrier streaming over a nested contraction is a **derived** structural property
(`dag.streaming`, computed on demand — never stored) that the one `MONOID` pass (`070_coop_reduce`) reads to pick the
streaming offer set; the build move (`monoid_build`) is shared with the cooperative reduce. Adding a model
architecture is **never** a new branch — it is the same move set on its algebra.

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
  legal iff the read is AFFINE and has fan-in reuse over a K-tower — a **non-degenerate intra-CTA** parallel axis
  (`THREAD` / `WARP`, extent > 1) absent from its free axes, so many threads in one CTA share the cached tile. A `GRID`
  axis is a separate CTA (own smem, no sharing) and an extent-1 axis is a single thread, so neither counts: a degenerate
  M-tile (`BM·FM=1`) does NOT stage the `wl[N,K]` weight into a slab that would blow the smem budget — it lowers
  gmem-direct, where the masked-overhang read is clamped by the kernel pass `kernel/009_clamp_masked_gmem_reads` (the
  read-side edge-clamp the staged path gets from `_stage_expand`).
  `assemble` synthesizes the smem slab + cooperative producer; nothing is stored in the algorithm.

> **Masked-tile index arithmetic lives in one module — `lowering/_masking.py`.** The edge-clamp
> (`mask_index(coord, bound)` → `coord<bound ? coord : bound-1`, a harmless in-bounds duplicate the gated `Write`
> drops), the zero-fill predicate (`in_bounds(coord, bound)` → `coord<bound`, gating the masked-K value→0 `Select`),
> the extent helpers (`ext_expr`/`ext_minus_one`), and the symbolic-K locator (`locate_symbolic_k` — which staged
> source dim carries the symbolic contraction extent) are defined once and consumed by `kernel/009` (gmem-direct
> clamp), `kernel/_stage_expand` (cooperative-load clamp + masked-K zero-fill), and `assembly/_slab` (`Source.kmask`
> stamp). The warp/MMA tier (`kernel/005`) renders its own M/N clamp + K zero-fill directly in CUDA C from
> `(coord, bound)` tuples on the `Mma`, so it builds no `Expr` clamps and does not route through `_masking` — it
> shares the *concept*, not the construction.

The only algebra-*conditioned* heuristic is a **ranking** cost model, not a code path: the free-axis register menu is
bandwidth-biased for a `MAP` nest (`map_reg_offers`) and compute/ILP-biased for a reduce regime (`reduce_reg_offers`).
This is the tile-phase instance of the global rule in [`../../ARCHITECTURE.md`](../../ARCHITECTURE.md) — *No
shape-specific pattern matching.*

## `split/` — pre-build structural forks

| Module | Role |
| ------ | ---- |
| `005_warp_chain.py` | Pre-build pass (**fused tensor-core flash**) — runs FIRST, before `010_split_demoted`. When the un-tiled nest is a streaming `MONOID(SEMIRING)` chain (`dag.chain`) in the fused-TC scope (fp16/bf16, causal or non-causal, `D%16==0`; **static `S%16==0` OR symbolic `seq_len`**; equal-head OR **GQA** `head//group`), it replaces the flash `LoopOp` with a single `TileOp` running the warp-chain kernel (`assembly/_warp_chain.build_warp_chain_tileop`); the engine splices it and it flows through the standard kernel passes. A **symbolic** `S` (`_flash_params` returns `seq_var`, B/H/D static) ceil-divides the KV-stream / query-tile extents over `Var(seq_var)` and masks the partial final tile (score-boundary mask + clamped loads + guarded store) — Phase 1; GQA (`group>1`) reads K/V at the kv-head `head//group` — Phase 2 of [`plans/smem-tiled-symbolic-flash.md`](../../../../../../plans/smem-tiled-symbolic-flash.md). **Routing (Phase 3):** a **symbolic** flash fires the warp chain BY DEFAULT (the deployed symbolic default — ~100× over the scalar streaming nest at seq=512, ~0.6× eager); a **static** flash stays a `DEPLODOCK_CHAIN=1` opt-in. Out of scope → `RuleSkipped` (falls through to `chain_build` / the materialized path — the correct fallback for fp32 / odd `D` / additive-mask symbolic flashes). |
| `010_split_demoted.py` | The **thin fork shell** (**structural**, R7) — holds no decision logic. `rewrite`: idempotent (`CUT` guard), runs before `enumeration/010_build` on the un-tiled body; calls `_extract.extract_block` (expressibility — `None` if not a cuttable demotion) + `_extract.seed_fused` (the keep(SMEM) realization) then `_cut.cut_offers` (the offer verdict), emitting the `[keep, cut]` fork (greedy default keep) — **forcing** the split `Graph` only when the fused body is `UNBUILDABLE` AND not on-chip-fusible. `CUT` is a width-1 `BINMASK` over the ranked cuttable edges (`"0"` keep / `"1"` cut). Honors `DEPLODOCK_CUT` (alias `DEPLODOCK_SPLIT_CONE`); stamps `CUT` on both branches. |
| `_cut.py` (in `enumeration/`) | The **offer policy** — the derived `Tier` lattice (`UNBUILDABLE` < `MAP` < `SCALAR_REDUCE` < `COOP_REDUCE` < `WARP_MMA`) + `tier(dag)` (built on `_atom.eligible_atoms` + `_classify.classify`) + `cut_offers(loop_op) -> CutDecision`. Offers iff `tier(inline) is UNBUILDABLE` (forces when also not `smem_fusible`): materializing the demoted operand strictly raises the consumer's tier. Returns the ranked `offers` tuple (one whole-cone offer today → width-1 `CUT` mask; the per-edge list is the additive-widening seam). One auditable place for the which-cuts policy; `eval`-introspectable. |
| `_extract.py` | The **fission** (`extract_block` + helpers, relocated from the legacy monolith): classify the body into `(leading, rows, prologue, outer_n, k_loop)`, backward-slice each computed/K-folded multiply-operand cone, build one `xn` producer per cone class + the rebuilt consumer (+ per-accum `mm_i` gemms for a multi-accum cell), wired into a `Graph` fragment by `_assemble_fragment` (which re-stamps `S_*` structural features). Reuses `lowering/_predicates` (`is_matmul_reduce` / `segmentable_k_extent`, the shared structural vocabulary) + `Body.backward_cone` / `defs_die_at`. Returns `None` (its expressibility check) for any shape it can't cleanly cut. |

## `enumeration/` — the move composer

| Module | Role |
| ------ | ---- |
| `010_build.py`        | Seed pass: `LoopOp` → `iter_dag` + `classify` → a `TileGraphOp` carrying the **logical** `TileGraph` (`seed_graph`) + dag + regime. Calls `_validate.validate_pins` first (greedy only) — a force-pinned env knob foreign to the op's tier is a hard error, not a silent drop. |
| `_validate.py`        | **Strict per-op knob-pin validation** (greedy `compile`/`run` only — the tune search sets `ctx.validate_pins=False`). Each kernel lowers on ONE `Tier` (MAP / scalar SEMIRING / warp MMA / **MONOID** — the flat cooperative reduce AND the streaming flash share the one `monoid_build` move + knob slice, so they are one tier; the streaming schedule is a derived structural property, not a separate tier), each owning a disjoint knob slice; `validate_pins(algebra)` intersects the tiers every force-pinned `DEPLODOCK_<KNOB>` is legal on and raises `KnobPinError` when empty (e.g. `BN`/`BM`/`BR`/`FK` beside a warp `MMA=<kind>`, or `WM`/`WN` on a scalar `MMA=0`). The K-chunk knobs `BK`/`FK` are legal on the MONOID tier (split-KV / serial re-bracketing is associativity-licensed on the nested monoid too — whether a given flash shape realizes them is the pipeline's job, like `SPLITK` on a non-linear matmul). Value-aware: a universal / OFF pin (`SPLITK=1`, `FK=1`, `BR=1`, `FM=1`, empty `STAGE`, `TMA=0`) constrains nothing; `SPLITK>1` is legal on scalar AND warp (R3 atomic-free split-K). `STAGE`/`TMA` are legal only on the **staged** tiers (scalar / warp) — a MONOID reduce is smem-free and a pointwise MAP has no K-tower, so a `STAGE`/`TMA` pin there refuses (`130_transport` promotes any staged matmul with a ringable K loop — the warp atom and the scalar register-tiled SGEMM alike). |
| `020_tensorize.py`    | Fork (warp tier, R4): atom-vs-scalar — `eligible_atoms` (gate) → `MMA=<kind>` per eligible atom + the scalar fallback (warp ranked first). An explicit scalar `BN`/`BM` pin defers to scalar. |
| `030_warp_geometry.py`| Fork (warp): the per-CTA warp counts — `warp_offers` → `(WM, WN)`. Knob-only. |
| `040_warp_reg.py`     | Fork (warp): the per-warp register cells — `warp_reg_offers` → `(FM, FN)`. Knob-only. A fully-pinned `(DEPLODOCK_FM, DEPLODOCK_FN)` is authoritative and bypasses the `_MAX_WARP_CELLS` *search* ceiling (the ceiling prunes auto-enumerated candidates, not explicit pins). |
| `050_warp_build.py`   | Fork (warp): the K chunk — `warp_bk_offers` → `BK`; **applies the `warp_build` body move** (four-way σ-split + K re-bracket + `atomize` the cell → `Mma`). A **symbolic (masked) K** (`_classify` now admits a symbolic-K `SEMIRING`, tiling K at the `Dim` hint) ceil-divides `K_o` in `_replace_k_warp` — the loop bound is the runtime `ceil(seq_len/(BK·atom_k))`, so seq > hint is covered **and** `seq_len` enters the kernel signature; the `dpl_mma_load_*_kzero` helpers (`kernel/005`) zero-fill the partial final K tile (a clamped duplicate would corrupt the reduction). |
| `070_coop_reduce.py`  | Fork (**MONOID**, R2/R6): the **single MONOID pass** — owns the flat cooperative reduce (softmax / rmsnorm / mean / max) AND the streaming flash (online-softmax over a nested QK^T — a twisted monoid is a monoid, selected structurally via `op.dag.streaming`). **Applies the `monoid_build` body move** (the reduce-decomposition tower on each contraction axis the DAG exposes — recursive for the nested stream — `K_c` cooperative-THREAD lane on the primary axis + free-axis σ-split, reg forced to 1). Only the offer set differs by regime: a flat reduce searches `(bk, fk, br)` (`coop_reduce_offers` / whole-CTA / strided-cooperative free tile); a streaming flash searches the free THREAD tile (`thread_offers`) with `BK=FK=SPLITK=1` and `BR` over the **static** KV axis (`streaming_br_offers` — cooperative-KV, opt-in, `streaming_coop_geometry_ok`-constrained). A **symbolic** (masked) streaming axis is serial-locked (`BR=BK=1` — symbolic-KV cooperative tiling is future work); with a carried-contraction chain `_streaming_leaves` routes it through **`chain_build`** (the FA-2 shared-score restructuring) BY DEFAULT — the QK^T score is computed once per KV step and shared across the P@V output `d` (register vector `O[d]`), not recomputed per `d` (which `monoid_build` would do, running unboundedly long — Finding 1, `plans/qwen3-embedding-0.6b-layer0-tune-findings.md`); the KV stream stays a serial runtime-bounded loop (no tiling → no masking, every `kv < seq_len` valid), so a symbolic hinge is in scope (`_chain_applicable` requires only the inner QK^T static). A symbolic stream WITHOUT a chain falls back to `monoid_build` and collapses the futile free-tile fork to one canonical leaf. Owns the `MONOID` regime end to end (`090`/`100`/`110`/`120_stage` gate off — smem-free). For a **static** stream `chain_build` stays a `DEPLODOCK_CHAIN=1` opt-in (`_build.chain_build`, Phase 1c); greedy default keeps the scalar `monoid_build` nest (the static search-fork integration is Phase 6). |
| `060_reduce_tile.py`  | Fork (scalar `SEMIRING`): the reduce decomposition — `reduce_offers` → `(bk, fk, splitk)`; **applies the `reduce_decomp` body move**. Skips on a warp variant. For an **fp16 matmul** (`_is_fp16_matmul`: every K-indexed operand `Load` is `F16`, no fused prologue/epilogue) an even `fk == bk` offer is reinterpreted as the **half2 accumulation window**: it builds the FK=1 fp32 K factorization (no `K_f` register fold) and stamps `FKWIN` so `kernel/015_pack_fk_window` packs the even bk inner loop into `__hfma2`. The register FK fold and the half2 window are mutually exclusive realizations of `FK`; fp32/bf16 keep the fold, and `fk=1` (greedy default) keeps the scalar fp32-accumulate path. |
| `090_thread_tile.py`  | Fork (scalar): the free-axis thread tile — `thread_offers` → `(thread_n, thread_m)`. Pins the thread knob, **no body move**. Skips on a warp or coop variant. For `SEMIRING` (matmul) it passes `balanced=True` so `thread_offers` drops degenerate-aspect tiles (`BN=1`/`BM=1`) and leads with a square-ish coalesced `BN >= BM` tile (the bare ≈256-thread sort ties `(BN=1, BM=256)` with `(16, 16)` and emits the degenerate one first — and emission order *is* the cold pick); MAP keeps the wide-N order. |
| `100_register_tile.py`| Fork (scalar): the free-axis register tile — `map_reg_offers` / `reduce_reg_offers` → `(reg_n, reg_m)`; **applies the `free_tile` body move** (the algorithm is fully tiled after). Skips on a coop variant. |
| `110_seal_scalar_tier.py`| Deterministic: stamp the reduce regime's scalar-tier OFF sentinels (`MMA=0 WM=0 WN=0 BR=1`). Knob-only; skips on a warp variant (it carries `MMA`) or a coop variant (it carries its own `BR`). |
| `120_stage.py`        | Fork (`Schedule`-move): `stage_candidates` off the stored tiled `TileGraph` → a `STAGE` bitmask → `Schedule.staged[edge] = SYNC` (scalar **and** warp operands; the transposed-B operand is excluded — gmem-direct). **Budget-aware**: the auto-enumerated masks are filtered to those whose slabs fit `ctx.max_dynamic_smem` (`_slab_bytes` matches `KernelOp.smem_bytes` exactly), so greedy's option-0 is the largest IN-budget staging (`STAGE=""` always fits) — without it a large pinned tile over-stages and the deterministic compile has no fallback. A `DEPLODOCK_STAGE` pin stays authoritative (no filter). Skips a `MONOID` coop variant (smem-free — no cross-thread reuse). |
| `130_transport.py`    | Fork (`Schedule`-move, R5): `promote_transport` — `TMA` BOOL on any fully-staged matmul with a ringable K loop (the **warp-tier** `mma.sync` atom **and** the **scalar** register-tiled SGEMM, the blogs' `TM=26` fp32 tile). `True` is **option-0 (the greedy / cold / analytic default) when eligible** — the bulk-async ring measures 1.3–1.9× faster than SYNC on every eligible tile, and making it the default keeps the deployed transport DETERMINISTIC (prior-independent: the old SYNC-first default let a pinned config's TMA flip with the loaded prior, 51 vs 93 µs); `False`/SYNC is the second offer the tuner still explores, and an ineligible tile stays SYNC. `True` promotes every staged `Edge` to `Schedule.staged[edge] = TMA` when the inlined `tma_eligible` oracle passes (sm_90+, affine box ≤ 256 / 16 B-aligned source + box inner, source inner ≥ 2× box, a ringable `serial_outer` K loop; ported from the deleted legacy `050_use_tma._source_eligible`). The 128 B box-alignment check is **tier-aware**: the warp tier's hardware-swizzled slab sizes off the lenient fp32 width (the B64/B128 atom realigns each ring slot), but the scalar tier deposits LINEARLY so it sizes off the TRUE dtype width — an fp16 `BK=32` = 64 B scalar slab thus declines TMA → cp.async instead of faulting `CUDA_ERROR_MISALIGNED_ADDRESS` (the #244 wedge). `assembly/_slab` then synthesizes the double-buffered `cp.async.bulk.tensor` ring. The slab is swizzled **only on the warp tier** (whose `ldmatrix` consumer reads the matching XOR); a scalar tile reads the slab with plain affine `Load`s, so its deposit stays linear (`SwizzleMode.NONE`) — the per-tier choice lives in `_slab._make_bundle`, keyed on `Block.atom`. Skips coop / pointwise / unstaged variants (they stage nothing ringable). |
| `140_atomic_free_splitk.py` | Fork (**structural**, R3): the split-K combine — `NOATOMIC` BOOL on a fully-tiled scalar `SEMIRING` matmul with `SPLITK > 1`. `False` keeps the codegen `atomicAdd`; `True` splices a two-node `Graph` (matmul writing `partial[K_s, M, N]` + the additive reduce kernel `_partition.additive_reduce_tilegraph` folding `K_s`). Skips a warp variant (v1 `SPLITK=1`) / non-split / non-2D-static output. |
| `_iterdag.py`         | `iter_dag` — the derived iteration-DAG view (axes tagged `PARALLEL` / `REDUCE` + carrier). For a streaming-flash nest `IterDag.chain` derives the **carried contraction chain** (`ContractionChain`): the dual-role hinge axis `kv` (free-output of the inner QK^T contraction, reduce of the carrier + the embedded P@V), the nested SEMIRING QK^T contraction, and the `Monoid` carrier whose first partial is the INLINE score edge. Derived on demand (`None` for a non-streaming nest), never stored. |
| `_classify.py`        | `classify` → `_Regime(algebra=AlgebraKind)`. A streaming-flash nest classifies as the **compositional** `MONOID(SEMIRING)` (`_Regime.inner_algebra=SEMIRING`): the twisted carrier's online-softmax combine is a SEMIRING accumulation (the embedded P@V on the hinge `kv`) twisted by the MONOID rescale, derived from `dag.chain`. `None` for a flat (non-twisted) carrier. |
| `_atom.py`            | The **atom layer** — the warp-tier gate `eligible_atoms` (per-atom eligibility over the dag + dtypes + cc) + `classify_matmul_operands` (the one A/B layout decision) + the `atomize` body edit `atomize_cell` (cell `[Load,Load,mul,Accum]` → `Mma`). `atomize_cell` is **provenance-agnostic** (names A/B by SSA value), so it lowers a cell whose operands are gmem `Load`s (the SEMIRING matmul) **or** a register/smem fragment (the Phase-2 MONOID flash inner contractions) — factored out of `_build.warp_build`'s matmul-staging geometry so the reuse has a fixed contract (`tests/compiler/passes/test_atomize_cell.py`). An `out_index` param supplies the output `(M, N)` coords for a cell whose result is an **INLINE register fragment with no `Write`** (the flash QK^T, whose transposed-B Q@K^T score never reaches gmem — M = query, N = kv); `None` keeps the `Write`-driven matmul path. A `frag_a` flag opts a cell into the **fragment-`A`** shape (one gmem `B` `Load` + a register `A` fragment — the flash **P@V** `O += P·V`, where `A = P` arrives from the QK^T C-fragment, the C→A handoff); OFF by default (the shape is ambiguous with a scalar-scaled reduce, so the warp-chain build that knows the cell is a P@V opts in). |
| `_moves.py`           | `Budget` + `legal_decomps` + the offers (`thread_offers`, `map_reg_offers`, `reduce_offers`, `reduce_reg_offers`, `coop_reduce_offers` / `coop_free_threads`, `warp_offers` / `warp_reg_offers` / `warp_bk_offers`) + knob deltas. Every scalar/warp offer honors its `DEPLODOCK_<KNOB>` env pin via `_pin` (the `thread_offers`/`map_reg_offers`/`reduce_reg_offers` narrow `BN`/`BM`/`FN`/`FM` to the pin, like `reduce_offers` does for `BK`/`FK`/`SPLITK`) — a pinned masked tile (e.g. `BN=8` over `N=47`) reaches the masked σ-split instead of being dropped for the best-first ≈256-thread default. |
| `_stage.py`           | `stage_candidates` — the `stage` move's ranked offer set (AFFINE + non-degenerate intra-CTA fan-in reuse + K-tower) off the derived `Block.reads`; excludes the transposed-B operand **and any buffer read at >1 distinct access** (`_multi_access_bufs` — a single slab can reconstruct only one access; the RoPE-fused score producer reads a rotary table at both the Q row `cos[m,d]` and K row `cos[n,d]`, and its projection both straight and rotate-half, so they stay gmem-direct — only same-access reads collapse to one slab, the `026` dedup by construction). |
| `_knobs.py`           | The knob schema (`BN`/`BM`/`BK`/`FK`/`STAGE`/`MMA`/`WM`/`WN`/… + the composer aliases `MAP_*` / `RED_*` / `TC_*`). |
| `_build.py`           | The F3-b incremental body moves — `seed_graph` (logical block), `reduce_decomp` (K re-bracket), `free_tile` (free-axis σ-split), `monoid_build` (the **one** MONOID move for the flat cooperative reduce AND the streaming flash — R2/R6: the reduce-decomposition tower `_replace_k_monoid` applied to each contraction axis the DAG exposes, recursive for the nested stream, `K_c` cooperative lane on the primary axis, carrier-aware masked-K fill via `_mask_carrier`; then `free_tile`), `chain_build` (the **shared-axis reduce_decomp**: restructure a `MONOID(SEMIRING)` chain nest into the FA-2 shared-score form — the P@V output `d` rides a REGISTER domain axis `O[BM,D]`, the score is computed once per KV step and shared across `d` (the register-replication pass keys on the `d` var), and `split_carrier` (the shared `ir/stmt/carrier_algebra` helper) splits the twisted carrier into a scalar stats `Monoid` + a register-tiled accumulation `Monoid` that reads the stats' rescale/probability temps — the two SEMIRING cells Phase 2 atomizes), `warp_build` (the warp-tier four-way split + K re-bracket — the matmul-staging geometry — composed with `_atom.atomize_cell`, the provenance-agnostic atom-layer body edit); `build_dag` is the scalar composition (the byte-identity oracle). |
| `_partition.py`       | The R3 split-K combine builders — `additive_reduce_tilegraph` (`Accum` sum) / `monoid_reduce_tilegraph` (carrier-general `combine_states` fold) emit a fully-tiled single-`Block` combine `TileGraph` (`GridTile(16×16) > ThreadTile > serial K_s reduce + boundary Cond`); `reduce_tilegraphop` wraps it with stamped fixed/OFF knobs so every enumeration fork skips it (fixed-schedule, not searched). |

The per-pass moves serve every regime: a `MAP` nest applies only `free_tile` (`reduce_decomp` is a no-op without a
contraction); a `SEMIRING` reduce applies both — `reduce_decomp` (gated on `target_names`) then `free_tile`; a `MONOID`
reduce (flat OR streaming) applies the single `monoid_build` (the reduce-decomposition tower on each contraction axis +
free split + the cooperative `K_c` THREAD lane, the cross-thread combine derived downstream from the carrier's `axes`).
The scalar composition order (reduce then free) reverses the old monolith (free then
reduce), but the two σ-rewrites touch disjoint axis sets (K vs the free N/M) so they commute — `build_dag` stays the
byte-identity oracle for the distribution.

## `assembly/` — `assemble`

| Module | Role |
| ------ | ---- |
| `010_assemble.py` | The pass: gates readiness (`assembly_ready` — every block that needs tiling has a `domain`) then calls the single `assemble_block` entry, which materializes the stored algorithm across the placement lattice (single `TileOp` / SMEM-fused `TileOp` / GMEM-cut `Graph` of `TileOp` kernels). **No build** here. |
| `020_peel.py`     | Deterministic post-`assemble` (R5): software-pipeline a ring-staged `serial_outer` K loop (a `StageBundle` with `policy ∈ {ASYNC, TMA}`, `pipeline_depth == 1`, static extent ≥ `buffer_count`) into prologue / `K_o-1` main loop with issue-next + `AsyncWait` + reduce / epilogue drains — the peeled shape D `kernel/005_lower_atom_tile` lowers. **No fork** (the transport decision was the fork; the peel has one output per ring depth). Ported from the deleted legacy `080_pipeline_stages`. |
| `030_mark_unroll.py` | Deterministic post-`assemble`: flag `SerialTile` / `StridedTile` nests whose total trip count (axis-extent product down the chain) is ≤ 64 with `unroll=True`, so the renderer emits `#pragma unroll`. Unrolls the FMA inner reduce (the SGEMM `BK` loop) for register-resident operand reuse + ILP — the lever that takes the `TM=26` fp32 hero tile from ~126 regs / 384 µs to 255 regs / ~293 µs (the article's ~96 %-of-cuBLAS number). A symbolic extent reports a placeholder above the threshold (left rolled — no compile-time bound to amortize). **No fork** (per-nest threshold, not a search dim). Ported back from the pre-block-DAG `tile/090_mark_unroll` (dropped in `7f764b26` with no replacement). |
| `_assemble.py`    | `assemble_block` — the **single entry that realizes the placement lattice**. Single block → `_assemble_one` synthesizes staging (`_slab`), then reconstructs the binding tower from `Schedule.binding` (ATOM/REGISTER/WARP/THREAD/GRID tiers) over the block's σ-rewritten `compute` (the warp tier's `Mma` cell rides inside the `AtomTile` for `kernel/005`). Same-launch-group multi-block (the **SMEM/INLINE fused edge**, R7) → `_assemble_group`: a MAP/MONOID producer `--xn-->` SEMIRING matmul consumer kept in **one kernel**, the `xn` intermediate riding an smem slab the producer fills (e.g. `relu(x) @ w`). Reuses the existing `StageBundle.compute` phase ("sibling-smem → own-smem"): stage `x` into a slab, `_fuse_producers` patches the `xn` source → `x_smem` + the producer transform as the bundle's `compute` phase writing `xn_smem`, the consumer matmul reads `xn_smem` (no gmem round-trip — the form that beats the cut); a MONOID (rmsnorm) producer additionally splits its per-row reduce off as a `CoopReduce` prologue (`_build_reduce_prologue`). The producer rides the consumer's slab cache axes (shared-knob, one knob set) and stays logical — `assembly_ready` exempts it from the tiled-readiness check. Multi-launch DAG (the **GMEM cut**) → `_assemble_multi` partitions `blocks` by `Schedule.launch` (one group = one kernel; v1 = one block per group, the two-launch cut — a multi-block group is the later cooperative `grid.sync` field), topo-sorts by the derived edge DAG, assembles each group, and wires a `Graph` of `TileOp` kernels with every cross-group edge materialized as an intermediate tensor (shape/dtype from the declared `TileGraph.buffers`). `is_fused_graph` detects the same-launch-group multi-block case. Deterministic — same `TileGraph` → byte-identical kernel set. (The former `_fused.py` is absorbed here.) |
| `_slab.py`        | `synthesize_staging` — materialize `Schedule.staged` into one `StageBundle` per K-tower (a `Source` per staged buffer, cache axes off the consumer `Load`, GRID + serial-outer K folded to the slab origin). SYNC by default; a `Transport.TMA` edge yields the double-buffered TMA ring (`buffer_count=2`, `phase = K_o % 2` prepended to the consumer slab Loads). The deposit is swizzled (per-source via `pick_swizzle_atom`) **only on the warp tier** (`Block.atom` set — its `ldmatrix` consumer reads back the matching XOR); a scalar tile's plain-`Load` consumer needs a linear deposit, so `_make_bundle` leaves its sources `SwizzleMode.NONE`. A **masked tile** (the K tower + `Write` wrapped in a boundary `Cond(σ(M\|N) < bound)`) goes through `_hoist_masked`: the K-pipeline is lifted **above** the guard (so every thread issues the cooperative load uniformly — a SYNC `__syncthreads` / cp.async / TMA inside divergent control flow hangs) and, for SYNC sources, `Source.gmem_extents` is stamped so `_stage_expand.emit_stage` clamps the overhang gmem read to `[0, extent)` (a static `int` or the symbolic `Var('seq_len')`); TMA sources rely on the hardware OOB zero-fill instead. **SSA-safety refusal**: if a hoisted K-tower stmt reads a name defined by a stmt staying inside the `Cond` (the fused-prologue shape), the hoist returns `None` and the caller keeps the `Cond` in place — hoisting would order a consumer above its definition (defense-in-depth; the planner doesn't emit such Conds today). `prospective_sources` exposes the slab `Source`s the transport fork's eligibility oracle + `120_stage`'s budget filter read pre-assemble. The per-axis `AffineAddressing.block` multiplier is **derived from the σ coefficients** — `()` for a scalar tile, atom-strided for a warp tile; a size-1 REGISTER cell is dropped first so its atom stride migrates to the warp axis. |
| `_tower.py`       | `_wrap_tower` — the shared innermost-first tower-building primitive (`Role` → tile flavor). Also `CarryScope` + `wrap_carry_tower` — a reduce axis that carries accumulator state ACROSS its serial loop (state init'd above, combined per-iteration: the `produce`/`merge`/`rescale`/`handoff`/`consume`/`update` phases of a `MONOID(SEMIRING)` carrier), the re-bracketable-reduction foundation; the warp-chain kv-stream is its first consumer. |
| `_warp_chain.py`  | The **fused tensor-core flash** assembler — `build_warp_chain_tileop` builds a `TileOp` (kernel-IR body, NOT a source string) that flows through the standard kernel passes and renders via `render_kernelop`, generalized over `(B,H,S,D)`: `GridTile > WarpTile >` QK^T `mma` (Q `ldmatrix.x4` A over `D/16` K-tiles, K transposed-B `dpl_mma_load_b_gmem_trans`) → score C-fragment → fragment online-softmax (`FragmentRowReduce` + `FragmentExp` + `FragmentScale` + the carried `m`/`l` recurrence via `Init`/`Reassign`) → C→A handoff (`P` `RegStore`→ the `flash_pv_smem` slab → `ldmatrix.x4` A, `ldm=16`) → P@V `mma` over `D/8` N-tiles → `RegStore` epilogue. The QK^T / P@V mma + the A/V loads fall out of the **same** kernel-IR ops (`MmaSyncPtx`/`LdmatrixLoad`/`RegStore`) as the warp-tier matmul — the kernel goes through the IR, no source template. The body is assembled through `_tower.wrap_carry_tower`: the kv-stream is a `MONOID(SEMIRING)` `CarryScope` (streaming `O` accumulator + online-softmax stats), the two SEMIRING cells bracketing its `merge`/`rescale`/`update`. The softmax phases (`init` stats / `merge` / `rescale` / `update` / `epilogue` scales) are no longer hand-listed: they are **generated from the `dag.chain.carrier` `Monoid`** by the fragment realizer `_frag_softmax.realize_fragment_softmax` (the m16n8 sibling of the cooperative `emit_combine`); only the `produce`/`handoff`/`consume` scaffolding (the SEMIRING cells + the C→A smem edge) stays hand-built. A **symbolic `seq_len`** (Phase 1, [`plans/smem-tiled-symbolic-flash.md`](../../../../../../plans/smem-tiled-symbolic-flash.md)) ceil-divides the kv-stream / query-tile extents over `Var(seq_var)`, uses `seq_var·D` for the (b,h) row stride, masks the partial final KV tile's score columns (`realize_boundary_mask`), stamps the QK^T / P@V operand load guards on the `Mma` (`m_guard`/`n_guard`/`k_zero` → `kernel/005` routes them: clamp Q rows / clamp K cols / zero-fill V rows), and guards the partial query-row store (`RegStore.m_guard`). A **GQA** flash (Phase 2, `group>1`) reads K/V at the kv-head `bh_kv = batch·H_kv + head//group` (Q/O keep the q-head `bh`; `group==1` collapses byte-identically). `warp_chain_eligible` gates the v1 scope (fp16/bf16, causal or non-causal, `D%16==0`; static `S%16==0` OR symbolic `seq_len`; equal-head OR GQA); driven by `split/005_warp_chain` under `CHAIN=1`. |
| `_frag_softmax.py`  | The **fragment-tier realizer** — generates the m16n8 tensor-core flash softmax from a streaming `Monoid` (the sibling of `lowering/kernel/_combine.emit_combine`, which realizes the same carrier across lanes/smem). `split_carrier` separates the stats `(m,l)` / accum `(O)` monoids; a fragment/scalar **taint** analysis maps the stats `merge` onto fragment ops (a reduce-eligible op over the score partial is a FOLD → `FragmentRowReduce` + row-distributed scalar; `exp(s-m)` → `FragmentExp`; the accum `O·α` → the rescale `FragmentScale`). The carrier supplies the algebra; the caller passes the fragment geometry (`FragGeom`). `realize_score_mask` is the fragment-tier score-partial mask (causal, `kv_col > q_row`); `realize_boundary_mask` is the symbolic-`seq_len` column-only sibling (`kv_col >= seq_len` — the partial final KV tile's padding keys masked to `-1e30` before the rowmax; composes with causal by ANDing the keep predicates). Imports only `ir.*` + the shared carrier algebra — never `enumeration`. |

## Coverage

Built today: **`MAP`** + scalar **`SEMIRING`** (including masked / symbolic free axes, split-K, and the `FK`
strip-mine) + the **`MONOID` reduce** (R2/R6 `monoid_build` — the one move for the flat cooperative reduce (softmax /
rmsnorm / mean / max, static **and** symbolic-K masked-fill, whole-CTA and strided-cooperative rows, the warp-shuffle /
hierarchical combine) AND the streaming flash, below) + the
**warp-tier `SEMIRING`** (tensor-core `mma.sync` via the R4 `atomize` move — matmul, residual / pointwise / causal-mask
epilogue fold, transposed-B, symbolic M/N), with **smem staging** (`stage` move) on both the scalar reduce regimes (R1)
and the warp operands (atom-strided slab + `ldmatrix`) + the **cross-CTA split-K combine** (R3 `140_atomic_free_splitk`
— the `NOATOMIC` structural fork: the matmul's per-`K_s` partials either `atomicAdd` into the output or write a
`partial[K_s, M, N]` workspace folded by a sibling additive / carrier-general combine kernel) + the **TMA transport**
(R5 `130_transport` + `assembly/020_peel`: the `promote_transport` fork promotes a static-shape matmul's staged
operands to a double-buffered `cp.async.bulk.tensor` ring, software-pipelined into prologue/main/epilogue — greedy stays
TMA-first when eligible — option-0 takes the ring; the tuner/`DEPLODOCK_TMA=0` can force SYNC). Both staged tiers promote: the **warp-tier** `ldmatrix` matmul (slab
swizzled per-source) and the **scalar register-tiled SGEMM** (the blogs' `TM=26` fp32 hero tile — its plain-`Load`
consumer reads an unswizzled `SwizzleMode.NONE` deposit) + the **streaming-flash
`MONOID`** (R6 — the **same** `070_coop_reduce` pass + `monoid_build` move as the cooperative reduce, selected
structurally via `op.dag.streaming` — SDPA / causal / GQA / additive-mask online-softmax, static **and**
symbolic-`seq_len` masked streaming, scalar-KV by default and **cooperative-KV** when `DEPLODOCK_BR>1` lays the
`K_c` THREAD lane on a static streaming axis). An **eligible** symbolic-`seq_len` flash now deploys the
**tensor-core warp chain** BY DEFAULT (Phases 1–3 of `plans/smem-tiled-symbolic-flash.md`, intercepted pre-enumeration by
`split/005_warp_chain`): the smem-tiled / tensor-core fused flash whose kv-stream / query-tile extents ceil-div the
runtime `seq_len` and whose partial final tile is masked at the score fragment (`kv_col >= seq_len` → `-1e30` before the
rowmax via `realize_boundary_mask`) with clamped Q/K loads + zero-filled V + a guarded query-row store (the operand
guards ride the QK^T / P@V `Mma`s, routed to the `LdmatrixLoad`s by `kernel/005`); causal or non-causal, equal-head OR
**GQA** (`head//group`), `D%16==0`. **Measured** at the Qwen3-Embedding fp16 attention shape (`1×16×512×128`): **68 µs**
(0.57× eager's cuBLAS flash) vs the old scalar streaming nest's ~8 ms — a ~100× kernel-level win. A **non-eligible**
symbolic flash (fp32, odd `D`, additive mask) falls back to the scalar **FA-2 shared-score** `chain_build` (the score
computed once per KV step, shared across the P@V output `d` in a register vector `O[d]`; removes the `monoid_build`
per-`d` recompute that made the symbolic SDPA reduce run unboundedly long, Finding 1). A **static** flash keeps the
scalar `chain_build` unless `DEPLODOCK_CHAIN=1` (the warp chain stays a static opt-in).
The **masked scalar-tile staging clamp** landed (R4 follow-up): scalar-offer
env-pin honoring (`_pin` in `thread_offers`/`reduce_reg_offers`) reaches the masked σ-split, the over-staging it
exposed is resolved by `120_stage`'s budget-aware mask filter (greedy falls back to the largest in-budget staging),
and `_slab._hoist_masked` lifts the cooperative load above the boundary `Cond` + clamps the SYNC gmem read to the
buffer extent (`test_masked_tile.py::test_planner_admits_non_divisor_n_with_real_extent` /
`…test_masked_n_clamps_cooperative_load_index` / `…test_symbolic_m_cooperative_load_clamps_to_runtime_extent`).
The remaining R4 follow-ups landed: the **gmem-direct unstaged atom** now compiles — a fully-pinned over-ceiling
`(FM, FN)` warp register tile is authoritative (`warp_reg_offers` bypasses the `_MAX_WARP_CELLS` *search* ceiling for a
full pin), so the warp build + assemble proceed, and with no `STAGE` pin `120_stage`'s budget-aware filter declines the
over-budget staging so the operands lower gmem-direct via `kernel/005_lower_atom_tile`
(`test_knob_pinning.py::test_unstaged_atom_lowers_gmem_direct`); and the **hoist-refuses** check is rewritten against
`_slab._hoist_masked`, which gained the SSA-safety refusal — a masked-tile hoist that would lift a K-tower stmt above an
SSA name defined by a stmt staying inside the boundary `Cond` (the fused-prologue shape) returns `None` and the caller
keeps the `Cond` in place (`test_masked_tile.py::test_hoist_refuses_lift_when_pipeline_reads_guarded_defs`).
