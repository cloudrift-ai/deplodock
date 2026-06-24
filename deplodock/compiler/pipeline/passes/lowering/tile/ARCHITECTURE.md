# Tile lowering — enumeration + assembly over the block-DAG IR

The tile phase lowers each fused `LoopOp` to a kernel-ready `TileOp` in **three passes** over the block-DAG Tile IR
(`ir/tile/ir.py`), following `plans/tile-ir-block-dag.md`:

- **`split/`** — the **pre-build structural-fork head** (R7 `005_split_demoted`). Runs FIRST, on the still-un-tiled
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
  (`split/_extract.py::extract_block`), and the **thin fork shell** (`005_split_demoted.py`, holding no decision logic).
  `split/` survives as a pre-build phase because the demoted matmul never classifies as a buildable seed
  (`000_build` would `RuleSkip` it), so the cut can't yet fold into `enumeration/` as an edge-placement move — see
  [`plans/dag-edge-placement-split-as-enumeration.md`](../../../../../../plans/dag-edge-placement-split-as-enumeration.md).
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
  warp operands too. The **cooperative-reduce** chain (R2 `MONOID`): `015_coop_reduce` is a single `(bk, fk, br)` fork
  applying the **coop build body move** (re-bracket K with the `K_c` cooperative-THREAD lane + free-axis σ-split with the
  register tile forced to 1); the scalar passes `020`/`030`/`040` + `050_stage` gate off `MONOID` (a cooperative reduce
  is one decision and stays smem-free). The algorithm is built up across the passes, never all-at-once.
- **`assembly/`** (`010_assemble` + `020_peel`) — the fully-tiled `TileGraphOp` → `TileOp`, in one deterministic step
  followed by deterministic post-passes. No build here: `assemble_block` only **materializes** the stored algorithm —
  the register/thread tower (`_wrap_tower`) + slab synthesis from `Schedule.staged`; `020_peel` then software-pipelines a
  ring-staged (TMA) K loop. Every scheduling decision already lives on the `TileGraph` / `Schedule`, so there is no
  search here.

```
              ┌─ split? (xn producer + clean gemm)        ┌─ scalar: reduce_decomp ─(thread)─ free_tile ─ 040_seal ─┐
LoopOp ─split─┤                              ─000_build─▶ logical TileGraph ─005_tensorize┤─ coop: 015_coop_reduce (coop_build) ─┼─ 050_stage ─ 052_transport ─▶ tiled TileGraphOp ─010_assemble ─ 020_peel ─▶ TileOp
              └─ keep fused ─────────────────▶            (per LoopOp)         └─ warp: 006/008 geom+reg ─ 009_warp_build ───────┘
```

## The block-DAG Tile IR (`ir/tile/ir.py`)

One file, two comment-block sections:

- **ENUMERATION** — the *invariant algorithm* + the *variant* `Schedule` the composer searches: `Block`
  (`name + domain + compute`), `Buffer`, `Edge`, `Schedule`, `TileGraph`. The derived projections
  (`Block.reads`/`writes`/`carrier`/`atom`, `TileGraph.edges`, `TileGraph.placement(edge)`) are **computed on demand,
  never stored** — the same discipline as `Loop.algebra_kind`, so they can't drift and don't enter `op_cache_key`.
  `TileGraph.placement(edge)` is the unifying **edge-placement** view
  (`plans/dag-edge-placement-split-as-enumeration.md`): `INLINE` (default, register/gmem-direct), `SMEM`
  (`Schedule.staged` — today's `stage` move), or `GMEM` (a cross-launch-group cut — the buffer lives in gmem) read off
  `Schedule.staged` + `Schedule.launch`, so `stage` and `split` are two values of one query. `TileGraphOp` wraps a
  chosen `TileGraph` as a graph node (the enumeration-pass output; `op_cache_key` keys it on `TileGraph.structural_key`
  + knobs).
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

## `split/` — pre-build structural forks

| Module | Role |
| ------ | ---- |
| `005_split_demoted.py` | The **thin fork shell** (**structural**, R7) — holds no decision logic. `rewrite`: idempotent (`CUT` guard), runs before `enumeration/000_build` on the un-tiled body; calls `_extract.extract_block` (expressibility — `None` if not a cuttable demotion) + `_extract.seed_fused` (the keep(SMEM) realization) then `_cut.cut_offers` (the offer verdict), emitting the `[keep, cut]` fork (greedy default keep) — **forcing** the split `Graph` only when the fused body is `UNBUILDABLE` AND not on-chip-fusible. `CUT` is a width-1 `BINMASK` over the ranked cuttable edges (`"0"` keep / `"1"` cut). Honors `DEPLODOCK_CUT` (alias `DEPLODOCK_SPLIT_CONE`); stamps `CUT` on both branches. |
| `_cut.py` (in `enumeration/`) | The **offer policy** — the derived `Tier` lattice (`UNBUILDABLE` < `MAP` < `SCALAR_REDUCE` < `COOP_REDUCE` < `WARP_MMA`) + `tier(dag)` (built on `_atom.eligible_atoms` + `_classify.classify`) + `cut_offers(loop_op) -> CutDecision`. Offers iff `tier(inline) is UNBUILDABLE` (forces when also not `smem_fusible`): materializing the demoted operand strictly raises the consumer's tier. Returns the ranked `offers` tuple (one whole-cone offer today → width-1 `CUT` mask; the per-edge list is the additive-widening seam). One auditable place for the which-cuts policy; `eval`-introspectable. |
| `_extract.py` | The **fission** (`extract_block` + helpers, relocated from the legacy monolith): classify the body into `(leading, rows, prologue, outer_n, k_loop)`, backward-slice each computed/K-folded multiply-operand cone, build one `xn` producer per cone class + the rebuilt consumer (+ per-accum `mm_i` gemms for a multi-accum cell), wired into a `Graph` fragment by `_assemble_fragment` (which re-stamps `S_*` structural features). Reuses `kernel/_helpers` (`is_matmul_reduce` / `segmentable_k_extent`) + `Body.backward_cone` / `defs_die_at`. Returns `None` (its expressibility check) for any shape it can't cleanly cut. |

## `enumeration/` — the move composer

| Module | Role |
| ------ | ---- |
| `000_build.py`        | Seed pass: `LoopOp` → `iter_dag` + `classify` → a `TileGraphOp` carrying the **logical** `TileGraph` (`seed_graph`) + dag + regime. Calls `_validate.validate_pins` first (greedy only) — a force-pinned env knob foreign to the op's tier is a hard error, not a silent drop. |
| `_validate.py`        | **Strict per-op knob-pin validation** (greedy `compile`/`run` only — the tune search sets `ctx.validate_pins=False`). Each kernel lowers on ONE `Tier` (MAP / scalar SEMIRING / warp MMA / cooperative MONOID / streaming TWISTED_MONOID), each owning a disjoint knob slice; `validate_pins(algebra)` intersects the tiers every force-pinned `DEPLODOCK_<KNOB>` is legal on and raises `KnobPinError` when empty (e.g. `BN`/`BM`/`BR`/`FK` beside a warp `MMA=<kind>`, or `WM`/`WN` on a scalar `MMA=0`). Value-aware: a universal / OFF pin (`SPLITK=1`, `FK=1`, `BR=1`, `FM=1`, empty `STAGE`, `TMA=0`) constrains nothing; `SPLITK>1` is legal on scalar AND warp (R3 atomic-free split-K). `STAGE`/`TMA` are legal only on the **staged** tiers (scalar / warp) — a cooperative MONOID / streaming flash is smem-free and a pointwise MAP has no K-tower, so a `STAGE`/`TMA` pin there refuses (scalar-tile `TMA` itself is unimplemented — `052_transport` is warp-only — but allowed-not-senseless; an xfail tracks it). |
| `005_tensorize.py`    | Fork (warp tier, R4): atom-vs-scalar — `eligible_atoms` (gate) → `MMA=<kind>` per eligible atom + the scalar fallback (warp ranked first). An explicit scalar `BN`/`BM` pin defers to scalar. |
| `006_warp_geometry.py`| Fork (warp): the per-CTA warp counts — `warp_offers` → `(WM, WN)`. Knob-only. |
| `008_warp_reg.py`     | Fork (warp): the per-warp register cells — `warp_reg_offers` → `(FM, FN)`. Knob-only. A fully-pinned `(DEPLODOCK_FM, DEPLODOCK_FN)` is authoritative and bypasses the `_MAX_WARP_CELLS` *search* ceiling (the ceiling prunes auto-enumerated candidates, not explicit pins). |
| `009_warp_build.py`   | Fork (warp): the K chunk — `warp_bk_offers` → `BK`; **applies the `warp_build` body move** (four-way σ-split + K re-bracket + `atomize` the cell → `Mma`). |
| `015_coop_reduce.py`  | Fork (cooperative `MONOID`, R2): the one `(bk, fk, br)` decision — `coop_reduce_offers`; **applies the `coop_build` body move** (K re-bracket with the `K_c` cooperative-THREAD lane + free-axis σ-split, reg forced to 1). Owns the `MONOID` regime end to end (`020`/`030`/`040`/`050_stage` gate off). |
| `017_streaming.py`    | Fork (streaming `TWISTED_MONOID`, R6 — e.g. flash attention): the free-axis thread tile × cooperative `BR` — `thread_offers` × `streaming_br_offers`; **applies the `streaming_build` body move** (serial-transform both contraction axes, `FM=FN=1`/`BK=FK=SPLITK=1`; a pinned `BR>1` lays the `K_c` THREAD lane on the **static** streaming axis so each lane streams a strided slice and the carrier's `combine_states` merges them at materialize, like the `MONOID` coop reduce). `streaming_coop_geometry_ok` constrains the free×BR layout (whole-CTA tree vs strided intra-warp); default `BR=1` is the serial-stream form, a symbolic streaming axis stays serial. Owns the `TWISTED_MONOID` regime end to end (`050_stage` skips — smem-free). |
| `010_reduce_tile.py`  | Fork (scalar `SEMIRING`): the reduce decomposition — `reduce_offers` → `(bk, fk, splitk)`; **applies the `reduce_decomp` body move**. Skips on a warp variant. For an **fp16 matmul** (`_is_fp16_matmul`: every K-indexed operand `Load` is `F16`, no fused prologue/epilogue) an even `fk == bk` offer is reinterpreted as the **half2 accumulation window**: it builds the FK=1 fp32 K factorization (no `K_f` register fold) and stamps `FKWIN` so `kernel/015_pack_fk_window` packs the even bk inner loop into `__hfma2` (`plans/fk-half2-fp16-matmul.md`). The register FK fold and the half2 window are mutually exclusive realizations of `FK`; fp32/bf16 keep the fold, and `fk=1` (greedy default) keeps the scalar fp32-accumulate path. |
| `020_thread_tile.py`  | Fork (scalar): the free-axis thread tile — `thread_offers` → `(thread_n, thread_m)`. Pins the thread knob, **no body move**. Skips on a warp or coop variant. |
| `030_register_tile.py`| Fork (scalar): the free-axis register tile — `map_reg_offers` / `reduce_reg_offers` → `(reg_n, reg_m)`; **applies the `free_tile` body move** (the algorithm is fully tiled after). Skips on a coop variant. |
| `040_seal_scalar_tier.py`| Deterministic: stamp the reduce regime's scalar-tier OFF sentinels (`MMA=0 WM=0 WN=0 BR=1`). Knob-only; skips on a warp variant (it carries `MMA`) or a coop variant (it carries its own `BR`). |
| `050_stage.py`        | Fork (`Schedule`-move): `stage_candidates` off the stored tiled `TileGraph` → a `STAGE` bitmask → `Schedule.staged[edge] = SYNC` (scalar **and** warp operands; the transposed-B operand is excluded — gmem-direct). **Budget-aware**: the auto-enumerated masks are filtered to those whose slabs fit `ctx.max_dynamic_smem` (`_slab_bytes` matches `KernelOp.smem_bytes` exactly), so greedy's option-0 is the largest IN-budget staging (`STAGE=""` always fits) — without it a large pinned tile over-stages and the deterministic compile has no fallback. A `DEPLODOCK_STAGE` pin stays authoritative (no filter). Skips a `MONOID` coop variant (smem-free — no cross-thread reuse). |
| `052_transport.py`    | Fork (`Schedule`-move, R5): `promote_transport` — `TMA` BOOL on a fully-staged **warp-tier** matmul. `False` (option-0, greedy-default) keeps SYNC staging; `True` promotes every staged `Edge` to `Schedule.staged[edge] = TMA` when the inlined `tma_eligible` oracle passes (sm_90+, affine box ≤ 256 / 16 B-aligned source + box inner, source inner ≥ 2× box, a ringable `serial_outer` K loop; ported from the deleted legacy `050_use_tma._source_eligible`) — `assembly/_slab` then synthesizes the double-buffered `cp.async.bulk.tensor` ring + per-source swizzle. Skips scalar / coop / unstaged variants. |
| `055_atomic_free_splitk.py` | Fork (**structural**, R3): the split-K combine — `NOATOMIC` BOOL on a fully-tiled scalar `SEMIRING` matmul with `SPLITK > 1`. `False` keeps the codegen `atomicAdd`; `True` splices a two-node `Graph` (matmul writing `partial[K_s, M, N]` + the additive reduce kernel `_partition.additive_reduce_tilegraph` folding `K_s`). Skips a warp variant (v1 `SPLITK=1`) / non-split / non-2D-static output. |
| `_iterdag.py`         | `iter_dag` — the derived iteration-DAG view (axes tagged `PARALLEL` / `REDUCE` + carrier). |
| `_classify.py`        | `classify` → `_Regime(algebra=AlgebraKind)`. |
| `_atom.py`            | The warp-tier gate: `eligible_atoms` (per-atom eligibility over the dag + dtypes + cc) + `classify_matmul_operands` (the one A/B layout decision, shared by the gate and the `atomize` move). |
| `_moves.py`           | `Budget` + `legal_decomps` + the offers (`thread_offers`, `map_reg_offers`, `reduce_offers`, `reduce_reg_offers`, `coop_reduce_offers` / `coop_free_threads`, `warp_offers` / `warp_reg_offers` / `warp_bk_offers`) + knob deltas. Every scalar/warp offer honors its `DEPLODOCK_<KNOB>` env pin via `_pin` (the `thread_offers`/`map_reg_offers`/`reduce_reg_offers` narrow `BN`/`BM`/`FN`/`FM` to the pin, like `reduce_offers` does for `BK`/`FK`/`SPLITK`) — a pinned masked tile (e.g. `BN=8` over `N=47`) reaches the masked σ-split instead of being dropped for the best-first ≈256-thread default. |
| `_stage.py`           | `stage_candidates` — the `stage` move's ranked offer set (AFFINE + fan-in reuse + K-tower) off the derived `Block.reads`; excludes the transposed-B operand **and any buffer read at >1 distinct access** (`_multi_access_bufs` — a single slab can reconstruct only one access; the RoPE-fused score producer reads a rotary table at both the Q row `cos[m,d]` and K row `cos[n,d]`, and its projection both straight and rotate-half, so they stay gmem-direct — only same-access reads collapse to one slab, the `026` dedup by construction). |
| `_knobs.py`           | The knob schema (`BN`/`BM`/`BK`/`FK`/`STAGE`/`MMA`/`WM`/`WN`/… + the composer aliases `MAP_*` / `RED_*` / `TC_*`). |
| `_build.py`           | The F3-b incremental body moves — `seed_graph` (logical block), `reduce_decomp` (K re-bracket), `free_tile` (free-axis σ-split), `coop_build` (the cooperative-reduce K re-bracket + free split, R2), `streaming_build` (the streaming `TWISTED_MONOID` serial K-transform + free split, R6 — e.g. flash attention; `BR>1` lays the `K_c` cooperative lane on the static streaming axis), `warp_build` (the warp-tier four-way split + `atomize`); `build_dag` is the scalar composition (the byte-identity oracle). |
| `_partition.py`       | The R3 split-K combine builders — `additive_reduce_tilegraph` (`Accum` sum) / `monoid_reduce_tilegraph` (carrier-general `combine_states` fold) emit a fully-tiled single-`Block` combine `TileGraph` (`GridTile(16×16) > ThreadTile > serial K_s reduce + boundary Cond`); `reduce_tilegraphop` wraps it with stamped fixed/OFF knobs so every enumeration fork skips it (fixed-schedule, not searched). |

The per-pass moves serve every regime: a `MAP` nest applies only `free_tile` (`reduce_decomp` is a no-op without a
contraction); a `SEMIRING` reduce applies both — `reduce_decomp` (gated on `target_names`) then `free_tile`; a `MONOID`
reduce applies the single `coop_build` (free split + the cooperative `K_c` THREAD lane, the cross-thread combine derived
downstream from `Accum.axes`). The scalar composition order (reduce then free) reverses the old monolith (free then
reduce), but the two σ-rewrites touch disjoint axis sets (K vs the free N/M) so they commute — `build_dag` stays the
byte-identity oracle for the distribution.

## `assembly/` — `assemble`

| Module | Role |
| ------ | ---- |
| `010_assemble.py` | The pass: fully-tiled `TileGraphOp` → `TileOp` (single block) or a `Graph` of `TileOp` kernels (multi-block — the edge-placement `GMEM` cut, R7). **No build** — `assemble_block` materializes the stored algorithm (tower + slab synthesis). |
| `020_peel.py`     | Deterministic post-`assemble` (R5): software-pipeline a ring-staged `serial_outer` K loop (a `StageBundle` with `policy ∈ {ASYNC, TMA}`, `pipeline_depth == 1`, static extent ≥ `buffer_count`) into prologue / `K_o-1` main loop with issue-next + `AsyncWait` + reduce / epilogue drains — the peeled shape D `kernel/005_lower_atom_tile` lowers. **No fork** (the transport decision was the fork; the peel has one output per ring depth). Ported from the deleted legacy `080_pipeline_stages`. |
| `_fused.py`       | `assemble_fused` — the **SMEM fused-edge** assemble (R7): a MAP producer `--xn-->` SEMIRING matmul consumer kept in **one kernel**, the `xn` intermediate riding an smem slab the producer fills (e.g. `relu(x) @ w`). Reuses the existing `StageBundle.compute` phase ("sibling-smem → own-smem"): stage `x` into a slab, `_fuse_producers` patches the `xn` source → `x_smem` + the producer transform as the bundle's `compute` phase writing `xn_smem`, the consumer matmul reads `xn_smem` (no gmem round-trip — the form that beats the cut). The producer rides the consumer's slab cache axes (shared-knob, one knob set). v1 = single-input MAP producer; a MONOID (rmsnorm) producer needs a compute-phase reduce (raises for now). `is_fused_graph` detects the same-launch-group multi-block case. |
| `_assemble.py`    | `assemble_block` — single block → `_assemble_one` synthesizes staging (`_slab`), then reconstructs the binding tower from `Schedule.binding` (ATOM/REGISTER/WARP/THREAD/GRID tiers) over the block's σ-rewritten `compute` (the warp tier's `Mma` cell rides inside the `AtomTile` for `kernel/005`). Multi-block DAG → `_assemble_multi` partitions `blocks` by `Schedule.launch` (one group = one kernel; v1 = one block per group, the two-launch cut — a multi-block group is the later cooperative `grid.sync` field), topo-sorts by the derived edge DAG, assembles each group, and wires a `Graph` of `TileOp` kernels with every cross-group edge materialized as an intermediate tensor (shape/dtype from the declared `TileGraph.buffers`). Deterministic — same `TileGraph` → byte-identical kernel set. |
| `_slab.py`        | `synthesize_staging` — materialize `Schedule.staged` into one `StageBundle` per K-tower (a `Source` per staged buffer, cache axes off the consumer `Load`, GRID + serial-outer K folded to the slab origin). SYNC by default; a `Transport.TMA` edge yields the double-buffered TMA ring (`buffer_count=2`, `phase = K_o % 2` prepended to the consumer slab Loads, per-source swizzle via `pick_swizzle_atom`). A **masked tile** (the K tower + `Write` wrapped in a boundary `Cond(σ(M\|N) < bound)`) goes through `_hoist_masked`: the K-pipeline is lifted **above** the guard (so every thread issues the cooperative load uniformly — a SYNC `__syncthreads` / cp.async / TMA inside divergent control flow hangs) and, for SYNC sources, `Source.gmem_extents` is stamped so `_stage_expand.emit_stage` clamps the overhang gmem read to `[0, extent)` (a static `int` or the symbolic `Var('seq_len')`); TMA sources rely on the hardware OOB zero-fill instead. **SSA-safety refusal**: if a hoisted K-tower stmt reads a name defined by a stmt staying inside the `Cond` (the fused-prologue shape), the hoist returns `None` and the caller keeps the `Cond` in place — hoisting would order a consumer above its definition (defense-in-depth; the planner doesn't emit such Conds today). `prospective_sources` exposes the slab `Source`s the transport fork's eligibility oracle + `050_stage`'s budget filter read pre-assemble. The per-axis `AffineAddressing.block` multiplier is **derived from the σ coefficients** — `()` for a scalar tile, atom-strided for a warp tile; a size-1 REGISTER cell is dropped first so its atom stride migrates to the warp axis. |
| `_tower.py`       | `_wrap_tower` — the shared innermost-first tower-building primitive (`Role` → tile flavor). |

## Coverage

Built today: **`MAP`** + scalar **`SEMIRING`** (including masked / symbolic free axes, split-K, and the `FK`
strip-mine) + the **`MONOID` cooperative-reduce** (R2 `coop_build` — softmax / rmsnorm / mean / max, static **and**
symbolic-K masked-fill, whole-CTA and strided-cooperative rows, the warp-shuffle / hierarchical combine) + the
**warp-tier `SEMIRING`** (tensor-core `mma.sync` via the R4 `atomize` move — matmul, residual / pointwise / causal-mask
epilogue fold, transposed-B, symbolic M/N), with **smem staging** (`stage` move) on both the scalar reduce regimes (R1)
and the warp operands (atom-strided slab + `ldmatrix`) + the **cross-CTA split-K combine** (R3 `055_atomic_free_splitk`
— the `NOATOMIC` structural fork: the matmul's per-`K_s` partials either `atomicAdd` into the output or write a
`partial[K_s, M, N]` workspace folded by a sibling additive / carrier-general combine kernel) + the **warp-tier TMA
transport** (R5 `052_transport` + `assembly/020_peel`: the `promote_transport` fork promotes a static-shape warp
matmul's staged operands to a double-buffered `cp.async.bulk.tensor` ring with per-source swizzle, software-pipelined
into prologue/main/epilogue — greedy stays SYNC, the tuner/`DEPLODOCK_TMA=1` selects TMA) + the **streaming
`TWISTED_MONOID` flash** (R6 `017_streaming` + `streaming_build` — SDPA / causal / GQA / additive-mask online-softmax, static
**and** symbolic-`seq_len` masked streaming, scalar-KV by default and **cooperative-KV** when `DEPLODOCK_BR>1` lays the
`K_c` THREAD lane on a static streaming axis). The **masked scalar-tile staging clamp** landed (R4 follow-up): scalar-offer
env-pin honoring (`_pin` in `thread_offers`/`reduce_reg_offers`) reaches the masked σ-split, the over-staging it
exposed is resolved by `050_stage`'s budget-aware mask filter (greedy falls back to the largest in-budget staging),
and `_slab._hoist_masked` lifts the cooperative load above the boundary `Cond` + clamps the SYNC gmem read to the
buffer extent (`test_masked_tile.py::test_planner_admits_non_divisor_n_with_real_extent` /
`…test_masked_n_clamps_cooperative_load_index` / `…test_symbolic_m_cooperative_load_clamps_to_runtime_extent`).
The remaining R4 follow-ups landed: the **gmem-direct unstaged atom** now compiles — a fully-pinned over-ceiling
`(FM, FN)` warp register tile is authoritative (`warp_reg_offers` bypasses the `_MAX_WARP_CELLS` *search* ceiling for a
full pin), so the warp build + assemble proceed, and with no `STAGE` pin `050_stage`'s budget-aware filter declines the
over-budget staging so the operands lower gmem-direct via `kernel/005_lower_atom_tile`
(`test_knob_pinning.py::test_unstaged_atom_lowers_gmem_direct`); and the **hoist-refuses** check is rewritten against
`_slab._hoist_masked`, which gained the SSA-safety refusal — a masked-tile hoist that would lift a K-tower stmt above an
SSA name defined by a stmt staying inside the boundary `Cond` (the fused-prologue shape) returns `None` and the caller
keeps the `Cond` in place (`test_masked_tile.py::test_hoist_refuses_lift_when_pipeline_reads_guarded_defs`).
See `plans/tile-ir-block-dag.md`
and `plans/algebra-licensed-decomposition-moves.md`.
