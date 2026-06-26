# Algebra-native knob naming schema — knobs per DAG element, not per GEMM letter

**Status:** in progress. Step 1 underway — the `_families` grammar + the **REDUCE@\<axis\>** and **SPLIT@\<axis\>**
families have landed:

- `REDUCE@<axis>` (`"s/f/c/t"`) replaces `BK`/`FK`/`SPLITK`/`BR`; the moves stamp it, `_build` decodes it, the
  enumeration passes gate on it.
- `SPLIT@<axis>` (`"par×reg"`) replaces `BN`/`BM`/`FN`/`FM`/`WN`/`WM`; the par (thread width / warp count) is stamped at
  the thread/warp-geometry fork as a par-only transitional and **completed** to `par×reg` at the register fork. The tier
  is read off the cell's `ATOM` (still `MMA` until the ATOM family lands), not the SPLIT value. The MONOID tier stamps
  the complete `SPLIT` (reg=1) in one leaf. The split-K combine is a dag-less, already-tiled op the geometry passes skip
  on a dag-None guard.
- `_validate`'s tier-foreignness policing of **both** geometry families is retired (plan step 3); only `MMA` / `STAGE` /
  `TMA` / `CHAIN` are still audited.
- `_knob_legacy` ingests legacy `DEPLODOCK_BN`/`BM`/`FN`/`FM`/`WN`/`WM` / `BK`/`FK`/`SPLITK`/`BR` to the native keys via
  the canonical free/reduce-axis ranking. The cold ranker (`AnalyticPrior`) is left to degrade per "break it, delete
  legacy" — tests that relied on its smart tile pick now pin the tile (legacy env pins route through ingest).

The **ATOM@\<cell\>** family (replacing `MMA`), **PLACE@\<edge\>** (step 5), and the golden YAML cutover (step 4) remain. No DB-compat constraint (clean-slate the storage) and **the learned prior is
assumed deleted** — so this plan designs no featurization and owes nothing to prior compatibility; greedy fork-picking
is whatever replaces the prior (out of scope here). Everything *native-facing* — `op.knobs` storage, `eval` display, new
goldens, new tests — speaks the new schema. A **legacy mapper** is **ingest-only** (legacy → new) and exists for one
reason: to keep old tutorials / existing env pins (`DEPLODOCK_WM=2`, …) and legacy-recorded golden YAMLs working *during
the transition*, by translating them to native knobs on read. It is a deprecation ramp, not a permanent fixture.

## Refactoring done after this plan was written (reconciliation notes)

The plan's assumptions were partly overtaken by work that landed afterward. Captured here so the execution stays
honest:

- **The learned prior is NOT deleted in code.** The precondition "the learned prior is assumed deleted" is satisfied by
  deleting the checkpoint **file** (`~/.cache/deplodock/prior.json`, which also holds the reservoir / -O3 evidence) — a
  cold prior, not a code excision. The `search/prior/` package (`CatBoostPrior` / `AnalyticPrior` / `FallbackPrior`)
  stays. **Decision:** the *cold ranker* is broken on purpose, not ported — greedy cold falls to **emission order**
  (the moves already emit best-first), and the legacy `D_*` tile-geometry featurization in `knob.py` /
  `search/analytic.py` is left to degrade (no native re-pointing). Tests that depended on the cold ranker's *smart*
  pick now pin the tile they need (legacy env pins route through the ingest mapper).
- **The carrier-algebra tile-IR landed after the plan.** `MAP`/`SEMIRING`/`MONOID` dispatch over the `IterDag`
  block-DAG, `ContractionChain` (the dual-role flash hinge `kv`), the TC flash **generated from the carrier** (the
  hand-built warp-chain phases were deleted), `split_carrier` relocated to `ir/stmt/carrier_algebra`, and the
  fragment-tier flash-softmax realizer. So the plan's flash example ("`CHAIN:true` + the warp-chain hand-assembled with
  empty knobs") is partly stale — flash is now carrier-generated. The `REDUCE@dd` / `REDUCE@kv` two-reduce-axis unlock
  still stands and is the right target; today both still ride one `REDUCE@<primary>` value (multi-axis is the next step).
- **The move/knob tests were reorganized/renamed** (`test_move_composer_matmul` → `test_matmul_rules`,
  `test_decompose_moves` → `test_decompose_rules`, etc.); stale `.pyc` for the old names linger in `__pycache__`.
- **The implementation reads native (or the IR), never legacy.** Per the maintainer steer: read from the IR structure
  where possible, else the new `MOVE@element` keys; legacy GEMM-letter names survive **only** in the ingest mapper
  (`_knob_legacy`) for backwards-compatible env pins / golden YAMLs.

## The problem

The tile composer dispatches on **carrier algebra** (`MAP`/`SEMIRING`/`MONOID`) over an arbitrary-rank iteration DAG —
axes tagged `PARALLEL`/`REDUCE` + a carrier, edges with a placement `∈ {INLINE, SMEM, GMEM}`, cells that may atomize.
But the knob vocabulary is a **rank-2 GEMM tile costume**: `BN` (N-thread), `BM` (M-thread), `BK` (K-chunk), `WM`/`WN`,
`FM`/`FN`, with `MAP_*`/`RED_*`/`TC_*` aliases stitched on so "the move code reads in move terms" (`_knobs.py`). Three
costs:

1. **The names presuppose M/N/K.** Only two free positions (M, N) and one reduce position (K) exist. A 3-free-axis
   pointwise, a two-reduce-axis contraction, or flash's dual-role `kv` hinge has nowhere to put its extra axes — flash's
   `dd` (QK^T reduce) **and** `kv` (stream reduce) cannot both be `BK`. The warp-chain flash carries `CHAIN=true` and
   otherwise **empty** knobs because the schema can't describe it.
2. **The placement lattice is fragmented across four knobs / four passes.** `STAGE` (→SMEM), `CUT` (→GMEM), `CHAIN`
   (→INLINE score), and the implicit gmem-direct default realize what the IR already computes as one query,
   `TileGraph.placement(edge)`. `CHAIN` is overloaded across two passes (scalar FA-2 restructure in `070_coop_reduce`
   AND warp-chain TC flash in `005_warp_chain`) precisely because there is no first-class "place this edge INLINE" knob.
3. **Move and realization are conflated, tiers coupled by magic sentinels.** `STAGE` decides SMEM *and* which sites;
   `TMA` separately decides the ring; `MMA` decides atomize *and* which atom *and* the tier. `BN.off=0` means "warp tier,
   BN unused"; `WM.off=0` means "scalar tier, WM unused". `_validate.py` exists largely to police that the vocabulary is
   tier-disjoint.

## The schema: `MOVE@element`

Each knob is **one move applied to one DAG element**. The key is the move family + the element's own IR identity:

- **free / reduce axes** → `Axis.name` (the `a0`/`a1`/`dd`/`kv` the iter-DAG assigns; stable per structural key)
- **edges** → the intermediate buffer name (`xn`, `score`)
- **cells** → the `Block.name` holding the cell

`op.knobs` keys read `SPLIT@a0`, `REDUCE@kv`, `PLACE@score`, `ATOM@out`. Four families, **instantiated per kernel** by
walking the DAG at `010_build`.

### `SPLIT@<free-axis>` — the `tile_axis` move

Value `"<par>x<reg>"`: the parallel-binding factor × the register-cell factor. **Grid is the launch residual** (extent /
(par·reg)) — not stamped. The *tier* of `par` (THREAD vs WARP) is **not a knob** — it is read off the consuming cell's
`ATOM` (scalar → thread, atom → warp). So legacy `BN`/`FN` and `WN`/`FN` collapse to one `SPLIT@<n-axis>`; the
thread-vs-warp distinction is recovered from `ATOM`, not from which knob was set.

### `REDUCE@<reduce-axis>` — the `reduce_decomp` move

Value `"s<serial>/f<fold>/c<cta>/t<coop>"`: the four reduce-decomposition tower components —

| field      | move                                                | legacy   |
|------------|-----------------------------------------------------|----------|
| `s` serial | serial re-bracket (intra-CTA K-loop trip = ext/s)   | `BK`     |
| `f` fold   | register strip-mine into independent accumulators   | `FK`     |
| `c` cta    | cross-CTA split (split-K)                           | `SPLITK` |
| `t` coop   | cooperative-thread partition (warp-shuffle combine) | `BR`     |

**Which fields are legal is the carrier's traits** — `associative → {s,f}`, `commutative → {c,t}`, `has_identity →`
masking is automatic (symbolic / non-divisible). An illegal field is forced to `1` (= identity = no decomposition). This
is the structural unlock: `REDUCE` is **per reduce axis**, so flash gets both `REDUCE@dd` and `REDUCE@kv`, which today's
K-only `BK/FK/SPLITK/BR` cannot express. The trait-gating *is* the legal value domain — replacing `_validate.py`'s
tier-foreignness policing with a value-domain check at the knob.

### `PLACE@<edge>` — the placement lattice + transport

Value `place[:xport]`: `inline` | `smem:sync` | `smem:cpasync` | `smem:tma` | `gmem`. One per-edge lattice coordinate
absorbs `STAGE` (→smem), `TMA` (→the `:xport` suffix), `CUT` (→gmem), and `CHAIN`'s INLINE score (→inline). The
transposed-B "don't stage" exclusion becomes `PLACE@B = inline` — no special case. `STAGE` was already a *bitmask over
DAG-ranked edges* (binary staged-or-not); this generalizes each edge's cell from `{0,1}` to the 3-valued lattice + a
transport sub-value.

### `ATOM@<cell>` — the `atomize` move

Value `scalar` | an `ATOM_REGISTRY` kind (`mma_m16n8k16_f16`). **Per cell**, so flash names both (`ATOM@score`,
`ATOM@out`); a fused producer+matmul names the producer cell + the matmul cell. `b_trans` / operand dtype stay
**derived** by `classify_matmul_operands`, never knobs.

### Env spelling + wildcards

`DEPLODOCK_<MOVE>_<ELEMENT>` (element upper-cased): `DEPLODOCK_REDUCE_KV=s16/f1/c1/t2`, `DEPLODOCK_PLACE_SCORE=inline`,
`DEPLODOCK_ATOM_OUT=mma_m16n8k16_f16`. A **bare family** pins all elements of that kind — `DEPLODOCK_PLACE=gmem`,
`DEPLODOCK_ATOM=scalar`, `DEPLODOCK_REDUCE=s16/f1/c1/t1` — for coarse pins / tests.

## Worked examples

**Warp-tier matmul `C[m,n]=Σ_k A·B`** — today `{MMA, WM:2, WN:2, FM:4, FN:4, BK:16, SPLITK:1, STAGE:"11", TMA:1}`:

```
ATOM@C   = mma_m16n8k16_f16
SPLIT@m  = 2x4            REDUCE@k = s16/f1/c1/t1
SPLIT@n  = 2x4            PLACE@A  = smem:tma   PLACE@B = smem:tma
```

**Flash (undescribable today)** — today `CHAIN:true` + the warp-chain is hand-assembled with empty knobs:

```
ATOM@score = mma_m16n8k16_f16     ATOM@out = mma_m16n8k16_f16
SPLIT@m    = 1x1                  SPLIT@d  = 1x4
REDUCE@dd  = s16/f1/c1/t1         # QK^T inner contraction
REDUCE@kv  = s1/f1/c1/t1          # the streaming axis (t = old BR cooperative-KV)
PLACE@score = inline             # the INLINE score edge (was CHAIN)
PLACE@out   = gmem
```

Two reduce axes, two cells, and the INLINE edge become first-class describable values.

## The legacy mapper (`_knob_legacy.py`)

The new schema is the **source of truth** everywhere native-facing — storage, `eval` display, new goldens, new tests all
speak `MOVE@element`. The legacy names survive only as an **ingest-only** translation at the read boundary, so an
existing env pin or a legacy-recorded golden YAML still resolves. The high-value set the user named — `MMA`, `WM`, `WN`,
`FM`, `FN`, `STAGE`, `TMA` — plus the core scalar/reduce/structural knobs (`BN`, `BM`, `BK`, `FK`, `SPLITK`, `BR`, `CUT`,
`CHAIN`). (The perf DB and the prior are not bridged: the DB is clean-slated to native keys, and the prior is deleted.)

### The element-resolution rule (the bridge)

The mapper recovers the M/N/K positions from the **canonical DAG ranking the move offers already use** — free axes
ranked innermost-first (rank-0 = "N", rank-1 = "M"); reduce axes ranked primary-first (rank-0 = "K"); edges ranked by
`stage_candidates` (bit `i` = ranked edge `i`):

| legacy                        | new target                                                  | element selector            |
|-------------------------------|-------------------------------------------------------------|-----------------------------|
| `MMA` (`ATOM_KIND`)           | `ATOM@<cell>` + sets `SPLIT.par` tier=WARP                  | the kernel's matmul cell(s) |
| `WN` / `FN`                   | `SPLIT@<free rank-0>` `.par` / `.reg`, tier=WARP            | innermost free axis         |
| `WM` / `FM`                   | `SPLIT@<free rank-1>` `.par` / `.reg`, tier=WARP            | next-out free axis          |
| `BN`                          | `SPLIT@<free rank-0>` `.par`, tier=THREAD                   | innermost free axis         |
| `BM`                          | `SPLIT@<free rank-1>` `.par`, tier=THREAD                   | next-out free axis          |
| `BK` / `FK` / `SPLITK` / `BR` | `REDUCE@<reduce rank-0>` `.s` / `.f` / `.c` / `.t`          | primary reduce axis         |
| `STAGE` bit `i`               | `PLACE@<edge rank-i>` = `smem`                              | ranked stageable edges      |
| `TMA`                         | the `:tma` suffix on every `smem`-placed edge               | all staged edges            |
| `CUT` bit `i`                 | `PLACE@<cuttable edge i>` = `gmem`                          | ranked cuttable edges       |
| `CHAIN`                       | `PLACE@<score edge>` = `inline` + `ATOM@{score,out}` = atom | the streaming score edge    |

The tier ambiguity (`WM` and `BN` both target `SPLIT.par`) is **resolved by which legacy knob is set**: `WM=2` ⟹
`ATOM@cell` = warp **and** `SPLIT.par=2`; `BN=16` ⟹ `ATOM@cell` = scalar **and** `SPLIT.par=16`. The legacy tier-split
*is* an `ATOM` choice in the new schema, so the mapper collapses it cleanly.

### One direction: `ingest`

```python
def ingest(legacy: dict[str, str], dag: IterDag) -> dict[str, str]:
    """Legacy knob dict (env pin / legacy golden YAML) -> new per-element knobs.
    DAG-aware: resolves M/N/K/edge positions via the canonical ranking. The only
    legacy-facing entry point — there is no reverse (display/storage are native)."""
```

`ingest` runs at exactly two read seams: (1) **env-pin read** — `Knob.raw()` for a legacy name routes through `ingest`
so `DEPLODOCK_WM=2` lands on `SPLIT@<rank1-free>`; (2) **legacy golden-YAML load** — legacy-keyed entries translate on
read. After that boundary everything is native: nothing ever projects back. There is deliberately **no** `project` (new
→ legacy) — `eval`, new goldens, and `op.knobs` all speak the native schema directly, so a reverse map would only invite
drift.

**Legacy pins can't express the richer decisions, by design.** A tutorial pinning `BK`/`WM`/`STAGE` reaches only the
rank-0 axis / ranked-edge slots; flash's 2nd reduce axis, a 3rd free axis, or an INLINE non-score edge have no legacy
name and simply aren't reachable through the ramp. That is the intended nudge — to touch those, move to the native pin.
`ingest` is the **only** place the canonical M/N/K ranking is named: one auditable table, the same role `_cut.py` plays
for the cut policy.

## What becomes DERIVED and drops out

The schema shrinks as structural facts move from knobs to DAG queries — the trajectory this work continues:

- **causal** — read off the `kv≤m` Select (already needs no knob; Phase 5).
- **b_trans / operand dtype** — `classify_matmul_operands` (derived).
- **streaming** — `dag.streaming` (derived).
- **`FLASH`** — stays a *recognizer* knob in the loop phase (it changes which ops exist before the DAG), not a tile knob;
  out of scope for this schema. `CHAIN` is **retired** — folded into `PLACE@score=inline` + `ATOM@{score,out}`.
- **`FKWIN`** (the half2 window) — a *realization* of `REDUCE.f`, not a separate knob; folds into the `kernel/015`
  lowering keyed on the fp16-matmul predicate.

The OFF-sentinel encoding goes away with the tiers: there is no `BN=0` "means warp tier" magic — a free axis simply has
a `SPLIT@a` whose binding tier is read from `ATOM@cell`, and an unused `REDUCE` field is `1` (identity), not a foreign
knob to suppress.

## Sequencing

1. **Families + grammars + per-kernel instantiation.** `KnobFamily` (per `AXIS_FREE` / `AXIS_REDUCE` / `EDGE` / `CELL`,
   a value grammar, a trait-legality gate for `REDUCE`). `010_build` walks the DAG and instantiates the concrete knobs;
   the move offers (`_moves.py`) stamp the new keys; `eval` display reads them natively. **Oracle:** a matmul / pointwise
   / reduce compiles to the byte-identical kernel it does today (the decisions are unchanged, only their names are).
2. **The legacy mapper** (`_knob_legacy.py`): `ingest` + the element-resolution table; wire `ingest` into the env-pin
   read and the legacy golden-YAML load (ingest-only — no `project`). **Oracle:** every existing `DEPLODOCK_<LEGACY>=…`
   test passes unchanged through `ingest`; every legacy golden YAML loads and compiles to the same kernel.
3. **Retire `_validate.py`'s tier policing** → the `REDUCE`/`SPLIT` value-domain + `ATOM` gate. Drop the `off=0`
   sentinels (no prior left to read the absent-vs-declined distinction, so the sentinels lose their last consumer).
4. **Cut over goldens / tests to native names**; the legacy mapper stays for env-pin convenience and external tooling.
5. **Fold `PLACE` across the placement passes** — `120_stage` (smem), `130_transport` (xport), `010_split_demoted`
   (gmem), `005_warp_chain` (inline score) all read/write one `PLACE@<edge>` per edge instead of four disjoint knobs.
   This is where the flash deploy becomes clean: the warp-chain is "this kernel's score edge is `PLACE=inline`, its cells
   are `ATOM=mma_*`" — one fork over the `PLACE@score` lattice coordinate, not a `CHAIN` BOOL smeared over two passes.
   (How greedy *chooses* that fork is the post-prior picking policy's job, out of scope here.)

## Open decisions (recommendations baked in above; flag if you disagree)

- **Axis identity = `Axis.name`** (canonical `a0`/`a1`, semantic `dd`/`kv` where the iter-DAG assigns them). Recommend
  teaching `iter_dag` to assign semantic role names (`row`/`col`/`head`/`reduce`) where structurally determined, so pins
  and goldens read well; the legacy mapper does **not** depend on this (it uses the canonical *ranking*), so it is a
  legibility nicety, not a blocker.
- **Grid implicit** for free axes (`SPLIT = par×reg`, grid = residual); the cross-CTA partition of a *reduce* axis is
  `REDUCE.c`, kept explicit. Alternative: explicit `g×p×r` to kill the residual magic — rejected as noisier for the
  common case.
