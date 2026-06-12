# Golden sweep findings — RTX 5090 (sm_120), 2026-06-12

- Sweep: `deplodock tune --dataset golden --clean` — 23 matmul shapes, 40 min wall (2411 s; the skill's ~30 min
  estimate is light by a third — the square.4096 pair dominates).
- A/B: `deplodock run --bench --golden NAME` per shape (28 names; the 5 reduce/pointwise names error out — see
  workflow notes), wins confirmed with two extra runs each (run-to-run spread ≤1.5%, far below the historical
  10–13% band — CUDA-graph capture has tightened the noise floor).
- Branch under test: `feature/strided-coop-symbolic-rows` (post M9 masked warp-tier MMA work).
- Tally: **8 replaced / 3 added / 3 re-recorded (same knobs, lower µs) / 9 worse / 5 unsweepable (reduce/pointwise)**.

## Per-shape outcome (all numbers -O3 live A/B, never the -O1 tune DB)

| shape                       | greedy µs | best golden µs | greedy/golden | category |
|-----------------------------|----------:|---------------:|--------------:|----------|
| square.512                  |       9.0 |            9.1 |          0.99 | same knobs — re-recorded 10.2 → 9.0 |
| square.1024                 |      45.8 |           48.3 |          0.95 | **replaced** |
| square.2048                 |     315.9 |          269.7 |          1.17 | worse (finding 2) |
| square.4096                 |    3143.6 |         2223.6 |          1.41 | worse (finding 2) |
| square.512.fp16             |       4.1 |            3.5 |          1.17 | worse (finding 1) |
| square.1024.fp16            |      16.1 |           14.9 |          1.08 | worse (finding 1) |
| square.2048.fp16            |     189.6 |           91.8 |          2.07 | worse (finding 1) |
| square.4096.fp16            |    5464.2 |          650.5 |          8.40 | worse (finding 1) |
| qwen3_06b.q_proj.s32        |       7.5 |            7.6 |          0.99 | **added** (parity, diff knobs) |
| qwen3_06b.kv_proj.s32       |       5.3 |            5.3 |          1.00 | same knobs — re-recorded 8.2 → 5.3 |
| qwen3_06b.o_proj.s32        |      10.8 |            8.7 |          1.24 | worse (finding 3) |
| qwen3_06b.gate_up_proj.s32  |       9.6 |           10.6 |          0.91 | **replaced** |
| qwen3_06b.down_proj.s32     |      15.0 |           12.1 |          1.24 | worse (finding 3) |
| qwen3_06b.q_proj.s128       |      16.3 |           16.5 |          0.99 | **added** (parity, diff knobs) |
| qwen3_06b.kv_proj.s128      |       9.8 |           10.6 |          0.92 | **replaced** (both old entries pruned) |
| qwen3_06b.o_proj.s128       |      17.7 |           19.8 |          0.89 | **replaced** |
| qwen3_06b.gate_up_proj.s128 |      28.8 |           24.4 |          1.18 | worse (finding 4) |
| qwen3_06b.down_proj.s128    |      24.9 |           29.1 |          0.86 | **replaced** |
| qwen3_06b.q_proj.s512       |      50.8 |           50.5 |          1.01 | same knobs — re-recorded 55.4 → 50.8 |
| qwen3_06b.kv_proj.s512      |      24.6 |           27.4 |          0.90 | **replaced** |
| qwen3_06b.o_proj.s512       |      50.5 |           55.6 |          0.91 | **replaced** |
| qwen3_06b.gate_up_proj.s512 |      67.6 |           70.6 |          0.96 | **replaced** |
| qwen3_06b.down_proj.s512    |      80.8 |           81.1 |          1.00 | **added** (parity, diff knobs) |

The wins are one coherent story: the prior now likes `SPLITK=2` + wider `BN=32` on the s32/s128 projections and the
`BK=32 FM=14 OVERHANG=['a0']` overhang family on the 512-row shapes — 9 of the 11 recorded configs carry one or both,
and every win reproduced 3/3 well above the ≤1.5% spread. The three same-knob re-records (square.512, kv_proj.s32,
q_proj.s512, −9% to −35%) are not codegen wins: the old numbers predate captured timing, so this is the YAML catching
up to the capture-mode measurement (kv_proj.s32's recorded 8.2 µs was a wall-clock-era number for the same kernel).

## Finding 1 — fp16 warp tier: the TMA+WARPSPEC class crashes at launch, locking the search out of every fp16 golden

The dominant regression — all four fp16 squares are worse, from +8% (1024) through 2.07× (2048) to **8.4×** (4096).

- `eval failures`: 7 bench_fail rows are `CUDADriverError('CUDA_ERROR_INVALID_VALUE: invalid argument')`, and every
  one shares `TMA=True, WARPSPEC=True, MMA=mma_m16n8k16_f16` (6 on `k_matmul_262948`, 1 on `k_matmul_207791`).
- The golden configs themselves are fine: `run --bench --golden square.2048.fp16` pins the recorded knobs and they
  run at 91.8 µs. Only the tune-time enumeration (which pairs WARPSPEC=True with TMA=True on this branch) crashes.
  Previous sweeps deployed these exact configs, so this is a regression — almost certainly from the masked warp-tier
  MMA (M9) / strided-cooperative-rows work, which touched the warp tier's load/launch paths.
- Knock-on 1: with the WARPSPEC=True class dead, no healthy fp16 warp-tier rows reach the prior's dataset — the fp16
  shapes are absent from `eval prior --dataset golden`'s rank list ("shapes with tuned data" lists only fp32). The
  greedy then deploys `WARPSPEC=False` (1024/2048.fp16) or worse.
- Knock-on 2: square.4096.fp16's greedy pick is `MMA=0` (no tensor cores at all) with `FK=32 SPLITK=2` — the *same
  family* as the three `benchmark run stage exceeded 2.0s of GPU time` bench_fails on `k_matmul_180e20`. The greedy
  deployed a config class whose only DB evidence is failure, at 5464 µs live.
- The cold heuristic is **not** the problem: `eval analytic` ranks the fp16 goldens **0** (1024/2048/4096.fp16) — the
  analytic prior would deploy them sight unseen.

**Recommendation (P0):** debug the `TMA=True + WARPSPEC=True + mma_f16` launch failure (CUDA_ERROR_INVALID_VALUE at
launch smells like a TMA descriptor / launch-arg mismatch introduced by the masked-MMA changes). Add a regression
test that compiles + launches the golden `square.1024.fp16` knobs with the planner-derived TMA transport. Separately,
greedy deploy should refuse a config whose op family only has bench_fail rows (the 4096.fp16 pick) — prefer the
analytic fallback over an unmeasured-and-failing class. square.512.fp16 (+17%) is collateral of the same lockout:
its greedy is in the right `WM4 WN4` family but lands `BK2 RING3` vs golden `BK4 RING2` with no fp16 data to rank by.

## Finding 2 — square.2048 / square.4096 fp32: big-register-tile goldens rank too deep for patience to reach

Greedy +17% / +41%. Per `eval golden`, greedy misses `FM 8/26` (2048) and `BM 8/16, FM 6/10, BK 64/32, SPLITK 2/1`
(4096) — the goldens are very wide register tiles (`FM=26`, `BM=16 FM=10`).

- `eval analytic`: golden rank 121 (2048) and 59 (4096) over a 1008 pool — the hand-coded weights misprice big FM.
- `eval prior --dataset golden`: rank **378** (2048) and 102 (4096) — the learned prior is even worse here, so the
  patience-bounded inner search never gets offered the golden region; `eval variants` for `k_matmul_180e20` confirms
  the golden config is absent from the 34 measured variants (reachability failure, not an -O1/-O3 inversion — the
  measured -O3 column tops out at 2450 µs vs the golden's live 2224 µs).

**Recommendation:** refit the analytic weights with the big-FM regime represented (`scripts/golden_knob_heuristics.py`,
tier-balanced), and add a `D_*` engineered feature capturing register-tile arithmetic intensity vs occupancy (FM·FN
per thread vs regs/occupancy ceiling) so both priors can see what distinguishes these configs. A patience bump alone
is a weak fix at rank 378.

## Finding 3 — o_proj.s32 / down_proj.s32: prior picks FM=1 over the measured-better FM=2 (+24% both)

- `eval golden`: the only knob misses are `FM 1/2` (o_proj also `RING` matches, down_proj adds `FN 4/2`).
- `eval variants --kernel bec8cc` (o_proj.s32): the golden `FM=2 FN=2 SPLITK=2` family holds measured ranks 1–4 with
  **-O3 re-bench rows at 8.0–8.7 µs**, while the prior deploys an `FM=1` config at rank 17/65 (1.61× of best -O1).
  The -O3 column rules out an -O1/-O3 inversion: the golden family is deployable-faster (8.7 vs 10.8 µs live).
- The learned prior ranks the goldens shallow (2–3/1008), i.e. the argmin misorders within its own top-3 — a
  calibration error in the tiny-M (M=32) regime, not a search/reachability problem.

**Recommendation:** the data is measured and the rank is shallow — this is the cheapest class to fix. Look at the
prior's featurization for M=32 shapes (the `S_ext_*` products collapse M=32×N tails); a `D_*` term separating
per-thread work at degenerate M (FM=1 halves work per thread but doubles CTA count at tiny M) should flip the top-3
ordering. Re-check with `eval prior --dataset db` reachability after the next refit.

## Finding 4 — gate_up_proj.s128: golden's BN16+OVERHANG config never measured this sweep

Greedy `BN=32 SPLITK=2` 28.8 µs vs golden `BN=16 FM=6 OVERHANG=['a0']` 24.4 µs (+18%). `eval variants` for
`k_matmul_2f6858` shows 18 measured configs with the deployed pick already rank 1 — the golden config was never
offered/measured this sweep. Learned rank 11, analytic rank 0: both priors think well of it; the inner search's
patience expired before reaching it. Note the contrast: on the same family one tier up (gate_up_proj.s512) the
BN32+SPLITK2 greedy genuinely beat the old BN16+OVERHANG golden by 4.3%, so the family direction is shape-specific.

**Recommendation:** patience bump is plausible here (rank 11 is reachable), but check first whether the OVERHANG
offer fires for N=3072 at M=128 in the enumeration this sweep took — `eval variants` showing zero OVERHANG rows for
this kernel (vs plenty on the s512 shapes) suggests the offer gate, not patience, decided it.

## Workflow notes

- **Tune wall time**: 2411 s, dominated by the four square.4096-class shapes. A `--kernel` exclusion (or per-shape
  patience scaling by shape size) would cut the routine re-sweep to ~25 min.
- **reduce/pointwise goldens are unsweepable**: `tune --dataset golden` targets only the 23 matmul shapes and
  `run --golden NAME` rejects the 5 reduce/pointwise names outright (`unknown golden config`). The YAML's
  ground-truth claim silently excludes them. Either extend `--golden` / the tune target list to non-matmul goldens
  or mark them explicitly as not part of the sweep. Cost this run: 5 failed A/B invocations and one re-loop.
- **The 10–13% noise band is stale.** Three runs × 14 shapes showed ≤1.5% spread under captured timing. Step 4 of
  the skill could drop to a single confirmation run, saving ~10 min.
- **Recorded golden numbers predate captured timing.** Same-knob shapes re-benched 9–35% lower (kv_proj.s32
  8.2 → 5.3 µs with identical knobs). The untouched "worse" entries carry the same skew (o_proj.s32 recorded 12.3,
  lives at 8.7), which also distorts `eval prior`'s `vs gold` column (kv_proj.s32's 0.65x is mostly a stale
  denominator). A one-shot re-record pass over `deplodock_us`/`cublas_us` for all entries would restore the ratio
  columns' meaning; until then treat the YAML latencies as ranking-era, not deployable, numbers.
- **`eval variants` kernel-name collisions.** fp32/fp16 twins and even different shapes share `k_matmul_<hash>`
  names, so per-shape drilldowns are unreliable: `k_matmul_bed174`'s group showed only 13 FK-split rows at 4–100 ms
  (square.2048's real ~300 µs rows were elsewhere), and down_proj.s32's rows were occluded by another shape's group
  under `k_matmul_75f220`. The view needs a shape-key (`S_*` sig) subgrouping or a `--shape` filter.
- **Silent fp16 omission in `eval prior --dataset golden`**: the rank list says "over 23 golden shapes with tuned
  data" while listing duplicates and omitting all fp16 shapes — the no-tuned-rows case should be a printed warning
  per shape, not a silent drop (it was the key clue for finding 1 and easy to miss).
- **A/B sweep is 28 cold process launches** (~25 min). A `run --bench --dataset golden` in-process loop (mirroring
  what `tune --dataset golden` got in `_tune_targets`) would amortize import/trace and emit the per-shape table
  machine-readably; this sweep assembled the outcome table by parsing 28 logs with awk.
- **Vs the 2026-06-07 report**: its action item (restore `STAGE=11`-first for fp32) is fixed and held — every fp32
  shape now deploys `STAGE=11`, and the fp32 counts flipped from 20-worse/0-better to 5-worse/11-better-or-parity.
  Its fp16 win, however, regressed via the new TMA+WARPSPEC crash (finding 1). That report predates the workflow-notes
  convention, so no notes to compare.
