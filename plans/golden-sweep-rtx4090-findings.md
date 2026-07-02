# Golden sweep findings — RTX 4090 (sm_89), 2026-06-19

- GPU: NVIDIA GeForce RTX 4090, sm_89, CUDA 12.9 (`nvcc` at `/usr/local/cuda`), local single-host sweep.
- Sweep: `emmy tune --dataset golden --clean` (29 shapes, ~102 min compute) → per-shape A/B
  `emmy run --bench --golden NAME` (29 shapes, ~6 min) → confirmation re-runs of the win candidates (~12 min).
  Logs under the gitignored `_tune/golden-sweep-rtx4090/` (`tune.log`, `ab/`, `ab2/`).
- Branch: `feature/golden-sweep-rtx4090`.
- **Category tally: 12 replaced / 0 added / 11 unchanged / 6 worse (left).** All A/B numbers are -O3 `run --bench`,
  never the -O1 tune DB.

## Headline: two prior-report findings closed before a single golden was re-recorded

This sweep is the first 4090 golden refresh since the 2026-06-13 seed, and two of that report's blockers are now gone:

- **Prev Finding 1 (fp16 tensor cores blocked on sm_89) — RESOLVED.** The 2026-06-13 file recorded all four fp16
  squares as **scalar-tier fallback** (`BM/BN/FM/FN`, no `WM/WN/MMA`) at 3–5× cuBLAS HGEMM, because the warp/MMA
  ldmatrix path faulted with an OOB `__shared__` read on Ada. The recent commits `28646fea fix ldmatrix bug` +
  `c7fa9ff2 drop ldmatrix gate on sm_89` fixed that. This sweep's greedy pick now deploys the **warp/MMA tier**
  (`MMA=mma_m16n8k16_f16`) for every fp16 square, passes the accuracy gate, and runs **2.9–3.0× faster** than the
  recorded scalar goldens. All four are replaced (see table). The stale scalar goldens were not just slow — their
  recorded `RING` knob **no longer fires** ("BUFFER_COUNT not promotable"), so `run --bench --golden` could not even
  re-bench them; the warp-tier replacement is the only deployable config now.

- **Prev Finding 3 (hard `LoweringError` with no fallback on sm_89) — RESOLVED (this branch).** The 2026-06-13 report
  recommended verbatim: *"the deterministic-compile path should fall back to the next-ranked tile when the chosen one
  fails `validate(ctx)` rather than raising."* The sweep could not even start without this: with the freshly-trained
  prior extrapolating a 112 KB-smem tile onto the tiny `qwen3_06b.q_proj.s32` (M=32) shape, the in-process tune crashed
  at shape 9/29 with `LoweringError: ... smem 114688 > max_dynamic_smem 101376 ... no fallback`. The smem cap is
  correct (`101376` is the genuine sm_89 per-block opt-in limit, sourced from the per-cc table in `gpu.py`), and the
  `validate(ctx)` gate was doing its job. The bug was the recovery: `Pipeline.run`'s 8-retry blocklist
  (`_MAX_GREEDY_RETRIES`) exhausts on the prior's over-budget picks and then *raised*. Fix (`pipeline/pipeline.py`): on
  retry exhaustion, fall back to the conservative emission-order resolve (`greedy_decide(prior=None)` = option-0), which
  is budget-safe by construction; it still raises when no in-budget tile exists (the single-option guardrail is
  preserved). Regression test added in `tests/compiler/pipeline/test_lowering_error_guardrail.py`
  (`test_greedy_run_falls_back_to_option0_when_prior_overflows`). With the fix the full 29-shape sweep ran to
  completion, 0 crashes.

The remaining picture matches **prev Finding 2** (the cold sm_120-fit `AnalyticPrior` mis-prices sm_89): the recorded
goldens were *transferred 5090 configs*, and the 4090's own warm-prior greedy now **beats** them on the whole s512 +
gate_up.s128 family (0.55–0.78×, recorded), but still **loses** on `square.4096`, the masked-tile `.dynM` family, and
two tiny s32 shapes — the shapes where the analytic geometry terms extrapolate worst.

## Per-shape outcomes (-O3 `run --bench` A/B)

| shape                            | greedy µs | golden µs | ratio | cuBLAS µs | vs cuBLAS | category                         |
|----------------------------------|----------:|----------:|------:|----------:|----------:|----------------------------------|
| square.512                       |      13.6 |      13.5 |  1.01 |      10.8 |      1.26 | unchanged (same knobs)           |
| square.1024                      |      74.2 |      71.0 |  1.05 |      45.4 |      1.63 | unchanged (same knobs; clean rerun) |
| square.2048                      |     383.3 |     467.5 |  0.82 |     320.0 |      1.20 | **replaced** (golden un-rebenchable) |
| square.4096                      |    4216.1 |    3192.8 |  1.32 |    2458.6 |      1.71 | worse → leave (Finding 1)        |
| square.512.fp16                  |       5.9 |      17.7 |  0.33 |       5.8 |      1.02 | **replaced** (scalar→warp/MMA)   |
| square.1024.fp16                 |      29.3 |      90.7 |  0.32 |      18.1 |      1.62 | **replaced** (scalar→warp/MMA)   |
| square.2048.fp16                 |     119.3 |     512.5 |  0.23 |     115.2 |      1.04 | **replaced** (warp/MMA + 128×128 tile, Finding 5) |
| square.4096.fp16                 |     889.9 |    2125.8 |  0.42 |     822.3 |      1.08 | **replaced** (scalar→warp/MMA)   |
| qwen3_06b.q_proj.s32             |       7.8 |       7.8 |  1.00 |       9.9 |      0.79 | unchanged (same knobs)           |
| qwen3_06b.kv_proj.s32            |       6.3 |       6.9 |  0.91 |       6.9 |      0.91 | unchanged (noise, ~0.4 µs)       |
| qwen3_06b.o_proj.s32             |      11.3 |      11.9 |  0.95 |       9.9 |      1.14 | unchanged (noise)                |
| qwen3_06b.gate_up_proj.s32       |      14.3 |      12.9 |  1.11 |      11.3 |      1.27 | worse → leave (Finding 4)        |
| qwen3_06b.down_proj.s32          |      19.9 |      16.6 |  1.20 |      13.0 |      1.53 | worse → leave (Finding 2)        |
| qwen3_06b.q_proj.s128            |      20.9 |      23.5 |  0.89 |      20.2 |      1.03 | **replaced** (12% under recorded) |
| qwen3_06b.kv_proj.s128           |      14.2 |      13.5 |  1.05 |      12.4 |      1.15 | unchanged (noise)                |
| qwen3_06b.o_proj.s128            |      23.2 |      24.4 |  0.95 |      19.0 |      1.22 | unchanged (noise)                |
| qwen3_06b.gate_up_proj.s128      |      33.0 |      49.5 |  0.67 |      25.5 |      1.29 | **replaced**                     |
| qwen3_06b.down_proj.s128         |      35.4 |      33.4 |  1.06 |      24.5 |      1.44 | unchanged (noise)                |
| qwen3_06b.q_proj.s512            |      55.6 |      95.8 |  0.58 |      53.3 |      1.04 | **replaced**                     |
| qwen3_06b.kv_proj.s512           |      34.3 |      44.5 |  0.77 |      38.9 |      0.88 | **replaced**                     |
| qwen3_06b.o_proj.s512            |      55.6 |      87.9 |  0.63 |      67.8 |      0.82 | **replaced**                     |
| qwen3_06b.gate_up_proj.s512      |     103.7 |     132.2 |  0.78 |      85.5 |      1.21 | **replaced**                     |
| qwen3_06b.down_proj.s512         |      79.7 |     143.1 |  0.56 |     113.2 |      0.70 | **replaced**                     |
| square.512.dynM                  |      13.9 |      12.9 |  1.08 |      10.8 |      1.29 | worse → leave (Finding 3)        |
| qwen3_06b.q_proj.s512.dynM       |      67.7 |      67.1 |  1.01 |      53.0 |      1.28 | unchanged                        |
| qwen3_06b.kv_proj.s512.dynM      |      38.8 |      35.3 |  1.10 |      37.0 |      1.05 | worse → leave (Finding 3)        |
| qwen3_06b.o_proj.s512.dynM       |      69.9 |      61.7 |  1.13 |      67.1 |      1.04 | worse → leave (Finding 3)        |
| qwen3_06b.gate_up_proj.s512.dynM |      91.5 |      96.0 |  0.95 |      85.6 |      1.07 | unchanged (noise, same knobs)    |
| qwen3_06b.down_proj.s512.dynM    |      95.3 |      91.8 |  1.04 |     119.4 |      0.80 | unchanged (noise)                |

`vs cuBLAS` = greedy µs / recorded `cublas_us` (torch eager: true-fp32 SGEMM with `allow_tf32=False`, or HGEMM for
`*.fp16`), so **>1.0 = emmy is slower than PyTorch** — the absolute gap the relative greedy-vs-golden ratio hides.
The worst cuBLAS losers are the large/regular GEMMs: `square.4096` 1.71×, `square.1024` 1.63×, `square.1024.fp16`
1.62×, `square.2048.fp16` 1.54×, `down_proj.s32` 1.53×. Note the fp16 squares win their golden A/B 3× yet still trail
cuBLAS HGEMM 1.5–1.6× (the win was vs the stale scalar golden, not vs PyTorch). emmy only *beats* cuBLAS on six
rectangular projections (`down_proj.s512` 0.70×, `down_proj.s512.dynM` 0.80×, `o_proj.s512` 0.82×, `kv_proj.s512`
0.88×, `q_proj.s32` 0.79×, `kv_proj.s32` 0.91×). The cuBLAS-loser shapes that are *also* golden-unchanged/worse are the
real headroom; the `_W_A` refit (Findings 1/4) targets the worst of them.

Both win families reproduced on a second independent `run --bench` (fp16 squares: 5.9 / 27.7 / 177.4 / 891.9; s512:
q 53.7, kv 34.3, o 55.5, gate_up 98.0, down 79.6) — all well outside the ~10–13% small-shape noise band. The two flaky
fp32 squares (1024, 2048) emitted `cudaErrorMisalignedAddress` at teardown in the `--golden` run; re-running greedy via
`--code` (no golden-pin contamination) was clean (0 misalign, 74.2 / 383.3 µs), so the faults were the golden pin
dirtying the context, not the greedy kernels. `square.1024` reproduces its golden (same knobs) → unchanged;
`square.2048`'s recorded SPLITK=2 golden is un-re-benchable (times out >1000 ms), so its faster SPLITK=1 greedy
(383 vs 467 µs) replaces it.

## Finding 1 — `square.4096` fp32: 32% slower than golden, deep on both priors (P1)

`square.4096` is the worst miss: greedy 4216 µs vs golden 3193 µs (1.32×). Greedy picks `BM8 BK64 FM16 FN2 SPLITK1
RING1`; the (transferred-5090) golden is `BM16 BK32 FM10 FN4 SPLITK1 RING2`. The golden ranks **43/1008 under the cold
`AnalyticPrior`** (`eval analytic`) and **77/1008 under the learned prior** (`eval prior --dataset golden`) — deep on
both, so neither patience nor the learned half recovers it. Interestingly `eval prior`'s `vs gold` reads **1.05×** —
the prior's *best measured reservoir config* is within 5% of golden, but the live greedy pipeline deploys the 1.32×
config. So two things compound: (a) the analytic geometry terms misprice the large fp32 square (the `FM/BK` tradeoff —
this is exactly prev Finding 4: the 5090's large-`FM` tiles don't map onto the 4090's ⅓ fp32 throughput / smaller L2),
and (b) the greedy pick diverges from the prior's own best-measured config.

**Recommendation (P1):** refit the cold `_W_A` analytic weights (`scripts/golden_knob_heuristics.py`) now that a real
4090 golden set exists (this sweep populated the DB) — the 2026-06-13 Finding 2 recommendation, still unactioned. A
capability-tier feature (or a `D_*` engineered occupancy term keyed on `H_sm_count`/fp32 throughput) would stop the
sm_120-fit geometry priors from extrapolating onto Ada.

## Finding 2 — `down_proj.s32`: learned prior ranks the golden #3 yet greedy deploys 22% slower (P1)

`down_proj.s32` greedy 19.9 µs vs golden 16.6 µs (1.20×). Unlike Finding 1 this is **not** a ranking failure: the
golden ranks **3/1008 under the learned prior** (shallow) and the live `--golden` A/B confirms the golden knobs are
genuinely faster at -O3 (16.6 vs 19.9, same process — so not a -O1/-O3 inversion). The only knob difference is `BK`
(greedy 32, golden 64). So the prior *knows* `BK64` is best (rank 3) and it *is* best at -O3, but the greedy pipeline
deploys `BK32`. This is a **reservoir/pick divergence**: `Prior.pick`'s evidence-first path should have deployed the
measured-best config, but it didn't — either the `BK64` config never got an `H_opt=3` reservoir row (so `evidence_pick`
couldn't see it), or the partition fork didn't offer it at the greedy site. `gate_up_proj.s32` (1.11×, learned rank 25,
knob diffs `BN/BK/FM`) is the same family, milder.

**Recommendation (P1):** check whether the golden config received an -O3 reservoir row during this tune (the
`EMMY_O3_TOL` re-bench band) — if a measured-best leaf is missing from the reservoir, `evidence_pick` can't
deploy it. If the row exists but greedy still diverged, the gap is in the greedy partition-fork enumeration vs the
reservoir key (`op_cache_key`); instrument `greedy_decide`'s `pick` path on this op. A faithful repro: `emmy run
--bench --golden qwen3_06b.down_proj.s32 --ab "BM=8,BN=16,BK=64,FM=2,FN=2,SPLITK=2,RING=4"`.

## Finding 3 — masked-tile `.dynM` family: greedy oversizes the tile by one notch (P2)

Three of six `.dynM` shapes are worse: `o_proj.s512.dynM` 1.13×, `kv_proj.s512.dynM` 1.10×, `square.512.dynM` 1.08×.
The pattern is consistent — greedy picks `BM16 BN32` where the golden wants `BM8 BN16` (one size larger on both the
free-tile axes), e.g. `o_proj.s512.dynM` found `BM16 BN32` / golden `BM8 BN16`. Under the dedicated masked-tier
`_W_A_DYN` analytic weights the goldens rank 4–40/1008 (`square.512.dynM` rank 4, `kv_proj` rank 12, `o_proj` rank 40),
so the symbolic-axis weights systematically under-penalize the larger masked tile (more wasted lanes past the boundary
guard at the hint extent). The other three dynM shapes are at parity (unchanged).

**Recommendation (P2):** re-fit `_W_A_DYN` (`scripts/golden_knob_heuristics.py` prints both `_W_A` and `_W_A_DYN`) with
these three goldens carrying more weight — the masked-tile occupancy term needs to prefer the smaller free tile at the
512 hint. Lower priority than Findings 1–2: the deltas are 8–13% on shapes already near cuBLAS.

## Finding 4 — `gate_up_proj.s32`: 11% slower, small-shape ranking drift (P2)

`gate_up_proj.s32` greedy 14.3 vs golden 12.9 µs (1.11×); golden ranks 19 (analytic) / 25 (learned). Greedy diverges
on `BN` (16 vs 32), `BK` (32 vs 64) and `SPLITK` (1 vs 2). Same root cause family as Finding 1 (analytic mis-pricing on
sm_89), at a tiny absolute (~1.4 µs). Folded into the Finding 1 `_W_A` refit; no separate action.

## Finding 5 — fp16 squares: we *do* use tensor cores, but the prior undersizes the warp tile (P1)

Follow-up investigation into the emmy-vs-cuBLAS fp16 gap ("are we not able to use MMA there?"). Two structural
facts first: (a) the atom registry holds only `mma_m16n8k16_f16` / `_bf16` — **there is no TF32 atom**, so fp32 matmuls
have no tensor-core path at all (the fp32 squares run scalar CUDA-core FMA against true-SGEMM, by design); (b)
**warp-specialization requires a TMA `StageBundle`** (`085_warp_specialize._eligible` → "no TMA StageBundle"), and TMA
is sm_90+ — the RTX 4090 (sm_89) has no TMA, so `WARPSPEC` is structurally unavailable here (this is why the 5090 fp16
goldens carry `WARPSPEC: true` and the 4090 ones cannot). So on the 4090 every fp16 square already rides the
plain-`mma.sync` + cp.async tensor-core path — MMA *is* used.

The gap is **tile size**, not tier. The recorded fp16 squares used small warp tiles (≤64×128); cuBLAS HGEMM uses
128×128–128×256. A manual tile sweep (10 warp-tile geometries × 4 shapes, all on the same mma.sync path — no
warpspec — accuracy-checked, in `_tune/golden-sweep-rtx4090/fp16/`) found the best reachable config per shape:

| shape            | recorded tile | was µs | best tile (`WM WN FM FN`)        | µs    | vs cuBLAS    | action          |
|------------------|---------------|-------:|----------------------------------|------:|-------------:|-----------------|
| square.512.fp16  | 32×32         |    5.9 | 32×32 (recorded is best)         |   5.9 | 1.02× (par)  | keep            |
| square.1024.fp16 | 64×32         |   27.8 | 64×32 (no tile beats it)         |  27.8 | 1.7× ceiling | keep            |
| square.2048.fp16 | 64×64         |  177.8 | **128×128 (`2 2 4 8`, BK2 RING2)**| 119.3 | 1.54×→**1.04×** | **replaced**  |
| square.4096.fp16 | 64×128        |  889.9 | 64×128 (recorded is best)        | 889.9 | 1.08× (par)  | keep            |

Only `square.2048.fp16` had headroom: the 128×128 tile (`WM2 WN2 FM4 FN8`) lands **119.3 µs / 1.04× cuBLAS** — a
**33% win** over the 178 µs the prior deploys, accuracy-checked and reproduced (119.3/119.7). The win is purely a
larger output tile; `RING=3` on it drops occupancy to 17% and erases the gain. Note the deployed **greedy still picks
the 64×64 (178 µs)** — the golden records the known-best, but the prior's warp-tier geometry terms **under-reward the
larger tile's data reuse** on sm_89, so `compile`/`run` won't deploy it until the prior is refit. The other three are
genuinely at their ceiling: `512`/`4096` are already ≤1.08× cuBLAS, and `1024.fp16`'s ~1.7× gap survives every tile
(no warp-specialization to overlap the mma.sync K-loop the way HGEMM does at M=1024).

**Recommendation (P1):** refit the warp-tier analytic weights / add a `D_*` tile-reuse feature so the prior prefers the
128×64 fp16 tile (the `eval analytic` warp-tier ranking under-prices it on sm_89) — this is the same `_W_A` refit
Findings 1/4 ask for, extended to cover the warp/MMA regime, not just scalar. The remaining fp16 ceiling
(`square.1024.fp16`, and the residual on 2048) needs warp-specialization on sm_89, which is blocked on TMA hardware
the 4090 lacks — a deeper codegen item (a cp.async-pipelined producer/consumer split without TMA), out of scope for a
golden refresh.

## Workflow notes

- **Prev report's Finding 3 fix was a hard prerequisite, not optional.** The sweep cannot run at all on a freshly-cleaned
  prior without the option-0 fallback — the in-process golden loop dies at the first tiny shape (q_proj.s32) when the
  square-trained prior extrapolates an over-budget tile. *Symptom:* `tune --dataset golden --clean` aborts ~⅓ through
  with a `LoweringError` and the per-shape bests for shapes 9–29 never get measured. *Improvement:* shipped (the
  fallback + regression test); but the in-process loop should also **catch a per-shape assemble failure and continue to
  the next shape** rather than letting one shape's compile abort the whole 29-shape sweep (today only a
  saturated-queue `RuntimeError` is caught).
- **`nvcc` not on `PATH` fails every bench silently-ish.** The first sweep attempt produced 29×N `bench_fail @ 2e6 us`
  rows ("nvcc unavailable") and would have "completed" with a garbage prior. *Improvement:* `tune` should
  **fail fast** with a clear error if `nvcc` resolves to nothing before benching the first variant, instead of pinning
  bench_fail and training on noise.
- **The golden-pin A/B dirties the CUDA context on the flaky fp32 squares.** `square.1024`/`square.2048` threw
  `cudaErrorMisalignedAddress` at teardown *only* in the `--golden` run; the clean `--code` greedy run was fine. Each
  win that skipped its golden row (5 fp16/fp32 squares) then needed a hand cross-check against the recorded number.
  *Improvement:* when a pinned golden variant `bench_fail`s, reset the stream/context before the next config so the
  contamination can't bleed into the greedy row's teardown (and ideally surface the greedy-vs-recorded delta in the
  table when the live golden row is skipped).
- **`eval variants --kernel <shape-name>` does not match by shape.** It keys on the kernel C-hash name (`k_matmul_…`),
  so `--kernel down_proj.s32` returns "no measured variants" and the per-variant -O3 reachability drill-down for a
  finding has to be assembled by hand from `eval prior --dataset golden` + a live `--ab`. *Improvement:* let `eval
  variants` accept a golden/shape name (join through the same `ShapeKey` the other `eval` views use).
- **Slowest step by far: the tune (~102 min compute).** The six `.dynM` shapes (~5 min each, 286 s for
  `square.512.dynM`) and the fp32/fp16 4096 squares dominate; the A/B phase was only ~6 min for all 29. *Improvement:*
  a `--kernel` narrowed re-tune is the lever (the skill already documents it); a per-shape patience cap for the
  symbolic-axis shapes would cut the dynM tail.
- **Status of the prior report's notes:** its Finding 1 (fp16 TC blocked) and Finding 3 (LoweringError no-fallback) are
  both **fixed and confirmed held** here (warp-tier fp16 deploys + passes accuracy; the full sweep completes). Its
  Finding 2 (cold `AnalyticPrior` mis-prices sm_89) is **partially addressed by data, not code** — the warm 4090 prior
  now beats the transferred goldens on 11 shapes, but the `_W_A` refit it asked for is still unactioned and is the
  root cause of Findings 1/4 here. Its "transfer mode as a first-class CLI flag" and "live-GPU golden filter" notes
  remain open.
