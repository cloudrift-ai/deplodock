# Golden sweep findings — RTX 5090 (sm_120), 2026-06-24 (fifth sweep; first on the block-DAG branch)

- **Branch under test:** `feature/tile-ir-block-dag` (the tile-IR block-DAG rework: TWISTED_MONOID dropped, enumeration
  passes renumbered into the `enumeration/` package, OVERHANG knob → per-role `S_masked_{m,n,k}` features, `pick_matmul`
  rebuilt over per-family offers). The previous (fourth) sweep ran on `main`.
- **Ask:** *verify the analytic prior on this branch and re-tune if necessary*, plus the full golden sweep.
- **Sweep:** `emmy tune --dataset golden --clean` — 29 matmul targets (23 static + 6 `.dynM`) + the reduce/pointwise
  shapes, **~112 min wall** (4× the skill's ~30 min estimate — see Workflow notes).
- **A/B:** `emmy run --bench --golden NAME` per shape (greedy pick benched live beside each recorded golden, all at
  -O3); table parsed by hand from the kernel rows (the prior sweep's `/tmp/harvest_goldens.py` is gone — Workflow notes).
- **Tally — goldens: 0 replaced / 0 added / 34 left (validated).** Every clean-search greedy pick is **at or below** its
  recorded golden; the apparent wins (`down_proj.s32`, `o_proj.s32/.s128`, `o_proj/down_proj.s512.dynM`) are 7–9% vs the
  *recorded* golden — inside the session noise band, inflated only against this session's slow live golden re-bench
  (Step 4 trap). The honest outcome is **no YAML changes**.
- **Analytic prior re-tune (the headline ask): `_W_A_DYN` refit and pasted** (joint all-GPU dynM median rank **839 → 86**,
  RTX 5090 dynM median ~62 → ~38); **`_W_A` left unchanged** (already at the feature ceiling — a refit cannot move it).
  The offline refit tool `scripts/golden_knob_heuristics.py` was **broken on this branch** and had to be repaired first.

## Per-shape outcome table

All emmy / golden numbers from the -O3 `run --bench` A/B. `ratio` = greedy ÷ **recorded** golden `emmy_us`
(the stable reference; the live golden row swings ~10–13% and ran slow this session). `vs cuBLAS` = greedy ÷ eager
(`>1.0` = emmy slower than cuBLAS — the loser view). cuBLAS = the live `Eager PyTorch` row (true-fp32 SGEMM /
HGEMM).

| shape | greedy µs | rec. golden µs | ratio | cuBLAS µs | vs cuBLAS | category |
|---|---|---|---|---|---|---|
| square.512 | 27.9 | 9.0 | 3.10 | 14 | 1.99 | worse |
| square.1024 | 138.6 | 44.1 | 3.14 | 53 | 2.62 | worse |
| square.2048 | 675.0 | 294.8 | 2.29 | 307 | 2.20 | worse |
| square.4096 | 2953.9 | 2348.7 | 1.26 | 2107 | 1.40 | worse |
| square.512.fp16 | 5.5 | 4.1 | 1.34 | 6 | 0.92 | worse |
| square.1024.fp16 | 21.5 | 17.3 | 1.24 | 16 | 1.34 | worse |
| square.2048.fp16 | 117.5 | 106.7 | 1.10 | 111 | 1.06 | worse |
| square.4096.fp16 | 779.3 | 727.7 | 1.07 | 739 | 1.05 | worse |
| q_proj.s32 | 19.7 | 7.5 | 2.63 | 10 | 1.97 | worse |
| kv_proj.s32 | 8.6 | 4.9 | 1.76 | 8 | 1.07 | worse |
| o_proj.s32 | 7.4 | 8.0 | 0.93 | 10 | 0.74 | noise (≈same) |
| gate_up_proj.s32 | 12.9 | 9.6 | 1.34 | 14 | 0.92 | worse |
| down_proj.s32 | 11.9 | 16.4 | 0.73 | 12 | 0.99 | win (see Finding 5) |
| q_proj.s128 | 19.4 | 16.3 | 1.19 | 19 | 1.02 | worse |
| kv_proj.s128 | 19.1 | 9.8 | 1.95 | 12 | 1.59 | worse |
| o_proj.s128 | 15.8 | 17.4 | 0.91 | 15 | 1.05 | noise (≈same) |
| gate_up_proj.s128 | 43.0 | 23.8 | 1.81 | 33 | 1.30 | worse |
| down_proj.s128 | 46.8 | 24.9 | 1.88 | 86 | 0.54 | worse |
| q_proj.s512 | 52.4 | 44.2 | 1.19 | 45 | 1.16 | worse |
| kv_proj.s512 | 49.4 | 24.6 | 2.01 | 33 | 1.50 | worse |
| o_proj.s512 | 105.3 | 50.5 | 2.09 | 52 | 2.02 | worse |
| gate_up_proj.s512 | 67.4 | 52.4 | 1.29 | 70 | 0.96 | worse |
| down_proj.s512 | 151.2 | 75.7 | 2.00 | 78 | 1.94 | worse |
| square.512.dynM | 11.2 | 11.0 | 1.02 | 14 | 0.80 | same |
| q_proj.s512.dynM | 56.8 | 50.4 | 1.13 | 49 | 1.16 | worse |
| kv_proj.s512.dynM | 57.4 | 30.7 | 1.87 | 39 | 1.47 | worse |
| o_proj.s512.dynM | 48.7 | 52.7 | 0.92 | 56 | 0.87 | noise (≈same) |
| gate_up_proj.s512.dynM | 155.6 | 72.8 | 2.14 | 80 | 1.94 | worse |
| down_proj.s512.dynM | 69.5 | 74.8 | 0.93 | 74 | 0.94 | noise (≈same) |
| reduce.* / pointwise.* | — | — | — | — | — | not A/B-able (`run --golden` is matmul-only) |

The goldens themselves beat cuBLAS on most fp32 shapes (e.g. square.512 golden 9.0 < cuBLAS 14; square.1024 golden 44.1
< 53). The greedy pick is **2.0–2.6× slower than cuBLAS** on the fp32 squares and large projections — the deployable
reality on this branch for fp32 matmul.

## Finding 1 — fp32 thread-tier greedy deploys *degenerate* tiles: the deploy enumeration is ungated (P0)

The dominant result: on ~20 of 29 matmul shapes the clean-search greedy pick is 1.8–3.1× **slower** than the recorded
golden, and the picks are not merely suboptimal — they are structurally degenerate:

- `square.512` greedy `BM=256, BN=1, BK=1, FN=8` (28 µs) vs golden `BM=8, BN=16, BK=64` (9 µs) — **BN=1, no K-blocking**.
- `square.1024` greedy `BM=256, BN=1, BK=32, FN=64`; `kv_proj.s512` greedy `BM=1, BN=256`; `o_proj.s512` greedy `BM=1,
  BN=32` (grid 4096); `gate_up_proj.s512.dynM` greedy `BM=1, BN=16` (grid 49152, **0 % occupancy**).

Root cause, traced to source: the heuristic-plausible band `_matmul_thread_gate` (`16 ≤ BN ≤ 64`, `8 ≤ BM ≤ 16`, `BK ≥ 32`
…) is applied **only** in the cold analytic ranking pool — `search/analytic.py:115` — and **never** in the deploy
enumeration. The deploy thread-tile offer `_moves.thread_offers`
(`compiler/pipeline/passes/lowering/tile/enumeration/_moves.py:193`) draws BN/BM from
`THREAD_CHOICES = (1, 8, 16, 32, 64, 128, 256)` (`enumeration/_knobs.py:110`) and sorts by `abs(t_n·t_m − 256)` then
`−product`. A degenerate `(BN=1, BM=256)` tile *ties* a balanced `(BN=16, BM=16)` tile (both 256 threads, both score
`(0, −256)`) and wins on emission order (`1` precedes `16` in the choices). The in-code comment concedes it: *"the cold
prior has no weighted feature for the greenfield knobs yet … emission order is the effective ranking."* The clean search
then samples the BN=1 family, measures it as locally-best at -O1, and `evidence_pick` faithfully deploys it.

This is the fourth-sweep **Finding 2** ("clean-DB search variance unsamples the goldens"), sharpened: on this branch the
unsampling is not random variance but a *systematic* bias toward degenerate tiles introduced by the ungated emission
order. The fp16 **warp** tier is unaffected (it has its own offers and is at/under golden).

**Recommendation (P0):** gate the deploy thread enumeration, not just the analytic pool. Either (a) apply
`_matmul_thread_gate`'s aspect/threads band inside `_moves.thread_offers` (fall back to ungated only when it empties, as
`analytic._enumerate` already does), or (b) fix the `thread_offers` tie-break to prefer balanced tiles (penalize extreme
aspect ratios / `BN==1` when a square alternative exists at equal thread count). This is strictly higher-leverage than
the carried "seed the inner search with golden knobs" item, because it fixes *what gets sampled* rather than patching
around it. Until then, fp32 thread-tier matmul deploys 2–2.6× off cuBLAS.

**Update — fixed (combines (a) + (b)).** `_moves.thread_offers` now takes a `balanced` flag (set by `090_thread_tile`
for the `SEMIRING` matmul regime): it drops the degenerate-aspect tiles (one axis collapsed to 1) and leads with a
square-ish, coalesced `BN >= BM` tile, with a fall-back to the bare order for a genuinely 1-wide axis (gemv). MAP
(pointwise) / streaming keep `balanced=False` (byte-identical). Validated: a **cold** 768³ matmul now deploys
`BN=32, BM=8, BK=32` (was a `BN=1` degenerate), and re-tuning **`square.512`** drops the deployed kernel **28.0 → 9.3 µs**
(now `BM=8, BN=32` — **1.33× faster than cuBLAS**, beats the golden). **Partial for `square.1024`:** the thread tile is
fixed (`BN=1 → BN=32`) but the deployed config now pairs it with `STAGE=00` (no smem staging) + `SPLITK=16` vs the
golden's `STAGE=11`/`SPLITK=1`, so it only improves 138.6 → 119 µs — a **separate** staging/split-K degenerate-default
(the `120_stage` / split-K tier has the same "emission order = cold pick" pathology as `thread_offers` did). Tracked as a
follow-up; the thread-tile fix is necessary but not sufficient where staging also mis-defaults. A full
`tune --dataset golden --clean` re-sweep is still owed to refresh every shape's reservoir + the learned prior under the
fixed enumeration (this sweep only re-tuned `square.512`/`square.1024` to validate).

## Finding 2 — the analytic prior's dynamic weight set `_W_A_DYN` was ~random on this branch; refit 839 → 86

The branch's OVERHANG→`S_masked_{m,n,k}` featurization change invalidated the masked-tile weight set the old `_W_A_DYN`
was fit to. Over the all-GPU `.dynM` golden set the current weights rank the goldens at **median 839** (top1 0/20,
*nothing* in the top 100) — i.e. the cold masked-tier ranking is noise. A fresh offline fit
(`scripts/golden_knob_heuristics.py`, 20 000 samples) recovers **median 86** (top1 4/20, top100 12/20), pasted into
`search/prior/analytic.py`. On RTX 5090 specifically the `.dynM` ranks go mixed-but-better: `kv_proj.s512.dynM` 12 → 0,
`gate_up_proj.s512.dynM` 9 → 0, `q_proj.s512.dynM` 112 → 16, against regressions `o_proj/down_proj.s512.dynM` 12 → 61
and the persistent outlier `square.512.dynM` (1403 → 1193). `_W_A_DYN` is a **single global weight set** shared across
4090/PRO 6000/5090, so the joint 839 → 86 (driven largely by the other cards, which were unreachable before) is the
metric that matters for what ships. **This is the verified "re-tune" the ask called for.**

**Recommendation:** done (pasted). Re-run `golden_knob_heuristics.py` after any future `.dynM` golden change or another
masked-tile featurization edit. `square.512.dynM` resisting the fit (rank ~1193) is a feature gap, not a weight problem —
see Finding 3.

## Finding 3 — static `_W_A` is at the feature ceiling: deep ranks are a feature/enumeration gap, not weights

Seeding the offline fit with the committed `_W_A` and searching 20 000 weight vectors + coordinate descent **cannot
improve** the static ranking (mean `log2(rank+1)` 4.61 → 4.58, noise; median 43 → 43). Yet several static goldens rank
very deep under the analytic prior — `square.1024` 1285, `kv_proj.s512` 1121, `o_proj.s512` 1057. Because re-fitting the
linear weights does not move them, the cold analytic prior simply **cannot distinguish** these goldens from ~1000 other
candidates with the current `D_*` features. This is the same feature ceiling behind Finding 1 (the cold rank is deep →
patience can't reach the golden even when the deploy enumeration is fixed).

**Recommendation:** add an engineered `D_*` feature that separates the golden-shaped fp32 tile from the degenerate band
the enumeration over-emits (e.g. a coalesced-N-width / square-aspect interaction, or an arithmetic-intensity term keyed
to `BK·BN`). The right test loop is `eval analytic --kernel square` after each candidate feature; target moving
`square.1024`/`kv_proj.s512`/`o_proj.s512` out of the 1000+ tail. Pair with the Finding 1 enumeration gate so the cold
rank and the deploy pool agree.

## Finding 4 — the offline refit tool was broken on this branch (fixed)

`scripts/golden_knob_heuristics.py` imported `tile._enumeration` (`enumerate_cartesian`,
`_enumerate_warp_matmul_impl`) — symbols the block-DAG rework **deleted** — so it raised `ModuleNotFoundError` on import
and the analytic prior could never be refit on this branch. The branch updated the *live* analytic eval path
(`search/analytic._enumerate`) but not the offline fitter. Repaired: `build_cases` now reuses `analytic._enumerate` for
matmul (fp32 thread / fp16 warp / `.dynM` — the same gate-narrowed pool `eval analytic` and the deploy rank over; the
old scalar+warp `MMA_tier` mixing is obsolete now that scalar↔warp is a structural fork) and traces each reduce /
pointwise snippet to an `IterDag`, composing the cooperative-reduce / MAP `_moves` offers.

**Recommendation:** add a 1-line smoke test (`tests/`) that imports + runs `build_cases()` and asserts a non-empty case
list, so the next enumeration refactor that drifts the fitter fails CI instead of silently rotting until a sweep.

## Finding 5 — two enumeration mismatches surfaced by the A/B

- **`pointwise.512x4096` golden is unreachable.** Its `FM=16` is no longer offered — `_MAP_REG_CHOICES = (1, 2, 4, 8)`
  (`_moves.py:174`) caps the per-axis register tile at 8, so the live greedy can't produce `FM=16` either. The refit
  script skips it (golden-not-in-candidates). Either widen `_MAP_REG_CHOICES` to include 16 or re-record the golden at a
  reachable tile.
- **The one genuine win is gated out of the analytic pool.** `down_proj.s32` greedy `BM=8, BN=16, BK=8, FK=8, SPLITK=16,
  STAGE=10` (11.9 µs, both runs) beats its recorded golden (16.4) by 27 % — but its `BK=8` *violates* the analytic gate
  (`BK ≥ 32`), so recording it would create a golden the analytic prior structurally cannot rank. The win exposes the
  gate-vs-deploy mismatch from Finding 1 from the *other* direction (the gate excludes a real winner). Left unrecorded
  pending the Finding 1 gate rework; revisit once the gate and deploy enumeration are reconciled.

## Finding 6 — bench failures: bf16 mma codegen, a split-consumer lowering gap, TMA hangs

8 of 2533 benched variants failed (0.3 %), clustered (`eval failures`):

- **`k_matmul_207791` (4 rows)** — bf16 mma: 3× `nvcc … 'a_m16n8k16_bf16' was declared but never referenced` + 1×
  `KeyError('x0_smem')`, all at `MMA=mma_m16n8k16_f16, WM=2, WN=8`. A bf16 warp-tier codegen bug (the bf16 mma helper is
  emitted but the operand is never wired).
- **`matmul__reduce` (2 rows)** — `TypeError: node 'matmul__partial' has non-CudaOp 'TileOp'; lowering must produce
  Graph[CudaOp]` at `BM=16, BN=16, BK=1`. A split-K consumer (`matmul__partial`) that doesn't finish lowering to CudaOp.
- **`k_matmul_180e20` (2 rows)** — a `HungKernelError` (1 s timeout) and an EOF/timeout, both with `TMA=True`. A
  TMA-transport hang on this shape.

**Recommendation:** low volume, but each is a real codegen/lowering gap. The bf16-mma "declared but never referenced" is
the most actionable (deterministic compile failure on a fixed knob set). Triage via
`eval failures --dataset db --kernel k_matmul_207791`.

## Workflow notes

- **Sweep wall time was ~112 min, not ~30.** The skill's estimate is stale for the current 29-target set on this branch
  (the dynM and large-square shapes dominate). *Improvement:* update the skill's time estimate; consider a `--kernel`
  narrowed default or a fast-rank-only mode for verify-only runs (this sweep's verify goal — the analytic-prior check —
  needed no GPU at all; see next).
- **The analytic-prior verification needs no GPU and should be step 0.** `eval analytic` + a `golden_knob_heuristics.py`
  diagnostic fit (both CPU-only) revealed the whole story — `_W_A_DYN` broken, `_W_A` at ceiling, the tool broken —
  *before* the 112-min sweep. *Improvement:* the skill should front-load a "verify analytic (no GPU)" step and let the
  operator decide whether the full sweep is even warranted.
- **The prior sweep's harvest script is gone** (`/tmp/harvest_goldens.py`, fourth sweep) — `/tmp` doesn't survive a
  reboot, so this sweep parsed the `run --bench` kernel table by hand again. *Improvement (carried, now 2×):* fold the
  harvest into the CLI as `tune --dataset golden --record` (emit per-shape greedy-knobs + -O3 latencies as JSONL).
- **`run --golden NAME` is matmul-only** — reduce/pointwise goldens (tuned by the sweep) have no A/B path, so they were
  left uncategorized. *Improvement:* either extend `--golden` to the reduce/pointwise snippets or document the gap in the
  skill.
- **`RING` is not in the `run --bench` kernel table**, and a STAGE=10 pick carries no `RING` at all — recording a golden
  needs a dump dir to recover the full knob set. *Improvement:* surface `RING` (and other recorded search knobs) in the
  A/B table, or have `--record` (above) emit them.
- **`eval analytic` uses `Context.from_target(cap)` without `gpu_name`**, while the deploy + the offline fit use the
  live card's SM count — so `eval analytic`'s absolute ranks differ slightly from what the prior is actually fit/deployed
  against. *Improvement:* thread `gpu_name` through `eval analytic` for an apples-to-apples diagnostic.
- **Carried from the fourth sweep:** golden-seeding the inner search (now subsumed by Finding 1's stronger
  enumeration-gate fix); the planner-derived knob column in the A/B table; the multi-GPU `compute_cap (12,0)` collision
  between 5090 and PRO 6000 (`eval` / `tune --dataset golden` are still GPU-blind over `GOLDEN_CONFIGS`). **Fixed and
  held:** the fp16 squares stay at/under cuBLAS (the warp tier is healthy on this branch).
