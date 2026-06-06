# Golden-knob heuristics — surfacing good matmul configs within patience

## Problem

The two-level autotuner's inner search explores a post-fusion kernel's knob space with an MCTS that stops on
**patience** (N consecutive measured terminals with no new best). Whatever *ordering* seeds that search decides which
leaves get visited first. For thread-tier fp32 matmul that seed is `_priority_matmul_thread`
(`compiler/pipeline/passes/lowering/tile/_enumeration.py`), and it ranks the recorded `GOLDEN_CONFIGS` catastrophically
— so a fixed patience budget never reaches them.

Measured baseline (reconstructing the exact enumeration each shape produces, sm_120 / RTX 5090):

| metric                  | `_priority_matmul_thread` |
| ----------------------- | ------------------------- |
| goldens in top-200      | **0 / 19**                |
| median golden rank      | **95 214**                |
| top picks               | degenerate `BN=1, BM=256` |

The warp tier (fp16/bf16 MMA) is **not** the problem — `_priority_matmul_warp` was retuned for the 5090 in 2026 and
already ranks goldens at ~7 (except `square.512.fp16` at 253, a tiny shape with many perf-equivalent tiles).

## Approach — offline learning-to-rank

`scripts/golden_knob_heuristics.py`:

1. **Dataset** — for each fp32 golden matmul, reconstruct the *exact* `enumerate_cartesian` candidate list and locate
   the golden row in it (≈100k–131k candidates/shape, ~2.3M total). Cached to a pickle so iterating is cheap.
2. **Features** — per candidate row + shape: geometry (threads, tile_m/tile_n, cells, BK, SPLITK, overhang), and
   crucially **shape-relative** terms (CTA count vs SM count, occupancy "waves", operand reuse).
3. **Search** — z-score features, random-search signed linear weights + coordinate-descent refine, maximizing golden
   top-k coverage (objective = mean `log2(rank+1)`).
4. **Exploration model** — softmax-sample leaves at a temperature; report P(golden hit within a patience budget).

## Findings

- A single **linear heuristic over geometry features alone** lifts the median golden rank from 95 214 → ~34 and gets
  11/19 within top-50, but **plateaus** — 8 large-tile shapes (the `s512` group, big squares) stay at rank ~600 even
  though the gate below shrinks the pool to ~1 000. One linear function cannot simultaneously prefer *small* cells
  (tiny `M=32` projections) and *large* cells (`4096²`).
- **Shape-relative occupancy is the missing lever.** Adding `near_waves` (keep #CTAs ≈ ~2 waves over the GPU's SMs)
  earned the largest weight and rescued the large-tile shapes: `square.2048` 13 118 → 63, `square.4096` 46 482 → 50,
  the whole `s512` group → ranks 5–100. Final: **median rank 8, 14/19 within top-50, 18/19 within top-200** (only
  `gate_up_proj.s32` remains hard — a small shape where many configs are near-equivalent).
- **Randomness helps only *after* the heuristic is good.** Softmax exploration over the full 100k-candidate pool with
  the weak heuristic *hurt* (hit-rate 0.58 → 0.02 as temperature rose — it dilutes the golden). With the strong
  heuristic, a little temperature *helped* (0.58 greedy → 0.65 at temp 0.5). The real lever for the tuner is a good
  ordering first; exploration is a secondary win.
- **A hard gate prunes the pool 125×** (2.28M → 19k, ~1 000/shape) and every golden survives it
  (`16≤BN≤64`, `8≤BM≤16`, `BN≥BM`, `BK≥32`, `SPLITK≤2`, power-of-two threads, `tile_n∈{32,64,128}`) — but gating alone
  doesn't fix ranking; you still need the occupancy-aware order *within* the gate.

## Deliverable

`compiler/pipeline/search/heuristic.py` — `score_matmul_thread(row, M, N, K, sm_count)` and
`evaluate_golden(...) → (pick, rank, pool)`. The weights are the validated linear model, folded onto raw features so
the score is a plain dot product (ranking is invariant to the dropped z-score constant). It reproduces the
median-8 / 14-of-19-top-50 / 18-of-19-top-200 result with **no prior, no GPU, no measurements**. Regenerate the
weights with `scripts/golden_knob_heuristics.py` (writes `/tmp/golden_heuristic_weights.json`).

Surfaced via `deplodock eval heuristic`: per-config golden **rank** in the heuristic order + a median/top-k summary.

## Empirical check: does a full `tune` make greedy-with-prior reproduce golden? — No (and why)

Tuned every golden shape into an isolated prior (`scripts/tune_golden_set.py`, patience 50, the same default
`find_golden_configs.py` used), then asked whether the greedy single-shot pick from that trained prior lands on the
golden. It does **not** — `eval golden` greedy reproduced **0/23** goldens; the picks are systematically
`SPLITK=32, BK=1/8, FM=1, BN=8/BM=32` (or `BN=1/BM=256` for big squares). Root cause, from the prior's training set:

- **The golden config was never benched.** For every shape checked, 0 of 46,752 training rows match the golden's tile
  geometry (even loosely on BN/BM/FM/FN/BK/SPLITK). The search used the golden's SPLITK+BK *strategy* sometimes, but
  never with the golden's *tile shape* — so the golden region was never visited under patience 50.
- **The prior is trained ~99.9% on -O1 latencies** (`H_opt`: 49 572 rows at -O1 vs 64 at -O3). `tune` ranks at
  `-Xcicc -O1`, where high-SPLITK / low-BK / FM=1 configs are artificially fast (the documented -O1↔-O3 inversion). The
  prior learns to love those; greedy (prior argmax) reproduces them; they're slow at -O3. The golden (SPLITK=1, BK=64,
  fatter tiles) is -O3-optimal but -O1-mediocre.
- **It's a feedback loop.** The inner search is PUCT over that same prior, so the -O1 bias steers exploration away from
  the golden region; cold start is the bad `_priority_matmul_thread` order (`BN=1/BM=256`). Golden never gets benched →
  the prior never gets evidence it's good → it keeps preferring the -O1 winners.

So the learned prior, trained on -O1 data, ranks golden poorly — while the hardcoded heuristic (no -O1 measurements to
mislead it) ranks golden at median 7. That is precisely the value of seeding/pruning the search with the heuristic.

Tooling: `deplodock eval <knobs|heuristic|prior|golden>` (`eval heuristic` = heuristic rank, `eval prior` = prior rank
+ greedy pipeline pick, `eval golden` = both, `--features` dumps the regressor input); `deplodock tune --golden NAME
[--clean]` tunes one golden shape at a time so the prior builds up incrementally. `scripts/tune_golden_set.py` runs the
whole tune→greedy-bench→eval check on an isolated prior/DB; `scripts/golden_knob_heuristics.py` fits the heuristic.

## Status & what's left

Landed: the heuristic IS now the matmul-thread **enumeration order** (`enumerate_cartesian`), so option-0 / cold-prior
greedy picks golden-shaped configs; a hard **gate** prunes the matmul enumeration to the golden-plausible band (every
golden passes; ungated fallback if it empties); with the heuristic order + gate + `tune --nvcc-flags "-Xcicc -O3"`, tune
benches golden and finds golden-or-better. Engineered `D_*` shape×knob features (occupancy / CTA-count / reuse) were
added to `knob_features` to give the learned prior the heuristic's signal.

**Open — greedy selection.** Even with the `D_*` features and clean `-O3` data, the learned `CatBoostPrior`'s
`mean_score` **argmax over the (gated) enumeration still does not select golden** (it extrapolates to unbenched configs;
golden ranks ~67 multi-shape `-O1` / ~840 single-shape `-O3`). Feature engineering improved *leaf calibration*
(+0.91 Spearman) but not the argmax. Per PR #205 review (`greedy.py:31`): make greedy **caller-configurable to use
either the prior or a hardcoded heuristic** (the heuristic could implement the `Prior` interface), and for a *tuned* op
prefer the measured best (the `lowering` best-child) over the prior's extrapolation. Also: pass the **live SM count**
into the heuristic instead of the hardcoded `DEFAULT_SM_COUNT = 170`; extend the offline fit to the warp small-shape
miss (`square.512.fp16`).
