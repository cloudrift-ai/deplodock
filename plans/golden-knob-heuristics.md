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

Surfaced via `deplodock knobs --golden`: per-config golden **rank** in the heuristic order + a median/top-k summary,
alongside the greedy pipeline pick (prior / option-0) vs golden.

## Not yet wired into the planner

The heuristic is currently a **diagnostic / no-prior baseline** only — it is not yet the seed order the live fork-tree
search consumes (that still comes from `_priority_matmul_thread` + the learned `CatBoostPrior`). Candidate next steps:

- Replace `_priority_matmul_thread`'s tiebreak with `score_matmul_thread` (or fold its features into the static prior)
  so a cold-prior `tune` starts near the goldens.
- Pass the **live device SM count** into the heuristic instead of the hardcoded `DEFAULT_SM_COUNT = 170` (the
  occupancy term is SM-count-relative).
- Use the hard gate to prune the enumeration the inner MCTS explores, shrinking the effective search 100×.
- Extend the offline fit to the warp tier's small-shape miss (`square.512.fp16`) if it matters.
