# Golden sweep findings — RTX 5090 (sm_120), 2026-07-02 (sixth sweep; first on the rebuilt tile IR)

- **Branch under test:** `feature/tile-ir-cleanup-and-debt` — the post-rebuild cleanup tree: recognize-time
  contraction nodify, scalar gmem→smem ring (`d2+`), staged split-K, the `ReduceStage.combine` fold-move selector,
  and the tie-pessimistic `_W_A`/`_W_A_DYN` refit. Deploy picks are **analytic-only** (the learned checkpoint was
  deleted mid-sweep — finding 2).
- **Sweep:** `emmy tune --dataset golden --clean`, 29 matmul shapes, ~2.5 h wall across three attempts (two
  environment kills + one crash, finding 5; restarts resume warm without `--clean`).
- **A/B:** `emmy run --bench --golden NAME` per shape, run TWICE — pass 1 exposed the deploy-tie disaster
  (finding 1), pass 2 ran after the fix. All emmy/golden numbers below are pass-2 -O3 live A/B rows; tune-DB
  latencies (-O1) appear only as ranking context.
- **Tally (pass 2):** 4 replaced / 2 added / 5 same / 14 worse / 4 at-parity-via-split / 5 no-A/B-path.

## Per-shape outcomes (pass 2, -O3 live A/B)

| shape | greedy µs | best-golden µs | ratio | eager/cuBLAS µs | vs cuBLAS | category |
|---|---|---|---|---|---|---|
| square.512 | 9.4 | 9.4 | 1.00 | 12 | 0.78 | same → parity knobs added |
| square.1024 | 56.6 | 43.3 | 1.31 | 45 | 1.26 | worse |
| square.2048 | 374.6 | 258.7 | 1.45 | 260 | 1.44 | worse (8.2 µs golden row discarded — finding 4) |
| square.4096 | 2713.5 | 2132.3 | 1.27 | 2123 | 1.28 | worse |
| square.512.fp16 | 5.4 | 8.9 | 0.61 | 6 | 0.90 | **replaced** |
| square.1024.fp16 | 21.9 | 21.4 | 1.02 | 14 | 1.56 | same → added; pruned the 2.7×-slower wspec entry |
| square.2048.fp16 | 132.6 | 133.5 | 0.99 | 97 | 1.37 | same |
| square.4096.fp16 | 744.7 | 919.0 | 0.81 | 651 | 1.15 | **replaced** (both wspec-era entries pruned) |
| qwen3_06b.q_proj.s32 | 6.2 | ~6.5 | ~0.95 | 8 | 0.78 | parity (split deploy — golden rows dropped, workflow) |
| qwen3_06b.kv_proj.s32 | 5.0 | ~5.9 | ~0.85 | 6 | 0.83 | parity (split deploy) |
| qwen3_06b.o_proj.s32 | 8.6 | ~9.1 | ~0.95 | 8 | 1.07 | parity (split deploy) |
| qwen3_06b.gate_up_proj.s32 | 11.8 | 10.4 | 1.13 | 12 | 0.98 | worse |
| qwen3_06b.down_proj.s32 | 13.1 | ~13.5 | ~0.97 | 10 | 1.31 | parity (split deploy) |
| qwen3_06b.q_proj.s128 | 17.3 | 17.0 | 1.02 | 16 | 1.08 | same |
| qwen3_06b.kv_proj.s128 | 11.9 | 9.7 | 1.23 | 10 | 1.19 | worse |
| qwen3_06b.o_proj.s128 | 22.5 | 18.2 | 1.24 | 16 | 1.41 | worse |
| qwen3_06b.gate_up_proj.s128 | 23.4 | 21.6 | 1.08 | 33 | 0.71 | worse |
| qwen3_06b.down_proj.s128 | 33.2 | 23.9 | 1.39 | 84 | 0.40 | worse |
| qwen3_06b.q_proj.s512 | 56.8 | 44.0 | 1.29 | 45 | 1.26 | worse (finding 3) |
| qwen3_06b.kv_proj.s512 | 32.8 | 25.1 | 1.31 | 34 | 0.96 | worse (finding 3) |
| qwen3_06b.o_proj.s512 | 64.2 | 45.8 | 1.40 | 51 | 1.26 | worse (finding 3) |
| qwen3_06b.gate_up_proj.s512 | 74.3 | 57.8 | 1.29 | 70 | 1.06 | worse (finding 3) |
| qwen3_06b.down_proj.s512 | 97.0 | 65.0 | 1.49 | 68 | 1.43 | worse (finding 3) |
| square.512.dynM | 8.2 | 9.3 | 0.88 | 12 | 0.68 | **replaced** (3 clean reproductions) |
| qwen3_06b.q_proj.s512.dynM | 47.1 | 46.9 | 1.00 | 45 | 1.05 | same |
| qwen3_06b.kv_proj.s512.dynM | 24.5 | 25.5 | 0.96 | 33 | 0.74 | same (within noise) |
| qwen3_06b.o_proj.s512.dynM | 47.7 | 44.8 | 1.06 | 51 | 0.94 | worse (marginal) |
| qwen3_06b.gate_up_proj.s512.dynM | 86.0 | 62.2 | 1.38 | 69 | 1.25 | worse |
| qwen3_06b.down_proj.s512.dynM | 70.9 | 63.7 | 1.11 | 67 | 1.06 | worse |
| reduce.* / pointwise.* (5) | — | — | — | — | — | no `--golden` A/B path (finding 6) |

## Finding 1 — tie-optimistic fitter ranks deployed the per-cell kernel on every static shape (FIXED)

Pass 1 had five catastrophic shapes (5.5–14.7× the golden: square.512, q_proj.s128, o/gate_up/down_proj.s512):
every static greedy pick was the **per-cell** row (`eval prior`'s golden-reproduction table showed found knobs `-`
on 29/29) while the `.dynM` twins picked staged tiles. `eval variants k_matmul_207791` put the tune's own pick at
rank 1/185 (`n32x8/f2x4 d2/tma/ring`, 9.4 µs) — the search was fine; the deploy was not. Root cause: the static
`_W_A` refit carried zero `D_stage_*` weights, so staged and gmem-direct rows TIED under the analytic score and
deploy tie-breaks to option-0 (per-cell) — and the fitter's rank metric counted ties optimistically
(`scores > golden`), reporting top-1 ranks for a fit that could not deploy. Fixed this sweep:
`scripts/golden_knob_heuristics.py::rank_of_golden` counts ties against the golden (`>=`; a tie IS a deploy miss),
and the refit carries live stage terms (`D_stage_tma +2.0`, `D_stage_ring +2.0`). Honest static median golden rank
is now 25 (the tie-honest seed measured 53). Pass 2 has no catastrophic class (worst ratio 1.49).

## Finding 2 — the learned prior checkpoint is mis-calibrated; deleted, deploy is analytic-only

`eval prior --dataset golden` after the sweep: median golden rank **186**, top10 **0/27**, with per-shape training
calibration lines like `Spearman pred vs latency: +0.06`. The learned model ranked worse than both the tune's inner
search and the analytic prior on the very shapes it had just trained on, and pass 1's deploy picks (which read it)
were the per-cell disaster above. `prior.json` was deleted, sending greedy deploys to the analytic path.
Recommendation: the trainer needs a dedicated investigation (57 k reservoir rows at near-zero calibration —
feature/label mismatch between the -O1 ranking family and the featurized rows is the prime suspect) before the
learned half owns deploy picks again.

## Finding 3 — the s512 static goldens are outside the enumerable space (reachability)

`eval prior` reports every static `.s512` golden `SKIPPED: recorded knobs not in the enumeration` — the recorded
register tiles (`f2x14`, `f4x8`, `f4x10`, `f4x26`) are not in `search/space._SCALAR_REG` (the pre-rebuild space
swept reg_m far deeper). These are exactly the shapes stuck at 1.29–1.49×: the pick cannot reach the recorded
geometry at any patience. Recommendation: widen `_SCALAR_REG` with the golden-informed deep-FM points (`(2,14)`,
`(4,6)`, `(4,8)`, `(4,10)`), re-sweep the s512 family, and only then re-judge; the same applies to the fp32 squares
(`f2x14`/`f4x26` entries).

## Finding 4 — a physically impossible golden bench row (8.2 µs for a 2048³ matmul)

square.2048's second golden row (the `REDUCE: g2a` atomic-split entry) benched 8.2 µs at grid 16384 — over
2 PFLOP/s, impossible; the plausible sibling row (258.7 µs) was used for categorization. The atomic-split re-bench
appears to skip the zero-init/finalize cost or fail silently. Recommendation: the A/B should sanity-gate golden rows
against an arithmetic-intensity floor and flag sub-floor rows, and the `g2a` re-bench path deserves an output
correctness assert against the greedy row.

## Finding 5 — `Tile.pretty` crashed the sweep at the first `.dynM` shape (FIXED)

The op-inventory persist (`op_cache_key` → `structural_key` → `pretty_body`) raised
`NotImplementedError: Tile: symbolic axis 'a0_b'`: `Tile.pretty` printed `N={n_elements}`, which raises on a
symbolic grid, and the recognize-time nodify now routes symbolic-M contractions through the tiled arm (block axes
carry symbolic extents). Fixed: `pretty` prints the `n_dim` product for a symbolic grid; the dynamic regression
suite stays green. The sweep also survived two environment kills — `setsid` + PID-file + resume-without-`--clean`
is the pattern that held.

## Finding 6 — reduce/pointwise goldens have no tune/A/B path (carried from the fifth sweep)

`tune --dataset golden` builds the 29 matmul shapes only, and `run --bench --golden reduce.2048x2048` fails with
"unknown golden config" (matmul-only registry). Carried unfixed from the fifth sweep's workflow notes.
Recommendation unchanged: extend `_tune_targets` + the `--golden` registry to the reduce/pointwise snippets (they
already carry `snippet()`), or drop the five entries.

## Workflow notes

Against the fifth sweep's notes: the *analytic-verify-first (no GPU)* advice was applied (the fitter diagnosis and
both refits ran CPU-only; the second A/B pass was the only extra GPU cost of finding 1); the *matmul-only
`--golden`* gap is still open (finding 6, now 2×); the *harvest script* gap repeated — this sweep again hand-wrote
a table parser (now 3×; `_tune/golden-sweep-rtx5090/parse_ab.py`, kept in the repo's gitignored `_tune/`).

- **Golden rows vanish when greedy deploys a split fragment** — four s32 shapes deployed a partial+finalize pair and
  the A/B printed no `golden` rows at all; categorization borrowed pass-1 numbers. The golden rows should attach to
  the shape, not the single kernel node.
- **A JSON A/B dump is overdue** — greedy/golden/eager numbers exist only in table text; `--record`/`--json` on the
  A/B would retire the parser and host the finding-4 sanity gate.
- **Pace**: the restored stage × split enumeration roughly doubles per-shape tune time vs the fifth sweep
  (~2.5 h vs ~112 min for the same 29 shapes); consider a coarser stage grid or scaled patience for the golden pass.
- **A/B noise**: the fp16 square golden rows swung 21.4 ↔ 53.0 µs across passes; the confirm-twice rule caught it
  (square.1024.fp16 recorded as parity, not the 2.6× win pass 1 suggested).
