# Analytic-prior per-card refit — findings (2026-06-13)

Refit the cold-start `AnalyticPrior` weights (`_W_A` / `_W_A_DYN`) over the multi-GPU golden set, making the offline
fit **per-card-aware**, and validate on three machines: RTX 5090 (local, sm_120/170 SM), RTX 4090
(`riftuser@176.124.69.200`, sm_89/128 SM), RTX PRO 6000 (`riftuser@38.108.83.78 -p 60002`, sm_120/188 SM).

**Not committed.** Two files left dirty in the working tree:
`scripts/golden_knob_heuristics.py` (the per-card fix) and `deplodock/compiler/pipeline/search/prior/analytic.py`
(the refit weights). The `_data/{5090,4090,pro6000}/` caches (each box's `autotune.db` + learned `prior.json`) were used
only as measured ground-truth context, per the design split: the **learned** `CatBoostPrior` consumes measured rows; the
**analytic** prior is a fixed golden-ranker, so its fit reads goldens only.

## What changed in the fit

`scripts/golden_knob_heuristics.py` previously reconstructed **every** golden under one global
`Context.from_target((cap, 0))` (default sm_120, no `gpu_name`) — so 4090 goldens were featurized as sm_120 and all
cards shared one SM count. Commit `46e3683a` had already made the *deployed* featurization per-card
(`Context.from_target(cap, gpu_name=…)`, `Sample.from_golden`), but the fit script was left behind, so the fit ranked
against a regime the eval no longer used. The fix builds each golden's context from its **own** `(compute_cap,
gpu_name)` (4090=128, 5090=170, PRO6000=188 SMs; 4090 at sm_89), mirroring `Sample.from_golden`. `--cap` removed (now
intrinsic).

## Result — neutral across all three cards

Methodology: `eval analytic --dataset golden` scores the AnalyticPrior weights directly (the real before/after). Cold
`eval golden` (learned prior disabled via `DEPLODOCK_PRIOR_FILE=<nonexistent>`) shows the analytic-driven greedy knob
pick — with a trained `prior.json` present the greedy pick is learned-prior-dominated and would not move. Both are
per-live-card (live-GPU golden filter), CPU-only.

| Card    | analytic median B→A | top10 B→A | top25 B→A | cold `eval golden` TOTAL B→A |
|---------|--------------------|-----------|-----------|------------------------------|
| 5090    | 12 → 15            | 18 → 18   | 28 → 31   | 7/29 → 7/29                  |
| 4090    | 34 → 32            | 11 → 11   | 18 → 18   | 2/29 → 2/29                  |
| PRO6000 | 16 → 17            | 17 → 17   | 28 → 28   | 4/29 → 4/29                  |

The weights take effect (numbers move) but only marginally, and net-neutral: 5090 median slightly worse (12→15, top25
better 28→31), 4090 slightly better (34→32), PRO6000 flat. No cold-golden knob-reproduction count changed on any card.

The fit log's headline "dynamic mean_log2 7.54 → 3.83" is **relative to a bad seed** (the freshly-fit *static* weights
applied to the dynamic cases — how the script seeds its dynamic fit), not relative to the committed `_W_A_DYN`. Against
the deployed weights it nets to neutral, as the per-card eval confirms.

## Why neutral — the basis, not the weights

- The static objective is **saturated**: seed sweep (samples 50k, seeds 1/2/3/7) all land at `mean_log2=3.48`
  (committed seed was 3.50). seed=0 has the best dynamic fit (3.83 vs 3.97/4.12/4.18). The deployed weights are the best
  found — no restart beats them.
- The 4090 stays the worst-ranked card (median ~32) under any linear weighting: its golden configs sit deep in their
  candidate enumerations under the current `D_*` feature basis (e.g. large `square.*` matmuls rank ~300–540). This is a
  **feature-basis** limit the per-card SM correction can't move, not a weight-tuning shortfall.
- The committed weights were effectively a 5090-regime fit; per-card-correcting the fit fixes the *featurization* but
  the linear model has no headroom left to exploit it.

## Recommendation

- **Keep** the `golden_knob_heuristics.py` per-card fix — it is a correctness improvement (the fit now ranks against the
  same per-card regime the deploy featurizes), independent of whether the weights move.
- The refit **weights** are a wash; safe to keep or revert. Reverting `_W_A`/`_W_A_DYN` to committed avoids the minor
  5090 median regression (12→15) at no cost elsewhere. Left deployed locally as the literal tuning output pending a
  decision.
- Real gains need a **richer feature basis** (or per-card weight sets — the prior carries one shared `_W_A`, selected
  only on the symbolic-axis flag, not on the card) so deep-ranked large-matmul goldens can be pulled toward the top —
  out of scope here.

## Reproduce

```
./venv/bin/python scripts/golden_knob_heuristics.py            # refit (seed 0 = best); prints _W_A / _W_A_DYN
./venv/bin/deplodock eval analytic --dataset golden            # per-live-card golden rank
DEPLODOCK_PRIOR_FILE=/tmp/nope.json ./venv/bin/deplodock eval golden --dataset golden   # cold analytic-driven greedy
```
