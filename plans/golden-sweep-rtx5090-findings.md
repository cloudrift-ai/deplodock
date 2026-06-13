# Golden sweep findings — RTX 5090 (sm_120), 2026-06-12 evening (third sweep; first after the evidence-pick prior fix)

- Sweep: `deplodock tune --dataset golden --clean` — 29 matmul targets (23 static + 6 dynamic `.dynM`), 52 min wall
  (15:40–16:32). 24 bench_fail rows total, all in fp32 `MMA=0, FK-split` classes on the big squares (slow compile /
  slow run) — the fp16 `TMA+WARPSPEC` launch-crash class from the previous two sweeps is **gone** (its goldens bench
  fine live now).
- A/B: `deplodock run --bench --golden NAME` per shape — and this time the full 29-shape pass took **~3.5 min**, not
  ~45: every kernel hit the cubin cache the tune had just filled (~7 s per cold process launch). Two full passes plus
  a 5-shape tie-breaker pass; greedy/golden ratios were stable to ≤2% across passes, and absolute µs barely drifted
  (back-to-back runs, no thermal swing).
- Branch under test: `main` @ `ad08f813` (the dynM end-to-end PR, which includes the `FallbackPrior.evidence_pick`
  measured-evidence fix the previous sweep's findings 2/4/5/6 asked for). Edits on `feature/golden-sweep-rtx5090-3`.
- Tally: **8 replaced / 5 added / 5 exact reproductions / 11 worse** (the 5 reduce/pointwise goldens remain
  unsweepable, as in both previous sweeps).
- The analytic weights were refit after the YAML edits (`scripts/golden_knob_heuristics.py` → `_W_A` + `_W_A_DYN`
  pasted into `search/prior/analytic.py`): the pre-refit `_W_A_DYN` ranked the *new* dynM goldens at median 355 (it
  was fit to the seeds this sweep replaced); post-refit five of eight dynM rows rank ≤1, dynM median ~6, static
  median 12.
- **Finding 1 fixed in this branch** (`FallbackPrior.score` analytic-tilt blend — see the rewritten finding below): the
  two fp16-square shapes the sweep deployed at 2.0–2.25× golden now deploy `BK=2` at golden parity (square.2048.fp16
  235→105 µs, square.4096.fp16 1490→761 µs). The per-shape table below still shows the pre-fix sweep numbers.

## Headline: the evidence-pick fix worked

The previous sweep's core finding — "the deploy argmin prefers unmeasured extrapolations over configs with winning
-O3 reservoir rows" (its findings 2, 4, 5, 6) — does not reproduce. Everywhere the search measured the good region,
the deploy pick *is* the measured -O3 best: down_proj.s512 and square.512.dynM deploy measured rank-1 configs with
-O3 rows; every dynM kernel's pick is exactly the config with the minimum -O3 reservoir latency (e.g.
`k_matmul_701d6c` deploys -O1 rank 3 because it holds the best -O3 row, 56.2 µs, over the -O1 rank-1's 59.1). The
previous sweep's finding-5 (down_proj.s32 FM1-over-FM2) and finding-3 (square.512 drift) are both exact
reproductions now. The dynM shapes — five of six of which the previous sweep could only record at parity or worse —
all came back better or equal, including a −29% on square.512.dynM.

What's left over is a different failure mode: when the clean-DB search never measures the golden's region at all,
evidence-pick (correctly) deploys its measured best, which can sit 5–20% above a golden the model itself ranks
shallow. See findings 2–3.

## Per-shape outcome (live -O3 A/B; pass-1 µs, pass-2 ratio in parens where it mattered)

| shape                            | greedy µs | best golden µs |     greedy/golden | category                                                |
|----------------------------------|----------:|---------------:|------------------:|---------------------------------------------------------|
| square.512                       |      10.6 |           10.5 |       1.01 (0.98) | same knobs — exact reproduction                         |
| square.1024                      |      53.4 |           49.2 |       1.09 (1.10) | worse (finding 3)                                       |
| square.2048                      |     305.7 |          294.3 |       1.04 (1.03) | worse-marginal (≤4%, left)                              |
| square.4096                      |    2525.8 |         2282.4 |       1.11 (1.10) | worse (finding 3)                                       |
| square.512.fp16                  |       4.4 |            3.9 |              1.13 | worse (finding 1)                                       |
| square.1024.fp16                 |      17.3 |           17.1 |              1.01 | **added** (parity, WM4 BK8 vs WM1 BK2)                  |
| square.2048.fp16                 |     235.4 |          104.8 |       2.25 (2.24) | worse (finding 1)                                       |
| square.4096.fp16                 |    1490.1 |          741.6 |       2.01 (2.01) | worse (finding 1)                                       |
| qwen3_06b.q_proj.s32             |       7.5 |            8.3 |       0.90 (0.89) | **replaced** (3 entries → 1, FM4 FN2)                   |
| qwen3_06b.kv_proj.s32            |       5.7 |            5.7 |              1.00 | same knobs — exact reproduction                         |
| qwen3_06b.o_proj.s32             |       9.2 |            9.2 |              1.00 | same knobs — exact reproduction                         |
| qwen3_06b.gate_up_proj.s32       |      10.9 |           11.0 |              0.99 | **added** (parity, RING 3 vs 4)                         |
| qwen3_06b.down_proj.s32          |      14.0 |           13.9 |              1.01 | same knobs (prev sweep's finding 5: fixed)              |
| qwen3_06b.q_proj.s128            |      19.4 |           18.5 |       1.05 (1.04) | worse-marginal (4–5%, left)                             |
| qwen3_06b.kv_proj.s128           |      11.4 |           11.3 |              1.01 | **added** (parity, BN16 SPLITK1 RING2)                  |
| qwen3_06b.o_proj.s128            |      19.9 |           19.9 |              1.00 | same knobs — exact reproduction                         |
| qwen3_06b.gate_up_proj.s128      |      23.8 |           28.1 |              0.85 | **replaced** (WARPSPEC pin dropped, finding 4)          |
| qwen3_06b.down_proj.s128         |      28.2 |           28.2 |              1.00 | **added** (parity, BM8 RING2)                           |
| qwen3_06b.q_proj.s512            |      54.3 |           49.1 |       1.11 (1.10) | worse (finding 2) — prev sweep's win not re-found       |
| qwen3_06b.kv_proj.s512           |      29.8 |           27.1 |       1.10 (1.09) | worse (finding 2, repeat)                               |
| qwen3_06b.o_proj.s512            |      49.5 |           49.8 | 0.99 (1.06, 1.06) | worse-marginal (left; bimodal greedy bench)             |
| qwen3_06b.gate_up_proj.s512      |      68.4 |           56.0 |       1.22 (1.19) | worse (finding 2) — prev sweep's −37% win not re-found  |
| qwen3_06b.down_proj.s512         |      75.7 |           81.8 |              0.93 | **replaced** (3 entries → 1, FM14 BK64 OVERHANG)        |
| square.512.dynM                  |      12.8 |           18.0 |              0.71 | **replaced** (BN16 FM4 FN2 RING4; sweep's big win)      |
| qwen3_06b.q_proj.s512.dynM       |      54.4 |           56.0 | 0.97 (0.99, 0.98) | **added** (parity, BM8 FM14 SPLITK2)                    |
| qwen3_06b.kv_proj.s512.dynM      |      30.7 |           35.0 |              0.88 | **replaced** (2 entries → 1, BM8 FM8)                   |
| qwen3_06b.o_proj.s512.dynM       |      52.7 |           62.0 |       0.85 (0.86) | **replaced** (2 entries → 1, BM16 FM8 BK64)             |
| qwen3_06b.gate_up_proj.s512.dynM |      72.8 |           77.0 | 0.95 (0.95, 0.94) | **replaced** (2 entries → 1, BM8 FM8 SPLITK2)           |
| qwen3_06b.down_proj.s512.dynM    |      74.8 |           81.9 |       0.91 (0.92) | **replaced** (BM16 FM8 BK64; prev finding 6 fixed)      |

All replaces verified post-edit: `run --bench --golden` on the edited shapes re-benches the new entries at parity
with the greedy pick (gate_up_proj.s128's unpinned entry re-derives the warp-specialized form at 24.0 µs ≈ the
23.8 µs pick).

The dynM family converged onto one knob family this sweep — `FM8 FN4 SPLITK2 RING2` with per-shape BM/BN/BK — and
it beats the seed-era `BM16 FM4 FN4` family by 5–15% everywhere; square.512.dynM (a `BN16 FM4 FN2 RING4` masked tile)
closed from 1.29× over its static twin to ~1.21× of the static 10.6 µs.

## Finding 1 — fp16 large squares: the learned prior buries the small-`BK` warp region; patience-bounded search never benches it (P0 — FIXED)

square.2048.fp16 deployed at **2.25×** the golden and square.4096.fp16 at **2.01×**. The sweep-time hypothesis ("the
warp-tier enumeration floors `BK` by shape size, so `BK=2` is never offered at 2048/4096") was **wrong** — disproven by
directly enumerating the warp variants (`_enumerate_warp_matmul_impl`, `lowering/tile/_enumeration.py`):

- The candidate set is `(64,32,16,8,4,2,1)` gated only by `k_cells_total % bk == 0`, and `k_cells_total = E_K/16` is
  32/64/128/256 at 512/1024/2048/4096 — **all divisible by 2**. A direct probe confirms `BK=2` (and the exact golden
  row `WM1 WN4 FM4 FN2 BK2`) is enumerated at every size: **316 / 373 / 404 / 417** `BK=2` rows for 512/1024/2048/4096.
  The BK histogram in the original finding was what the **search benched**, not what was enumerated.

The real gate is the **search/prior**, and the chain is:

1. The learned `CatBoostPrior` regresses `log(µs)` on `knob_features`. It measured `BK=2` warp tiles only at small fp16
   squares (512), so at 2048/4096 it **extrapolates** the fp32-thread-tier pattern (large shape → large `BK`) and
   predicts `BK=2` is slow there — `eval prior` ranks the golden 335–592 deep. The cold `AnalyticPrior` prices it
   correctly (`D_w_near_bk`, rank 0/2632), but `FallbackPrior` used the learned half once fitted.
2. PUCT selection weights an unvisited child by `1/score`, so the buried `BK=2` region gets a tiny weight and patience
   (50) expires before reaching it — a **self-reinforcing blind spot** (never benched → never measured → never learned
   fast). Reproduced in isolation with a 6-shape "polluter" prior (sweep targets 1–6): tuning 2048.fp16 benched `BK=2`
   **0 times**.
3. `evidence_pick` then faithfully deploys the best **measured** `-O3` config — `BK=64` @ 235 µs — because the reservoir
   has no `BK=2` row to deploy. Not an enumeration gate, not an evidence-pick bug: the search simply never produced the
   evidence.

**Fix (this branch): an analytic-tilt blend in `FallbackPrior.score` — the MCTS *selection* signal only.** The two
priors are on different scales (CatBoost = calibrated µs; analytic = an ordinal learning-to-rank proxy, neutral value
1.0), so a naive `min`/blend collapses to "always analytic" (the reverted 2× regression). Instead `score` keeps the
learned µs as the per-shape scale and folds the analytic in as a dimensionless multiplier centered at its neutral 1.0:

```
score = learned_µs · analytic(knobs) ** W            # W = config.analytic_tilt(), default 0.3
```

A config the heuristic favors (`analytic < 1`) gets its predicted µs pulled down → larger PUCT exploration weight → it
is explored; a no-opinion config (`analytic = 1`) is untouched. So the heuristic *perturbs* the learned order without
replacing it, and the learned half still owns the scale. `mean_score` / `mean_scores` / `pick` (deploy, eval,
diagnostics) stay **pure-learned + evidence** — only exploration changed. No MCTS machinery: the blend lives entirely
in the prior, behind the regular `score` interface PUCT already calls.

**Validation** (polluter prior, fresh DB, isolated; `W` swept via `DEPLODOCK_ANALYTIC_TILT`):

| `W` | `BK=2` benched | tuned best µs |
|----:|---------------:|--------------:|
| 0 (baseline) | **0** | 99.0 (`BK=4`) |
| 0.2 | 40 | 92.5 |
| 0.3 (default) | 41 | 92.3 |
| 0.5 | 45 | 92.4 |

End-to-end, re-tuning the two shapes on the real cache with the fix and re-deploying: square.2048.fp16 **235 → 105 µs**
(`BK=2`, golden parity) and square.4096.fp16 **1490 → 761 µs** (`BK=2`, golden parity). No regression where the learned
prior is already right: square.1024 fp32 46.5 → 44.9 µs (improved), q_proj.s512 40.3 → 40.7 µs (within bench noise). The
existing fp16 goldens already record the `BK=2` region, so no YAML change is needed — the fix makes them *reachable*.
square.512.fp16's residual 1.13× is the same story at its mildest (a within-tier `WM 1/4` misorder, `BK=4` already
benched) and is not load-bearing.

**Follow-up:** the fix only refreshes a shape's deploy once that shape is re-tuned (it changes selection, not the
stored reservoir). A full `tune --dataset golden` re-run on this branch would re-record every shape under the fixed
exploration; only the two fp16 squares were re-tuned here.

## Finding 2 — fp32 s512 family: clean-DB search variance unsamples the goldens; evidence-pick then deploys a worse measured best (q_proj +10%, kv_proj +10%, gate_up +20%)

q_proj.s512, kv_proj.s512 and gate_up_proj.s512 are all worse with the same signature: `eval variants` shows the
deploy pick at measured rank 1 (or the only -O3-evidenced config) — evidence-pick did its job — but the golden's
config is **absent from the measured set** (21–39 measured configs per kernel; grepping `--top 0` for the golden's
knob row finds nothing). Two of these goldens are the *previous sweep's own wins* (q_proj.s512's `BM16 BK32 RING4`,
gate_up.s512's −37% `FM10 FN4 SPLITK1 RING3`): the accumulated DB that contained them was wiped by `--clean`, and
this sweep's patience-bounded re-search took different trajectories (the learned prior ranks the q_proj.s512 golden
3/1008 — shallow — yet the inner search still never benched it before patience ran out; kv_proj.s512's ranks 140
learned / 331 analytic, a genuine mispricing on top).

**Recommendation:** seed the inner search with the recorded golden configs. `tune --dataset golden` knows exactly
which shape it is tuning and what the golden knobs are — force-bench each recorded golden as the first variant(s) of
its op's inner search. Cost is one bench per golden entry (~30 s across the whole sweep); benefit is that
evidence-pick can never deploy worse than the recorded golden after a clean tune, and "worse" findings would then
*only* mean real regressions. This also stops wins from silently un-happening across sweeps (gate_up.s512's −37%
existed this morning, was recorded, and is now unreachable again purely by sampling luck).

## Finding 3 — big fp32 squares: FM misorder within shallow ranks persists (square.1024 +9%, square.4096 +11%)

square.1024 deploys `FM10 RING3` vs golden `FM14/FM8 RING2/4` (learned ranks 3 and 47 for its two entries);
square.4096 deploys `FM28` vs golden `FM10` (learned rank 50, `vs gold` 1.00x — the -O3 reservoir number says
parity, the live A/B says +11%). The previous sweep's recommendation to represent the big-FM regime in the analytic
fit was applied (FM26-class analytic ranks improved 121→147 vs 394 learned previously), but the searched-and-measured
sets again don't contain the goldens (same mechanism as finding 2 at lower stakes). Covered by finding 2's
golden-seeding recommendation; no separate action.

## Finding 4 — the A/B knob columns hide planner-derived knobs that differ (WARPSPEC), and gate_up_proj.s128's golden was pinning the slow choice

gate_up_proj.s128's greedy pick benched **−15%** vs a golden row showing *identical* knob columns — only `block`
(160 vs 128) and `regs` betrayed the difference. `eval golden` resolved it: `WARPSPEC True/False`. The pick's variant
space (`eval variants`) has no WARPSPEC column at all — in the fp32 thread tier it is planner-derived, not searched —
while the YAML entry recorded `WARPSPEC: false` from an earlier era, actively pinning the kernel away from the
warp-specialized form the planner now prefers. The fix was to drop the pin (the replacement entry records only the
search knobs), verified at 24.0 µs.

**Recommendation:** two small ones. (a) The `run --golden`/`--ab` knob table should add a column for any knob that
*differs* between the greedy pick's resolved config and a golden row even when it isn't in the searched space —
block-size archaeology shouldn't be needed to see a WARPSPEC diff. (b) Audit the remaining `WARPSPEC: false` pins
(square.2048, square.4096 static) — they may be pinning those shapes away from better planner-derived forms too.

## Finding 5 — `eval variants`' ◄ pick marker disagrees with the actual deploy when no -O3 evidence exists

For the small s32/s128 kernels (no -O3 reservoir rows at all — e.g. `k_matmul_60e20f`, `2f6d1b`, `39c474`,
`aaa3b2`), the ◄ marker lands on `FM6 FN4` rows while the live deploy (and `eval golden`'s found column) picks
`FM4 FN2`. With no evidence, deploy argmins the model over the **full enumeration** while the variants view argmins
over the **measured rows only** — both labeled "the prior's pick". Harmless this sweep (the live picks beat the
goldens anyway) but it makes the view's `pick: rank N` line untrustworthy exactly when it's needed. Also from the
same neighborhood: kernels this small never get -O3 re-bench rows despite the 10%-of-best tolerance band — worth a
look at whether the re-bench is failing or the reservoir is dropping them.

**Recommendation:** make the ◄ marker call the same `Prior.pick` path the deploy uses (full enumeration, evidence
first), and flag when the resulting config is unmeasured instead of silently marking the nearest measured row.

## Workflow notes

- **The A/B pass collapsed from ~45 min to ~3.5 min** because it ran right after the tune with a hot cubin cache
  (~7 s per shape, 29 shapes). The previous report's "in-process `run --bench --dataset golden` loop" proposal is
  much less urgent now; sequencing A/B directly after tune is the cheap fix. Three full passes cost ~10 min total,
  which made the noise-floor confirmation (step 4) painless for the first time.
- **Ratios were rock-solid across passes** (≤2% drift on 28/29 shapes; absolute µs steady too — back-to-back runs,
  one thermal state). The exception: o_proj.s512's greedy bench was bimodal (49.5 / 53.2 / 53.3 µs across passes,
  same kernel) — categorized worse-marginal on the 2-of-3 majority.
- **`eval golden` is now the WARPSPEC oracle** (finding 4): the live A/B table hid the only differing knob, and
  `eval golden` exposed it. Until finding 4(a) is fixed, run `eval golden` before recording any "same-knobs but
  faster" result — a hidden planner-derived diff may be the real story.
- **Kernel-name collisions persist** (third report in a row): square.NNNN and square.NNNN.fp16 share kernel names
  (`bed174`, `180e20`, `262948`, `207791`), so their `eval variants` groups interleave fp32 FK-split rows with
  fp16 warp-tier rows. This sweep's finding-1 evidence had to come from the tune log (grepping per-target BK
  histograms) because the variants view couldn't separate the tiers. The ShapeKey-subgrouping / `--shape` filter
  proposal stands, now with a concrete victim.
- **The tune log was the only source for "which knob values were ever offered"** (finding 1's BK histograms). A
  per-op enumeration summary — even just min/max per knob in `eval variants`' header — would have answered it in
  one command.
- **Fixed since the previous report and held**: (a) the evidence-pick work (its findings 2/4/5/6) — verified across
  every kernel with -O3 reservoir rows; (b) the fp16 TMA+WARPSPEC launch crash (its finding 1) — zero
  CUDA_ERROR_INVALID_VALUE rows this sweep, the fp16 golden rows bench fine; (c) fp16 shapes still appear in
  `eval prior` rank lists (held). **Not fixed**: kernel-name collisions (above); reduce/pointwise goldens still
  outside the tune/A-B loop (third report in a row).
- The `--clean` sweep cost finding 2's regressions: two recorded wins became unreachable again because the DB that
  knew about them was wiped and the re-search didn't re-find them. Until golden-seeding lands, consider running
  routine sweeps *without* `--clean` (accumulate) and reserving `--clean` for prior-hygiene checks.
