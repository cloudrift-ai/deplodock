# Golden-set tune / bench sweep — report (2026-06-07)

End-to-end sweep over the 23 `GOLDEN_CONFIGS` matmul shapes: for each shape `tune --golden NAME` then
`run --bench --golden NAME` (after-tune bench), then a final `run --bench --golden NAME` for every shape once all are
tuned. The first config is tuned with `--clean` so the learned prior is rebuilt from scratch over the sweep. Driver +
analyzer live under `/tmp/golden_sweep/` (`driver.py`, `analyze.py`).

Two comparison bases are reported:

- **after-tune** — each shape's greedy pick right after *its own* tune (per-shape freshly-fit prior). Shows what the
  tuner can find for a shape in isolation.
- **final** — the deployable greedy pick after the whole set is tuned (the number a user actually gets from
  `run` / `compile`). This is the one that matters for deployment.

## Headline

The codegen fix + regenerated goldens + heuristic adjustment moved the sweep from "mostly broken" to "half at-or-better,
no crashes":

| run | crashes | BETTER | SAME-DIFF | SAME-MATCH | WORSE | regressions |
|-----|--------:|-------:|----------:|-----------:|------:|------------:|
| before codegen fix      | 2 | 0 | 3 | 0 | 18 | 15 |
| after fix (prev sweep)  | 0 | 0 | 2 | 0 | 21 | 16 |
| **this sweep (final)**  | **0** | **2** | **6** | **1** | **14** | **12** |

(categories: BETTER = greedy >3% faster than golden; SAME-DIFF = within 3% but different knobs; SAME-MATCH =
reproduces a golden; WORSE = >3% slower / couldn't reach golden; regressions = final >3% slower than after-tune.)

## Fix that unblocked this: duplicate thread-axis codegen bug

`rename_ssa_sequential` canonicalized axis names from every binding scope **except**
`WarpSpecialize.consumer_thread_axes`. After `085_warp_specialize` removes the consumer `ThreadTile`, those thread
coords live only as free Vars in the consumer body + that off-body field. An unstaged coord (e.g. the N-thread index of
a matmul whose B operand isn't TMA-staged, so it's absent from every StageBundle cache axis) kept an uncontrolled name
and could collide with a renamed cache axis — emitting two `int a3` in one scope → `nvcc` error
(`"a3" has already been declared`). The learned-prior greedy was selecting a warp-specialized fp32 config for
`square.2048` / `square.4096`, so both failed to compile.

Fix (commit `9d2853ed`): record `consumer_thread_axes` in the rename pass so each thread coord claims its own canonical
slot. Regression test: `tests/compiler/ir/stmt/test_rename_ws_thread_axes.py` (reproduces the `['a3','a3']` collapse
pre-fix). Full compiler suite green.

## This sweep — final re-bench (deployable greedy vs golden)

**BETTER (2)** — genuine deployable wins, ~20% faster than the recorded golden:

| shape | greedy | golden | gain | knobs |
|-------|-------:|-------:|-----:|-------|
| `square.1024`           | 55.3 µs | 68.9 | +19.7% | `BM=8 BN=32 BK=64 FM=8 FN=4 FK=1 SPLITK=1 RING=2 STAGE=11` |
| `qwen3_06b.q_proj.s512` | 55.4 µs | 70.0 | +20.9% | (same) |

**SAME-DIFF (6)** — at parity, prior consistently lands on a uniform
`BM=8 BN=32 BK=64 FM=4 FN=2 SPLITK=2 RING=2` config: `gate_up_proj.s32`, `kv_proj.s128`, `gate_up_proj.s128`,
`kv_proj.s512`, `o_proj.s512`, `down_proj.s512`. **SAME-MATCH (1):** `square.512` reproduces its golden.

**WORSE (14)** — the dominant remaining gap is **fp16 tensor-core selection**: the prior picks a ~2–3× slower config for
every fp16 square:

| shape | greedy | golden | slowdown |
|-------|-------:|-------:|---------:|
| `square.512.fp16`  | 17.9 µs | 6.1   | **+193%** |
| `square.1024.fp16` | 31.5 µs | 18.5  | +70% |
| `square.2048.fp16` | 131.8 µs | 107.7 | +22% |
| `square.4096.fp16` | 840.4 µs | 738.8 | +14% |

Other WORSE: fp32 `square.2048` (+38%) / `square.4096` (+22%), and several qwen `s32`/`s128` shapes (+24–56%).

**Regressions (12)** — final vs after-tune drift is now mild (mostly +3–46%, vs +50–190% in the previous sweep), e.g.
`square.512` 12.3→14.3, `down_proj.s128` 26.9→39.2. The catastrophic prior collapse from earlier sweeps is gone.

## Golden updates applied

The 2 BETTER configs were recorded (commit replaces `square.1024` and `qwen3_06b.q_proj.s512` in `golden.py`):
deplodock 69.2/69.6 → **55.3/55.4 µs** at ratio 0.96 (now golden). The `cublas_us` reference is reused (config-
independent torch SGEMM). The 6 SAME-DIFF variants were **not** added — they're parity-only and the warm-prior pick that
produces them isn't reproducible by the isolated `find_golden_configs.py` search (see below), so they'd be churn.

### Note on the regenerator

`scripts/find_golden_configs.py` was repaired (commit `28d6dcee`): it pointed at a nonexistent `golden_configs.py`,
dumped the `S_`/`H_` feature columns into recorded knobs, and clobbered the set. It now targets `golden.py`, strips
feature + control knobs, and merges (better→replace, parity+diff→add, worse→keep) so a weak tune can't degrade a golden.

**Important limitation:** its isolated per-shape search (cold DB, `-O1` ranking) consistently finds **worse** configs
than the warm-prior greedy pick — e.g. for `square.1024` it lands on ~117 µs while the deployed greedy is 55 µs. So the
authoritative regen reproduces almost none of the sweep's wins; the wins only surface through the warm-DB greedy path.
That gap (isolated search ≪ warm greedy) is itself a tuner-quality signal worth chasing.

## Next targets (highest leverage first)

1. **fp16 tensor-core mis-pick** — the prior avoids the fast `mma_m16n8k16_f16` config on every fp16 square (up to
   +193%). Investigate the warp-tier scoring / enumeration for fp16.
2. **Isolated-search vs warm-greedy gap** — `find_golden_configs.py`'s cold `-O1` search can't reach configs the warm
   greedy deploys. Closing this would let the authoritative regen capture wins directly.
3. **fp32 large squares** (`2048`/`4096`, +20–38%) and several qwen `s32`/`s128` shapes remain off golden.

---

# fp16 warp-tier deep-dive (2026-06-07 follow-up — Next-target #1)

Investigation of the report's #1 gap (fp16 tensor-core mis-pick, up to +193%). The cause is **not one thing**, and it
is **not** a regression from the explicit-knob-OFF work (a fresh no-prior greedy still picks `WM=2/WN=2,
WARPSPEC=False`, exactly what the heuristic dictates).

## What the numbers show (RTX 5090, sm_120)

- `eval heuristic --kernel fp16`: goldens sit at **median rank 7** in the enumeration (decent, not 1). Not the
  bottleneck on its own.
- `tune --golden square.512.fp16 --clean`: finds **6.17 µs** vs golden 6.1 / cuBLAS 6.2 — *isolated* tune is fine, but
  **91% of benches are `silly` (≥2× best)** → very wasteful exploration. (The report's *final* 17.9 µs for this shape is
  the warm-prior-over-all-shapes pick — the prior generalizes the warp tier badly; see cause C.)
- `tune --golden square.2048.fp16 --clean`: settles at **104–106 µs**; golden is **94.3 µs**. Decomposing it:

  | config | tile | -O1 (rank pass) | -O3 (deployable) |
  |---|---|---:|---:|
  | greedy pick `WM=2 WN=2 FM=2 FN=4` | 64×64 square | 163.1 | 110 |
  | golden-shape `WM=1 WN=4 FM=4 FN=2` **WS=False** | 16×32 skew | 162.2 (fastest -O1) | 112 |
  | **golden** `WM=1 WN=4 FM=4 FN=2` **WARPSPEC=True** | 16×32 skew | *never benched* | **94.3** |

## Root causes (ranked)

**A. `WARPSPEC=True` is never explored (dominant).** A full 2048² tune benched **0 of ~59 configs with
`WARPSPEC=True`**. The inner MCTS (`search/policy/mcts.py`) is fully deterministic — one PUCT exploration term, strict
`>` tie-break (ties → first-in-list = heuristic order), no sampling/noise/forced-visit. With a cold prior it fans out
in heuristic order and `patience` (~60 benches) fires before the `085_warp_specialize` option-1 branch (deep under the
planner forks) is ever selected. The 94.3 µs golden lives behind that unexplored fork, so the search cannot reach it.

**B. The -O1 ranking pass is blind to the -O3 win (fixed, see below).** The golden-shape tile (162.2 µs) and the
heuristic tile (163.1 µs) tie within 0.5% at -O1 but are 94.3 vs 110 at -O3. `tune` ranks at -O1 and previously
re-benched at -O3 only a *strict new -O1 best*, so near-tied contenders never got a deployable sample and the prior
trained on flattened -O1 numbers.

**C. Warp tier has no occupancy features.** `knob._tile_features` early-returns `{}` for warp rows (the `is_warp`
gate). A warp config's vector is only `WM/WN/FM/FN/BK/SPLITK` + `MMA_*` + coarse `S_ext_*`/`H_*` — **no CTA-count /
waves / occupancy**, which is exactly the quantity that picks skewed-vs-square per shape. So even with -O3 truth for a
few configs the prior generalizes the warp tier poorly — the mechanism behind the report's catastrophic *final* fp16
picks (17.9 µs @ 512²) vs the healthy *isolated* tune (6.17 µs).

**D. Heuristic prefers the square 64×64 tile** (`_priority_matmul_warp`), so greedy/cold-start deploys `WM=2/WN=2`
while goldens are skewed (rank ~7). The warp priority is hand-tuned, not learned (`golden_knob_heuristics.py` fits the
scalar tier only).

**E. (minor) Offline diagnostics can't match fp16 data to fp16 goldens** — `tune` offline reports `golden coverage
0/32` after tuning an fp16 shape (coverage keys on free·reduce product, conflating fp16/fp32 and not recognizing the
warp signature). Hides whether the prior learned the shape; not a perf cause.

## Implemented: -O3 re-bench tolerance (fixes cause B)

Re-bench at -O3 any config **within `DEPLODOCK_O3_TOL` (default 10%) of the best -O1 so far**, not just a strict new
best — so configs that tie at -O1 but diverge at -O3 each get a deployable prior sample.

- `search/policy/mcts.py`: `observe` sets `last_o3_worthy` when `median ≤ best_-O1·(1+tol)`, deduped via `_o3_done`
  (`_o3_sig` str-ifies values so list-valued `OVERHANG` stays hashable, excludes `H_opt`). `O3_REBENCH_TOL = 0.10`.
- `pipeline.py`: drive loop triggers `_rebench_o3` on `last_o3_worthy` (was `last_improved_best`).
- `config.py`: `float_env` + `o3_tol()` (`DEPLODOCK_O3_TOL`).

**Validated**: re-tuning `square.2048.fp16 --clean` captures **4 `H_opt=3` (-O3) rows** (was 1); among explored configs
the prior now ranks by deployable cost and correctly prefers the genuine -O3 best (`WM=2/WN=2 @105.9`). The residual gap
to 94.3 µs is entirely cause A — `WARPSPEC=True` is still unexplored — which the exploration work must close.

## "Add randomness" — necessary but not sufficient

Stochastic/forced exploration is the highest-leverage next change and directly fixes cause A (surfaces `WARPSPEC=True`
and skewed tiles). But alone it doesn't fix B (the prior would still mis-rank by -O1 — now fixed) or C (the prior can't
*generalize* warp tiles without occupancy features). Recommended order: **exploration → -O3 tolerance [done] → warp
features.**

## Recommended next steps

1. **Stochastic / forced exploration in `mcts.py`.** MVP: force every live fork child to be visited once before
   patience can fire (guarantees both `WARPSPEC` branches + all `WM/WN` siblings get a bench). Stronger: softmax/
   Boltzmann sampling over PUCT values, or Dirichlet noise on `P`, seeded reproducibly by op + round (no `random`/
   `Date.now` — counter-seed an RNG). Re-validate: 2048² benches `WARPSPEC=True` and reaches ≤95 µs.
2. **Warp occupancy features** (`knob._tile_features`): drop the blanket warp early-return; compute warp `D_*` from
   `tile_m=WM·FM·atom_m`, `tile_n=WN·FN·atom_n`, CTAs≈`S_ext_free_prod/(tile_m·tile_n)`, waves vs `H_sm_count`
   (mirroring `heuristic`), so the prior can pick skewed-vs-square per shape.
3. **Fix offline golden coverage match** (`prior/diagnostics.py`): key on dtype + per-axis extents so fp16 shapes are
   recognized — needed to *measure* that (1)/(2) worked.

## Resolution (2026-06-07): the fp16 win was a fork-ordering bug, not exploration

Implementing the exploration work (above) and validating it on `square.2048.fp16` overturned the "add randomness"
hypothesis and found the real cause:

- **ε-greedy** (random child with prob ε): implemented as an opt-in knob (`--explore-eps` / `DEPLODOCK_TUNE_EPS`,
  **default 0**). At ε=0.2 it did **not** surface `WARPSPEC=True` (still 0 benched) — the random budget is diluted
  across ~10–15 fork levels per descent, so it rarely lands on one specific deep fork.
- **Random tie-break** (randomize among equal-PUCT children under a cold prior): **tried and reverted** — it discards
  the heuristic enumeration order, which does real work, and **regressed 2048² tuning ~2×** (random walk into degenerate
  wide-`FM/FN` tiles; best 238 µs vs ~104 deterministic) while *still* benching 0 `WARPSPEC=True`.
- That 0-under-full-random was the tell: `WARPSPEC=True` wasn't a *selection* casualty, it was barely a *candidate*.
  Pinning `DEPLODOCK_WARPSPEC=1` during tune **benched it for `WM=2/WN=2`** (and crashed on tiles where WS is
  ineligible — `producer_threads(32)` must divide the inner thread-axes product). So for *eligible* warp tiles the
  `WS=True` child exists; for ineligible ones the pass stamps `False` (no fork).

**Root cause.** `085_warp_specialize` emits the `WS` fork `False`-first and relies on its `score_fn` (`WS=1 → 1.0`) to
make pickers prefer `True`. But `TuningSearch._select` consults the **CatBoost prior**, not the fork `score_fn`; a cold
/ unfit prior (a fresh `tune --clean`, and every no-prior `compile`) falls back to uniform `P` → PUCT tie → **emission
order** → `WS=False`. Under patience the search then advances to the next tile and never benches `WS=True`. So both the
deployed greedy pick and the tuner missed the warp win.

**Fix (`085_warp_specialize.py`): emit `WARPSPEC=True` FIRST for the warp tier.** Emission order now deploys the win
cold, and the -O3 tolerance feeds the prior its deployable latency.

**Measured (`run --golden square.2048.fp16 --bench`, no prior → heuristic pick):**

| | before | after |
|---|---:|---:|
| deployed greedy fp16 2048² | `WS=False` **110 µs** (0.88× eager) | `WS=True` **93.4 µs** (1.05× eager) |

The deployed pick now **beats** the recorded golden (94.6 µs) and eager. The earlier three changes (-O3 tolerance, warp
occupancy features) still matter — they let the prior *rank* the explored warp configs by deployable cost and pick the
skewed tile — but the fp16 cliff was this ordering bug. ε-greedy stays as opt-in infrastructure for shapes where the
heuristic order is known-bad; it is not the fp16 fix.

Remaining follow-ups still open: warp-tier `producer_threads` divisibility narrows WS eligibility for some tiles (a
pinned `DEPLODOCK_WARPSPEC=1` hard-fails rather than pruning during a sweep — worth softening); and the offline golden
coverage match (#3 above) still conflates fp16/fp32.
