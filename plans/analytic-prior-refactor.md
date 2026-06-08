# AnalyticPrior refactoring — one ranking path: analytic prior (cold) or learned CatBoost (warm)

## Context

The compiler has had **two** parallel ranking mechanisms with **two** parallel featurizations:

- the hand-coded heuristic — `score_matmul_thread` (+ fitted weights `_W`) and the `_priority_matmul_*` /
  reduce / pointwise enumeration sort keys — which ranks the *cold* path purely as **enumeration (emission)
  order** (greedy takes option-0; MCTS ties → first-in-list), featurized by `heuristic._featurize` /
  `golden_knob_heuristics._featurize`;
- the learned `CatBoostPrior`, featurized by `knob.knob_features`, which ranks once it has data.

They answer the same question ("which config is fastest") over near-identical features (the `D_*` terms in
`knob_features` already mirror `score_matmul_thread`), yet are wired as a heuristic-order *fallback* vs a
learned scorer with a `fitted` special-case in every policy. The recent fp16 fix had to be an *emission-order
reorder* precisely because the heuristic only influences ranking through sort order, not a score.

**Goal:** collapse to a single ranking path — a config is ranked by a `Prior`, which is the hand-coded
**`AnalyticPrior`** when there's no learned data and the **`CatBoostPrior`** once trained, composed behind a
`FallbackPrior`. Delete `score_matmul_thread` / `_priority_matmul_*` and the per-policy cold special-cases;
`knob_features` becomes the single featurization (enriched with whatever discriminative terms the old
heuristic had). Outcome: less code, one feature set, cold-start ranked by a real heuristic *score* (not raw
emission order), and a clean analytic→learned handoff.

## Design

### 1. Enrich `knob_features` (`knob.py`) — the single featurization

Add the discriminative terms the old heuristic had that aren't already in `knob_features`, as engineered
`D_*` features (computed for both tiers, alongside the existing `D_threads/D_cells/D_tile_*/D_log2_area/
D_reuse/D_aspect/D_log2_ctas/D_log2_waves`):

- **band indicators** the fitted `_W` leaned on hardest: `D_bn_band` (16≤BN≤64), `D_bm_band` (8≤BM≤16),
  `D_tilen_clean` (tile_n ∈ {32,64,128}), `D_bk_ge32`, `D_splitk_le2`, `D_near_threads` (|threads−256| or
  |threads−128| warp), `D_square` (|aspect|). Most are cheap functions of the raw knobs already in the dict.
- **K-chunk** term `D_kchunks` from `S_ext_reduce_prod` (= K for matmul) + BR/BK — the one feature the agent
  flagged as needing K; `S_ext_reduce_prod` already carries it.

Keep the ceil-free CTA approximation (`D_log2_ctas/waves` from `S_ext_free_prod`); exact per-axis ceilings are
a noted follow-up, not in this prototype.

### 2. `AnalyticPrior(Prior)` — `search/prior/analytic.py`

A stateless, fixed **linear model over `knob_features`**:

- `score(knobs)` / `mean_score(knobs)` → `exp(-(w·knob_features(knobs)))` — a positive latency-proxy, **lower
  is better** (matches `CatBoostPrior` polarity; quality `w·f` higher → score lower). One weight dict `_W_A`
  over `knob_features` keys covers **both tiers** (the `D_*` are tier-aware). Configs the model has no opinion
  on (non-matmul) get the neutral baseline so ties fall to enumeration order.
- `fitted` → `True` always (it's the untrained prior); `fit` / `add_rows` / `maybe_refit` / `checkpoint`
  no-ops; `to_json` → `None`.
- Weights: **port the dominant `_W` terms** onto the new `D_*` keys (so cold ranking reproduces today's
  heuristic — `bn_band`/`bm_band`/`tilen_clean`/`bk_ge32`/`splitk_le2`/`near_waves`/`ctas_ge_sm` map directly),
  plus hand-set warp terms from `_priority_matmul_warp`'s targets (`D_log2_area`→64×64, `D_aspect`→0,
  `D_near_threads`→128, BK≈2). Re-fitting over `knob_features` via `golden_knob_heuristics.py` is the tuning
  path (step 6) but not required for the prototype to R rank correctly.

### 3. `FallbackPrior(Prior)` — `search/prior/fallback.py` + `load_prior()` factory

Wrap `(learned: CatBoostPrior, analytic: AnalyticPrior)`:

- `fitted` → `True` always; `score`/`mean_score` → `learned.{score}` when `learned.fitted` else
  `analytic.{score}`. Training surface (`add_rows`, `maybe_refit`, `checkpoint`, `to_json`, dataset, summary)
  **delegates to `learned`** — tune trains CatBoost exactly as today.
- `load_prior(*, seed=0, path=None) -> Prior` returns `FallbackPrior(CatBoostPrior.load(seed, path),
  AnalyticPrior())`. Export `AnalyticPrior`, `FallbackPrior`, `load_prior` from `search/prior/__init__.py`.

### 4. Policies always rank via the prior (`greedy.py`, `mcts.py`)

Because `FallbackPrior.fitted` is always `True`, the cold special-cases go:

- `greedy.py`: `_ensure_prior` → `load_prior()`; drop `not fitted → option-0` — always `argmin mean_score`
  (option-0 only if load catastrophically returns `None`).
- `mcts.py`: `_select` drops the `fitted`-uniform-`P` branch — always `P = (1/prior.score)/global_best`. The
  ε-greedy hook stays. `two_level.run_two_level_tune` builds the shared prior via `load_prior()`.

### 5. Delete the old heuristic (`heuristic.py`, `_enumeration.py`)

- `_enumeration.py`: remove `_priority_matmul_thread` / `_priority_matmul_warp` / the reduce + pointwise
  priority fns and the `priority_fn` sort in `enumerate_cartesian` — rows stay in cartesian construction order
  (deterministic; the prior ranks the frontier). **Keep `_matmul_thread_gate`** (a pruning filter, not a
  scorer).
- `heuristic.py`: remove `score_matmul_thread` + `_W`. `_enumerate` / `evaluate_golden` / `pick_matmul`
  (used only by `eval` + tests) get repurposed to rank via `AnalyticPrior` (or move into `diagnostics.py`),
  so `eval heuristic` becomes "golden rank under the AnalyticPrior" — reusing the existing
  `diagnostics.golden_prior_eval` machinery with an `AnalyticPrior` instance.

### 6. Weight fitting (`scripts/golden_knob_heuristics.py`)

Re-point the offline fitter at `knob_features` (drop its parallel `_featurize`): enumerate each golden's
candidates, featurize via `knob_features`, fit `_W_A` to minimize median golden rank — the same random-search
+ coordinate-descent it does now, over the unified feature set, covering the warp tier too. Output feeds
`AnalyticPrior._W_A`. Prototype ships hand-ported weights; this is how they get tuned.

### 7. Docs + tests

`pipeline/ARCHITECTURE.md` + `search/ARCHITECTURE.md` (the heuristic/prior sections), `CLAUDE.md`
(`eval heuristic` semantics, the single-ranking-path note).

## Files to modify

- `deplodock/compiler/pipeline/knob.py` — enrich `_geom_feats` / `knob_features` with the band + kchunks `D_*`.
- `deplodock/compiler/pipeline/search/prior/analytic.py` (**new**), `…/fallback.py` (**new**),
  `…/prior/__init__.py` (exports + `load_prior`).
- `deplodock/compiler/pipeline/search/policy/greedy.py`, `…/mcts.py`, `…/two_level.py` — `load_prior()`, drop
  cold branches.
- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` — remove priority fns + the sort.
- `deplodock/compiler/pipeline/search/heuristic.py` — remove `score_matmul_thread`/`_W`; repoint
  `evaluate_golden`/`_enumerate` to `AnalyticPrior` (or fold into `diagnostics.py`).
- `deplodock/commands/eval.py` — `eval heuristic` → AnalyticPrior golden eval; `eval prior` → `load_prior()`.
- `deplodock/commands/tune.py` — offline refit uses the learned half (`CatBoostPrior.load`) directly.
- `scripts/golden_knob_heuristics.py` — fit over `knob_features`.
- Tests: `tests/compiler/pipeline/search/test_heuristic*.py` (rewrite for AnalyticPrior), new
  `test_analytic_prior.py` / `test_fallback_prior.py`, `test_online_prior.py` (FallbackPrior), planner
  enumeration-order tests (rows no longer heuristic-sorted), `eval` CLI tests.
- Docs: `pipeline/ARCHITECTURE.md`, `search/ARCHITECTURE.md`, `CLAUDE.md`.

## Reuse (don't reinvent)

- `knob.knob_features` / `_geom_feats` — the single featurizer; extend, don't fork.
- `diagnostics.golden_prior_eval` — already evaluates a `Prior` over goldens; feed it `AnalyticPrior` for
  `eval heuristic`.
- `Prior` ABC + `CatBoostPrior.load` — `AnalyticPrior`/`FallbackPrior` subclass `Prior`; `FallbackPrior`
  delegates training to the existing `CatBoostPrior`.
- `_matmul_thread_gate` — keep as the enumeration pruning filter.

## Verification

1. `make test` (xdist flags per CLAUDE.md) — new prior unit tests + rewritten heuristic/planner tests green.
2. `deplodock eval heuristic` (now AnalyticPrior) — golden **median rank** should match the old heuristic
   (~7 for fp16, comparable for fp32); this is the regression gate that the ported weights reproduce the old
   ranking. `deplodock eval prior` unchanged (learned).
3. Cold compile (no prior file) — `deplodock run -c "<fp32 512²>"` and `"<fp16 2048²>"` `--bench`: accuracy
   PASS and the deployed pick is ≥ the old emission-order pick (the AnalyticPrior cold ranking should match or
   beat option-0; fp16 should still land WARPSPEC=True at ~94 µs).
4. `deplodock tune --golden square.512.fp16 --clean` — confirms `FallbackPrior` trains the CatBoost half
   (rows added, checkpoint written) and warm picks unchanged; `tune` offline diagnostics still work.
5. `make lint`.

## Notes / risks

- **Cold-ranking parity is the key risk:** AnalyticPrior must reproduce the old heuristic's golden ranks.
  Mitigated by porting `_W` onto matching `D_*` keys (same features, same weights) — the only deltas are the
  ceil-free CTA approximation and the dropped low-weight `kchunks` exactness. Gate on `eval heuristic` median
  rank before/after.
- **Persisted state:** unaffected — `FallbackPrior` delegates `to_json`/dataset to `CatBoostPrior`; the
  existing `prior.json` keeps working, no schema bump.
- **Per-axis exact shape** (`S_ext_m/n/k`) is deferred — both priors keep the coarse `S_ext_free_prod`
  approximation noted in `plans/golden-sweep-report.md`.
