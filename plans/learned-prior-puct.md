# Learned online prior for `deplodock tune` (Bayesian-linear Thompson → LinUCB)

## Context

Today the inner MCTS (`TuningSearch`, SP-MCTS) ranks unvisited siblings with `TileOp.score(ctx)` — a hand-written
heuristic. It is computed lazily pre-materialization (`lazy_score`, `tile/ir.py:1985-2003`), cached per
`(smem_cap, compute_capability)` (`tile/ir.py:1934-1982`), and the geometry formula lives in
`tile/ir.py:1501-1610`. It feeds **only** the tiebreaker slot of `_ucb_key` (`mcts.py:147-157`): unvisited children get
UCB `= +∞` and the prior just orders the mandatory first-visits among them — it never enters UCB arithmetic.

Two facts drive this plan:

1. **The heuristic is low quality and we want to stop relying on it.** It is a hand-tuned geometry proxy (thread-count
   target, register-cell budget, MMA aspect reward, staging bonus) — not fit to any measured data. We do *not* want it
   as a prior mean. (We may keep it as one input feature with a learned weight; see §3.)
2. **The tiebreaker is a real latency lever despite never touching UCB arithmetic.** The inner search stops on
   **patience** (N consecutive measured terminals with no new best), and at every node breadth is forced (`+∞` on
   unvisited). So the prior's leverage is *which subtrees get deepened first*: surface the winner early → high reward →
   UCB deepens the right subtree → patience closes sooner. Cost ≈ `(how early the good branches are ordered) + patience`.
   A better ranking over the **top quantile** directly cuts benches-to-best. (Not just top-1 — UCB spends its bonus
   across the competitive branches, so the upper slice of the order has to be trustworthy, not merely the single best.)

We already accumulate the training signal: `~/.cache/deplodock/autotune.db` holds measured
`(op_cache_key, knobs, hardware) → latency` rows in `perf` (`db.py:164-177`, columns include
`latency_us_median/min/max/mean`, `n_samples`, `knobs` JSON), keyed structurally so the *same kernel transfers across
models* (`keys.py:31-55`). The `knobs` command (`commands/knobs.py`) already computes per-knob regret + a
knob-interaction matrix from these rows — we reuse both. Nothing today reads this history to warm-start a search;
`TuningSearch.push` explicitly ignores the greedy DB hint (`mcts.py:123`).

## Why this design (and why not the existing LightGBM plan)

The previous revision of this file proposed an **offline LightGBM `lambdarank`** prior retrained periodically. We are
replacing that with an **online Bayesian linear model**, for reasons specific to this problem:

- **Signal arrives fast and asymmetric.** High-level knobs (BR, ATOM_KIND, BM/BN — the high-regret knobs per the `knobs`
  command) swing latency hard, so a handful of benches already separate good from bad. Early data is mostly *bad/ok*;
  *good* is rare. A point-estimate ranker (LightGBM, River-style SGD) learns "everything seen is bad-to-ok" and has **no
  drive toward the rare good region**. The posterior **variance** is exactly what pulls exploration into the
  unexplored-and-maybe-good corner. The asymmetry argues *for* keeping uncertainty in the loop, not against it.
- **It updates per bench, in closed form.** Sherman-Morrison rank-1 updates, no SGD step size, no replay buffer, O(d²)
  per observation — fits "learn during the tune run." LightGBM only retrains in batches between runs.
- **Uncertainty unifies the prior and the exploration bonus.** The same posterior that ranks (Thompson/LinUCB) can later
  replace the hand-set `ucb_c` and the `+∞`-unvisited hack — the natural graduation to a true PUCT-style `P(s,a)` term.
- **It persists and transfers.** The posterior `(θ̄, Σ)` keyed structurally is the *high-quality replacement for the
  heuristic*: a new kernel starts from the cross-run fit, not a hand formula.

LightGBM/GBT stays in reserve as the nonlinearity escape hatch (§9) if linear + interactions proves too weak — but it
loses the closed-form uncertainty, so it is not the default.

## Design

### 1. Core model — online Bayesian linear regression on log-latency

Per regime (see §6), maintain a Bayesian linear model `y ≈ θ·φ(x)` in the **Normal-Inverse-Gamma** conjugate form
(unknown noise σ², closed-form online updates, **Student-t** predictive — heavier tails when data is scarce → more
exploration early, auto-tightening as benches land).

- **Target** `y = −log(latency)`. Log compresses the dynamic range so a few catastrophic configs don't dominate the
  least-squares fit (load-bearing given the high variance), homogenizes the noise, and is monotone so ranking is
  unaffected. Higher `y` = faster = explore first.
- **Prior mean 0** (pure ridge — no heuristic mean). Justified by the fast-signal point above: a prior mean only earns
  its keep when data is scarce *and* per-bench SNR is low; here SNR is high, so cold-start is short.
- **State**: precision `Λ` (d×d) and info vector `b = Λμ`, plus NIG scalars `(a, β)` for σ². Sufficient statistics
  accumulate as `Λ += φφᵀ/σ²`, `b += φy/σ²`.

### 2. Features `φ(x)` — shape-relative, presence-aware

Available pre-materialization (same inputs `lazy_score` takes: `KernelShape` + `TileParams` + `compute_capability`) so
the prior never forces body materialization.

- **Shape-relative, not raw dims** — required for cross-kernel transfer. Use ratios: cells/thread, predicted occupancy,
  tile-to-problem ratio, smem/`smem_per_sm`, grid/`sm_count`, arithmetic intensity. Raw extents don't generalize across
  kernels of different sizes.
- **Knobs**: high-regret knobs as **main effects** (these move fast); **interaction terms only for the coupled pairs**
  the `knobs` command flags as high-interaction (data-driven feature selection — keeps d small, regularizes the cross
  terms). **Presence-aware** encoding (`has_BK ∈ {0,1}` + value-or-NaN) so partial states (some knobs fixed) and
  terminals share one feature space — needed because the search ranks partial states.
- **Hardware**: `cc_major/minor` + a `CUDA_DEVICE_SPECS` lookup (`sm_count`, `smem_per_sm`, `l2_size`,
  `mem_bandwidth_gb_s`, tensor-core gen). The keyed lookup is what lets a new GPU start warm.
- **Optional heuristic-as-feature**: include `TileOp.score(x)` as one input. If it's as low-quality as we believe, its
  learned weight decays toward 0 (harmless); if it has residual signal, it earns a small weight. No-regret way to not
  *trust* it while not *discarding* it.
- **Standardize** all features (running mean/var, or fixed from the DB). Interaction products blow up the condition
  number of `Λ` otherwise — this is what keeps Sherman-Morrison numerically stable, not optional.

For a *shared* cross-kernel posterior, fit log-latency with a **per-kernel intercept** (mixed-effects): the intercept
absorbs per-kernel scale, the slopes carry the transferable knob signal. Matches "ordering, not absolute".

### 3. Cold-start — zero-mean prior + DB warm-start + golden seeding

Two warm-start tiers on top of the zero-mean prior:

1. **DB posterior warm-start (structural prior).** Offline-fit `(θ̄, Σ)` on the whole `perf` DB for the regime, load it
   as the new kernel's prior. This learned posterior *is* the heuristic's replacement.
2. **Golden-neighbor seeding (first real measurements).** Seed the first benches with the **golden configs of the
   nearest kernels in shape-feature space**, then hand off to Thompson. This gives real, high-SNR labels at known-good
   points immediately, and pairs with the zero-mean prior: no bad heuristic bias, but good empirical starting points
   instead of wild first Thompson draws.

**Seed, don't anchor — and this is a correctness point, not a preference.** Two ways to use goldens:

- *Seeding* (chosen): **bench** the golden configs ourselves → they get real labels **in the current `-O1` ranking
  regime**, so the O1↔O3 reorder problem never bites.
- *Anchoring* (rejected): inject goldens as low-noise pseudo-observations / prior mean. This **trusts goldens as good in
  the current regime** — false under O1↔O3 reorder (goldens are `-O3` deployable winners; `tune` benches at `-O1`) and
  false when the new kernel's optimum differs from its neighbors'. Only safe if anchored with goldens measured at the
  *same* opt level.

Seeding **measures**; anchoring **trusts**. We measure.

### 4. Selection — Thompson sampling, two integration depths

- **Depth 1 (drop-in, build first).** The posterior just *orders unvisited siblings* — Thompson draw `θ̃ ~ N(θ̄,Σ)`,
  score the frontier, replace the `child.score` tiebreaker in `_ucb_key`. **Nothing else in `mcts.py` changes**;
  `+∞`-unvisited and `ucb_c` stay. Minimal blast radius; directly tests whether the ranking leverage is real.
- **Depth 2 (acquisition, only if depth-1 leverage is too weak).** Promote the posterior into the UCB arithmetic —
  LinUCB (`θ̄·φ + α·√(φᵀΣφ)`) or LinTS guides descent; `ucb_c` and the `+∞`-unvisited breadth hack go away. Bigger
  rewrite — earn it with depth-1 benches-to-best numbers first.

**Batched selection** (if benching is ever parallel): draw an **independent posterior sample per slot** (batched
Thompson) — do *not* draw once and take top-k (collapses exploration). But note `DEPLODOCK_GPU_LOCK` likely serializes
benching within a run, in which case selection is effectively sequential and single-sample is the natural mode.

### 5. Updates — Sherman-Morrison, mini-batch semantics

- **Single-sample** rank-1 Sherman-Morrison per bench (the serialized-GPU common case), with a **periodic Cholesky
  refactor of `Λ` every N steps** for numerical hygiene (incremental inverse updates drift; the refactor is the cheap
  fix and is where any re-regularization clamps live).
- **Mini-batches are statistically free for updates**: the posterior is a function of the accumulated sufficient
  statistics `(Σφφᵀ, Σφy)`, so per-sample / rank-k Woodbury / periodic refactor all give the **identical**,
  order-independent posterior. Batching is purely a compute choice. It matters in practice only for the **offline
  warm-start refit** over the whole DB and for genuine multi-GPU parallel benching.

### 6. Regularization & regime keying

Most of the regularization is free from the Bayesian formulation:

1. **The prior is ridge.** `Λ₀ = (σ²/τ²)·I` → ridge with `λ = σ²/τ²`. `τ²` controls how fast the model leaves the flat
   start; broad is fine given high SNR. Auto-annealing: heavily regularized early, likelihood takes over as data lands.
2. **σ² from data, not guessed.** `perf` stores `min/max/mean` over `n_samples` — a direct measurement-noise estimate.
   Feed it in (or let the NIG posterior infer it). σ² matters twice: ridge ratio *and* Thompson exploration scale.
3. **Feature selection = regularization.** Only the high-interaction knob pairs (`knobs` command) become cross-terms.
4. **Conditioning.** Standardized features (§2) keep `Λ` well-conditioned.
5. **Forgetting only cross-run.** `Λ ← γΛ` (γ<1) for the *persisted* model to handle driver/GPU drift; keep `γ=1`
   within a stationary single-GPU run.

**Regime key = `(compute_capability, nvcc_flags)`** — the same partitioning `perf` rows already use. O1 and O3 reorder
variants, so an O1-trained model and O3 data never share a posterior. `tune`'s `-O1` benches are a *ranking* signal;
this model ranks, it does not predict deployable latency.

### 7. Persistence

New table in `db.py`, keyed by regime + version:

```sql
CREATE TABLE IF NOT EXISTS prior_model (
    regime_key   TEXT NOT NULL,   -- digest(compute_capability, nvcc_flags)
    version      INTEGER NOT NULL,
    trained_at   TEXT NOT NULL,
    n_rows       INTEGER NOT NULL,
    theta_blob   BLOB NOT NULL,   -- posterior mean μ̄
    cov_blob     BLOB NOT NULL,   -- Σ (or Λ Cholesky) for Thompson draws
    nig_blob     BLOB NOT NULL,   -- (a, β) noise posterior
    scaler_blob  BLOB NOT NULL,   -- feature standardization stats
    feature_spec TEXT NOT NULL,   -- JSON: feature names + knob/interaction spec
    metrics      TEXT NOT NULL,   -- JSON: held-out regret@budget, NDCG@topq, calibration, golden-top-K
    PRIMARY KEY (regime_key, version)
);
```

`SearchDB.latest_prior(regime_key)` / `load_prior` / `write_prior` helpers. A `Prior` wrapper owns the loaded posterior
+ featurizer + scaler; `prior.thompson_scores(candidates, ctx)` returns ranking scores. Falls back to `TileOp.score`
when no model exists for the regime — day-one parity.

### 8. Lazy recompute in MCTS

Keep the version-tagged lazy scheme: `SearchNode` caches `(prior_value, prior_version)`; on selection, recompute iff the
registry's current version for the regime has advanced. Nodes never re-selected pay nothing. `_ucb_key` keeps shape
`(ucb_value, prior_value)` at depth 1 — only the prior *source* changes. (At depth 2 the key changes; defer.)

### 9. Escape hatch — nonlinearity

If linear + selected interactions can't separate the top quantile (check via the offline harness §10), escalate to an
incrementally-fit GBT or online RankNet pairwise-SGD. Cost: lose closed-form uncertainty → hand exploration back to UCB
(depth-1 only). Try linear first; it's the only option that keeps the whole loop closed-form.

## Validation

Two levels, sharply separated. Note the memory rule: greedy/learned picks are GPU-dependent, so anything asserting a
*pick* needs pinned regime — offline ranking validation is GPU-less, benches-to-best is a GPU/`perf`-marked test.

### 10. Offline harness (GPU-less, from the `perf` dump) — the cheap falsifier

Build this **first**; only proceed if it beats static `score`.

- **Fit** the zero-mean log-latency NIG model on `perf` rows for a regime (reuses the `knobs` command's join +
  interaction matrix). Optional heuristic-as-feature.
- **Group split by kernel** (`op_cache_key` group split, never row-shuffle — variants of one kernel leaking across
  train/test fakes transfer).
- **Metrics** (all on held-out groups):
  - **Regret-within-budget** — primary: rank held-out variants by the prior, report the best-found regret after B
    pseudo-benches. This is the quantity that converts to saved benches.
  - **NDCG@top-quantile / within-group Spearman** — smooth proxies over the upper slice (not top-1; §Context).
  - **Golden-top-K** — does the prior surface the kernel's golden config in its top-K? Direct, label-efficient.
  - **Leave-one-kernel-out (LOKO)** — for each kernel, train on all *others*, check it ranks this kernel's golden
    highly **cold**. Measures pure transfer / warm-start — the whole value proposition. Near-zero LOKO ⇒ memorizing,
    do not promote.
  - **Posterior calibration** — do held-out latencies fall in the 90% Student-t interval ~90% of the time? Validates
    the **uncertainty**, not just the mean; an overconfident posterior silently kills Thompson exploration and the
    mean-ranking metrics won't catch it.
- **Off-policy held-out coverage.** Validate on the broad exploration already in the DB from past full tune runs, *not*
  on a Thompson-selected trajectory (which is on-policy biased).
- **Regime match.** Validate the `-O1` model against `-O1`-best targets. Reserve `-O3` goldens for validating an
  `-O3`-trained model or an end-to-end "final pick == golden" check — never validate an O1 model against O3 goldens.
- **Promotion gate**: promote a new version only if it beats static `score` on held-out regret-within-budget *and* is
  calibrated. Else log the regression, keep the old version.

### 11. Online truth (GPU, `perf`-marked) — benches-to-best

The thing that matters. On target ops with wide knob spaces (Qwen matmul `(M=32,K=3584,N=18944)`; SDPA TinyLlama `s32`;
`matmul_add s128`), run `tune` with `--prior off` vs `--prior auto` (cold) vs `--prior auto` (warm cache + golden
seeding) against a snapshot DB; record `terminals_to_best` and `best_us`. Script: `scripts/bench_prior_reduction.py`.

- **Pass**: median `terminals_to_best` reduction ≥ 30% with no op regressing `best_us` > 10% (the prior must not trade
  quality for convergence speed).
- **Held-out variant**: exclude the target ops' rows from the fit (LOKO). Pass: median reduction ≥ 15%. Near-zero ⇒
  memorizing, do not ship.

## Files to modify

- `deplodock/compiler/pipeline/search/prior/` *(new package)* — `model.py` (NIG Bayesian-linear: Sherman-Morrison
  update, Thompson draw, Cholesky refactor), `features.py` (shape-relative + presence-aware knob + hardware featurizer,
  standardizer, interaction selection from the `knobs`-command matrix), `seed.py` (nearest-kernel golden seeding),
  `registry.py` (`Prior` wrapper + regime dispatch, heuristic fallback), `fit.py` (offline DB fit + group-split
  evaluation + promotion gate), `__init__.py`.
- `deplodock/compiler/pipeline/search/policy/mcts.py` — route the tiebreaker through `TuningSearch._prior_for(child)`
  (Thompson score), version-tagged lazy recompute on `SearchNode`, `observe()` feeds `(features, latency)` to the online
  update. Depth-2 LinUCB deferred.
- `deplodock/compiler/pipeline/search/db.py` — `prior_model` table; `latest_prior` / `load_prior` / `write_prior`.
- `deplodock/compiler/pipeline/pipeline.py` — online update + (optional) mid-run version bump after each terminal bench;
  pass the loaded `Prior` into `TuningSearch`.
- `deplodock/commands/tune.py` — `--prior {auto,off}` (default `auto`: learned if a `prior_model` row exists for the
  regime, else heuristic fallback; `off`: zero tiebreaker).
- `deplodock/commands/prior.py` *(new)* — `deplodock prior fit` / `prior eval` offline subcommands (bootstrap from an
  existing cache, report held-out regret-within-budget / NDCG@topq / golden-top-K / calibration vs the static-`score`
  baseline).
- `scripts/bench_prior_reduction.py` *(new)* — the §11 benches-to-best harness.
- Tests under `tests/compiler/search/` — featurizer determinism + presence-aware partial-state encoding; Sherman-Morrison
  vs batch-refactor give identical posteriors; group-split correctness; promotion gate rejects a worse model; calibration
  coverage on a toy fixture; lazy-recompute version invariants; golden seeding picks nearest-kernel configs.

## Build order

1. **Offline harness** (`prior fit` + §10 metrics) — the cheap falsifier. Fit zero-mean log-latency NIG on the `perf`
   dump, evaluate LOKO regret-within-budget + golden-top-K + calibration vs static `score`. **Stop here if it doesn't
   win.**
2. **Online path behind the tiebreaker** (depth 1) — wire Thompson scoring + Sherman-Morrison `observe`, golden seeding,
   DB warm-start. Measure §11 benches-to-best.
3. **LinUCB acquisition** (depth 2) — only if depth-1 tiebreaker leverage is too weak.

## Open questions

- Reward shape: `−log(latency)` vs a rank-transform per group — log keeps it Gaussian-NIG; rank-transform is fully
  scale-free but loses magnitude. Decide from calibration results in step 1.
- Does the per-kernel intercept (mixed-effects) transfer better than a global zero-mean fit? A/B in the offline harness.
- Outer fusion-fork prior: forks are deterministic today (one terminal), so the prior is inner-search-only for now.
  Revisit once `plans/structural-forks-in-two-level.md` makes forks branch.
