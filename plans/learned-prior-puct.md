# Learned prior for `deplodock` selection — replace static + DB heuristics, ship by default

## Goal

**North star:** a trained prior model that *is* the search selection policy — shipped with the package, on by default
for `tune`, and the **only** prior in the tree. The static `TileOp.score`, the DB-greedy fork lookup (`_best_fork`), and
any deterministic **argmax / greedy pick** are **removed entirely**, not kept as fallbacks. We test the prior purely on
how it steers the search (Thompson sampling, no greedy shortcut). How `compile` / `run` should consume the prior —
argmax, a short prior-guided search, or something else — is deliberately deferred until we see how the prior does.

**De-risk first.** Before wiring a shipped model into every path, answer one question with a minimal in-memory probe:

> Within a **single** `tune` run on **one reduction op**, can an online Bayesian-linear Thompson prior learn to choose
> good tile-op knobs — i.e. stop picking silly variants — better than pure UCB?

If the probe fails on one op, the shipping goal isn't worth pursuing. If it passes, generalize → persist → ship.

## Scope

**Probe (milestones 1–2):** the **second stage** of the two-level search — inner per-kernel **tile-op** tuning — on one
reduction kernel (`k_rms_norm`, or a softmax/`k_sdpa_reduce`). In-memory model, single shape, single GPU, judged on
**pick quality** (does it stop choosing silly variants) vs pure UCB.

**Returns for shipping (milestones 3–4):** persistence + a shipped artifact, hardware / shape-relative features for
cross-GPU + cross-kernel generalization, default-on loading **for `tune`**. Deferred only until the probe proves the
idea.

**Out of scope entirely (for now):** any deterministic argmax / greedy prior pick, and how `compile` / `run` consume the
prior; the stage-1 fusion-fork prior (forks are deterministic today); golden seeding; LinUCB acquisition (depth-2).
Revisit after the prior proves out in search.

## Why reduction first

Reduction knobs (block size / threads, BR, split-K reduce factor, num warps, vectorization) swing latency hard, so the
run produces strong good/bad signal within a handful of benches — the regime where an online model can actually move.
Cooperative-reduce kernels keep their symbolic axis degenerate, so the knob space is small and self-contained: a clean
single-op testbed.

## Training data: leaf vs non-leaf

Real benches exist only for **leaves** — fully-specified TileParams, materialized + benched → `−log latency`. But the
prior ranks **siblings at every level** of the blockify fork tree (`BR → (BM,BN) → (FM,FN) → (BK,SPLITK)`), and those
siblings are mostly **non-leaf** partial-knob states with no direct measurement. Training on leaves alone asks the model
to rank a feature subspace (partial knobs) it never saw labeled.

So the label for any node is its **value-of-position = max reward over benched descendants** — the standard MCTS
bootstrap. `mcts.py` already maintains exactly this: max-reward backprop writes `node.best_reward` on every ancestor
(`mcts.py:76-84`). Training data is then just a walk over the live tree:

- **rows** = every node with ≥1 benched descendant: `(φ(node), node.best_reward)` — leaves *and* intermediate nodes.
- features `φ` are **presence-aware**, so a partial-knob node encodes its unset knobs as `has_K=0 / NaN`.
- **prediction targets** = the unvisited siblings (`visits==0`, no label yet) the prior must order.

The labels are **non-stationary**: `best_reward` only ever rises as new leaves bench under a node. A streaming rank-1
update would bake in stale labels — so the model **refits from a tree snapshot** (below) rather than updating
incrementally.

## Model (minimal)

In-memory Bayesian linear regression, discarded at end of run.

- **Target** `y = −log(latency_us_median)` at leaves, propagated to intermediate nodes as `best_reward` (max over
  descendants). Log so a few catastrophic configs don't capture the fit; monotone, so ranking is unaffected. Higher =
  faster = explore first.
- **Prior mean 0**, Gaussian (ridge `λ`). No heuristic mean. Noise `σ²` a small constant (or from leaf bench
  repeatability — `perf` exposes `min/max/mean` over `n_samples`).
- **Features** `φ(x)` = just the TileOp reduction knobs, **presence-aware** (`has_K`, value-or-NaN) since the fork tree
  fixes knobs progressively and the prior ranks partial states. log2-encode size knobs, ordinal for enums. Add the few
  coupled-pair interactions if a knobs-command sweep on this op flags any; else main effects only. No cross-kernel/shape
  features — single shape.
- **Fit**: periodic **batch ridge/NIG solve over the live-tree snapshot** (every K benches, or lazily before a selection
  that needs the prior): `θ̄ = (ΦᵀΦ + λI)⁻¹ Φᵀy`, `Σ = σ²(ΦᵀΦ + λI)⁻¹`. The tree is small (≤ hundreds of nodes, d ~
  tens), so the solve is trivial, and re-reading `best_reward` each refit sidesteps the non-stationary-label problem.
  (Streaming Sherman-Morrison is unnecessary here and wrong for moving labels.) Optionally weight rows by `log(1+visits)`
  — better-explored nodes have a tighter, more trustworthy max.
- **Selection**: Thompson — draw `θ̃ ~ N(θ̄, Σ)`, score the unvisited siblings, use as the tiebreaker. Cold start (no
  benched descendants anywhere) → broad prior → first picks ~random, fine here because the high-SNR knobs separate fast.

## Integration (drop-in)

- `TuningSearch` holds one in-memory model for the run. The `score` tiebreaker (`mcts.py:42-54`, consumed in `_ucb_key`
  at `mcts.py:147-157`) routes through `model.thompson_score(child)` when `--prior online`; with `--prior off` the
  tiebreaker is `0` (pure UCB). The static `child.candidate.score()` path is removed from the tuning search (M1).
- After each bench, mark the model dirty. Before a selection that consults the prior (and the tree has grown ≥ K since
  the last fit), **refit from the live-tree snapshot**: walk nodes with ≥1 benched descendant, featurize each, read
  `best_reward` as the label, solve the ridge system. No per-bench streaming update — the refit reads current labels.
- `_ucb_key` keeps shape `(ucb_value, prior_value)` — only the tiebreaker source changes. `+∞`-unvisited and `ucb_c`
  untouched.

## Validation — watch the picks, not the clock

Wall-clock / benches-to-best alone is too coarse and noisy. The real question is **which nodes the search chooses**: once
the prior has data, it should stop descending into silly subtrees. Instrument selection and judge from the trajectory.

Record every benched leaf `(knobs, latency)` in order. At end of tune, print a stats block — the simple sanity check:

- **best** latency + the bench index it was found at + its knobs.
- **silly-pick rate**, split **warmup vs post-warmup**: fraction of benched leaves ≥ 2× the run's best ("silly"). With a
  working prior this drops sharply warmup → rest; with `--prior off` (pure UCB) it stays ~flat. The *contrast* is the
  evidence, and it controls for the op. (Forced breadth means some silly leaves are unavoidable — the prior's win is not
  *deepening* silly subtrees, so fewer get benched overall.)
- **prior self-calibration**: Spearman between the prior's predicted score and the measured reward over its post-warmup
  picks. Positive and rising = the prior learned the knob structure.
- **knob concentration**: per high-level knob, the bench index after which picks settle to one value — shows the search
  locking onto the good region instead of churning.

Run a few seeds (vary the run, not `Math.random`) — Thompson cold-start is stochastic; report the trajectory across
seeds, not a single run. Benches-to-best stays a secondary number, not the headline.

## Milestones

1. **Total nuke — remove every existing selection prior.** Strip the static `TileOp.score` tiebreaker from selection
   (all paths) *and* the DB-greedy fork lookup `_best_fork` from `compile` / `run` (`pipeline.py:697-762`); `push`
   already drops the tuning DB hint (`mcts.py:123`). After this: `tune` selection is pure UCB1 (`prior_value = 0`) and
   greedy `compile` / `run` falls back to the rule default — and **stays there**, with no argmax replacement; the prior
   is a search-only policy for now. This regresses compile pick quality — accepted; reconnecting `compile` / `run` is a
   later question. Confirm the suite still runs.
2. **Online probe** (one reduction op, in-memory) — tree-snapshot featurize + batch ridge fit + Thompson tiebreaker +
   the end-of-tune stats block. Validate the silly-pick-rate contrast vs pure UCB across seeds. **Gate: pass or stop.**
3. **Generalize + persist.** Bring back hardware + shape-relative features; train across ops / runs; persist `(θ̄, Σ,
   scaler, feature_spec, regime_key)` to a small artifact. Validate cross-kernel transfer (leave-one-kernel-out) and
   posterior calibration.
4. **Ship + default.** Bundle a trained artifact in the package; `tune` Thompson-samples it **by default** (`--prior off`
   for pure UCB; a user file / DB override loads a locally-retrained model). No argmax, no greedy shortcut — the prior is
   judged purely as the search policy. `compile` / `run` consumption is a separate question opened only once the prior's
   search quality is established.

## Files

- `deplodock/compiler/pipeline/search/policy/mcts.py` — **M1**: strip the static-`score` tiebreaker (pure UCB1). **M2**:
  route the tiebreaker through the model; mark dirty after each bench; record the benched-leaf trajectory; expose the
  live-tree node walk (nodes with ≥1 benched descendant + their `best_reward`) the fit consumes.
- `deplodock/compiler/pipeline/pipeline.py` — **M1**: remove the `_best_fork` DB-greedy fork selection
  (`pipeline.py:697-762`) from `compile` / `run`; greedy then uses the rule default. **No argmax rewire** — `compile` /
  `run` prior consumption is deferred.
- `deplodock/compiler/pipeline/search/prior.py` *(new)* — `OnlinePrior`: tree-snapshot featurizer + ridge/NIG batch fit,
  Thompson score (search only — no argmax mode), TileOp knob featurizer, end-of-tune stats summarizer (silly-pick rate,
  calibration, knob concentration). **M3**: persistence (load/save artifact) + hardware / shape-relative features.
- `deplodock/commands/tune.py` — `--prior {off,online}` (default `off` for the probe; **M4**: default loads the shipped
  model); print the stats block at end of run.
- shipped artifact (**M4**) — a small committed `prior_model` file (θ̄, Σ, scaler, feature_spec, regime_key) loaded by
  default.
- `tests/compiler/search/test_online_prior.py` — featurizer determinism + presence-aware partial-knob encoding; the
  batch ridge solve matches a hand-checked toy fit; value-of-position labels read `best_reward` correctly (a 2-leaf tree
  sharing an intermediate ancestor labels that ancestor with the better leaf); Thompson ordering deterministic under a
  fixed drawn `θ̃`.

## Risks / open questions

- **Broken tune→compile handoff.** Nuking `_best_fork` removes how a tuned winner reaches `compile` / `run` — they now
  always use the rule default, ignoring any tuning, and we deliberately add no argmax to reconnect them. Accepted for the
  experiment: we are measuring the prior as a *search* policy, not deploying. Reconnecting `compile` / `run` to the prior
  (argmax, a short prior-guided search, or a measured-best replay fast path) is the explicit follow-up once the prior
  proves out — and is itself a design question, since argmax of a *prediction* would discard measured ground truth.
- **Regime coverage of the shipped artifact.** A bundled model trained on our GPUs / opt-level must generalize via the
  hardware features (M3), or it mis-ranks on an unseen card. Validate cross-GPU before defaulting it on.
- **Reduction op choice for the probe** — start `k_rms_norm` (small, clean knob space); stress with softmax /
  `k_sdpa_reduce` (wider, more knob coupling).
