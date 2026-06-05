# Online learned prior — minimal probe on one reduction op

## Goal

Answer one question before building anything bigger:

> Within a **single** `tune` run on **one reduction op**, can an online Bayesian-linear Thompson prior learn to rank the
> reduction knobs well enough to reach the best variant in **fewer benches** than the static `TileOp.score` tiebreaker?

Purely online, in-memory, no DB, no cross-run transfer, no golden seeding. If this doesn't beat static `score` on one
op, the larger design isn't worth building.

## Scope

**In:** the **second stage** of the two-level search — the inner per-kernel **tile-op** tuning — on one reduction kernel
(`k_rms_norm`, or a softmax/`k_sdpa_reduce`). Its TileOp knob space, an in-memory model that orders unvisited siblings,
judged on **pick quality** (does it stop choosing silly variants) against a pure-UCB baseline.

**Out (deferred until this works):** the stage-1 fusion-fork prior, DB persistence, regime keying, cross-run warm-start,
golden seeding, shape-relative features / cross-kernel transfer, mixed-effects intercept, LinUCB acquisition. None of it
is needed to answer the question.

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
  tiebreaker is `0` (pure UCB). The static `child.candidate.score()` path is removed from the tuning search (step 1).
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

## Build order

1. **Nuke existing selection priors first (clean slate).** In the tuning search path, drop the DB greedy hint (`push`
   already does `del best`, `mcts.py:123`) and the static `TileOp.score` tiebreaker — `_ucb_key` becomes pure UCB1
   (`prior_value = 0`). Confirm tune still runs and converges. This isolates the variable so the only prior under test
   is the learned one; `--prior off` now means honest pure UCB.
2. **Add the learned prior + trajectory stats.** Tree-snapshot featurize + batch ridge fit, Thompson tiebreaker, the
   end-of-tune stats block. `--prior online`.
3. **Compare** `off` vs `online` on the silly-pick-rate contrast (and calibration / knob concentration) across seeds.

## Files

- `deplodock/compiler/pipeline/search/prior.py` *(new, single file)* — `OnlinePrior`: tree-snapshot featurizer +
  ridge/NIG batch fit, Thompson score, the TileOp reduction-knob featurizer, and the end-of-tune stats summarizer
  (silly-pick rate, calibration, knob concentration).
- `deplodock/compiler/pipeline/search/policy/mcts.py` — **step 1**: strip the static-`score` tiebreaker + DB hint from
  the tuning selection (pure UCB). **step 2**: route the tiebreaker through the model; mark dirty after each bench;
  record the benched-leaf trajectory; expose the live-tree node walk (nodes with ≥1 benched descendant + their
  `best_reward`) the fit consumes.
- `deplodock/commands/tune.py` — `--prior {off,online}` (default `off`); print the stats block at end of run.
- `tests/compiler/search/test_online_prior.py` — featurizer determinism + presence-aware partial-knob encoding; the
  batch ridge solve matches a hand-checked toy fit; value-of-position labels read `best_reward` correctly (a 2-leaf tree
  sharing an intermediate ancestor labels that ancestor with the better leaf); Thompson ordering deterministic under a
  fixed drawn `θ̃`.

## If it works

Then — and only then — promote toward the full design: DB-persisted posterior keyed by `(cc, nvcc_flags)`,
shape-relative features for cross-kernel transfer, golden-neighbor seeding, and the LinUCB acquisition. Captured
separately when we get there.
