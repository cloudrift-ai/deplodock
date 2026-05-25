# Learned-prior PUCT for `deplodock tune`

## Context

Today `TuningSearch` (SP-MCTS) breaks ties among unvisited siblings using `TileOp.score(ctx)` — a hand-tuned heuristic
defined at `deplodock/compiler/ir/tile/ir.py:608-700`. The heuristic is rank-only and feeds the tiebreaker slot in
`_ucb_key()` at `deplodock/compiler/pipeline/search/policy/mcts.py:127-137`. Across runs we accumulate measured
`(kernel, knobs, hardware) → latency` rows in `~/.cache/deplodock/autotune.db` (`perf` and `lowering` tables in
`deplodock/compiler/pipeline/search/db.py:90-150`); the heuristic ignores all of this signal.

We want to replace the hand-tuned `score()` with a **learned LightGBM `lambdarank` prior** fit on the SQLite cache. The
prior should:

1. Improve over runs (more data → better ranking → fewer wasted rollouts).
2. Generalize across GPUs via numeric hardware features (not a categorical GPU id) so a new GPU starts warm.
3. Be retrained periodically — triggered when the cache grows ≥10% since the last fit, or when the prior's calibration
   drifts.
4. Apply **lazily**: each `SearchNode` caches `(prior_value, model_version)`; if the global model version has advanced
   since the cached value was computed, the node re-scores on next selection. Nodes that are never re-selected pay
   nothing.

## Design

### 1. Prior models (one per dialect)

The candidate types at each search level are structurally different — `LoopOp` decisions (fusion, loop tiling) share
almost no feature schema with `TileOp` knobs (BM/BN/BK, num_warps, stages) or `KernelOp` lowering choices. We train and
persist **three separate boosters**: `prior_loop`, `prior_tile`, `prior_kernel`. A single `PriorRegistry` dispatches on
the candidate's dialect to the matching model. Each dialect has its own featurizer, its own version counter, and its
own retrain trigger (see §4).

For each dialect:

- **Learner**: `lightgbm.LGBMRanker(objective="lambdarank")`.
- **Group**: rows sharing `(parent_op_signature, depth_in_chain, hardware_tag)` form one ranking group — the unit of
  "siblings PUCT has to rank against each other." Depth-in-chain matters because the same kernel appears at multiple
  pipeline depths with progressively-more knobs fixed, and siblings always share a depth.
- **Label and training-data extraction**. `perf` rows only exist for *terminals* (fully-specified lowerings that were
  benched), but PUCT asks the prior to rank *partial* states with only some knobs fixed. Training only on terminals
  puts inference in a feature subspace the model never saw. Instead, we use **value-of-position labels via ancestor
  unrolling**:

  1. For each terminal in `perf`, walk its lowering chain back through `lowering(parent_key → child_key)` until the
     dialect root.
  2. Emit a training row at *every* intermediate state `(parent_op, knobs_so_far, hardware, depth)` along the chain.
  3. Label = the best terminal latency reachable from that state (max over descendants in the chain DAG). Standard MCTS
     bootstrapping — "value of this state assuming optimal downstream play."

  Filter rows to the dialect being trained. The extraction is a single SQL pass: recursive CTE on `lowering` joined to
  `perf`, aggregating `MIN(latency_us_median)` per intermediate state.
- **Features** (all numeric / one-hot, no sequence input):
  - *Parent op signature* (tabular fingerprint from the dialect's parent op body): op-kind histogram, in/out shape stats
    (rank, log-bucketed extents, dtype one-hots), loop-nest depth, reduction count, broadcast count, signature hash
    bucket (categorical).
  - *Knobs* from `lowering.knobs` JSON — schema is dialect-specific (LoopOp fusion choices vs TileOp BM/BN/BK/num_warps
    vs KernelOp lowering knobs). **Presence-aware encoding**: each knob becomes *two* features — a presence flag
    (`has_BK ∈ {0,1}`) and its value (`BK_log2`, or `NaN` when absent). LightGBM handles `NaN` natively and learns
    splits like "if `has_BK==0` (early state, no BK fixed yet) go left, else split on `BK_log2`." This makes partial
    and full knob sets share a single feature space, which is required for the value-of-position labels above to be
    usable at inference.
  - *Depth-in-chain* as an explicit integer feature, so the model can condition on pipeline progress directly rather
    than inferring it from presence-flag patterns.
  - *Hardware features* derived from `Context.compute_capability` and `cuda` device props — `sm_count`, `smem_per_sm`,
    `l2_size`, `mem_bandwidth_gb_s`, `cc_major/minor`, tensor-core gen flags, plus derived ratios
    (`tile_smem_bytes / smem_per_sm`, `grid_size / sm_count`). Shared across all three models.
- **Output**: pre-sigmoid relevance score, used as the tiebreaker tuple element in `_ucb_key` (replacing
  `child.score`). Rank-only — absolute value irrelevant.

### 1b. Train/test split and evaluation

Group-aware split — never shuffle rows, since the same `(parent, knobs)` pair appearing on both sides leaks measurement
noise and inflates NDCG. PUCT faces unseen kernels at inference time; evaluation must reflect that.

- **Split**: `GroupKFold(n_splits=5)` keyed on `parent_op_signature_hash` (single 80/20 group split for CI cheapness;
  full k=5 cross-val for the offline `deplodock prior fit` command).
- **Metrics reported per fit**: `NDCG@1`, `NDCG@5`, within-group Spearman ρ. NDCG@1 is the primary metric — it answers
  "would the prior put the actual winner first among unvisited siblings?"
- **Heuristic baseline**: compute the same metrics for the hand-tuned `TileOp.score` heuristic on the *same held-out
  groups*. A new model is only promoted (i.e., its row written to `prior_model` and picked up by `latest_prior_version`)
  if it beats the heuristic on held-out NDCG@1. Otherwise the run logs the regression and keeps the prior version
  unchanged.
- **Persistence**: extend the `prior_model` schema with `ndcg_at_1`, `ndcg_at_5`, `spearman`, `heuristic_ndcg_at_1`,
  `n_groups_train`, `n_groups_eval` so model quality is inspectable across versions.

### 2. Persistence: model artifact + version

- New table in `deplodock/compiler/pipeline/search/db.py`, scoped by dialect:
  ```sql
  CREATE TABLE IF NOT EXISTS prior_model (
      dialect              TEXT NOT NULL,             -- 'loop' | 'tile' | 'kernel'
      version              INTEGER NOT NULL,          -- monotonic per dialect
      trained_at           TEXT NOT NULL,
      n_rows               INTEGER NOT NULL,          -- rows used when fit
      n_groups_train       INTEGER NOT NULL,
      n_groups_eval        INTEGER NOT NULL,
      ndcg_at_1            REAL NOT NULL,
      ndcg_at_5            REAL NOT NULL,
      spearman             REAL NOT NULL,
      heuristic_ndcg_at_1  REAL NOT NULL,             -- baseline gap
      model_blob           BLOB NOT NULL,             -- lightgbm.Booster.model_to_string()
      feature_spec         TEXT NOT NULL,             -- JSON: feature names + categorical list
      PRIMARY KEY (dialect, version)
  );
  ```
- `SearchDB.latest_prior_version(dialect) -> int | None` and `load_prior(dialect, version) -> Prior` helpers.
- A `Prior` wrapper owns the loaded booster + featurizer for one dialect; `prior.score(candidate, ctx) -> float`.
- A `PriorRegistry` holds the three boosters and dispatches `score(candidate, ctx)` on the candidate's dialect, falling
  back to `candidate.score()` (heuristic) for dialects with no trained model yet.

### 3. Lazy recomputation in MCTS

Modify `SearchNode` (`mcts.py:32-43`):

```python
@dataclass
class SearchNode:
    ...
    _prior_value: float | None = field(default=None, repr=False)
    _prior_dialect: str | None = field(default=None, repr=False)
    _prior_version: int | None = field(default=None, repr=False)
```

The `score` property becomes a method on `TuningSearch` (since prior needs the loaded booster):
`_prior_for(child) -> float`. Cache hit when `(child._prior_dialect, child._prior_version)` matches the registry's
current `(dialect, version)` for the candidate's dialect; otherwise recompute and update all three fields. Because the
key is per-dialect, retraining one model never forces recompute for nodes whose candidates belong to a different
dialect. Fallback to `child.candidate.score()` (the existing heuristic) when no prior is loaded for the candidate's
dialect — gives day-one parity.

`_ucb_key` keeps shape `(ucb_value, prior_value)`; only the prior source changes.

### 4. Retrain triggers

A new `PriorTrainer` (`deplodock/compiler/pipeline/search/prior/trainer.py`) owns retraining **per dialect** (loop /
tile / kernel). Triggers fire independently — tile retrains far more often than loop, which is correct.

- `maybe_retrain(db, mcts_state) -> list[str]` — returns the list of dialects whose model version was bumped.
  Per-dialect conditions (OR):
  - **Row growth**: `current_dialect_rows / rows_at_last_fit_for_dialect >= 1.10`, with a floor of `>= 200` new rows to
    avoid thrashing early.
  - **Calibration drift**: rolling NDCG@1 of prior predictions vs measured rank over the last K terminals *for that
    dialect* drops below `0.6`. The trainer keeps one rolling buffer per dialect; `TuningSearch.observe()` feeds it
    `(dialect, predicted_rank, measured_reward)` per terminal.
  - **Hard cap**: after `terminals_since_retrain[dialect] >= N_forced` (default 500), retrain regardless.

- **Promotion gate**: after fitting on the training groups, evaluate NDCG@1 on the held-out groups and compare to the
  heuristic on the same groups. Promote (write the new `prior_model` row) only if `learned_ndcg_at_1 >
  heuristic_ndcg_at_1`. Otherwise log the regression with the metrics and keep the previous version.

- Retrain writes a new row to `prior_model` (when promoted); `TuningSearch` reads
  `latest_prior_version(dialect)` between rollouts. Because node priors are version-tagged per dialect, in-flight tree
  state stays consistent — stale nodes recompute on next visit; nodes never revisited cost nothing.

Called from the `Pipeline.tune()` loop body (`deplodock/compiler/pipeline/pipeline.py:466-526`) after `_bench_terminal`.

### 5. Featurizers

Three things to featurize per training/inference row: the **parent Op** (its structural fingerprint), the **knobs**
fixed so far (dialect-specific schema), and the **hardware context**. Op-structural features and hardware features are
dialect-agnostic — only knobs need per-dialect treatment.

**Unified Op featurizer.** Every Op (LoopOp, TileOp, KernelOp) shares the same `body / inputs / outputs` shape after
the BodyOp unification (PR #145), so a single function walks any Op and emits the same column set regardless of which
dialect it lives in:

```python
# deplodock/compiler/pipeline/search/prior/op.py
def op_features(op: Op) -> dict[str, float]:
    """Dialect-agnostic structural features. Same columns whether op is a
    LoopOp / TileOp / KernelOp — they share Op.body / .inputs / .outputs.
    Used by all three per-dialect models."""
    return {
        # input/output tensor stats
        "n_inputs": ..., "n_outputs": ...,
        "in_rank_max": ..., "in_extent_log2_max": ...,
        # ... dtype one-hots (fp16/bf16/fp32/i32/...)
        # body fingerprint — recursive walk
        **op_kind_histogram(op),               # counts of MatMul/Add/Mul/Exp/Reduce/...
        "body_depth": ...,                     # max loop-nest depth in body
        "n_reductions": ..., "n_broadcasts": ...,
        "signature_hash_bucket": ...,          # categorical: hash(canonical_pretty) % N
    }
```

The op-kind histogram is itself dialect-agnostic — it counts whatever `Op` subclasses appear in `body`. A new dialect
or new op kind just shows up as a new column in the histogram (existing column set extends, models retrained on the
new schema pick it up).

**Hardware features come from Context, unified with `structural_key`.** `Context.structural_key()` today
(`deplodock/compiler/context.py:76-88`) hand-picks the codegen-affecting fields. The prior must consume the *same* set
of fields — otherwise the cache key and the model drift as Context grows (forced TMA, splitk overrides, etc.). Single
source of truth:

```python
def codegen_view(self) -> dict[str, Any]:
    """Codegen-affecting fields, in a stable order. Single source of truth
    for both ``structural_key`` (digested) and prior featurization."""
    return {
        "compute_capability": self.compute_capability,
        # extend as forced-TMA / splitk-overrides land
    }
```

- `structural_key()` becomes `digest("Context", *sorted(self.codegen_view().items()))` — same key as today.
- `hardware_features(ctx)` reads `ctx.codegen_view()` and materializes numeric features: cc fields go in as-is, plus
  looked-up specs from a `CUDA_DEVICE_SPECS` table keyed by `cc_major.cc_minor` (sm_count, smem_per_sm, l2_size,
  mem_bandwidth_gb_s, tensor_core_gen flags). The keyed lookup is what lets the prior generalize cross-GPU.
- A parity test asserts `codegen_view()`'s key set matches what `hardware_features` consumes.

**Unified knob featurizer.** Knobs are already a uniform key-value dict in `lowering.knobs` JSON — same shape across
all dialects, just different key sets. One function handles every dialect:

```python
# deplodock/compiler/pipeline/search/prior/knobs.py
def knob_features(knobs: dict[str, Any], spec: KnobSpec) -> dict[str, float]:
    """Presence-aware encoding for every knob in ``spec``. Dialect-agnostic
    — the spec is just the union of knob keys observed in the training
    rows for the dialect being fit."""
    out: dict[str, float] = {}
    for name, meta in spec.items():
        if name in knobs:
            out[f"has_{name}"] = 1.0
            out[name] = meta.encode(knobs[name])   # log2 for sizes, raw for counts, ordinal for enums
        else:
            out[f"has_{name}"] = 0.0
            out[name] = float("nan")
    return out
```

The encoding (log2 vs raw vs ordinal vs categorical) is a property of the Knob, not of the featurizer. Today the
encoding can be inferred heuristically (int → raw, power-of-two int → log2, str → ordinal hash); when richer typing is
needed, extend the `Knob` class (e.g. `class Knob: name; kind: Literal["count", "log2_size", "enum"]; choices: ...`)
and the featurizer reads `Knob.kind`. No per-dialect branching.

`KnobSpec` for a dialect is the union of all knob keys ever observed in `lowering.knobs` for that dialect's rows,
discovered during training-data extraction and persisted alongside the model in `prior_model.feature_spec`. New knobs
appearing in future cache rows trigger a retrain (row-growth signal), and the new spec is captured then.

**Top-level `featurize(parent_op, knobs, ctx, spec)`** concatenates `op_features(parent_op)` +
`knob_features(knobs, spec)` + `hardware_features(ctx)` + `{"depth_in_chain": …}` into a numpy row in the spec order
persisted in `prior_model.feature_spec`. No dialect-specific code path — only the loaded `feature_spec` (per-dialect)
distinguishes models.

All prior-related code lives in a single `search/prior/` package:

```
deplodock/compiler/pipeline/search/prior/
    __init__.py    # public API: Prior, PriorRegistry, PriorTrainer, featurize
    registry.py    # Prior (loaded booster + spec), PriorRegistry (per-dialect dispatch)
    trainer.py     # PriorTrainer, maybe_retrain, group-aware split, promotion gate
    features.py    # featurize(parent_op, knobs, ctx, spec) concatenator + KnobSpec discovery
    op.py          # op_features(op)              — dialect-agnostic
    knobs.py       # knob_features(knobs, spec)   — dialect-agnostic
    hardware.py    # hardware_features(ctx)       — dialect-agnostic
```

### 6. CLI surface

- `deplodock tune --prior {auto,off,heuristic}` (default `auto`):
  - `auto`: use latest learned prior if `prior_model` table has a row, else fall back to heuristic.
  - `off`: pure UCB1 with no tiebreaker (`prior_value = 0`).
  - `heuristic`: force `TileOp.score`.
- `deplodock prior fit [--db PATH] [--dialect {loop,tile,kernel,all}] [--cv-folds N]` — one-shot offline retrain that
  ignores the trigger logic; defaults to `--dialect all --cv-folds 5`. Prints per-dialect held-out NDCG@1/NDCG@5/Spearman
  and the heuristic baseline gap, then writes promoted models. Useful for bootstrapping from an existing cache.
- `deplodock prior eval [--db PATH] [--dialect ...]` — load the latest promoted model per dialect and re-report
  held-out metrics without retraining; useful for inspecting model drift over time.

## Files to modify

- `deplodock/compiler/context.py` — add `Context.codegen_view()` helper; rewrite `structural_key()` to digest it so the
  cache-key contract and prior featurization share a single source of truth.
- `deplodock/compiler/pipeline/search/policy/mcts.py` — add `_prior_value/_prior_version` to `SearchNode`; route
  tiebreaker through `TuningSearch._prior_for(child)`; expose `observe_prediction(child, reward)` hook for calibration
  tracking.
- `deplodock/compiler/pipeline/search/db.py` — add `prior_model` table to `_SCHEMA`; add `latest_prior_version`,
  `load_prior_blob`, `write_prior_model` methods.
- `deplodock/compiler/pipeline/search/prior/` *(new package, all prior code lives here)*:
  - `registry.py` — `Prior` (loaded booster + feature_spec), `PriorRegistry` (per-dialect dispatch).
  - `trainer.py` — `PriorTrainer`, `maybe_retrain` (per-dialect triggers), group-aware split + held-out evaluation +
    heuristic-baseline promotion gate.
  - `features.py` — top-level `featurize(parent_op, knobs, ctx, spec)` concatenator + `KnobSpec` discovery from cache.
  - `op.py`, `knobs.py`, `hardware.py` — dialect-agnostic sub-featurizers (Op-structural, knob presence-encoding,
    hardware lookup).
  - `__init__.py` — re-exports the public API.

  The only per-dialect artifact is the `feature_spec` JSON persisted in `prior_model`.
- `deplodock/compiler/pipeline/pipeline.py` — call `prior_trainer.maybe_retrain(db, search)` in the tune loop after each
  bench; pass loaded `Prior` into `TuningSearch` (via constructor or `set_prior()`).
- `deplodock/commands/tune.py` — `--prior` flag; load latest prior from DB on startup.
- `deplodock/commands/prior.py` *(new)* — `deplodock prior fit` subcommand; register in CLI root.
- Tests under `tests/compiler/search/` — unit tests for featurizer determinism, lazy recomputation invariants, retrain
  trigger conditions; integration test that runs `tune` on a tiny kernel with a forced retrain mid-run and asserts
  consistency.

## Verification

1. **Unit**: `./venv/bin/pytest tests/compiler/search/test_prior.py -v` — covers unified `op_features` determinism
   across dialects (same Op subclass passed at LoopOp/TileOp/KernelOp wrapping levels yields the same structural
   columns), unified `knob_features` determinism (a TileOp knob set and a LoopOp knob set both featurize through the
   same function, each against its own `KnobSpec`), group-aware split correctness (no group appears in both train and eval), held-out NDCG
   computation matches a hand-checked toy case, heuristic-baseline promotion gate rejects a worse model,
   `SearchNode` lazy recompute (set `(dialect, version)`, assert recompute happens; bump version for one dialect,
   assert nodes of other dialects are not invalidated), **`Context.codegen_view()` parity** (the helper's key set
   matches what `hardware_features(ctx)` consumes; `structural_key()` digest is byte-stable across reordering of
   `codegen_view()` insertion order), **ancestor-unrolling extraction** (a fixture DB with two terminals sharing an
   intermediate ancestor produces the expected per-depth groups and best-of labels), and **presence-aware knob
   encoding** (a partial knob set featurizes with `NaN` in the unset slots and the same column order as a terminal).
2. **Integration**: `./venv/bin/pytest tests/compiler/search/test_tune_prior.py -v -p no:randomly` — small matmul tune
   run with `--prior heuristic` vs `--prior auto` (cold) vs `--prior auto` (warm cache); assert warm-cache run reaches
   the known optimum in fewer terminals than cold/heuristic.
3. **End-to-end node-expansion reduction** (primary success metric). The whole point of the prior is to make MCTS
   expand fewer terminals to reach the optimum. Measure it directly on operations with non-trivial search spaces:

   - **Target ops** (each has a wide TileOp knob space that the current heuristic only partially constrains):
     - Qwen-shape matmul `(M=32, K=3584, N=18944)` — the shape the existing `TileOp.score` was hand-tuned to avoid
       blowing up on (see docstring at `deplodock/compiler/ir/tile/ir.py:608-700`).
     - SDPA TinyLlama `s32` — non-trivial fused softmax+matmul knob space; the 2026-05-15 autotune sweep
       saw this regress from 8.08x → 4.17x, indicating real sensitivity to knob choice.
     - `matmul_add s128` — already flagged in the autotune-failures memory as a wedged case under the heuristic.

   - **Procedure** (script at `scripts/bench_prior_reduction.py`):
     1. Snapshot the current `~/.cache/deplodock/autotune.db` to a fresh DB.
     2. For each target op, run `deplodock tune --prior heuristic --patience 100` and record `terminals_to_best`
        (number of bench-evaluated terminals before the patience window closed on the best variant).
     3. Run `deplodock prior fit --db <snapshot>` to fit the learned prior from cache rows.
     4. For each target op, run `deplodock tune --prior auto --patience 100` against the *same* snapshot and record
        `terminals_to_best`.
     5. Report a table `(op, terminals_heuristic, terminals_prior, reduction_pct, best_us_heuristic, best_us_prior)`.

   - **Pass criteria**: median `reduction_pct ≥ 30%` across the target ops, with no individual op regressing more than
     10% on `best_us` (the prior must not trade quality for speed of convergence). Numbers persisted to
     `bench_results/prior_reduction.csv` so we can chart improvement across model versions.

   - **Held-out variant**: repeat (3.3) with the target ops' rows *excluded* from the training set (group-level
     leave-one-out by `parent_op_signature_hash`). Pass criteria: median `reduction_pct ≥ 15%`. This is the honest
     test that the prior generalizes — if the held-out reduction is near zero, the prior is memorizing rather than
     learning structure, and we should not promote it.

4. **Smoke**: `deplodock prior fit --db ~/.cache/deplodock/autotune.db` then
   `deplodock tune --code "torch.nn.Linear(2048, 2048)(torch.randn(32, 2048).cuda())"` — confirm the run loads the
   prior (stderr line) and that terminal count to first record is lower than the heuristic baseline on the same
   patience.
5. **Full suite**: `make test` and `make lint` before commit.
6. **Cross-GPU sanity**: train prior on sm_120 cache rows only, predict on synthetic sm_90 hardware features, confirm
   ranker still produces sensible relative order on a fixed `(kernel, knobs)` sweep (no NaN, monotone with knob
   sweeps that should monotonically improve).
