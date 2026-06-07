# Make every tuning knob explicit on emitted variants (OFF-value mechanism)

## Context

The learned `CatBoostPrior` featurizes a kernel variant from its knob dict (`knob.knob_features`) and **NaN-fills
absent knobs** (`catboost.py` `f.get(c, np.nan)`, `nan_mode="Min"`). That NaN bucket today conflates **two different
absences**:

- **not-yet-decided** — a partial-prefix row scored mid-descent (greedy `_path_knobs`, MCTS `_node_knobs`), where a knob
  legitimately hasn't been chosen yet, and
- **tier-unused** — a *complete* scalar/FMA leaf simply has no `WM`/`WN`/`MMA` (warp-tier-only), and a *complete* warp
  leaf has no `BM`/`BN`/`BR`/`FK` (scalar-only).

Because both collapse to NaN, the prior cannot tell "this knob doesn't apply to this kernel" from "this knob is still
open," which weakens its predictions (the original motivation behind commit `a01cdd74`, which only separated absent from
present-zero — not the two flavors of absent).

**Goal:** every *emitted variant* carries an explicit value for every declared knob. Tier-unused knobs get an explicit
OFF sentinel (`WM/WN=0`, `MMA="0"`, `BM/BN/BR/FK=0`); NaN then means *only* "not-yet-decided." Per the chosen approach,
this is done declaratively: each `Knob` declares its OFF value, and the **pipeline fills any unspecified declared knob
with that OFF value** on emitted variants (covering passes that decline to stamp, are skipped, or return no variants) —
replacing the current manual off-stamp discipline with one mechanism. Scope: **all planner leaves uniformly**
(matmul, reduce, pointwise).

## Design

### 1. `Knob.off` field + a shared fill helper (`knob.py`)

- Add `off: Any = _UNSET` to the `Knob` dataclass (`knob.py:57`). `_UNSET` is a module sentinel meaning "no OFF declared
  — this knob is expected to always be stamped by its pass" (universal knobs like `BN/BM/FM/FN/BK/SPLITK`).
- Add `def apply_off_defaults(knobs: dict, declared: Iterable[Knob]) -> None` (or pure-return variant): for each knob in
  `declared` with `off is not _UNSET` whose `name` is absent from `knobs`, set `knobs[name] = off`. Idempotent.
- Declare OFF on the conditional knobs:
  - Tier knobs (in `_enumeration.py`): `WM.off=0`, `WN.off=0`, `MMA.off="0"`, `BM.off=0`, `BN.off=0`, `BR.off=0`,
    `FK.off=0`. (`FM/FN/BK/SPLITK` stay `_UNSET` — always set by both tiers.) `"0"` is already MMA's scalar value
    (`mma_mode("0")==(False,None)`, `_mma_features("0")=={"MMA_tier":0.0}`).
  - Toggle/marker passes: `TMA.off=False`, `ASYNC_COPY.off=False`, `HOIST_COMPUTE.off=False`, `PAD_SMEM.off=False`,
    `PIPELINE_STAGES.off=False`, `WARPSPEC.off=False`, `NOATOMIC.off=False`, `RING.off=1`, `GROUP_M.off=1`,
    `VECTORIZE_LOADS.off=False`, `INTERLEAVE_LOADS.off=False`, `PERMUTE_LANES.off=False`.
  - `STAGE` (BINMASK) stays `_UNSET` — its all-zero off mask is width-dependent and the stage pass already stamps it with
    the correct width; a static OFF can't encode the width.

### 2. Pipeline completion at pass boundaries (`pipeline.py`)

- Compute, per `Pass`, its **declared knobs**: in `Pass.load` (`pipeline.py:170`) scan each rule module's `vars()` for
  `Knob` instances. This naturally captures *imported* knobs too (e.g. `010_partition_loops` imports `BN…MMA` from
  `_enumeration`, so they belong to the `lowering/tile` pass; `RING` from `040`, etc.). Store as `Pass.declared_knobs`.
- In the run loop, when a pass finishes and its realized variant op(s) advance to the next pass (the `Cursor.advance` /
  `is_last` boundary in `pipeline.py`), call `apply_off_defaults(op.knobs, pass.declared_knobs)`. This fills any of *that
  pass's* OFF-declared knobs the variant left unspecified — including a pass that was skipped or returned no variants.
  Scoping to the just-completed pass avoids prematurely stamping a *later* pass's knob (which would trip that pass's
  `if KNOB.name in op.knobs` idempotency guard).

### 3. Planner enumeration fill for tier knobs (`_enumeration.py`)

The prior is queried on **fork.knobs prefixes during the planner's own fork descent** (greedy `prior.mean_score`), and
trained on realized leaves. For the unused-vs-undecided distinction to actually reach the prior, the tier sentinels must
be in the **variant identity from enumeration** (fork.knobs → `lazy_score` → DB key → materialized `TileOp.knobs`), not
only added at the post-pass boundary. So: apply `apply_off_defaults(row, _PLANNER_KNOBS)` to each enumerated row in
`enumerate_cartesian` (both scalar and warp impls). The pass-boundary fill (step 2) then becomes a no-op for tier knobs
and a safety net for the rest.

### 4. Convert tier discriminators from presence/truthiness to value-based (the bulk of the work)

Once `MMA="0"` appears on scalar variants, every `"MMA" in knobs` / `knobs.get("MMA")` / `ATOM_REGISTRY[knobs["MMA"]]`
breaks (`"0"` is a truthy string; `ATOM_REGISTRY["0"]` raises `KeyError`). Add to `knob.py` (pure dict logic, no pipeline
imports — it's imported everywhere):

- `is_warp(knobs) -> bool`: `MMA` present and not in the falsy set `{"0","false","no","off",""}`.
- `mma_atom(knobs) -> str | None`: the atom-kind string, or `None` for the `"0"`/absent scalar case (value extraction).
- Move the MMA falsy/truthy sets into `knob.py` and have `_enumeration.mma_mode` delegate to a `knob.py` decoder (removes
  duplication, keeps `ATOM_REGISTRY` out of `knob.py`).

Convert call sites (replace presence/truthy with `is_warp`; replace `ATOM_REGISTRY[str(knobs["MMA"])]` with
`mma_atom(knobs)` + guard):

- **`compiler/ir/tile/ir.py`** (the scorer — runs ~40k×/kernel; CRITICAL): gates `1544`, `1574`; value lookups
  `1732-1733`, `1774-1775`, `1972-1973`, `2063-2067` (these `ATOM_REGISTRY[str(knobs["MMA"])]` would KeyError on `"0"`).
- **`010_partition_loops.py`**: MMA fork-tree `Level` lambda `283` (`() if scalar` via `is_warp`); warp/scalar build
  dispatch `if "MMA" in params` `866`. The `BR`/`(BM,BN)`/`(WM,WN)` `Level` lambdas (`284-286`) need no logic change —
  with completion their tier-foreign keys are uniformly `0` within a subtree and still single-value-collapse (verified in
  `fork.py:193`); simplify `.get(name,1)`→`p[name]` for clarity.
- **`020_stage_inputs.py:160`** (value consumer, not a flag): `atom_kind = mma_atom(root.op.knobs)` so scalar leaves get
  `None` (preserving the no-stage/partial-staging masks; `"0"` would wrongly force the MMA-only fully-staged path).
- **`015_gate_splitk_residual.py:71`**, **`017_atomic_free_splitk.py:281`**, **`040_use_ring_buffers.py:144`**,
  **`kernel/060_permute_lane_accesses.py:137`**: `if op.knobs.get("MMA")` / `if "MMA" in op.knobs` → `if is_warp(op.knobs)`
  (these also run on reduce kernels, which now carry `MMA="0"` under uniform completion).
- Leave `010_partition_loops.py:971` (`"FKWIN" in params`) as-is — `FKWIN`/`OVERHANG` are conditional sub-mode knobs,
  out of scope (no OFF; `OVERHANG` is list-valued).

### 5. Featurizer (`knob.py`)

- **Tier-gate `_tile_features`**: today warp rows omit `BM/BN` so `_tile_features` KeyErrors → `{}` (warp gets no scalar
  `D_*` features). After completion warp rows carry `BM=BN=0`, which would compute garbage `D_*`. Add `if is_warp(knobs):
  return {}` at the top to preserve current behavior.
- Keep the `MMA_tier` setdefault (`knob.py:384`) — still load-bearing for partial branch rows that lack `MMA`.
- Update `knob_features` / `catboost.py` `fit` docstrings: NaN now means **only** "not-yet-decided (partial prefix)";
  an OFF value (`0`/`"0"`/`False`) means "decided: unused/declined." Note the intended within-kernel branch-vs-leaf
  distinction (a scalar branch has `WM`=NaN, its leaf has `WM=0`).

### 6. Persisted state (`search/db.py`, prior reservoir)

- `op_cache_key` (`search/keys.py:53`) folds the knob dict into the `tile_op`/`kernel_op` digest, so sentinels shift
  those keys (the `perf` row is `CudaOp`-keyed on source+geometry, so measured latencies survive). **Bump
  `db.py:_SCHEMA_VERSION` 1→2** so the stale `lowering` replay table is dropped on first open; add a version-log note.
- Reservoir: old checkpoint rows lack sentinels (absent→NaN) while new rows carry `0` — mixing them reintroduces a
  NaN-vs-0 split on the very signal we're fixing. **Document `tune --clean`** as the supported rebuild path (consistent
  with prior commits).

## Files to modify

- `deplodock/compiler/pipeline/knob.py` — `Knob.off`, `apply_off_defaults`, `is_warp`/`mma_atom`, MMA decoder + falsy/
  truthy sets, `_tile_features` tier gate, docstrings.
- `deplodock/compiler/pipeline/pipeline.py` — `Pass.declared_knobs` (vars scan in `Pass.load`), pass-boundary fill.
- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` — declare tier-knob `off`s, enumeration fill,
  `mma_mode` delegate, docstrings (`13`, `181-188`).
- `deplodock/compiler/ir/tile/ir.py` — 7 MMA discriminator sites → `is_warp`/`mma_atom`.
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — `Level` lambda `283`, dispatch `866`.
- `020_stage_inputs.py`, `015_gate_splitk_residual.py`, `017_atomic_free_splitk.py`, `040_use_ring_buffers.py`,
  `kernel/060_permute_lane_accesses.py` — discriminator swaps; declare remaining toggle/marker `off`s in their passes.
- `deplodock/compiler/pipeline/search/db.py` — `_SCHEMA_VERSION` bump.
- `deplodock/compiler/pipeline/ARCHITECTURE.md` — knob-stamp-invariant prose + knob table (OFF column / sentinel notes).
- Tests (below).

## Reuse / do-not-touch

- **Goldens (`golden.py`)**: leave the static dicts unedited. `eval`/`diagnostics` match through the heuristic
  `THREAD_KNOBS`/`_WARP_KNOBS` projection (`heuristic.py:186`, `eval.py:320`), so completed leaves still match; the
  found/golden tables iterate `gold.keys()` (`eval.py:236`), so absent-in-golden tier knobs don't show as spurious
  mismatches. `find_golden_configs.py` pulls knobs off the realized op, so a future regen adds the sentinels
  automatically. (If `run.py:490` flags `0 != absent`, add a display filter that drops OFF-valued tier-foreign knobs.)
- Reuse the existing `_PLANNER_KNOBS` tuple (`_enumeration.py:158`) as the planner's declared set.
- Reuse the fork-tree single-value collapse (`fork.py:193`) — no change needed beyond the MMA `Level` lambda.

## Tests

- Extend `tests/compiler/passes/test_knob_stamp_invariant.py`: add `WM/WN/MMA` to the mandatory leaf set; assert a scalar
  matmul leaf carries `WM=0,WN=0,MMA="0"` and a warp leaf carries `BM=0,BN=0,BR=0,FK=0`.
- New unit tests in `tests/compiler/pipeline/test_knob.py`: `Knob.off` + `apply_off_defaults` fills only unspecified
  OFF-declared knobs; `is_warp`/`mma_atom` return scalar for `"0"`/absent and warp for a real atom.
- A pass-level test that a *skipped* toggle pass still yields its OFF value on the variant (pipeline fill).
- Pin-knob CUDA-assert tests must pin full knobs + `--target` (per `[[compile-pick-tests-need-target-pin]]`).

## Verification (end-to-end)

1. `make test` (esp. `tests/compiler/` with the xdist flags from CLAUDE.md) — discriminator/fill/featurizer unit + e2e
   invariant tests green; no `KeyError: '0'` from the scorer.
2. `deplodock compile -c "torch.matmul(torch.randn(512,512),torch.randn(512,512))" --ir cuda` (scalar) and an fp16 square
   (warp) — confirm the rendered leaf's `knobs` carry the full set with sentinels; no crash in scoring.
3. `deplodock eval golden` and `eval prior` — no spurious found/golden mismatches; picks unchanged vs main for the fp32
   goldens (the scorer re-rank risk from `ir.py` is neutralized by the value-based gates — verify ranks didn't move).
4. `deplodock tune --golden square.512 --clean` then `--golden square.512.fp16`, then `eval golden` — prior trains on
   sentinel-complete rows; confirm scalar pick stays sane and fp16 pick is non-degenerate.
5. Dump-diff control per `[[dump-diff-needs-noise-floor-control]]`: compare a change-run dump vs a main-vs-main control to
   confirm generated CUDA is byte-identical for a scalar kernel (sentinels are codegen no-ops).
