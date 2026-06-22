# Drop the `OVERHANG` knob → split per-role structural masked features

Replace the `OVERHANG` tuning knob with **structural (`S_`-prefixed) features derived from the realized body**,
split per masked axis role: `S_masked_m`, `S_masked_n`, `S_masked_k`. The guard codegen does not change — only the
recorded identity does.

## Why

`OVERHANG` is registered/stamped as a *tuning* knob, but the search never *chooses* it: both producers compute it
deterministically from `(axis.extent, tile geometry)`. Masking is a **consequence** of the shape/tile pairing, not a
free search dimension — so its correct home is a structural feature, the same bucket as `S_ext_n_symbolic_axis`. The
only behavioral read of its value today is `knob.py:550` (`len(...)` → `D_neg_overhang`); everything else is
list-vs-tuple plumbing that exists *solely* because a tuning knob carries a sequence value.

The one design constraint: some consumers have the body, some don't.

- **Have the body** (`materialize.py`, greedy pick): can derive masked-ness by inspecting statements.
- **Body-less** (DB-trained prior, `eval` over golden YAML, MCTS `_o3_sig` dedup): only ever see a persisted knob dict.

So "derive from the body" is the producer side; the body-less consumers still need the derived scalars **persisted** —
but as `S_` structural identity, not a tunable knob. Splitting per role (vs today's flat count) lets the prior learn
that K-masking and M/N-masking have different perf consequences (a masked K pins the SYNC transport and declines ring
buffers — `_enumeration.py:751-757`), which a single count cannot express. Since we are touching every site anyway, the
split costs nothing extra.

## Feature definition

Three `S_`-prefixed floats stamped into the realized kernel's `knobs`:

- `S_masked_m` — 1.0 if the M (outer free) output axis is boundary-masked, else 0.0
- `S_masked_n` — 1.0 if the N (inner free) output axis is boundary-masked, else 0.0
- `S_masked_k` — 1.0 if the K (reduce) axis is masked (symbolic-K zero-fill), else 0.0

Body signal (the "leverage the body" primitive — all already present post-materialize):

- **M / N masking** → output axis carries `real_extent is not None` (`materialize.py:434`, `:68`) + a boundary `Cond`.
- **K masking** → `gmem_extents` stamped on the staged load (`materialize.py:486`) + smem zero-fill; no store guard.

A body walk counting `{axes with real_extent} ∪ {axes with a masked gmem_extent}`, classified by axis role (M/N free
vs K reduce), reproduces today's `overhang` tuple exactly. Live in / next to `structure_features` in
`passes/loop/stamp/020_stamp_structural_features.py`.

## Where to stamp

`020_stamp_structural_features` runs at the **loop dialect**, before tiling — static non-divisor masking is not yet
known there (it depends on the chosen `BN/BM`); only symbolic masking is, and that is already `S_ext_n_symbolic_axis`.
So stamp the three flags at **materialize time**, where `overhang` is already computed:

- `materialize.py:520-521` (composer warp path) — replace `knobs_full["OVERHANG"] = overhang` with the three flags,
  classified from the `(inner_n, outer_m)` bound assembly at `:493-495` plus the K-mask state.
- `_enumeration.py:695-696` (scalar `_enumerate_cartesian_impl`) and `:866-867` (warp impl) — replace the
  `params["OVERHANG"] = overhang_axes` / `row["OVERHANG"] = overhang_axes` stamps with the three flags, classified from
  the `overhang_axes` tuple already assembled at `:758-764` (it knows `m_axis_name` / `n_axis_name` / `k_axis_name`).

## Changes by file

### Producers
- `passes/lowering/tile/partition/materialize.py:520-521` — stamp `S_masked_m/n/k` instead of `OVERHANG`.
- `passes/lowering/tile/_enumeration.py:695-696`, `:866-867` — same; classify `overhang_axes` by role.
- `passes/lowering/tile/_enumeration.py` doc block `:179-184`, `:742`, `:756` — update the knob inventory comment
  (scalar tier no longer lists `OVERHANG`; note the masked axes surface as `S_masked_*` structural features).
- `passes/loop/stamp/020_stamp_structural_features.py` — add the role-classified body-walk helper that the materialize
  sites call (keeps the structural-feature logic in its owning module; the pass itself still runs at loop dialect and
  is unchanged for the symbolic case).

### Featurizer
- `knob.py:547-550` — drop the `overhang = len(knobs.get("OVERHANG", ()))` line; read the three `S_masked_*` flags.
- `knob.py:574` — replace the single `D_neg_overhang` with per-role penalties (`D_neg_masked_m/n/k`) so the prior can
  weight K-masking distinctly from M/N. Keep the sign convention (negative = penalty) so existing analytic weights
  port with minimal refit. `S_masked_*` themselves also pass through `knob_features` as raw floats automatically via
  the `STRUCT_PREFIX` branch (`knob.py:476-477`) — no new passthrough code.
- `scripts/golden_knob_heuristics.py` — refit the `AnalyticPrior` weights over the renamed features (the joint
  tier-balanced fit); `D_neg_overhang` → `D_neg_masked_{m,n,k}`.

### Body-less consumer cleanups (all exist only for the sequence-valued knob)
- `search/prior/base.py:50-53` — `_norm_knob` normalized list↔tuple for evidence matching. `OVERHANG` was the only
  sequence knob; delete `_norm_knob` and its call sites once all knob values are scalars. (Confirm no other sequence
  knob first — current grep shows none.)
- `search/policy/mcts.py:216-224` — `_o3_sig` `str()`-ified values to survive the unhashable list. Keep the function
  (the `H_opt` exclusion is still needed); drop the `OVERHANG` rationale from the docstring. `str()` on floats is
  harmless, so the body can stay or simplify to raw values.
- `commands/eval.py:386-393` `_knob_eq` and `:720-725` `_hashable` — both purely cope with list-valued `OVERHANG`.
  Delete both; use `==` and the raw value directly once no knob is a sequence.
- `compiler/pipeline/fork.py:173` — comment cites `OVERHANG` as a sequence-knob example; swap for `FK`.

### Persistence migration (the real work)
- **Golden YAMLs** — `goldens/rtx4090_sm89.yaml`, `goldens/rtx5090_sm120.yaml`, `goldens/rtxpro6000_sm120.yaml` record
  `OVERHANG: ['a0']` on many entries, static *and* `.dynM`. Drop the `OVERHANG:` key from every entry — the value is
  re-derived now. The `S_masked_*` flags are structural, so they are NOT recorded in goldens (goldens carry only
  tunable knobs + shape); the greedy-vs-golden `eval` compares them as derived structural features, not recorded ones.
  Verify the golden loader/schema does not require `OVERHANG` (grep found no explicit validator).
- **Tune DB** — existing `perf` rows carry `OVERHANG` in their knob JSON and lack `S_masked_*`. After the switch,
  featurizing an old row yields the masked penalties = 0 (missing feature) — silent drift for historical rows. Choose:
  (a) accept and let the reservoir/prior refit over fresh tunes, or (b) one-time DB migration mapping
  `OVERHANG=['a0',...]` → the three flags by axis-name heuristic. **Flag explicitly; do not let it pass silently.**
- **Cache-key shift** — `OVERHANG` currently sits in the *tunable* bucket (`sample.py:_split_by_prefix`), so it is part
  of `op_cache_key` / fork-tree / perf identity. Moving the signal to `S_` keys moves it to the structural bucket
  (semantically correct — deterministic from shape), but the dict-as-key changes, so old DB keys won't match new ones.
  This invalidates replay for masked kernels; acceptable on a `--clean` retune, must be called out.

### Tests
- `tests/compiler/test_matmul_mma_masked.py`, `tests/compiler/pipeline/test_fork.py`,
  `tests/compiler/pipeline/search/test_online_prior.py`, `tests/compiler/cli/test_eval.py` assert on `OVERHANG`.
  Flip to asserting `S_masked_m/n/k` and/or the body's `Cond` / `real_extent` directly. Masked-codegen tests should
  assert on the **body** (guard emitted) rather than the knob — the more honest test.

## Correctness invariant

Guard *emission* does not move. `_warp_axis` (`materialize.py:428`) and the scalar `_assemble` already derive `masked`
from `extent % per_block` (or symbolic) and emit the `Cond` / `real_extent` / `gmem_extents`. This change only deletes
the redundant knob carrier and records a derived structural feature — **codegen is byte-identical**; only the recorded
identity changes. That is the safety margin: any test comparing emitted CUDA before/after should diff empty.

## Sequencing

1. Add the role-classified body-walk helper in `020_stamp_structural_features.py`; stamp `S_masked_m/n/k` at all three
   materialize/enumeration sites **alongside** the existing `OVERHANG` (dual-write). Land + retune so DB/goldens carry
   both. → verify: `eval prior` / `eval golden` ranks unchanged vs pre-change baseline.
2. Switch `knob.py` to read the new flags; refit analytic weights. → verify: golden-rank median and per-knob
   `found/golden` in `eval analytic` / `eval golden` hold within noise.
3. Migrate golden YAMLs (drop `OVERHANG`); delete `_norm_knob` / `_knob_eq` / `_hashable` + the `fork.py` comment;
   flip tests. → verify: `make test` green.
4. Stop stamping `OVERHANG`; decide DB migration (a) vs (b); `--clean` retune. → verify: a fresh tune of a masked
   golden (e.g. a `.dynM` shape) reaches the same config and the same emitted CUDA as before.

## Prerequisite: drop the legacy `enumerate_cartesian` enumerator first

`enumerate_cartesian` (and `_enumerate_cartesian_impl` / `_enumerate_warp_matmul_impl`) are the legacy enumerator,
fully replaced by the move composer (`partition/`) in the live pipeline — `010_partition_loops.py` now calls
`partition.compose.try_compose`, not the enumerator. Dropping it **before** the OVERHANG work collapses this plan's
three stamp sites to **one** (`materialize.py`), since `_enumeration.py:695-696` and `:866-867` disappear with the
impls.

### What stays vs goes in `_enumeration.py`

The file is two modules under one name:

- **Stays (knob schema — imported by `partition/knobs.py`, `tuning.py`, `knob.py`):** the `Knob` objects
  `BN/BM/FM/FN/FK/BK/SPLITK/BR/WN/WM/MMA`, candidate constants, `mma_mode` / `_mma_features`, `_PLANNER_KNOBS`,
  `planner_pin_set` / `planner_pin_snapshot`. (`apply_off_defaults` already lives in `knob.py`.) The file survives as
  the knob-schema home; an optional follow-up renames it (`_knobs.py`) — wider import churn, keep surgical for now.
- **Goes (the enumerator):** `enumerate_cartesian`, `_enumerate_cartesian_impl`, `_enumerate_warp_matmul_impl`,
  `_prior_order`, `_matmul_thread_gate`, `_divisors_up_to`. Verified no external callers (only a stale comment in
  `ir/kernel/ir.py:1271` and ARCHITECTURE.md references).

### Consumers to rewire

- **`search/analytic.py:18,40,48`** — the only production caller. It reconstructs a matmul shape's candidate pool for
  offline golden eval/diagnostics (`eval analytic` / `eval prior`). The composer has no flat "enumerate all rows" entry
  (it builds a Fork tree in `tree.py` whose `expand()` chain yields leaf `TileOp`s with full knob dicts). **Add**
  `partition/enumerate.py::candidate_knob_rows(loop_op, ctx, graph) -> list[dict]` that calls `try_compose` and walks
  the fork via `fork.flatten_leaves` / recursive `expand()`, harvesting each terminal's `.knobs`. Then `_enumerate`
  builds the matmul LoopOp (reuse the snippet→loop-dialect compile already cached in
  `data/sample.py::compiled_s_features`) and calls it. `THREAD_KNOBS` / `WARP_KNOBS` match keys are unchanged (the
  composer stamps the same legacy names). **More correct than today:** the golden-rank metric then measures against the
  *same pool the pipeline deploys from*, removing legacy-vs-composer drift.
- **Tests** `test_strided_coop_rows.py`, `test_fk_reduce_enumeration.py`, `test_knob_pinning.py` — port to the
  composer's `moves.py` offers (`matmul_reduce_offers`, `coop_reduce_offers`, the fork-tree leaves) or to the new
  `candidate_knob_rows`. Knob pinning already flows through the composer via `Knob.raw()` / `planner_pin_snapshot`.
- **ARCHITECTURE.md** `:915` (claims `010_partition_loops` imports `enumerate_cartesian` — false now) and `:856`
  (`_PLANNER_KNOBS` / `MMA` declaration prose) — update to the composer reality.

### Ordering across both plans

1. Add `candidate_knob_rows`; rewire `analytic.py`; port the 3 tests. → verify: `eval analytic` / `eval prior` golden
   ranks match the pre-change baseline (the composer pool should rank goldens at least as well — investigate any
   regression, it signals a real enumeration gap, not just a port bug).
2. Delete the enumerator functions from `_enumeration.py`; update ARCHITECTURE.md. → verify: `make test` green,
   no dangling imports.
3. Proceed with the OVERHANG plan above — now a **single** `materialize.py` stamp site (the `_enumeration.py` sites
   are gone).

## Open / decided

- **Decided:** split per-role (`S_masked_m/n/k`), not a flat count — lets the prior learn K-mask ≠ M/N-mask.
- **Open:** DB migration strategy (accept drift vs one-time map). Lean (a) accept + `--clean` retune, since masked
  kernels are a small slice and the reservoir refits; revisit if a trained prior must survive the transition without a
  retune.
