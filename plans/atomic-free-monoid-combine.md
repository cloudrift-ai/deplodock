# Atomic-free, monoid-general cross-partition reduction — design plan

## Status

- **Step 1 — `Combine.combine_states` — DONE.** `Combine` now carries `state_b` (the second-operand state
  names, default `"<s>__o"`) + `combine_states` (the state-merges-state program), auto-derived from `merge` for
  additive carriers and authored explicitly by `flash_combine` (the LSE state-merge). `as_state_merge(other)`
  renders a two-partition merge through the streaming machinery. Threaded through `rewrite`. Verified by
  `tests/compiler/ir/test_combine_forward.py` (two-partition CPU forward vs numpy + the additive auto-derive)
  and `tests/compiler/ir/stmt/test_reduce_carrier.py`.
- **Step 5 (analytic preference) — partially DONE (the prior term).** `AnalyticPrior.score` carries the gated
  `NOATOMIC` term (hardcoded `atomic_free_split_threshold` / `atomic_free_weight` `__init__` params); verified
  by `tests/compiler/test_analytic.py::test_atomic_free_split_preference_above_threshold`. The cold-default
  *flip* of the `ATOMIC_FREE_SPLITK` knob + golden re-tune are still pending (gated on Steps 3a/3b landing).
- **Steps 2 / 3a / 3b / 4 — PENDING.** The carrier-general combine on the materializer / `017` / partition
  planner + flash split-KV. These are coupled: a cooperative / split `Combine` does not exist in any lowered
  kernel until the partition planner offers the KV split (Step 4 / the `010_partition_loops` hook), so Step 2's
  intra-CTA combine emission can only be GPU-verified end-to-end alongside that producer. Recommended to land as
  one focused follow-up (materializer combine emission + coordination recognition + planner offer + `017`
  carrier-general reduce), each piece bit-correctness-checked on GPU.
- **Step 6 (deletion decision) — PENDING (bench-gated).** Per the rule below, decided by the atomic-vs-workspace
  bench across split-K goldens on the target GPUs once both tiers have an atomic-free path.

---


Make the **separate-kernel cross-CTA combine the canonical reduction**, generalized over the `Combine` monoid carrier
(`merge` + `identity`), and demote `atomicAdd` to a measured fast-path fork. This unifies matmul split-K, flash split-KV
/ flash-decoding, and any future monoid (max, Welford, argmax) under one mechanism — and it is the *only* mechanism that
works for non-additive carriers like flash's log-sum-exp `Combine`, which cannot `atomicAdd` at all.

Builds directly on the `Combine` generalization (the loop-carried monoid carrier: `state` / `partial` / `merge` as a data
program / `identity` / `commutative`). That carrier exists precisely so the cross-partition combine can be *driven by the
carrier's algebra* instead of a hardcoded `+`: `associative` / `commutative` gate legality, `identity` seeds each
partial, and `merge` *is* the combine the reduce kernel (or cooperative tree) emits.

## Why — what the current path does, what this changes

A reduction whose reduce axis is **split across partitions** (the matmul's K split `K_s`, a cooperative-K thread split,
a future flash split-KV across CTAs) has to combine the per-partition partials. Today that combine takes one of three
shapes, all **additive / `Accum`-specific**:

1. **Cross-CTA via `atomicAdd`** — `SPLITK > 1` fans each `K_s` CTA's partial into the output with `atomicAdd`. The
   trigger is `Body.coordination.atomic_axes` (an enclosing block axis missing from a `Write` index ⇒ atomic); codegen
   is `Write.render`'s `atomicAdd` branch. **Additive only**, non-deterministic (FP add order), and `atomicAdd(__half*,
   …)` is a precision hazard the code already special-cases.
2. **Cross-CTA atomic-free** — `lowering/tile/017_atomic_free_splitk.py`: a two-kernel pair. The matmul writes per-`K_s`
   partials into a workspace `partial[S, M, N]` (`K_s` now in the `Write` index ⇒ `atomic_axes = ∅` ⇒ plain store), and a
   sibling pre-tiled reduce `TileOp` **sums** along `S` into the output. The pass already **splits the kernel exactly like
   `005_split_demoted`**: `rewrite` returns a `list[TileOp | Graph]` fork — `[atomic single `TileOp`, atomic-free two-node
   `Graph` (matmul→workspace + reduce)]` — with `NOATOMIC` (alias `ATOMIC_FREE_SPLITK`) stamped on each branch under an
   idempotence guard, the same shape as `SPLIT_CONE`. So the kernel-split + fork machinery is in place; what's missing is
   generality. Two limits: it is **hardwired to `+`** (the reduce `TileOp` sums), and it is **scalar-matmul only** —
   `is_warp(op.knobs)` early-returns the off-decision (`017:282`), so **MMA / warp-tier split-K still uses the codegen
   `atomicAdd` rewrite**, never the workspace split. Cold default `False` = atomic.
3. **Intra-CTA cross-thread** — `lowering/kernel/_combine.py` (`emit_combine` → warp-shuffle / `TreeHalve` / smem tree by
   thread count) combines an `Accum`'s per-thread partials when its reduce axis is split across the CTA's threads
   (cooperative-K). **Keys off `Accum`** (`find_nested_reduce_accums` collects `Accum`s; combines via the scalar `op`).

None of the three can express a general monoid's merge, so flash's `Combine` (max + rescale) has no cross-partition
reduce at all — which is why the deployed flash today keeps the KV reduce fully **serial** (one thread per output
element, no atomics, no combine). That serial reduce is correct but leaves flash with no path to parallelize the KV axis
(cooperative-K, split-KV/decoding) — the regime that matters for long sequences and low query count.

This plan makes the combine **monoid-general** and **atomic-free-by-default**:

- Generalize the intra-CTA combine (`_combine.py`) and the cross-CTA workspace-reduce (`017`) to consume **any
  `ReduceCarrier`** — seed partials with `identity`, combine with `merge` (or the `Accum.op` / `Mma` additive fold as the
  degenerate cases). One mechanism for `Accum`, `Mma`, and `Combine`.
- Make the **separate-kernel cross-CTA combine the canonical lowering** for a split reduce; demote `atomicAdd` to a
  fast-path the analytic prior / tuner picks only for small additive splits (see "Analytic prior" below).

Two non-perf properties make atomic-free-by-default the right call beyond just flash: **determinism** (a fixed-order
reduce kernel is bit-reproducible; this project's golden + accuracy infra depends on that, and atomics inject FP-order
noise) and **fp16 safety** (an f32 workspace + downconvert dodges the `atomicAdd(__half*)` hazard). And a non-commutative
monoid *can only* reduce in a fixed order — atomics can't provide it.

## Scope of "separate kernel" — cross-CTA only

Be precise about which combine becomes a separate kernel:

- **Cross-CTA** (split-K `K_s`, split-KV across CTAs): a **separate reduce kernel** over a gmem workspace. Atomic-free,
  monoid-driven. This is the path that *replaces* `atomicAdd`.
- **Intra-CTA** (cooperative-K across the CTA's threads): **stays in-kernel** — warp-shuffle / smem `TreeHalve`. Already
  atomic-free; a separate kernel here would force a pointless gmem round-trip. This path is *generalized* over the
  carrier (so it can combine a `Combine`'s state across threads), not moved out of kernel.

## Target design — one carrier-driven combine

### Intra-CTA: `_combine.py` over any carrier

`find_nested_reduce_accums` → `find_nested_carriers` (returns the body's `ReduceCarrier`s, not just `Accum`s).
`emit_combine` takes the carrier and emits the cross-thread fold:

- `Accum` → the scalar `op` shuffle/tree (unchanged).
- `Mma` → the fragment additive fold (the existing cooperative-K matmul path).
- `Combine` → the monoid fold: each lane holds a full `state` tuple (declared from `identity`); the warp-shuffle / tree
  step shuffles **every state component** down and applies the carrier's `merge` to the (this-lane-state, shuffled-state)
  pair. `commutative` is required (a tree/shuffle reorders); `_combine` already needs `associative`, and `Combine`
  advertises both. For flash this combines `(m, l, O)` per row segment without atomics.

The merge for a *cross-state* shuffle reads two full states (not state + partial). The flash `merge` is written as
state-folds-partial; the cross-partition form is the **state-merges-state** monoid law (`m = max(m_a, m_b)`,
`l = l_a·exp(m_a−m) + l_b·exp(m_b−m)`, `O = O_a·exp(m_a−m) + O_b·exp(m_b−m)`). So `Combine` likely needs a second program
— `combine_states` (merge two carried states) — alongside `merge` (fold a partial). `flash_combine` builds both; for
`Accum` / additive `Mma` the two coincide (the op is the same), so the carrier can default `combine_states` from `merge`
when the partial *is* a lifted state.

### Cross-CTA: `017` workspace-reduce over any carrier

The kernel-split + fork is already there (it mirrors `005_split_demoted`); `017` is generalized along **two axes**, both
needed:

**(a) Carrier-general reduce (sum → `combine_states`).** The reduce `TileOp` `017` builds is hardwired to sum
`partial[S, …]` along `S`. Make it a **carrier-driven reduce**: seed the accumulator from `identity`, fold each `S` slice
with `combine_states`. For `Accum` that is `+` (unchanged); for flash's `Combine` it is the LSE state-merge over the `S`
KV-split partials (the workspace then carries the partial `(m, l, O)` per split, not just a scalar). The producer side
stays the same shape — write per-partition partials with the split index in the `Write` index so `atomic_axes = ∅`.

**(b) Warp / MMA tier (drop the `is_warp` early-out).** Today `017` bails on warp tiles (`is_warp(op.knobs)` →
`_off()`), so split-K on an MMA matmul still fans the C-fragment to the output with the codegen `atomicAdd` rewrite. The
generalization wires the MMA producer to write its C fragment to `workspace[S, M, N]` (the `RegStore`'s output retargeted
to the workspace, `K_s` in the index) and reuses the same carrier-driven reduce kernel for the `S` sum. This is a
strictly larger surface than the scalar path `017` covers today and lands as its own step — but it is what lets the
*matmul* split-K become atomic-free across both tiers (and a tensor-core flash, once that tier exists, reuse the same
reduce). Until it lands, MMA split-K keeps the `atomicAdd` rewrite (the demoted fast-path), so the deletion decision
below is **scalar-only** at first.

A flash split-KV is exactly path (a) with the `Combine` carrier: split the KV axis into `S_kv` CTA chunks, each writes
its `(m, l, O)` partial to `workspace[S_kv, …]`, the reduce kernel LSE-merges them. The single-CTA serial flash is the
`S_kv = 1` degenerate case.

## Where it lives in the pipeline

- **`lowering/kernel/_combine.py`** — generalize `find_nested_*` + `emit_combine` from `Accum` to `ReduceCarrier`
  (carrier-typed combine; the `Combine` branch shuffles the full state tuple and applies `combine_states`).
- **`lowering/tile/017_atomic_free_splitk.py`** — the reduce `TileOp` becomes carrier-driven (`identity`-seed +
  `combine_states`-fold), the workspace element type widens to the carrier's state tuple for a non-scalar `Combine`, and
  the `is_warp` early-out is dropped so the **MMA/warp tier** routes its C-fragment store to the workspace + shared reduce
  (instead of the codegen `atomicAdd` rewrite). The list-of-variants fork + `NOATOMIC` stamping stay as-is.
- **`lowering/tile/010_partition_loops.py`** — the cooperative-reduce / split planner offers the split for a `Combine`
  reduce (the `nonmatmul_reduces` path), reading the carrier's `commutative` to gate split-KV the way it reads
  `Accum.op.associative` today.
- **`Write.render` + `coordination.atomic_axes`** — unchanged for now (the `atomicAdd` branch stays for the demoted
  fast-path); revisited if the deletion decision (below) lands.
- **`ir/stmt` `Combine`** — add `combine_states` (the two-states monoid merge) beside `merge`; default it from `merge`
  when the partial lifts to a state (additive carriers). `flash_combine` builds both programs.

## Analytic prior — prefer atomic-free beyond a size threshold (mostly default)

Atomic-free should be the **cold default**, with `atomicAdd` kept only as a measured fast-path for the *small* additive
split where atomic contention is cheap and the workspace + extra launch aren't worth it. The split regime where atomics
might win is narrow: split-K is chosen exactly when M·N is too small to fill the GPU, so the workspace `S·M·N` is small
*and* the few output addresses see high per-address atomic contention — i.e. atomic-free usually wins, and wins *more*
as the split count grows. So the gate is: **prefer atomic-free once the reduction is split beyond a threshold**, with the
threshold a hardcoded `__init__` param (calibratable, like the `FLASH` term's `flash_seq_threshold`).

Implement as one gated term, **local to `AnalyticPrior.score`**, so it neither perturbs the shared featurizer nor forces
a refit of `_W_A` / `_W_A_DYN`. The signal is the split count `SPLITK` (the partial count = workspace multiplier =
atomic-contention driver), already a registered `INT` knob that `knob.knob_features` passes through as a float
(`S_ext_reduce_max`, the reduce extent K, is the size-based alternative — they correlate, since larger K is split more
ways; calibration picks the better predictor):

```python
# AnalyticPrior.__init__ — hardcoded parameters (overridable for eval analytic sweeps)
atomic_free_split_threshold: float = 4.0   # prefer atomic-free once K is split at least this many ways
atomic_free_weight:          float = <W>   # how hard the cold pick leans

# AnalyticPrior.score — after  feats = knob.knob_features(knobs)
af_on        = feats.get("ATOMIC_FREE_SPLITK", 0.0)                              # 0 / 1
many_splits  = feats.get("SPLITK", 1.0) >= self._atomic_free_split_threshold     # the "size" gate
quality     += self._atomic_free_weight * af_on * (1.0 if many_splits else -1.0) # +reward big split, −penalty small
```

`af_on · (±1)` is the **interaction** the gate needs — a plain weight on the boolean can't say "good when split wide, bad
when split narrow." Above the threshold it *rewards* `ATOMIC_FREE_SPLITK=True` (raises quality → lower
`exp(−scale·quality)` → better); below it *penalizes* it, so a narrow split keeps the `atomicAdd` fast-path.
`ATOMIC_FREE_SPLITK=False` scores zero either way (`af_on = 0`), so the atomic path keeps its geometry-driven rank.
Set the threshold **low** so that essentially every real split-K (which only fires for large-K shapes, typically split
≥4–8 ways) lands above it → atomic-free is the de-facto default, and the small `SPLITK ∈ {2,3}` band is the only place the
atomic fast-path is preferred. Threshold and weight are hardcoded `__init__` params, **not** fit — the learned
`CatBoostPrior` takes over the moment real atomic-vs-atomic-free `H_opt=3` rows exist for these shapes.

Note this term only steers the cold *pick*; the knob already exists (`ATOMIC_FREE_SPLITK`), so unlike the `FLASH` term it
has a live consumer today and can be wired independently of the carrier-generalization work.

## The "delete atomics entirely?" decision — data-driven

Making atomic-free *canonical* is unambiguous. *Deleting* the `atomicAdd` path (the `Write.render` branch,
`coordination.atomic_axes`, the fp16-atomic handling, the `ATOMIC_FREE_SPLITK` fork's `False` arm — ~40 references) is a
real simplification but should be gated on measurement, because split-K is tuned and real (~15–20 golden configs per GPU
use `SPLITK > 1`, and the atomic arm is the current cold default). The deletion can only happen once **both tiers** have
an atomic-free path (Steps 3a + 3b) — until the warp/MMA workspace split lands, MMA split-K *needs* the `atomicAdd`
rewrite, so the atomic branch can't be removed even if the scalar bench favors deletion. The decision:

1. Land the carrier-general atomic-free combine on **both** tiers (Steps 3a + 3b) + the analytic preference (atomic-free
   default, atomic a fork).
2. Re-tune the split-K goldens on the target GPUs (5090 / Pro6000 / 4090) and `compare` atomic vs workspace per shape,
   scalar **and** warp.
3. If atomics never win by more than `DEPLODOCK_O3_TOL` across both tiers → **delete** the atomic arm and its
   escape-analysis branch. If they win on the narrow small-split band → **keep** the demoted fork and stop there.

Hypothesis: on modern GPUs with fast L2 atomics *and* CUDA-graph-amortized launches, the workspace approach is within
noise on most split-K shapes, so deletion is likely viable — but the bench, not aesthetics, decides.

## Incremental steps (each independently verifiable)

- **Step 1 — `Combine.combine_states`.** Add the two-states monoid-merge program beside `merge` (default it from `merge`
  for additive carriers); `flash_combine` builds the LSE state-merge. Verify: unit test on the carrier surface +
  `LoopOp.forward` of a hand-built *two-partition* reduce that merges two `(m, l, O)` states and matches numpy.
- **Step 2 — intra-CTA `_combine.py` over any carrier.** Generalize `find_nested_carriers` + `emit_combine`; the
  `Combine` branch shuffles the state tuple and applies `combine_states`. Verify: a cooperative-K **scalar** monoid
  (a max-with-rescale or Welford) reduces across threads, accuracy on GPU; the existing `Accum` / `Mma` cooperative-K
  matmul tests stay green (degenerate cases unchanged).
- **Step 3a — cross-CTA `017` carrier-general (scalar tier).** Make the reduce `TileOp` carrier-driven (`identity`-seed +
  `combine_states`), workspace widened to the state tuple. Scope unchanged (scalar matmul; `is_warp` still bails). Verify:
  scalar matmul split-K still bit-matches (additive degenerate case); a hand-built scalar monoid split reduces correctly.
- **Step 3b — cross-CTA `017` warp / MMA tier.** Drop the `is_warp` early-out: route the MMA producer's C-fragment store
  to `workspace[S, M, N]` (retarget the `RegStore`, `K_s` in the index) and reuse the Step-3a reduce kernel. Verify: an
  fp16 MMA split-K matmul is bit-correct with `NOATOMIC=True` and emits no `atomicAdd`; the atomic arm stays available.
- **Step 4 — flash cooperative-K / split-KV.** Let `010_partition_loops` offer the KV split for the flash `Combine`
  reduce (reading `commutative`), so flash's KV axis parallelizes across threads (Step 2) and CTAs (Step 3a). Verify: a
  long-seq / low-query flash bench shows the split kernel beating the serial one; accuracy vs torch SDPA unchanged.
- **Step 5 — analytic preference + demote atomics.** Add the gated term above; flip `ATOMIC_FREE_SPLITK` so atomic-free
  is the cold default beyond the threshold. Verify: `eval analytic` picks atomic-free past the threshold and the atomic
  fast-path below; split-K goldens re-tune without regression.
- **Step 6 — deletion decision.** Run the atomic-vs-workspace bench across split-K goldens; delete or keep per the rule
  above. Verify: if deleted, the suite is green with the atomic branch gone; if kept, the fork is documented as a
  small-split fast-path.

## Open questions / risks

- **Cost-model direction.** The gate assumes atomic-free wins as the split widens (small output + high per-address
  contention). Confirm on the target GPUs before committing the threshold sign; an off-by-direction threshold would
  prefer the wrong path. The bench in Step 6 settles it.
- **Workspace memory.** A non-scalar `Combine` workspace carries the full state tuple per partition (`S · M · N ·
  |state|`). For flash split-KV the state is `(m, l, O[d])`, so the workspace is `S_kv ·` the output plus the `(m, l)`
  stats — modest for small `S_kv`, watch it for large splits.
- **Non-commutative monoids.** The intra-CTA shuffle/tree reorders, so it needs `commutative`; the cross-CTA reduce can
  fix order and serve a non-commutative `Combine`. The planner must read `commutative` to pick which split is legal.
- **`combine_states` vs `merge` duplication.** Two programs per carrier is a smell; mitigate by defaulting one from the
  other for additive carriers and only authoring both for genuinely-asymmetric monoids (flash). Reassess whether a
  single state-merge program (with the partial lifted to a state) can subsume both.
- **Determinism as a feature.** If the project commits to deterministic compilation, atomic-free becomes mandatory (not
  a perf fork) and the deletion decision is forced regardless of the bench — worth deciding the policy explicitly.
- **Warp/MMA C-fragment store retarget.** Step 3b retargets the `RegStore` to the workspace, which the kernel tier emits
  late (`kernel/005_lower_atom_tile`); confirm the `K_s`-in-index workspace store survives the fragment lowering the same
  way the scalar Write does, or the retarget moves into the kernel tier instead of `017` (a tile pass).

## Scope

**In:** the carrier-general cross-partition combine — intra-CTA `_combine.py` + cross-CTA `017` on **both** the scalar
and warp/MMA tiers (drop `017`'s `is_warp` early-out), driven by the `Combine` monoid (`identity` + `combine_states`);
the analytic preference for atomic-free beyond a split threshold; flash cooperative-K / single-node split-KV as the
motivating consumer.

**Out (future):** multi-CTA flash-decoding scheduling (paged-KV, persistent kernels); deleting the atomic path (gated on
Step 6's bench, tracked separately); the learned-prior takeover of the atomic-vs-free choice (rides on tuned data, no new
work).
