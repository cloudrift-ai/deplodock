# Monoid reduction combine as an array of stages

**Status:** partially executed. M1 / M3 / M4 landed (full `make test` green after each); M2 resolved as
representation-only; M5 / M6 are the remaining flash split-KV feature (see Milestones).

- **M1 (done)** — the intra-CTA combine is a derived `CombineStage(width, fold, kernel_boundary)` array
  (`kernel/_combine.py`); `emit_combine` consumes it; output byte-identical. `Fold.ATOMIC` reserved for the cross-CTA
  finalize.
- **M2 (resolved, representation-only)** — the literal "shrink the REDUCE codec to `s/f`" conflicts with "`make test`
  unchanged" (the `s/f/c/t` codec is the perf-DB / learned-prior / golden variant identity, and goldens reference the
  cross-CTA/coop widths via the legacy `SPLITK` / `BR` knobs). Decision: keep the codec string intact; the array is the
  structure the finalize fork reads. No DB/golden/prior migration.
- **M3 (done)** — `140_atomic_free_splitk`'s `NOATOMIC` fork reframed as the cross-CTA stage's finalize policy
  (`False` = in-place `Fold.ATOMIC`, `True` = deferred `kernel_boundary`); legality `atomic_finalize_legal` (additive
  `Accum` + atomicAdd dtype) added to `lowering/_predicates` (shared layer — no tile→kernel back-edge); the fork narrows
  to deferred-only when ATOMIC is illegal. Knob identity + reachable behavior unchanged; the legality gate is the hook
  M5's twisted finalize trips.
- **M4 (done)** — flash-vs-flat (Axis A) routes on `MonoidReduction.nested` (folded onto the reduction, set even when
  `inner is None` — the non-separable stream); the standalone `IterDag.streaming` routing property is removed.
- **M5 (deferred finalize, carrier-generic — done)** — the deferred finalize is now ONE carrier-generic entry point,
  `_partition.deferred_combine_tilegraph(carrier, …)`, dispatched on the carrier's state: a 1-component additive `Accum`
  (matmul split-K) → `additive_reduce_tilegraph`; a tuple `Monoid` (online-softmax / flash) → `monoid_reduce_tilegraph`.
  This is the blog's thesis as code — a regular sum, online softmax `(m, d)`, Welford `(n, μ, M₂)`, and flash `(m, d, o)`
  are the SAME cross-partition reduce, differing only in carrier state, so the finalize is not a flash special case.
  `140` routes through it (the additive matmul = the trivial carrier; byte-identical). Both carriers are GPU-verified
  through the one entry point (`tests/compiler/test_monoid_reduce_kernel.py::test_deferred_finalize_*`).
- **M5 producer / M6 (remaining)** — the twisted finalize is reachable from `140`'s legality gate but not yet *driven* by
  greedy/tune, because the **producer** — a cross-CTA split that writes the carrier's partial state (`(m, l, O)`) to the
  per-component `partial[S, M, N]` workspaces — is itself carrier-generic future work (the cooperative + warp MONOID
  reduce currently pin `cta=1`; the additive matmul is the only `cta>1` producer). The generic producer = thread the
  split-K `grid` through `monoid_build` (`_rebracket_k` already accepts it) + write each carrier state component to its
  workspace (via `split_carrier`, like `chain_build`) + offer the `cta` width on the MONOID reduce. M6 then surfaces the
  finalize knob in the inner search. Neither is a representational tweak; both reuse the generic finalize above.

---

Decouples the monoid reduction *combine* from *flash*, and represents the combine as an
ordered array of stages — a **derived** structure with a **narrow tunable overlay** (the cross-CTA finalize), not a
freely-searched knob.

## The thesis (two orthogonal axes)

A monoid reduction has two independent decisions that today are tangled together:

- **Axis A — WHAT is folded.** A flat reduce vs a flash reduce. **Flash means exactly one thing: there is an inner
  contraction** (`IterDag.reduction.inner is not None` — the embedded SEMIRING P@V on the hinge). Nothing else. A flat
  softmax/RMSNorm/mean carries `inner = None`; a flash carries `inner` set. Already a derived structural fact
  (`_iterdag.MonoidReduction`), so Axis A needs no new state — only the removal of the *other* signals that currently
  stand in for it (see "Retire the streaming schedule bit").

- **Axis B — HOW the partials are combined.** Independent of A. The cross-execution-unit reduction hierarchy
  (lanes → warps → CTAs → grid) is represented as an explicit **array of stages**, each carrying a width and a fold
  primitive. This array **subsumes** the `c`(cta) / `t`(coop) factors of today's REDUCE codec — one structure owns the
  widths *and* their folds, replacing two bare ints that `emit_combine` re-interprets. But it is mostly **derived**: the
  one genuinely free choice in it is the cross-CTA finalize. See "Representation vs policy".

A flat reduce and a flash both combine through the same stage array. Flash just additionally has the inner contraction
sitting in the carrier; it does not change the combine.

## What is tangled today

1. **The combine realization is geometry-implicit.** `kernel/_combine.py::emit_combine` *picks* "warp-shuffle /
   hierarchical / block-wide smem tree-halve" **by thread count** (its docstring, lines 5-7). The fold is re-derived
   from a bare count at codegen time, not represented.
2. **Atomic-vs-separate-kernel exists only for cross-CTA split-K.** `140_atomic_free_splitk` forks `atomicAdd` (default,
   `escape_analysis` emits the atomic when the split axis is missing from the Write index) vs a workspace +
   `_partition.additive_reduce_tilegraph` / `monoid_reduce_tilegraph` combine kernel. Gated on `SPLITK>1`, not offered as
   a general monoid finalization.
3. **The schedule routes on flash-ness.** `070_coop_reduce.py::reduction_build` ends with
   `_streaming_leaves if op.dag.streaming else _coop_leaves`, so "is it flash" picks the emit body — instead of being
   orthogonal to the combine.
4. **`coop`/`cta` are bare counts.** The REDUCE codec `s<serial>/f<fold>/c<cta>/t<coop>` (`_families.enc_reduce`) folds
   four factors into one string; `c`(cta) and `t`(coop) are the two that *need* a cross-unit combine, but they say only
   *how many* partials, never how they fold — so the width and the fold live in two places that must agree.

## Representation vs policy (the core of this revision)

The combine array is the right *representation*, but its cells are not all free parameters. Separate them:

**Derived (not searched):**

- **Intra-CTA fold primitive is forced by the level.** Lane-level → `SHFL` (smem-within-a-warp is strictly dominated);
  across-warps → `SMEM` tree (warps cannot shfl across each other). There is no perf choice here worth a search
  dimension — hierarchical (shfl-then-smem) is the dominant intra-CTA scheme. So a stage's `fold` is *computed* from its
  execution level, not tuned.
- **The default array reproduces today's `emit_combine` exactly.** A `coop` of 128 derives to `[(32, SHFL),
  (4, SMEM)]`; no behavior change until the overlay below is touched.

**Tunable (the levers):**

- **The partition widths** (`coop`, the split-K `cta` degree). These are *already* tuned today via the REDUCE codec —
  occupancy, wave quantization, split-K degree are real perf knobs. Subsuming them into the stage array changes *where*
  they live (one typed structure), **not** whether they are tuned. This is not new search surface.
- **The cross-CTA finalize** — `ATOMIC` in-place vs a deferred fold in a separate kernel (`kernel_boundary`). This is
  the **one new tunable choice**, and not by coincidence the only reduction-combine fork the codebase already bothered
  to build (`140`). It is where contention-vs-launch-overhead actually trades off.

So the net new search dimension is **exactly one** (cross-CTA finalize). The intra-CTA `all-smem` vs `hierarchical`
alternative *exists* in the representation but stays defaulted to hierarchical and is **not** forked unless a measured
case justifies it. Everything else is derived or already-tuned.

## Core abstraction: the (derived) stage array

```
CombinePlan = tuple[CombineStage, ...]      # outer→inner, applied after the in-thread (s/f) accumulation
CombineStage = (width: int, fold: SHFL | SMEM | ATOMIC, kernel_boundary: bool)
                                            # width   — DERIVED from the partition (the already-tuned coop/cta degree)
                                            # fold    — DERIVED from the execution level (lane→SHFL, warp→SMEM,
                                            #           cross-CTA→ATOMIC); the level is implied by the width's place
                                            # kernel_boundary — the cross-CTA stage's POLICY knob: deferred fold in a
                                            #           fresh kernel (write partials → combine kernel) vs in-place atomic
```

The array is the whole cross-unit hierarchy made explicit, but assembled from the partition + the level→fold derivation
+ the single finalize policy — not stored as free knobs per cell. The product of the widths equals the old `coop·cta`.

| Stage example                  | Subsumes        | Origin of each field                                   |
|:-------------------------------|:----------------|:-------------------------------------------------------|
| `(32, SHFL, false)`            | part of `coop`  | width: partition · fold: lane-level · finalize: n/a    |
| `(n_warps, SMEM, false)`       | rest of `coop`  | width: partition · fold: warp-level · finalize: n/a    |
| `(n_cta, ATOMIC, false)`       | `cta`           | width: split-K degree · fold/finalize: **the one knob**|
| `(n_cta, SMEM, true)`          | `cta`           | width: split-K degree · finalize: **deferred kernel**  |

Only the last two rows differ by a tunable; the rest is derivation. "Each stage can be a separate kernel" stays the
general form (any boundary *may* be a kernel boundary), but in practice only the cross-CTA stage exercises it.

## Design changes (where each lands)

1. **The stage array as a derived view** (`_combine.py` + `_families.py`): build the array from the partition widths
   (`coop`/`cta`) + the level→fold derivation, with the cross-CTA stage's finalize read from one new policy field. Move
   `coop`/`cta` into this structure as the representation (honoring "subsume"), but keep their *values* sourced from the
   same partition decision that sets them today — `reduce_fields` / the partition planner / the cooperative-axis
   derivation read the widths off the array instead of the bare `Decomp` fields. `emit_combine` consumes the array
   instead of choosing by count.

2. **One finalize knob, not a combine-plan search** (`_knobs.py` + the enumeration fork): the new tunable is the
   cross-CTA `kernel_boundary` (`ATOMIC` in-place vs deferred kernel), offered only where there is a cross-CTA stage
   (`cta>1` or an `atomic_axes` level). This **absorbs `140_atomic_free_splitk`** and generalizes it from split-K-only
   to any additive monoid. Legality: `ATOMIC` needs an additive `Accum` + an `atomicAdd`-capable dtype; deferred is
   always legal. The intra-CTA `all-smem` alternative is representable but not enumerated (defaulted to hierarchical).

3. **Route Axis A on `reduction.inner`** (`070_coop_reduce::reduction_build`): stop branching on `dag.streaming`. Decide
   flash vs flat from `reduction.inner` (warp-tier when it tensorizes + is buildable + deployable; else the scalar
   build), independent of the combine. Both emit bodies consume the same derived array.

## Retire the streaming schedule bit

Axis A's "flash = inner only" requires removing the stand-in signals. `IterDag.streaming` (nested-reduce ∧ monoid
carrier) currently distinguishes a non-separable stream (`streaming=True, inner=None`) from a flat cooperative reduce
(`streaming=False, inner=None`) — both have `inner=None`, so the dispatch needs the extra bit. Fold the residual
nested-ness fact onto `MonoidReduction` (a `nested` field set even when `inner=None`) so `_classify` keeps the one case
it genuinely needs, and drop `IterDag.streaming` as a standalone routing input. (Companion to the earlier `dag.streaming`
analysis.) Note this is the *one* MONOID distinction that is **not** subsumed by the combine array — it is an Axis-A
fact (what is folded), not an Axis-B one (how).

## The twisted-monoid finalize (the one place A and B touch)

A deferred-kernel fold is trivial for an additive `Accum` (sum the partials). For a flash `Monoid` the cross-partition
merge carries the `e^{Δm}` rescale, so the combine kernel must fold the *twisted* combine, not a plain sum.
`_partition.monoid_reduce_tilegraph` already exists beside `additive_reduce_tilegraph`, so the mechanism is there. Plan:
offer the deferred finalize for additive carriers first (reuse the `140` path), then enable the twisted variant via
`monoid_reduce_tilegraph` once the additive path is green. This is the only coupling between the axes, and it is
contained to the cross-CTA stage's one knob.

## Milestones (single branch, `make test` green after each)

1. **Derive the stage array; `emit_combine` reads it.** Build the array from the existing `coop`/`cta` + the level→fold
   derivation; `emit_combine` consumes it. Output byte-identical (the default array reproduces today's pick). Verify via
   `--ir cuda` that the emitted combine is unchanged across the reduction/matmul examples.
2. **Subsume `coop`/`cta` widths into the array.** Re-point `reduce_fields` / the partition planner / the
   cooperative-axis derivation to read widths off the array; shrink the REDUCE codec to `s/f`. `make test` unchanged
   (values still come from the same partition decision; only their home moved).
3. **Add the cross-CTA finalize knob; absorb `140`.** Make `kernel_boundary` (`ATOMIC` vs deferred) the cross-CTA
   stage's one tunable, generalized beyond `SPLITK>1`, reusing `_partition.additive_reduce_tilegraph` + the `140`
   Graph-fragment splice. A/B atomic vs deferred on a split-K shape; both reachable by the knob.
4. **Route Axis A on `reduction.inner` only.** Replace the `dag.streaming` branch in `reduction_build`; fold the
   nested-ness fact onto `MonoidReduction`; drop `IterDag.streaming` as a routing input. `_classify` unchanged.
5. **Twisted-monoid deferred finalize.** Enable the deferred finalize for flash carriers via `monoid_reduce_tilegraph`
   (fold the rescale). Accuracy vs eager on an SDPA shape.
6. **Search integration.** Surface the *finalize* knob (only) as a fork in the inner per-kernel search; record the
   derivation rules + legality in the tile-lowering ARCHITECTURE.

## Risks / constraints

- **Scope discipline, not search blowup.** Because only the finalize is tuned (folds derived, widths already tuned), the
  per-kernel search grows by one binary dimension, not a combinatorial array. Resist the temptation to enumerate
  intra-CTA fold variants without a measured win.
- **Launch overhead of deferred folds.** A separate combine kernel costs a launch; at tiny sizes it loses. Keep
  atomic/in-place reachable; the finalize knob is a tradeoff, not a forced win.
- **Doc drift to fix in passing.** `_partition.py` and `070`/`_classify` docstrings reference `017_atomic_free_splitk`
  and an `080_streaming` fork that no longer exist as files (now `140` / folded into `070`).

## Verification

- Structural: `deplodock compile --code "<reduce>" --ir cuda` unchanged after M1–M2; after M3 a deferred-finalize pin
  shows a second `__global__` and no `atomicAdd`.
- Accuracy: `deplodock run --code "torch.nn.RMSNorm(2048)(torch.randn(1,32,2048))" -v` and an SDPA `--code` snippet →
  `Accuracy vs eager … PASS` for both finalize choices.
- A/B: `deplodock run ... --bench --ab "<finalize=deferred>"` vs the atomic default on a split-K shape.
- `make test` + `make lint` green at each milestone.
