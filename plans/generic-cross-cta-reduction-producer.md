# Generic cross-CTA reduction producer (carrier-agnostic split-K / split-KV)

**Status:** planned (design). Closes the last asymmetry in the monoid-combine work: the cross-execution-unit reduction
*combine* is carrier-generic at every level, but the *producer* that emits the partials is split — intra-CTA
(cooperative) is carrier-generic, cross-CTA (split-K) exists **only** for the additive matmul. This plan makes the
cross-CTA producer carrier-generic too, so any reduction — plain sum, online softmax, Welford, flash attention — can
split across CTAs by writing its **carrier state** to a workspace, with no shape or attention special-casing.

## Required reading (AI context)

Read these before touching code — they hold the algebra this plan rests on:

- **The article:** `/home/dikobraz/Projects/cloudrift-landing/content/blog/learning-flashattention-the-hard-way-part-1/index.md`
  ("Learning FlashAttention the Hard Way — Part 1", Dmitry Trifonov). The thesis this plan implements: a plain `+`
  reduction, safe/online softmax `(m, d)`, Welford `(n, μ, M₂)`, and flash attention `(m, d, O)` are the **same**
  associative reduction — *"only the combine operator changes, never the schedule."* It names the three schedule
  realizations explicitly — **blocking**, **cooperative reduction**, and **split-K** (each CTA owns a chunk, writes its
  partial to scratch, a small second kernel folds them) — and the **twisted monoid** (transport of structure) that keeps
  attention's `(m, d, O)` a monoid under the `e^{Δm}` rescale. Split-K for the twisted carrier is exactly Flash-Decoding
  / split-KV. This plan is the compiler realization of the article's "split-K is generic over the carrier" claim.
- `plans/monoid-combine-stages.md` — the completed predecessor (M1/M3/M4/M5): the combine is a derived `CombineStage`
  array, the finalize lives in the `REDUCE` codec's `c` field (`c<cta>a` atomic / `c<cta>k` deferred), and
  `_partition.deferred_combine_tilegraph(carrier, …)` is the carrier-generic combine kernel. **The combine half is
  done;** this plan is the producer half.
- The tile-lowering map: `deplodock/compiler/pipeline/passes/lowering/tile/ARCHITECTURE.md`.

## The gap (precise)

The **combine** is unified: `kernel/_combine.emit_combine` (intra-CTA lanes/warps) and
`enumeration/_partition.deferred_combine_tilegraph` (cross-CTA) both drive off the carrier's algebra surface
(`carried_names` / `combine_partials` / `combine_states`) — `Accum` and `Monoid` alike.

The **producer** is split. Today, three pins keep the cross-CTA split additive-matmul-only:

1. `enumeration/_moves.coop_reduce_knobs` emits `enc_reduce(serial, fold, coop)` — **no `cta`** (the MONOID reduce
   defaults `cta = 1`); `_streaming_leaves` hardcodes `cta=1` too.
2. `enumeration/_build.monoid_build` calls `_rebracket_k(…)` **without `grid=`**, so even a pinned `cta>1` never threads
   the split-K `K_s` GRID axis (the matmul build passes `grid=(k_s, d.cta)`; the monoid build does not).
3. `enumeration/140_atomic_free_splitk` gates `op.algebra is AlgebraKind.SEMIRING` (+ `mma_atom is None`) — it won't
   fire on a MONOID reduce.

Net: a non-matmul reduction **cannot** split cross-CTA, whatever its carrier. The combine that would fold those partials
already exists and is already GPU-verified for the additive `Accum`, the `(m, l)` softmax monoid, and the flash
`(m, l, O)` carrier (`tests/compiler/test_monoid_reduce_kernel.py::test_deferred_finalize_*`).

## Design: the cross-CTA producer mirrors the intra-CTA one

The intra-CTA cooperative producer already does the carrier-generic thing: it partitions the reduce axis across threads
(the `K_c` lane) and lets each thread accumulate the **carrier's state** in registers; `emit_combine` folds the state
across the partition. The cross-CTA producer is the **same move, one level up**: partition the reduce axis across CTAs
(the `K_s` GRID axis), let each CTA accumulate the carrier's state, and write **each state component** to a workspace
`partial_i[K_s, …]`; `deferred_combine_tilegraph` folds them.

So the producer is parametrized only by the carrier's state arity:

- **additive `Accum`** (`state = (acc,)`): one workspace; the matmul's existing Write-retarget (the `acc` value IS the
  output). The degenerate 1-component case.
- **online-softmax `Monoid`** (`state = (m, d)`) / **Welford** (`(n, μ, M₂)`) / **flash** (`(m, d, O)`): one workspace
  **per state component**; the producer writes the partial *state* (not the finalized output), and the combine kernel
  finalizes (`o / d`, `M₂ / n`, …) via `deferred_combine_tilegraph`'s `finalize` / `out_value`. `split_carrier`
  (`ir/stmt/carrier_algebra`, already used by `chain_build`) separates the state the producer must expose.

The finalize fork is **one generalized pass** (the `140` rename we agreed on — e.g. `150_cross_cta_finalize`): fire on any
fully-tiled op with `cta > 1`, read the carrier off the block, and splice the carrier-generic combine. `atomic` stays
legal only for an additive `Accum` (`_predicates.atomic_finalize_legal`); a twisted `Monoid` is non-additive → the fork
narrows to the deferred kernel (the `e^{Δm}` rescale can't be an `atomicAdd`) — which is why **attention has only a
kernel arm, by design**.

## Where each change lands

1. **Offer `cta > 1` on the MONOID reduce** (`_moves.coop_reduce_offers` / `coop_reduce_knobs`): enumerate a split-K
   degree for the cooperative reduce (mirror `reduce_offers`' `SPLITK_CHOICES`, pin-honored). Start additive-only
   (`Accum` carrier); a `Monoid` carrier opts in at milestone 3.
2. **Thread `grid` through `monoid_build`** (`_build.monoid_build`): pass `grid=(k_s, d.cta)` to `_rebracket_k` when
   `cta > 1` (the `_k_s_axis` derivation + the GRID binding the matmul build already use — reuse, don't fork).
3. **Emit the carrier state to workspaces** (the new bit): when `cta > 1`, the producer's terminal must write each of the
   carrier's `state` components to its own `partial_i[K_s, …]` instead of the finalized output. For an additive `Accum`
   this is the existing single Write-retarget; for a `Monoid` it is `split_carrier` + N Write-retargets. The
   escape-analysis / `K_s`-in-index machinery is shared with the matmul.
4. **Generalize the finalize fork** (`140` → `150_cross_cta_finalize`): drop the `SEMIRING`-only gate; fire on any
   `cta > 1` carrier; build the combine via `deferred_combine_tilegraph` with the carrier + its N workspaces +
   `init_ops` (the carrier's `identity`) + the op's `finalize`.
5. **Generalize the combine geometry** (`_partition`): `additive_reduce_tilegraph` / `monoid_reduce_tilegraph` hardcode a
   2-D `(M, N)` output tile (`GridTile(16×16)`). A general reduction output is N-D (`(M,)`, `(B, M, N)`, …) — generalize
   `_grid_thread_axes` / `_combine_block` to the actual output rank (the partition axis `K_s` stays the serial fold).

## Milestones (single branch, `make test` green after each)

1. **Additive cross-CTA reduce (plumbing checkpoint).** Offer `cta>1` for an `Accum` MONOID reduce; thread `grid`;
   reuse the single-Write retarget; generalize the finalize fork's gate + the combine geometry for the reduce's output
   rank. Verify `x.sum(dim=…)` with `cta=2` splices a combine kernel and matches torch. *Low marginal value (cooperative
   already parallelizes a reduce) — its job is to land the generic plumbing under a 1-component carrier.*
2. **Rename + unify the finalize.** `140` → `150_cross_cta_finalize`, carrier-driven, no `SEMIRING` gate (the matmul is
   now just the 1-component instantiation). `make test` unchanged for the matmul path (byte-identical deferred output).
3. **Twisted carrier cross-CTA (flash split-KV — the payload).** Offer `cta>1` on a `Monoid` carrier; emit the partial
   `(m, l, O)` state to 3 workspaces via `split_carrier`; the generalized finalize folds them
   (`monoid_reduce_tilegraph`, already GPU-verified for this carrier). Accuracy vs eager on an SDPA shape with a long KV
   split across CTAs. This is Flash-Decoding / split-KV; the article's twisted-monoid split-K made real.
4. **Search + docs.** Surface the `cta` degree as a reduce-offer dimension for carriers (the finalize stays the codec's
   `c`-letter); record the carrier→producer derivation in the tile-lowering ARCHITECTURE; note that the producer now
   mirrors the intra-CTA cooperative producer exactly (one move, two partition levels).

## Risks / constraints

- **Value asymmetry.** Milestone 1 (additive cross-CTA reduce) is mostly plumbing; the cooperative reduce already covers
  reduce-scale parallelism. Don't over-invest there — it is the checkpoint, not the goal. Milestone 3 (flash split-KV) is
  the win.
- **State emission is the hard part.** The producer must expose the carrier's *partial state*, not the finalized output
  — the same restructuring `chain_build` does with `split_carrier`. Reuse it; do not re-derive the flash geometry.
- **Combine geometry generalization.** The matmul's 2-D `(M, N)` combine kernel must generalize to the reduce output
  rank without regressing the matmul (its 2-D path stays a special case).
- **Accuracy-critical.** Every milestone is gated on `Accuracy vs eager … PASS` (the twisted carrier especially — the
  rescale is where a wrong fold hides). Do not ship an unverified arm.
- **Atomic stays additive-only.** The twisted carrier's cross-CTA finalize is kernel-only by construction
  (`atomic_finalize_legal` returns `False` for a `Monoid`) — not a missing feature.

## Verification

- Structural: `deplodock compile --code "<reduce>" --ir cuda` under a `cta=2` pin shows a second `__global__` (the
  combine kernel) and the `c<cta>k` codec; the matmul path stays byte-identical after milestone 2.
- Accuracy: `deplodock run --code "<sum/softmax/sdpa>" -v` → `Accuracy vs eager … PASS` for each carrier at `cta>1`.
- Kernel-level (already passing): `tests/compiler/test_monoid_reduce_kernel.py::test_deferred_finalize_*` proves the
  combine for the additive / `(m,l)` / flash `(m,l,O)` carriers; this plan supplies the producers that feed them.
- `make test` + `make lint` green at each milestone.
