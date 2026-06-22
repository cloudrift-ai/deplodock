# Algebra-licensed decomposition move space — moves as axis factorizations, legality + recombine from the carrier

**Branch:** `feature/move-composer` (caps the axis-walk / monoid-DAG / carrier-analysis line of work)
**Status:** in progress — phases 1–3 landed (byte-identical). Phase 1: `ReduceCarrier.combine_partials` /
`combine_operands` (uniform recombine accessor; `kernel/_combine.py` routes through it). Phase 2: `partition/iterdag.py`
— the `iter_dag` derived view; `walk_nest` builds it and the four skeletons are projections of its nodes. Phase 3:
`partition/decompose.py` — `AxisDecomp` + `legal_decomps` (carrier-trait legality); `matmul_reduce_offers` /
`coop_reduce_offers` delegate their legality to it. Phases 4–8 (materialize-on-DAG, split-KV, skeleton dissolution,
free-axis/tensorize fold-in, streaming-vs-tree) remain. Reframe the move composer's search space so each scheduling move
is an **index-axis factorization**, with its
**legality and its recombination operator read off the carrier's algebra** instead of hand-coded per regime. Split-K, split-KV,
cooperative-reduce, strip-mine, tensorize, and streaming-vs-tree collapse into ONE carrier-parameterized "factor an axis,
recombine via the carrier" move. The cost pick and the hardware realization of each factored piece stay OUTSIDE the algebra. The
terminus: the four typed skeletons (`PointwiseSkeleton` / `MatmulSkeleton` / `CoopReduceSkeleton` / `FlashSkeleton`) **dissolve** into
one derived **iteration-DAG view over the `LoopOp` body** — the body *is* the DAG, and one `build_partition(dag)` factors its axes.
See `plans/move-composer-axis-walk-scheduler.md`, `plans/monoid-dag-carrier-annotation.md`, `plans/algebraic-carrier-analysis.md`.

## The thesis

Every move in the composer is already secretly an index-set decomposition justified by the carrier's algebraic laws — the code
just checks legality ad hoc (divisibility + budget) and hand-codes each recombine per regime. State it at full generality:

> **move space = algebra · pick = cost · realization = hardware**

- **Algebra** defines the *equivalence class* of factorizations that compute the same value, and supplies the *recombination
  operator* for each. A decomposition law is many-to-one: all factorizations are equal by construction.
- **Cost** picks the representative within the class (which split, what size, tensor vs scalar). Algebra is silent on speed —
  associativity says streaming and tree-reduce are equal, not which is faster (linear-scan often beats parallel-scan).
- **Hardware** realizes each factored piece (placement on grid/thread/serial, smem staging, async/TMA, bank conflicts). None of
  this is a decomposition of the computation; it is how already-decomposed work maps to the machine.

This is a *sharpening* of the thread's "algebra says legal, cost picks," not a reversal: the legal move **space itself** becomes
algebra-generated rather than three hand-written per-regime trees.

## Every current move is `(factor axis, placement, recombine-op, realization)`

The unification, with today's hand-coded instances mapped onto it:

| move (today)         | axis factored                  | recombine operator (algebra)        | placement (hardware)  | realization (hardware)   | code today                                  |
|----------------------|--------------------------------|-------------------------------------|-----------------------|--------------------------|---------------------------------------------|
| strip-mine `fk`      | reduce `K → K_f × K_i`         | carrier combine (serial in-thread)  | `K_f` REGISTER        | serial accumulate        | `_replace_k_scalar`                         |
| split-K              | reduce `K → K_s × K_rest`      | carrier combine across CTAs         | `K_s` BLOCK grid      | atomic / atomic-free     | `_replace_k_scalar` (`splitk`, `k_s`)       |
| cooperative-reduce   | reduce `K → K_c × K_rest`      | carrier combine across threads      | `K_c` THREAD          | warp-shuffle / smem-tree | `_replace_k_coop` (`br`, `k_c`)             |
| streaming (flash)    | reduce `KV` serial             | carrier streaming merge (left fold) | `KV_o` SERIAL_OUTER   | serial                   | `build_flash_tile`                          |
| **split-KV (flash)** | reduce `KV → partition`        | `combine_states` cross-partition    | grid / thread         | tree / atomic-free       | **not built**                               |
| tensorize            | `M,N,K → atom blocks`          | semiring `+` over blocks            | WARP/REGISTER/ATOM    | `mma.sync`               | `_replace_k_warp`, `build_warp_matmul_tile` |
| free-axis tile       | parallel `A → A_b × A_t × A_r` | none (degenerate)                   | BLOCK/THREAD/REGISTER | —                        | `_split_free_axis`                          |

Read the rows: split-K, cooperative-reduce, split-KV, and strip-mine are **the same move** — factor a reduce axis, recombine with
the carrier — differing only in *where the partition lands* and *how the combine is realized*, both of which are cost/hardware
choices. Tensorize is "a semiring axis admits a block-decomposition whose block-op is a hardware atom." Free-axis tiling is the
degenerate case (a product decomposition of a parallel index, no recombine).

## Legality is a carrier-trait query

The per-regime divisibility + combine hand-coding becomes a query on traits the carrier already exposes
(`ir/stmt/base.py::ReduceCarrier`: `associative` / `commutative` / `has_identity`; `ir/elementwise.py`: `identity` / `_IDENTITY`,
the `_SEMIRING` pairing / `distributes_over`):

| trait                                                 | licenses                                                                                                         |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| `associative`                                         | splitting the reduce axis at all (partition + recombine)                                                         |
| `commutative`                                         | reordering partials → unordered tree / atomic combine / split-KV (flash's `commutative=True`)                    |
| `has_identity`                                        | masking a non-divisible / symbolic axis: ceil-div + **identity-fill** (`carrier.identity`, not a hard-coded `0`) |
| semiring `distributes_over` + `(⊕,⊗)` matches an atom | block-tensorizing the contraction (the atom is the block-op)                                                     |

So the move enumerator stops asking "is this regime allowed this knob" and asks "does the carrier's algebra license this
factorization."

## We are already half-built

The recombination operator is the part most people would expect to be the hard new machinery, and it already exists:

- **`combine_states`** is the carrier's partial-combine, **already authored** for flash (`_flash.py:94-117`) and documented as "the
  form the cross-partition combine (cooperative-tree / split-KV / split-K cross-CTA reduce) folds."
- **`kernel/_combine.py`** (`emit_combine_states`, `MonoidWarpShuffle`, `MonoidTreeHalve`) **already reuses that one operator
  unchanged** across warp-shuffle and smem-tree realizations — i.e. the recombine is already realization-agnostic for `Monoid`.
- `Accum`'s recombine is its own scalar op-fold; `Monoid`'s is `combine_states`. The one missing seam is a uniform
  **`ReduceCarrier.combine_partials`** so the decomposition move reads ONE operator regardless of carrier kind.

The genuine gap is only in `moves.py` / `tree.py`: per-regime offer functions (`matmul_reduce_offers`, `coop_reduce_offers`,
`warp_offers`, strip-mine) each hand-code legality + an implied combine, and split-K / coop / split-KV are three code paths for
one law.

## Proposed interface — three layers, only two are new code

The design has three layers; the realization layer **already exists** and must not be duplicated:

1. **Input — the iteration-DAG view** (`IterDag`), a *derived* read over the `LoopOp` body (not a stored dataclass, not a dialect).
2. **Move — `AxisDecomp` + `legal_decomps`**, the carrier-parameterized decomposition, replacing the per-regime offer functions.
3. **Realization — `TileOp`**, the existing Role-tagged σ-rewritten tower. The chosen schedule is *already* stamped structurally
   here; no new schedule dialect.

**Layer 1 — the body as a DAG.** The four skeletons are lossy typed extractions of structure already in the body. Replace them with
a derived view:

```python
# partition/iterdag.py  (new — a VIEW, computed on demand like Loop.algebra_kind; no stored field, no serialization)
@dataclass(frozen=True)
class AxisNode:
    axis: Axis                       # the index axis (extent, symbolic flag)
    role: AxisRole                   # PARALLEL | REDUCE
    carrier: ReduceCarrier | None    # the reduce carrier (None for PARALLEL); the algebra the moves query
    parent: AxisNode | None          # nesting/containment (the loop-nest order)
    body: tuple[Stmt, ...]           # the statements scoped at this axis

def iter_dag(loop_op: LoopOp) -> IterDag:
    """Walk the body once → axis nodes (role + carrier from `Loop.algebra_kind`/the carrier) plus edges:
    NESTING (free-axis chain, QK^T-d nested in flash-KV), READ_AFTER_REDUCE (softmax sum reads max —
    the monoid-DAG edges), PRODUCER_CONSUMER / ABSORBED_SEMIRING (flash's P@V folded in the carrier).
    Subsumes PointwiseSkeleton/MatmulSkeleton/CoopReduceSkeleton/FlashSkeleton — they become projections
    of this one view, then are deleted."""
```

**Layer 2 — the move.** A single carrier-parameterized decomposition:

```python
# partition/decompose.py  (new)
@dataclass(frozen=True)
class AxisDecomp:
    axis: Axis                  # the index axis being factored
    factors: tuple[int, ...]    # e.g. (K_s, K_f, K_i) — extent product ≤ axis.extent (== unless masked)
    placement: tuple[Role, ...] # BLOCK / THREAD / WARP / SERIAL_OUTER / STAGE_INNER / REGISTER / ATOM, per factor
    # recombine + realization are NOT stored here — derived:
    #   recombine operator = carrier.combine_partials   (algebra; None for a parallel axis)
    #   realization (atomic / shuffle / tree / mma / serial) = chosen by cost+hardware from placement

def legal_decomps(node: AxisNode, budget) -> list[AxisDecomp]:
    """Factorizations the node's carrier algebra licenses (associative→split, commutative→reorder,
    has_identity→mask, semiring→atom-block), pruned + ranked by the cost heuristics. A PARALLEL node
    (carrier None) → product decompositions only, no recombine."""
```

`ReduceCarrier` grows one accessor (the uniform recombine):

```python
class ReduceCarrier:
    def combine_partials(self) -> tuple[Assign, ...]:
        """The operator that folds two partial reductions — combine_states for Monoid,
        the scalar op-fold for Accum. The ONE recombine the decomposition move uses,
        independent of realization (atomic / shuffle / tree / mma)."""
```

One `build_partition(dag)` walks the `IterDag`, branches the Fork tree on `legal_decomps(node)` per axis, and `materialize`
realizes the chosen `AxisDecomp`s — the existing `_replace_k_*` functions become *realizations* selected by `placement`, not
separate regimes, and the four `build_*_tree` entry points collapse into this one.

## Dissolving the skeletons — the body is the DAG

The skeletons are not the structure; the body is. `PointwiseSkeleton` / `MatmulSkeleton` / `CoopReduceSkeleton` / `FlashSkeleton`
each hold "(free axes, reduce axes + carrier, body, leading)" — which the `LoopOp` body already encodes as nesting + carriers +
Loads. So the end state is not "four skeletons demoted to data containers"; it is **no skeletons** — `iter_dag(loop_op)` is the one
view the partition consumes. Three deliberate stances keep this from over-reaching into a new IR:

- **The DAG is a derived view, not a fifth dataclass and not a dialect.** `iter_dag` is computed on demand from the body, exactly
  as `Loop.algebra_kind` is — zero serialization, zero `op_cache_key` surface, always consistent with the body, re-derived after
  every rewrite. It caches nothing the body doesn't already determine.
- **Staleness is an argument against stamped *copies*, not against a representation.** A thing goes stale only if it is a second
  copy of truth a mutation can fail to update — a decision object stamped alongside the tower, or labels on tile nodes a rewrite
  doesn't maintain. After `080`/`085`/`017` mutate the tower, such a copy drifts; this is exactly why the existing tile IR
  *derives* split-K / cooperativity from structure (`Body.coordination()`) instead of tagging them. Consequence: the chosen
  `AxisDecomp`s stay a **transient handoff** (move tree → `materialize`), NOT threaded forward to downstream passes; downstream
  facts are recomputed as **derived views**. This is the derive-from-structure-vs-stamp-labels choice — and it is **orthogonal to
  whether a schedule dialect exists** (a dialect that *is* the source of truth has no copy to go stale).
- **The realized-schedule IR already exists — it is `TileOp`; a *separate* dialect is a distinct question, not settled by
  staleness.** The Role-tagged tower *is* the structural record of every scheduling decision; the σ-map *is* the factorization.
  Whether to ALSO add a schedule dialect turns on one test (per the audit): *would the tower-mutating passes be easier to write
  against it?* The audit's answer is **not forced either way**: most tower-mutating passes (`020` staging, `080` pipeline-peel,
  `085` warp-specialize, `040`/`070`) are **hardware-realization** rewrites that inherently operate on the concrete tower, so a
  dialect sitting *above* it would not host them and would not ease them. The genuine pain (re-derivation — §Grounding) is
  addressable without a separate stage. The one real case for a dialect — *schedule-then-lower* (compose/legalize an abstract
  schedule, realize the tower once) — is an unforced architectural bet, deferred here on cost/benefit, **not** on staleness.

So "an IR reflecting decomposition and scheduling decisions" is satisfied by what exists plus derived views: input = `iter_dag`
(derived), the move-tree→materialize handoff = `AxisDecomp` (transient), realization = `TileOp` (the schedule IR, already there),
and downstream facts = centralized derived views (the `Coordination` pattern), recomputed across rewrites. A separate
schedule-then-lower dialect is left as an open architectural option (it is not refuted — only not forced).

## Grounding — tile-IR pass audit

A four-cluster read of the ~15k lines of tile/kernel passes (structural/split-K; staging/transport/pipeline; atom/tensor-core;
tile→kernel materialization) grounds the no-dialect decision and sharpens the derived-view stance:

- **No serialized schedule dialect — confirmed in every cluster.** No pass reconstructs or serializes a separate schedule object;
  each consumes the `TileOp` tower directly by `isinstance` flavor-dispatch (`SerialTile.kind`, `StageBundle.policy`,
  `AtomTile`/`WarpTile` presence) or generic `nested()` recursion. The codebase's existing model for derived scheduling facts is
  the right one: `Body.coordination()` (a cached derived view, `ir/stmt/body.py:631`) and `kernel/_tma_groups`'s transient
  read-only dataclass ("No IR is rewritten here"). A dialect would have to be re-parsed back into exactly the tower that exists.
- **The real debt is re-derivation — fixed by a centralized derived view, NOT a carried object.** Several passes pay to recover
  facts the partition knew: `020_stage_inputs` re-solves the per-axis source-dim factorization + block multipliers from
  `Load.index` σ-decomposition (and three sites re-invert it via `affine_decode_per_dim`); `050_use_tma` re-derives the TMA box
  collapse that must stay hand-locked to the materializer's identical math (disagreement → deadlock); `015`/`017` re-derive the
  split-K axis from `Coordination.atomic_axes` 3–4×; `085` runs `_split_by_role` twice; the MMA tier is double-encoded
  (`knobs["MMA"]` vs `AtomTile` presence). The decisive constraint: schedule-to-schedule rewrites exist (`017`/`080`/`085`/`020`/
  `015` mutate the tower), so a stamped/carried decision object (a sidecar) would go stale across them — the correct fix is to
  extend the `Body.coordination()` derived-view pattern to cover the factorization/source-dim facts (computed once per
  tower-state, many readers). NB: this is the derive-from-structure-vs-stamp-labels choice; it is **orthogonal to** whether a
  separate schedule dialect exists (which an authoritative dialect would not stale) — see the stances above.
- **`combine_partials` is a small unification — confirmed.** The recombine is already realization-agnostic *in data*
  (warp-shuffle / smem-tree behind one signature per kind; `Monoid.combine_states` reified as a program, `as_state_merge`
  packaging it for the cross-CTA split-K combine in `017_atomic_free_splitk.py:242`). Only the DISPATCH is split across four sites /
  two files (`kernel/_combine.py` scalar+monoid; `017` sum+state-merge). The `Monoid` half exists today; only the scalar `Accum`
  op-fold needs exposing through the same accessor.
- **The `knobs["MMA"]` bridge → structure-tightening, not a dialect.** `011_lower_atom_cell` and `kernel/005_lower_atom_tile`
  already lower the tier purely off `AtomTile`/`Mma` structure (no knob read); the live knob-readers are `020`/`040`/`015`/`060`/
  `025`/`010`. Converting them to read `AtomTile`/`WarpTile` presence is mechanical (proven by the two already migrated) and adds
  no dialect — the `Atom` spec is already on the node. Caveat: `MMA`/`is_warp` also serves as a tuner *search-space coordinate*
  (the `_enumeration` featurizer, partition `Level`), so it stays a search knob; the promotion only removes it as a *lowering*
  discriminator.

Net: (1) `combine_partials` is small; (2) downstream decision facts are best served by centralized **derived views** (the
`Coordination` pattern) rather than stamped sidecars — because a sidecar (not a dialect) is what schedule-rewrite passes would
stale, the same derive-from-structure principle that makes `iter_dag` a view rather than a dataclass; (3) a *separate* schedule
dialect is neither forced nor refuted by the audit — the tower-mutating passes are mostly hardware-realization rewrites convenient
on the concrete tower, and the only real upside (schedule-then-lower) is an unforced architectural bet, left open. The skeleton
dissolution itself needs no dialect either way; the realized schedule stays in `TileOp`.

## What stays OUT of the algebra

Three things must remain external, or the frame over-reaches:

1. **The cost pick** — irreducibly external; the whole point of enumerating is to choose by predicted/measured latency (the prior).
2. **Hardware realization** of each piece — placement on grid/thread/serial, smem staging, async/TMA pipelining, warp
   specialization, bank-conflict avoidance, register pressure. The FA3 levers are pure hardware mapping with zero algebra.
3. **Candidate pruning.** Algebra *over-generates* — associativity licenses a million K-splits; only a few are worth benching.
   Today's `*_offers` encode the pruning (best-first, ~256 threads, ~16 cells). The legality moves to traits; the pruning/ranking
   stays — just separated from the legality so it can't silently reject a legal-but-unranked decomposition.

## First consumer — flash split-KV is the forcing function

Do NOT build this as a speculative grand refactor. The cheapest validating consumer is **flash split-KV** (deferred in
`plans/online-softmax-flash-attention.md`): it needs "factor the KV axis, recombine via `combine_states`" — which is split-K's
mechanism with a different carrier. Build split-KV by **generalizing split-K into the carrier-parameterized decomposition move**
rather than hand-writing a third copy of partition-and-combine. The carrier already advertises legality (`commutative=True`) and
supplies the operator (`combine_states`); only the move enumerator + a realization (atomic-free, cf.
`plans/atomic-free-streamk.md`) are missing. If the `AxisDecomp` abstraction cleanly expresses split-K AND split-KV AND coop with
one move, it is validated; only then fold tensorize and streaming-vs-tree in.

## Phasing & the byte-identical gate

Land behind the discipline used by the sibling plans: per-kernel `deplodock compile` compare under a fixed `PYTHONHASHSEED`, green
`make test`. Each refactor phase must be **byte-identical** on kernels already covered; only phases 5, 7, 8 add new coverage. The
order front-loads the byte-identical refactors (1–4) that build the DAG + move substrate, validates the move abstraction on a real
new carrier (5, split-KV), THEN dissolves the skeletons (6) once the substrate is proven, and finally generalizes the remaining
moves (7–8).

1. ✅ **`ReduceCarrier.combine_partials`** — add the uniform recombine accessor (`combine_states` for `Monoid`, scalar op-fold for
   `Accum`, fragment-add for `Mma`) plus `combine_operands` (the second-state names). Pure addition; `kernel/_combine.py`'s
   `MonoidWarpShuffle` / `MonoidTreeHalve` emission routes through it. No schedule change.
2. ✅ **`iter_dag` view, with the skeletons derived from it (shim).** Built `partition/iterdag.py` — the `IterDag` derived view
   (axis nodes: role + carrier; the reduce-node set matches the legacy `iter_of_type(Loop)` recursion). `walk_nest` builds the DAG
   and reads the skeleton fields off its nodes — the four skeletons are *projections* of one source of truth. Byte-identical
   (the move-composer tests that pass keep passing).
3. ✅ **`AxisDecomp` + `legal_decomps` over DAG nodes, reduce axes only, byte-identical.** Built `partition/decompose.py`;
   `matmul_reduce_offers` / `coop_reduce_offers` delegate their legality to `legal_decomps` reading the carrier traits
   (associative → split, commutative → partition, has_identity → mask). Byte-identical on the covered matmul / coop-reduce
   kernels (all covered carriers are associative + commutative + have identity), proven by a reference-enumeration test.
4. **`materialize` consumes the DAG + chosen `AxisDecomp`s, byte-identical.** `_replace_k_scalar` / `_replace_k_coop` /
   `_replace_k_warp` become *realizations* selected by `placement`; `_assemble` reads `IterDag` nodes (free axes, reduce axes,
   body slices) instead of typed skeleton fields. The skeleton projection from phase 2 is now unused by the move/materialize path.
5. **Flash split-KV via the move** (the forcing consumer — the third instance, a NEW carrier). `legal_decomps` offers a KV
   partition on the `FlashCombine` carrier (`commutative=True`), recombined by `combine_partials`, realized atomic-free (cf.
   `plans/atomic-free-streamk.md`). First genuinely new coverage; validate accuracy + `run --bench` on decode-shaped attention.
   This proves the `AxisDecomp` abstraction expresses split-K **and** split-KV **and** coop with one move before any deletion.
6. **Dissolve the skeletons (the target state).** With the substrate validated, delete `PointwiseSkeleton` / `MatmulSkeleton` /
   `CoopReduceSkeleton` / `FlashSkeleton` (and the phase-2 projection shim), collapse `build_pointwise_tree` / `build_matmul_tree` /
   `build_coop_reduce_tree` / `build_flash_tree` into one `build_partition(dag)`, and have `walk_nest` yield the `IterDag` directly.
   Byte-identical on every covered kernel (pointwise / matmul / warp / coop / flash) — pure removal of the typed-extraction layer.
7. **Fold in free-axis tiling and tensorize** — the parallel-axis (no-recombine) case and the semiring-block (atom-as-block-op)
   case become `AxisDecomp` instances over `IterDag` nodes, retiring `thread_offers` / `reg_offers` / `warp_offers` as special
   trees. Byte-identical on pointwise / matmul / warp kernels.
8. **Streaming-vs-tree as a cost choice** — once split-KV is a decomposition, flash streaming and tree-reduce are two placements of
   one associative law; the prior picks per shape (the research's #1 author-acknowledged gap — FA3/FlashInfer fix this heuristically).

## End state & the irreducible residue

Completing axis-walk + this plan through **phase 6** is the point where **moveset generation and classification-based
partitioning end**: every Fork branch offers the *same* `legal_decomps(node, budget)` family filtered by carrier traits, the
per-regime trees (`build_pointwise_tree` / `build_matmul_tree` / `build_coop_reduce_tree` / `build_flash_tree`) collapse into one
`build_partition(dag)` that factors the next axis, and the regime skeleton types (`PointwiseSkeleton` / `MatmulSkeleton` /
`CoopReduceSkeleton` / `FlashSkeleton`) are **gone** — `iter_dag(loop_op)`, a derived view over the body, is the one structure the
partition consumes. There is no per-regime move vocabulary and no typed-skeleton layer left — that is "generic moves in all
branches," and it is reachable, bounded, and byte-identical-gated (no open research question blocks it).

What does **not** dissolve is **carrier recognition** — you cannot apply a generic move to a carrier you do not have, and minting
the carrier is the one step that stays a classification. It splits sharply:

- **MAP / MONOID / SEMIRING carriers are already generic.** `classify_algebra` reads them bottom-up from the body in one line
  (`ir/algebra.py`); no bespoke recognizer. For these regimes "classification" is just the derived algebra tag and genuinely
  dissolves into the walk.
- **TWISTED_MONOID is irreducible (but shrinkable).** The twisted carrier does not exist in the loop body — the body holds a
  coupled-accumulator cluster, and the bespoke flash recognizer (`loop/recognize`) *mints* the `Monoid` from it (the `algebra.py`
  "expensive match, done once" boundary). The monoid-DAG fusion rule (`plans/monoid-dag-carrier-annotation.md`) can *shrink* this —
  derive the twist from an unfused monoid-DAG instead of hand-writing `flash_combine` — but cannot eliminate it: you still must
  structurally identify the monoid-DAG (plus the exp-stabilizer semantics).
- **Attention index-recovery never goes generic.** Q/K disambiguation, the GQA group, and the mask kind (`_extract_qk` /
  `_classify_rowmax`) recover a particular computation's data plumbing, not algebra; they stay bespoke regardless.

So the boundary is precise: **"bypass classification-based partitioning and moveset generation" — yes, this plan reaches it;
"bypass classification entirely" — no.** Generic moves stop at the carrier boundary. Everything below the carrier (decomposition,
legality, recombine, realization) is generic; producing a non-trivial carrier (the twisted monoid) is the surviving classification,
and the most that can be done there is to shrink the flash recognizer to "identify the monoid-DAG + recover the attention indices,"
not to delete it.

## Hard constraints

- **Algebra generates and recombines; cost picks; hardware realizes.** No cost or hardware-mapping decision moves into
  `legal_decomps`. The realization (atomic/shuffle/tree/mma) is selected downstream from `placement`, never stored as legality.
- **Pruning is preserved, not dropped.** `legal_decomps` over-generates by law; the existing best-first ranking must still bound the
  offered set, or the Fork tree explodes.
- **Derived, not stamped.** `iter_dag`, carrier traits, and `combine_partials` are reads over the body/carrier (like
  `algebra_kind`), zero serialization / `op_cache_key` surface. No fifth skeleton dataclass, no schedule dialect — the realized
  schedule stays in `TileOp`; the decision object stays transient until a schedule-rewrite pass forces a dialect.
- **Byte-identical on covered shapes.** Phases 1–4, 6, and 7 are pure refactors; the per-kernel compare must match. Only phases 5
  and 8 add coverage.
- **Consumer-driven.** The move abstraction proves itself on flash split-KV (phase 5) — a real third instance (split-K / split-KV /
  coop) — *before* the skeletons are dissolved (phase 6) or the remaining moves generalized (phases 7–8).

## Open questions

- **Atomic combine legality.** A split-K atomic realization needs the combine to be a hardware atomic op (plain `add`), which is
  stricter than `commutative`. Is that a realization-level gate (cost/hardware) or does it need a carrier trait
  (`atomic_realizable`)? Leaning realization-level — the algebra licenses the split; the atomic is one of several realizations,
  and a non-atomic carrier (flash) just uses the tree/atomic-free realization.
- **Cross-axis coupling.** Matmul register offers read the pinned `fk` (`materialize.py:177`); a clean per-axis `AxisDecomp` may
  need to carry a dependency on a sibling axis's factorization. Does the move stay per-axis with a budget thread, or do coupled
  axes share one decomposition node?
- **Placement legality vs algebra.** Some placements are illegal for non-algebraic reasons (a `SERIAL_OUTER` split-K can't also be
  the STAGE_INNER). Where does that constraint live — in `legal_decomps` (mixing concerns) or a separate placement validator?
- **Tensorize's two absorbed semirings.** Flash's P@V is absorbed into the carrier merge (not a standalone contraction), so the
  "semiring axis → atom block" decomposition can't see it. Does the decomposition frame need the absorbed-semiring annotation from
  `plans/monoid-dag-carrier-annotation.md` to tensorize flash, or does that stay a separate flash-MMA concern?
- **Centralize the re-derivation (audit follow-on, separable from the dissolution).** `020_stage_inputs` and `050_use_tma` re-solve
  the per-axis source-dim factorization / TMA box collapse independently, and three sites re-invert `affine_decode_per_dim`. The
  audit's fix is one `Coordination`-style derived view computed per tower-state (many readers), not a carried decision object. Is
  this worth doing as its own cleanup phase (it removes real duplication and the 050↔materializer hand-locked-math deadlock risk),
  or does it ride along with phase 4 when `materialize` is reworked to consume the DAG + `AxisDecomp`s? It is **not** required for
  the skeleton dissolution — flag it so it isn't conflated with it.
