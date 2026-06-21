# Algebraic carrier analysis — trait-carrying ops, a bottom-up algebra analyzer, and a trait-driven planner

> **Status (implemented).** Parts **A**, **B**, and **C1a** are landed and byte-identical (verified by `deplodock
> compile` per-kernel compare on Qwen3-Embedding-0.6B layer 0 under a fixed `PYTHONHASHSEED` — both the default and
> `DEPLODOCK_FLASH=1` paths — and a green `make test`):
> - **A** — `ir/elementwise.py` is the complete algebraic registry; the ~10 string-match sites are op-name-free trait
>   queries (`selecting`, `distributes_over` / `is_semiring_product`, `reduce_canon`, the `_REDUCE_SPELLING` registry).
> - **B** — `ir/algebra.py`'s `AlgebraKind` + `classify_algebra` / `Loop.algebra_kind` tags each reduce loop bottom-up
>   from its carrier (implemented as a **derived property**, not a normalize-stamped field — the plan's "derived cache,
>   read back from the carrier" framing, with zero serialization/codegen risk). `matmul_reduce` moved to `ir/algebra`;
>   `lowering/tile/_helpers.is_matmul_reduce` delegates.
> - **C1a** — `010_partition_loops` routes on the tag (`SEMIRING` → matmul tiling; static `TWISTED_MONOID` →
>   cooperative-reduce; else cooperative-K / pointwise).
> - **C6** — effectively satisfied: after A no op-name string-match gates remain (the cooperative combine / split-KV
>   legality already read `carrier.associative` / `commutative`; dtype reads `op.selecting`).
>
> **Deferred (C1b / C2 / C3 / C4 / C5 — the generic MMA twisted-monoid carry).** These supersede the bespoke flash
> rule by giving the MMA tensor-core tier a `(init, action, fold)` carrier interface. They are **blocked on a consumer
> that does not yet exist**: twisted monoids today route to the cooperative *scalar* path, and the MMA tensor-core
> flash tier is itself future work (the landed flash is the *scalar* tier — see `plans/masked-gqa-mma-flash-attention.md`
> Part C / `plans/masked-gqa-mma-flash-attention.md` sub-step C1). Adding the `action`/`fold` protocol + the MMA
> fragment-carry generalization now would be machinery with no live consumer; they should land *with* the MMA-flash tier
> work, behind the `action = identity` byte-identical gate the plan describes below. The analyzer (B) and the tag-driven
> planner (C1a) are the groundwork that makes them reachable.

Derive each kernel's **algebra** bottom-up from its loops, carry it as a first-class structure, and let the partition
planner + downstream passes dispatch on it instead of re-deriving the archetype from syntactic shape. This is the
inverse of the bespoke top-down recognizers (`loop/recognize/010_recognize_flash`): rather than hand-match a known
pattern and stamp a carrier, *analyze* the carrier that's already there and expose a uniform interface the scheduler
consumes.

The north-star payoff: the MMA tensor-core tier for flash (and any other twisted monoid — Welford, argmax) deploys
through a **generic `(init, action, fold)` carrier interface** rather than a bespoke per-carrier lowering rule. This
supersedes the "bespoke `FlashCombine`-keyed rule" decision in `plans/masked-gqa-mma-flash-attention.md` (Part C) once
the action hook lands — see Part C5 below.

## The framing (why this is mostly consolidation, not new machinery)

Three facts about the current tree make this tractable:

1. **The op trait tables already exist.** `ir/elementwise.py::ElementwiseImpl` is the single source of truth:
   `_COMMUTATIVE` / `_ASSOCIATIVE` (lines 59–64), `_IDENTITY` (lines 67–75), and the `commutative` / `associative` /
   `identity` / `has_identity` properties (lines 96–113). The docstring already says these exist so "the reassociation
   gates query instead of matching op names."
2. **The carrier already *is* the algebra.** `ir/stmt/base.py::ReduceCarrier` (lines 562–578) exposes
   `associative` / `commutative` / `has_identity` + `carried_names` / `partial_deps`; `Accum` forwards to its scalar
   `op`, `Mma` reports the additive-fold constants, `Monoid` (FlashCombine) reports a monoid with a per-instance
   `commutative` flag (`ir/stmt/leaves.py:461–780`). The base docstring already states rules "don't switch on the
   carrier type — they read the algebraic traits."
3. **The planner already classifies by predicate, not by op-name string.** `lowering/tile/010_partition_loops.py`
   routes on `is_matmul_reduce` / `Loop.is_reduce` / presence of a `Monoid` in the body (lines 599–617), not on
   `"add"` / `"maximum"`.

So the infrastructure is half-built. What's missing — and what this plan delivers — is (A) **finishing** the op trait
system so the ~15 remaining string-match sites become trait queries, (B) a **bottom-up classifier** (run in `LoopOp`
normalize) that tags each reduce loop with its algebraic kind, and (C) **rewiring** the planner + downstream to consume
those tags, including the `(init, action, fold)` generalization that unlocks generic MMA-tier twisted monoids.

## Hard constraints

- **Behavior-preserving through Parts A–B.** The trait system must give byte-identical answers to the current
  hardcoded sets for every known op; the analyzer is additive (it produces a structure; nothing yet *requires* it).
  Validate with `deplodock compare` per-kernel on a real model (Qwen3-Embedding-0.6B layer 0) — the three existing
  regimes (matmul / pointwise / cooperative-reduce) must not move.
- **`action = identity` is byte-identical (Part C).** Generalizing the MMA fragment-carry to call a carrier-supplied
  action between K-steps must reduce to today's zero-init + store-at-end for plain `Accum` / `Mma` (the action is the
  no-op). Plain matmul codegen does not change.
- **Traits are algebraic over the reals; fp reordering is the accepted relaxation.** float32 `add` is not
  bit-associative; `associative` means "the schedule may reorder, accepting rounding drift," which the pipeline already
  tolerates (accuracy is checked with tolerances, not bit-equality). Do NOT try to prove fp associativity — prove over
  exact algebra, keep the existing tolerance-based validation.
- **No frontend / decomposition changes.** All work lands in `ir/` (the classifier in `LoopOp` normalize), `loop/` (the
  reframed `loop/recognize/`), and `lowering/` (`tile` / `kernel`). Recognizers are reframed, not deleted.

## Part A — finish the op trait system (the "stop switching on names" layer)

`ElementwiseImpl` becomes the complete algebraic registry, and every site that string-matches op identity queries a
trait instead.

### A1 — extend the trait tables

Add to `ElementwiseImpl` (data, not code — adding an op is editing a table):

- **`selecting`** — True for ops that *select* an existing value rather than accumulate magnitude (`maximum` / `amax` /
  `minimum` / `max` / `min`). Replaces `_SELECTING_OPS` in `lowering/kernel/020_place_inits.py:79`.
- **Semiring pairing** — `_SEMIRING: dict[str, frozenset[str]]` mapping a reduce `⊕` to the products `⊗` that
  distribute over it: `{"add": {"multiply"}, ...}`. A `distributes_over(mul, add)` helper lets matmul detection ask
  "does `⊗` distribute over `⊗`'s reduce?" instead of hardcoding `"multiply"` + `"add"`
  (`recognize/010_recognize_flash.py`, `011_lower_atom_cell.py:117`, `_splitk_residual.py:111`, `_split_demoted.py:175`,
  `015_pack_fk_window.py:102`). Start minimal — only `(+, ×)` is exercised today; the table is structured so tropical
  `(min, +)` etc. are *data* if ever needed, but DO NOT add unused semirings (simplicity-first).
- **One render-spelling registry.** Consolidate the per-op CUDA/numpy spelling currently scattered across
  `Accum.render` (`leaves.py:545–552`), `kernel/ir.py::_tree_halve_op` (1245–1254), `stmt/base.py::op_to_expr`
  (206–237), and `tensor/ir.py::ReduceOp.forward` / `ScanOp.forward` (133–176) into one op-keyed table (`+=` / `*=` /
  `fmax` / `fmin`, and the numpy reduce callable). These four sites are the same dispatch written four times.

### A2 — convert the string-check sites to trait queries

The audited call sites (file:line → trait):

| site                                                                                     | today                                  | becomes                                                                     |
|------------------------------------------------------------------------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `lowering/kernel/020_place_inits.py:79,82`                                               | `_SELECTING_OPS` frozenset             | `op.selecting`                                                              |
| `lowering/kernel/010_split_register_axes.py:46,174`                                      | `_ASSOCIATIVE_REDUCE_OPS` frozenset    | `op.associative` (already a trait!)                                         |
| `loop/lifting/020_lift_reduce.py:25,54`                                                  | `_COMBINE` name map                    | keep as a tensor→loop alias, but key off the op, not a literal dict         |
| `loop/recognize/010_recognize_flash.py` (`_SUM`/`_MAX` tuples + `"multiply"`)            | `_SUM` / `_MAX` tuples + `"multiply"`  | semiring-role + `selecting` helpers (the recognizer survives — see Part B3) |
| `tensor/ir.py:133–176`, `kernel/ir.py:1245–1254`, `leaves.py:545–552`, `base.py:206–237` | per-op spelling switches               | the A1 render registry                                                      |
| `lowering/kernel/030_stamp_types.py:203`                                                 | `"multiply" and args[0]==args[1]`      | `op.name`-free square detection via semiring role                           |
| `ir/loop/ir.py:414`, `ir/stmt/blocks.py:108,320`, `ir/tile/ir.py:1179,1248`              | identity-presence / op-conflict checks | `op.has_identity` / carrier-trait checks (mostly already trait-shaped)      |

`010_split_register_axes` is the cleanest example of the win: it carries its *own* `_ASSOCIATIVE_REDUCE_OPS` set that
duplicates `ElementwiseImpl._ASSOCIATIVE` — pure redundancy, delete it and read `op.associative`.

### A3 — tests

A trait-table test asserting every known op's `(associative, commutative, identity, selecting, semiring-role)` matches
the pre-refactor hardcoded sets exactly (behavior-preservation). Lint + `make test` green; `deplodock compare` on a
model dump unchanged.

## Part B — the bottom-up algebra analyzer (computed in `LoopOp` normalize)

Rather than a one-shot stamp pass, the classification is computed in **`LoopOp` normalize** (`loop/ir.py:102`
`__post_init__` → `normalize_body`, which already runs on every (re)construction — i.e. after every rewrite). Each
reduce loop is tagged with its **algebraic kind** as a derived field. It does NOT build a separate tree: the loop nest
already carries the nesting and the carrier already lives in the loop body, so the classification is just a cheap
function of the body, re-run whenever the body changes. The scheduler then walks the loop nest it already walks,
dispatching on each reduce loop's tag.

Computing it in normalize (vs. a stamp pass) is **only safe because the tag is excluded from `op_cache_key`** (B1): a
knob re-stamped on every reconstruction would give one logical kernel many identities — which is exactly why
`020_stamp_structural_features` must run once at the end. Our tag dodges that by being out of the key, so continuous
maintenance is free and there is no "must run after recognize" ordering hazard: the tag is always consistent with the
current body.

**Why this stays cheap (the expensive match is elsewhere).** Normalize must not re-run an expensive analysis on every
rewrite — and it doesn't. The costly part — the **catalog match** that turns a raw coupled-accumulator cluster into a
verified twisted monoid (eventually SMT-backed) — is done ONCE by the recognizer (`loop/recognize`, B3), which produces
a tagged carrier (`FlashCombine` ⟺ `lse`). Normalize's classification is then a cheap read: `Accum` → `MONOID` (or
`SEMIRING` if matmul-shaped), `Mma` → `SEMIRING`, a recognized `Monoid` → `TWISTED_MONOID` by the carrier's id; an
unrecognized coupled accumulator stays `MAP` (no regression). For this the classifier must be **ir-level** (it reads
carriers, an ir concept) — an `is_matmul_reduce`-style structural check may have to move from `lowering/tile/_helpers`
down to `ir`.

**Axis assignment — why it lives on `LoopOp`, not `TileOp`.** The tag is well-defined on the `LoopOp` because the
carrier is present there. It is NOT recomputed in `TileOp` normalize: the partition planner's axis assignment **lowers
the carrier structure away** (a matmul becomes `Mma`/`ldmatrix` register fragments; a `Monoid` combine becomes an
explicit rescale + `Accum`), so the bottom-up signals don't survive in the same form. The planner (the first
`lowering/tile` pass) reads the tag from its `LoopOp` *input*, before tiling; the C6 legality gates run during that
pass, on the same pre-tile form. Post-partition nothing re-derives it — the schedule embodies the decision; if a later
tile pass ever needs it, it carries the tag forward via provenance (`Op.source`) rather than re-classifying the tiled
body.

**Why no separate structure.** Everything a richer node object would store is already recoverable:

- **Nesting → the loop nest.** Kernels are compositions of algebras (flash is `twisted-monoid { semiring(QK^T) ;
  map(scale,mask) ; <P@V inside the combine> }`), but that composition IS the loop nest. A decomposable kernel's
  sub-algebras are its inner loops / surrounding stmts; walking the nest recovers them, so a `children` field would only
  duplicate the IR.
- **Carrier → the loop body.** The `Accum` / `Mma` / `Monoid` is a stmt in the body, found by scanning, and stays the
  single source of truth: traits read from it (`carrier.associative` …, the `ReduceCarrier` surface at
  `base.py:562–572`), operand roles recompute from its loads (`classify_matmul_operands`), and the twisted catalog
  identity is the carrier's own identity (`FlashCombine` ⟺ `lse`).
- **The twist is NOT loop-isomorphic — and that's the point.** Flash's two matmuls aren't both loops: QK^T is an inner
  reduce loop, but P@V lives *inside* the `combine` carrier, not as a loop, so children-as-loops couldn't capture it
  anyway. What defines that internal structure is the **catalog entry** (`lse` ⇒ "two matmuls + a rescale"), read off
  the carrier — not a `children` list. The catalog absorbs exactly the non-loop structure, so the loop-nest-as-tree view
  loses nothing.

**The tag is a derived cache, not a second source of truth.** Because traits and structure are always read back from the
carrier, the tag can never contradict the algebra — the `MonoidNode(associative=False)` class of bug is unrepresentable,
there is no trait field to set.

### B1 — the analysis output

A single enum assigned to each reduce loop (non-reduce loops are `MAP` by default and need no tag):

```python
class AlgebraKind(Enum):
    MAP             # pointwise / functor — a non-reduce scope (the default; not stamped)
    MONOID          # a plain associative reduce      (carrier: Accum)
    SEMIRING        # a contraction / matmul          (carrier: Mma, or a matmul-shaped Accum)
    TWISTED_MONOID  # an online / coupled reduce       (carrier: Monoid — flash, Welford, …)
    SCAN            # prefix / causal — reserved, out of scope v1
```

The kind is a derived field on the reduce `Loop`, set during `LoopOp` normalize, and — like `Axis.source_axis` — is
**excluded from equality / `op_cache_key`**, which is what makes normalize-maintenance safe. Everything else the
scheduler needs is fetched from the loop on demand:

- **traits** — `carrier.associative` / `commutative` / `has_identity` (the existing `ReduceCarrier` surface);
  `distributive` is the A1 semiring-pairing query on the loop's `(⊕, ⊗)`. Read from the carrier, so the tag can never
  contradict them.
- **operands** (SEMIRING) — `classify_matmul_operands` over the loop's loads (cache if it proves hot).
- **catalog_id** (TWISTED_MONOID) — the carrier's identity (`FlashCombine` ⟺ `lse`); a future coupled-accumulator match
  that isn't yet a `Monoid` carrier records its id on the carrier it constructs.

This subsumes the current prologue/epilogue machinery without a tree: plain matmul is a SEMIRING reduce loop wrapped by
non-reduce (MAP) loops/stmts for its prologue/epilogue, so `_classify_fused_prologue` /
`has_nonlinear_post_reduce_epilogue` become "is the surrounding scope a non-reduce loop," read straight off the nest.
The kernel's top-level kind (for routing-only consumers, Part C1) is its outermost reduce loop's tag.

### B2 — the classifier

The per-reduce-loop classification (a cheap function of the loop body, run in normalize):

- **`MONOID`** — a single `Accum` whose `op.associative` (and `has_identity` for maskability). Traits from the op.
- **`SEMIRING`** — `is_matmul_reduce` (≥2 distinctly-indexed K-Loads feeding a reduce) **and** the body's `⊗`
  `distributes_over` the reduce `⊕` (A1 helper). Operand roles come from `classify_matmul_operands` on demand. Replaces
  the planner's inline `is_matmul_reduce` + the `"multiply"`/`"add"` checks with one trait-verified classification.
- **`TWISTED_MONOID`** — a recognized `Monoid` carrier (its catalog id is the carrier's own identity). The raw
  coupled-accumulator → catalog match that *creates* that carrier is the recognizer's job (B3), not normalize's — the
  key inversion is that `lowering/tile/_helpers.py::accums_independent` currently *rejects* cross-`Accum` reads
  (`l_i`/`O_i` read `m_i`) as un-fusable, and `_atom.py::classify_fragment_epilogue` bails on "accumulator consumed
  inside the reduce loop"; the recognizer treats that coupling as the **signal** of a twisted monoid, not a failure.
- **`MAP`** — any non-reduce loop (the default; not tagged).

Tiering for the recognizer establishing a novel combine's laws (B3): (1) registry/catalog lookup (cheap, covers
everything today); (2) a randomized associativity/commutativity probe as a filter for a novel combine; (3) SMT
verification of the survivor — (2)/(3) are future (catalog-only to start), noted in Scope.

### B3 — recognizers become catalog matchers feeding the analyzer

`loop/recognize/010_recognize_flash` + `loop/fusion/_flash.py` are **not deleted** — they are reframed as the **`lse`
catalog entry**: a verified matcher that recognizes the softmax-P@V shape and emits a `FlashCombine` `Monoid` carrying
its `lse` identity. Normalize's classifier then tags the loop `TWISTED_MONOID` by reading that carrier, instead of the
planner re-deriving `combine_reduces` from `isinstance(s, Monoid)` (`010_partition_loops.py:612`). New twisted monoids
(Welford, argmax) are added as new catalog matchers in the same shape — each ships its pre-verified combine + identity
(the way `_flash.py:91` hardcodes the LSE identity `(−1e30, 0, 0)`).

### B4 — tests

The classifier tags the known kernels from the dump set correctly: `k_linear_*` reduce loop → `SEMIRING` (its epilogue a
surrounding non-reduce loop); RMSNorm / `k_mul_*_reduce` → `MONOID`; flash `k_sdpa_*` → `TWISTED_MONOID` with the
carrier's id `lse`; gated-MLP prologue and SDPA P@V keep their current routing. Assert each reduce loop's tag (plus, for
`SEMIRING`, its `classify_matmul_operands` roles, and for `TWISTED_MONOID`, the carrier's catalog id) matches an
expected fixture. A separate check asserts the tag survives idempotently across a normalize round-trip but does NOT
enter `op_cache_key`.

## Part C — planner + downstream consume the analysis

### C1 — planner dispatches on the per-loop tags (staged)

The planner already walks the loop nest; now it reads each reduce loop's tag instead of re-deriving the archetype. Stage
it:

- **C1a — top-level routing first (behavior-preserving).** Replace the inline classification at
  `010_partition_loops.py:599–617` (`matmul_reduces` / `nonmatmul_reduces` / `combine_reduces`) with a dispatch on the
  outermost reduce loop's tag: `SEMIRING` → matmul-output tiling; `MONOID`/`TWISTED_MONOID` → cooperative-reduce; no
  reduce (`MAP`) → pointwise. The routing decisions stay; their *inputs* come from the tag, not re-derivation.
  Byte-identical.
- **C1b — use the nested tags where it unlocks something.** Only for the cases that need the nesting — first a
  `TWISTED_MONOID` loop whose inner `SEMIRING` loops must be tensorized as MMA cells while the carrier holds the
  rescaled fragment (C2–C4). Each reduce loop consults its carrier traits to certify its axis transform (assoc→tile,
  distributive→split-K, identity→mask). Plain matmul / pointwise stay on the C1a routing path until there's a reason to
  go deeper, so the blast radius is confined to the twisted-monoid case it enables.

### C2 — promote `(init, action, fold)` onto `ReduceCarrier`

The carrier interface gains three methods naming its reduce surface for a *tiled* schedule:

- **`init`** — seed the accumulator (today: zero-init `c` for `Mma`, `op.identity` for `Accum`).
- **`action(acc, partial_stats)`** — the **twist**: transform the carry given a new partial. For `Accum` / `Mma` this
  is the **identity** (no rescale) — which is *why* the plain monoid is already MMA-transparent. For `Monoid` /
  `FlashCombine` it is the per-step rescale (`O *= alpha`, `l *= alpha`), the content of `FlashCombine.combine_states`.
- **`fold(acc, partial)`** — accumulate the contribution (today: `+=` / `mma.sync`).

### C3 — generalize the MMA fragment-carry to call `action`

Today the `Mma` `c` fragment is zero-init-at-declaration + store-at-end (`ir/kernel/ir.py:705–707`). Generalize the
fragment lifecycle across a serial-outer (`K_o`) loop to: `init → (fold ; action)* → finalize`. For plain matmul,
`action` is identity → the emitted CUDA is byte-identical to today (the hard-constraint gate). For a twisted monoid,
`action` is a `RegScale` on the fragment regs → the loop-carried, per-step-rescaled accumulator the MMA flash kernel
needs.

This is exactly sub-step **C1 of `plans/masked-gqa-mma-flash-attention.md`** ("loop-carried, per-step-scaled `Mma`
fragment"), generalized: instead of a flash-only carry, *any* carrier supplies its action.

### C4 — generic twisted-monoid MMA path

With C2 + C3, the MMA-tier flash kernel (two chained MMAs — QK^T score fragment feeding P@V, with the rescale between
them) falls out of the generic path keyed on a `TWISTED_MONOID` loop with inner `SEMIRING` loops, reusing the
existing
`Mma` / `RegFragment` / `ldmatrix` staging for the two cells. The cross-cell fragment routing (`c:f32` score →
`a:f16` P operand) and the softmax-stat reduction over the `c`-fragment row layout remain the flash-specific pieces,
but they hang off the catalog entry's `action` / finalize, not a standalone lowering rule.

### C5 — supersede the bespoke flash rule

`plans/masked-gqa-mma-flash-attention.md` Part C recommends a **bespoke `FlashCombine`-keyed lowering rule** explicitly
to dodge blast radius ("generalizing `010_partition_loops` to model a reduced-N matmul-reduce is invasive"). This plan's
C2/C3 is the *general* version it set aside — and it is now justified because the generalization is **gated on
`action = identity` being byte-identical**, so plain matmul is provably untouched. The decision flips: land the action
hook (small, test-guarded, additive to the carrier protocol) and the flash MMA tier is a catalog client, not a bespoke
rule. If C3 proves to move plain-matmul codegen at all, fall back to the masked-gqa plan's bespoke rule for flash and
keep the generic interface for the scalar tier only.

### C6 — downstream gates read the analysis

Consolidate the remaining schedule-legality gates onto the nodes' derived trait properties: SPLITK legality
(`010_partition_loops.py` `force_splitk_one` reasoning) reads the reduce loop's `distributive` (A1 semiring query);
cooperative tree-combine reads `carrier.associative` / `carrier.commutative` (split-KV legality); dtype selection
(`020_place_inits`) reads `op.selecting`. These already *want* traits (A2); the tag + carrier are the one source.

## Risks / open questions

- **Blast radius of C3** (touches the shared MMA lowering every matmul uses). Mitigate: the `action = identity`
  byte-identical check, compare-tests on existing kernels before merge, and the C5 fallback to the bespoke flash rule if
  it isn't clean.
- **Coupled-accumulator detection vs the `accums_independent` / `classify_fragment_epilogue` bail.** The recognizer must
  run *before* (or those gates must defer to) the twisted-monoid verdict — the dependency they reject is the dependency
  it needs. Concretely: `accums_independent` should consult the loop's tag and allow a coupling already classified as a
  known twisted monoid.
- **fp associativity** — algebraic-over-reals only; keep tolerance-based validation (hard constraint).
- **SMT / verified lifting of *novel* carriers is out of scope** — catalog-only v1. A loop matching no catalog entry
  stays on its current path (no regression), and `log()`s that it was unclassified so the gap is visible.
- **Semiring-table generality** — resist adding tropical/boolean semirings until a consumer exists; structure it as
  data so it's a one-line add when one does.

## Scope

**In:** complete the op trait system + delete the duplicated string-match sets (A); a bottom-up classifier in `LoopOp`
normalize that tags each reduce loop with its algebraic kind, with the recognizers reframed as catalog matchers (B);
planner dispatch on the tags + the `(init, action, fold)` carrier interface + the generic MMA twisted-monoid carry that
supersedes the bespoke flash rule (C). Static + dynamic seq at existing test parity.

**Out (future):** SMT / verified-lifting synthesis of novel coupled carriers (catalog-only for now); semirings beyond
`(+, ×)` and the LSE log-semiring; `SCAN` / segmented-scan (causal attention as a prefix, not a fold) — the algebra
makes it reachable but it is its own work.

## Sequencing

A (pure refactor, zero behavior change — safe to land alone) → B (additive analysis, nothing depends on it yet) → C
(consume it; the flash MMA payoff and the only real blast radius live here). A and B are low-risk and independently
mergeable; C is the one to land behind the byte-identical gate + compare-tests.

## Test strategy

- **Behavior-preservation:** full `make test` green; `deplodock compare <before> <after>` per-kernel unchanged on
  Qwen3-Embedding-0.6B layer 0 through Parts A–B and for the three existing regimes through C.
- **New:** the A3 trait-table test; the B4 analyzer-classification fixtures; MMA-flash parity reusing
  `tests/compiler/e2e/test_flash_attention.py` extended with the masked-gqa plan's GQA / masked / MMA-tier cases.
- **Integration:** re-run the `plans/masked-gqa-mma-flash-attention.md` validation commands and confirm the flash MMA
  kernel deploys via the generic twisted-monoid path (one fused `k_sdpa…` kernel with `MMA=mma_m16n8k16_f16`).
