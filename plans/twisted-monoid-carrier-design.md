# Twisted-monoid carrier: no class, no ψ field — and drop the `TWISTED_MONOID` algebra kind

A design-decision record, prompted by the FlashAttention Part-1 blog post
(`cloudrift-landing/.../learning-flashattention-the-hard-way-part-1/index.md`), which proves a *twisted monoid* is a
plain monoid `(C, ·, e)` transported through a bijection `ψ`: `x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`. Three questions, in
increasing depth:

1. Should the tile IR carry a separate `TwistedMonoid` carrier class?
2. Or parametrise the existing `Monoid` carrier with a `ψ` field?
3. Should `TWISTED_MONOID` exist as a member of the `AlgebraKind` enum *at all*?

**Decision: no, no, and no.** Keep one `Monoid` carrier with the twist baked into `merge` (Q1/Q2), **and drop
`TWISTED_MONOID` from `AlgebraKind`** (Q3) — it is not a distinct algebra, and a "carrier *algebra*" enum should not
carry it. The genuinely-distinct thing (the streaming flash *schedule*) is a structural/tier property, selected one layer
below the algebra. This is the conclusion that makes the architecture's "dispatch on the carrier algebra" invariant and
the blog's "no attention code path" actually hold.

## Q1 — separate `TwistedMonoid` class: no

Models a distinction that does not exist: transport of structure proves a twisted monoid *is* a monoid — same axioms
M1–M3, isomorphic via `ψ` (blog, **Twisted Monoid** subsection). A second class would fork the shared machinery
(`identity`, `associative`/`has_identity`, `combine_states`, split-KV legality via `commutative`, the cooperative-tree
combine) and fight the blog's punchline: "deplodock's compiler has no attention code path… FlashAttention falls out as
the … case of the reduction lowering it already had" (blog **Wrap-Up**).

## Q2 — a `ψ` field on the `Monoid` carrier: no

The carrier would store the plain monoid `·` plus `ψ` and reconstruct `⊕ = ψ(ψ⁻¹·ψ⁻¹)` at lowering. Two problems:

- **Codegen wants `⊕`, never `·`/`ψ` separately.** The kernel emits the rescale-and-merge directly; `Monoid.merge` already
  *is* that fused `⊕` as a short `Assign` program (`loop/recognize/_flash.flash_combine`, `_flash.py:55-117`). Storing
  `ψ` and recomposing is indirection you immediately collapse back.
- **Materializing `ψ⁻¹` is the numerical sin online softmax exists to avoid.** `ψ⁻¹` un-rescales to absolute coordinates
  — `D = d·eᵐ`, `O = o·eᵐ` (blog, **Why it had to cancel — the semiring underneath**), the overflow-prone quantities the
  relative-to-max carrier never forms. Keeping `⊕` as the primitive matches the numerics by construction.

`ψ` has real value for *verification/discovery* — holding the trivially-associative direct product `·` and asserting `ψ`
bijective gives associativity of `⊕` for free (the blog's "monoidization as a pass" direction: **When Is a Loop Secretly
Associative?**, **Wrap-Up**). But that is a **construction-time** concern. The expensive match — coupled-accumulator
cluster → verified twisted monoid — is already done once in the recognizer (`_flash.flash_combine`), which *emits* a
`Monoid`; everything downstream just reads the carrier (`ir/algebra.py:11-17`). If a monoidization synthesizer ever
lands, `ψ` is an *input* to that recognizer step, not a field on the lowered carrier.

## Q3 — `TWISTED_MONOID` as an `AlgebraKind` member: drop it

This is the substantive change. `AlgebraKind` is the **carrier-algebra** dispatch key (`MAP` / `SEMIRING` / `MONOID` /
`TWISTED_MONOID`), and the whole-architecture invariant is *dispatch on the algebra, no specializations*. But
`TWISTED_MONOID` is **not a distinct algebra** — by transport of structure it is the `MONOID` algebra. What it actually
encodes is a *carrier representation* (a tuple `Monoid` combine vs a scalar `Accum`) plus a *schedule* (streaming flash
vs cooperative reduce). Encoding a schedule fact as an algebra member is a category error, and the code already
half-collapses it.

### The code already treats them as one algebra in 4 of 6 sites

| Site | `MONOID` vs `TWISTED_MONOID` today |
| --- | --- |
| `_cut.py` `_ALGEBRA_TIER` | **identical** → both `Tier.COOP_REDUCE` (comment: *"streaming flash — a cooperative reduce regime"*) |
| `120_stage.py` | **identical** → `if op.algebra in (MONOID, TWISTED_MONOID): skip` (both smem-free) |
| `010_build.py` `_BUILDABLE` | **identical** → both listed, both buildable |
| `_classify.py` `coop_monoid` (line 36) | a twisted carrier with **no** nested reduce returns `_Regime(MONOID)` — the enum value is already discarded |
| `_validate.py` `_ALGEBRA_TIERS` (line 79) | **diverge** → `MONOID`→`{COOP}`, `TWISTED_MONOID`→`{STREAMING}` (disjoint knob slice) |
| `080_streaming.py` + `_classify.py:38` | **diverge** → the streaming build move, gated on `TWISTED_MONOID in algebras and nested_reduce` |

The real streaming discriminator is **structural, not algebraic**: `_classify.py:38` splits on `nested_reduce` (flash's
QK^T score reduce nested inside the KV stream), not on the algebra. A `Monoid` carrier *without* that nesting is already
the plain cooperative reduce. So the two genuinely-divergent sites are about *tier/schedule*, which the `Tier` lattice
(`MAP < SCALAR < COOP < STREAMING < WARP`) already models independently of `AlgebraKind`.

### The rehoming (what "drop it" entails — a refactor, ~6 files, not a delete)

1. **`ir/algebra.py`** — remove `TWISTED_MONOID` from `AlgebraKind`; `classify_algebra` returns `MONOID` for a `Monoid`
   carrier (it *is* a monoid). The `isinstance(s, Monoid)` test stays — it now feeds the structural streaming check, not
   a separate algebra value.
2. **`enumeration/_classify.py`** — select the streaming regime structurally: *a tuple `Monoid` carrier* (the same
   `isinstance(s, Monoid)` read, since `dag.algebras` can no longer distinguish it) **and** `nested_reduce`. Record it as
   a `streaming: bool` (or a `tier: Tier`) on `_Regime`/the seed. Delete the `coop_monoid` un-routing hack — moot once
   twisted *is* `MONOID`.
3. **`enumeration/_validate.py`** — keep `Tier.STREAMING` (a real, disjoint knob slice). Select it from the structural
   streaming flag, not an `AlgebraKind` member: `MONOID` maps to `{COOP, STREAMING}` and the seed's tier picks, or the
   seed declares its tier directly.
4. **`enumeration/080_streaming.py`** — gate the build move on the structural streaming marker instead of
   `op.algebra is AlgebraKind.TWISTED_MONOID`.
5. **`_cut.py` / `120_stage.py` / `010_build.py`** — drop the separate `TWISTED_MONOID` entries; `MONOID` now covers
   them (they already behave identically).
6. **Docs + tests** — `tests/compiler/ir/test_algebra_kind.py` expects `Monoid`→`TWISTED_MONOID`; flip to `MONOID` plus a
   structural streaming assertion. Update the `AlgebraKind` / `_classify` docstrings, `stmt/blocks.py:90`, and the tile
   `ARCHITECTURE.md` so the algebra list reads `MAP / SEMIRING / MONOID` and "streaming flash" is described as a **tier**,
   not an algebra.

### One genuine open sub-decision

Where the streaming signal lives once the enum member is gone:

- **(a) a `streaming: bool` on `_Regime`/seed** — minimal, local; the build/validate gates read it. Recommended.
- **(b) a `tier: Tier` on `_Regime`** — more general (the seed carries its target tier directly), but a wider change to
  how regimes hand off to the fork passes.

Either keeps `Tier.STREAMING` as the disjoint knob slice; the difference is only how the seed advertises it.

## Net result

`AlgebraKind` becomes exactly the distinct carrier algebras — `MAP / SEMIRING / MONOID` (+ reserved `SCAN`) — and flash
attention is the `MONOID` algebra lowered on `Tier.STREAMING`, chosen by structure. That is the faithful form of the
invariant: adding a model architecture is never a new algebra; the streaming schedule is the same move set on the same
algebra, selected by the shape of the reduce. Q1/Q2 reject adding *more* structure (a class, a field); Q3 removes the
*one* piece of structure that was masquerading as algebra.

## Bulletproofing — claims tested against the codebase

| Claim | Evidence (verified) |
| --- | --- |
| Twist baked into one `Monoid.merge`, no ψ | `_flash.flash_combine` builds a single `Monoid`; fields `state/partial/merge/identity/commutative/axes/state_b/combine_states` (`stmt/leaves.py:715-722`) — no `psi`. |
| No `ψ` symbol in source | `grep -rin 'psi' deplodock --include=*.py`: no carrier/recognizer hit. |
| `Accum`→`MONOID`, `Monoid` leaf→`TWISTED_MONOID` today | `classify_algebra` branches on `isinstance(s, Monoid)` (`ir/algebra.py:113-119`). |
| `_cut.py` already lumps both into `COOP_REDUCE` | `_ALGEBRA_TIER` maps both → `Tier.COOP_REDUCE` (`_cut.py:84-85`). |
| `120_stage.py` already lumps both | `if op.algebra in (MONOID, TWISTED_MONOID)` (`120_stage.py:70`). |
| Twisted-without-nesting already routes to `MONOID` | `_classify.py:36,62-72` (`coop_monoid` → `_Regime(MONOID)`). |
| Streaming split is structural (`nested_reduce`), not algebraic | `_classify.py:35,38`. |
| Only `_validate` + `080_streaming` genuinely diverge | `_validate.py:78-79` (`{COOP}` vs `{STREAMING}`); `080_streaming.py:65`. |
| `Tier.STREAMING` is a tier, independent of `AlgebraKind` | `_validate.py:67`; `_cut.py` `Tier` lattice. |
| Blog proves twisted monoid IS a monoid via ψ | Blog **Twisted Monoid** subsection (transport of structure + `ψ(m,D,O) = (m, D·e⁻ᵐ, O·e⁻ᵐ)`). |

Blog references are anchored to **section headings**, not line numbers, on purpose: the post is in a separate,
actively-edited repo (`cloudrift-landing`) and its line numbering drifts (verified — `# References` shifted ~8 lines
between drafts). Deplodock code line anchors were each re-checked against the current tree.

Open follow-up noted while verifying: `plans/algebraic-carrier-analysis.md` is referenced by `ir/algebra.py`,
`tests/compiler/ir/test_algebra_kind.py`, and `tests/compiler/ir/test_algebra_traits.py` but is **not present** in
`plans/`. Out of scope; flagged for the maintainer (restore or drop the stale references).
