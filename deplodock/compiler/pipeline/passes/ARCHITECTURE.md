# Pass-authoring invariants

Rules that apply to EVERY pass in this tree (`frontend/`, `loop/`, `lowering/`). Per-dialect details live in
[`../ARCHITECTURE.md`](../ARCHITECTURE.md) (pass order, knob table, fork semantics). The **tile-lowering** phase
(`lowering/tile/`) is the canonical instance of the invariant below — a **purely algebraic moveset, no
specializations**: it dispatches on the carrier algebra (`MAP` / `SEMIRING` / `MONOID`), never on a named shape
(matmul / pointwise / attention) — flash attention is the `MONOID` algebra on the streaming schedule (a twisted
monoid is a monoid), selected structurally, not a distinct kind.

## No shape-specific pattern matching

A pass must not dispatch on enumerated shapes ("if this is the gated-MLP body do X, if it is the QK^T body do Y").
Each named shape that needs handling is evidence of a missing GENERAL rule; find the per-element formulation that
makes the old and new shapes degenerate cases of one code path. Shape dispatch compounds: every new model
architecture would add a sibling branch to every pass it touches — the combinatorial explosion of compiler
complexity this invariant exists to prevent. It also breeds divergent incidental behavior (per-branch dtype or
layout rules that drift apart) and silently narrows coverage to the shapes someone happened to name.

How to comply:

- **Write the rule per element, not per shape.** Example: `lowering/tile/split/010_split_demoted.try_split_demoted`
  classifies each multiply operand independently (plain `Load` stays put; computed cone becomes a producer
  materialized over exactly the axes it reads). Norm→linear, scale→matmul, SDPA P@V, and rotary QK^T are
  *instances* of that one rule, not branches — and a shape nobody designed for (a weight-side dequant cone) is
  covered for free.
- **Gate in the negative.** Enumerating admissible shapes is shape matching by another name. Walk the body and
  report the first thing the transform *fundamentally cannot do*, like `lowering/_predicates.classify_fragment_epilogue`
  (the epilogue folds unless it has an ineligible op/dependency) — the eligible set then grows with the renderer
  instead of with a hand-maintained list.
- **Bail conservatively on well-formedness, never on shape identity.** `return None` / `RuleSkipped` for a body
  the rule doesn't fully understand is fine; the conditions must be structural properties (escaping values,
  symbolic extents, mixed dtypes), not "is this the X kernel".
- **Phrase dataflow conditions over cones, don't hand-roll the walk.** `Body.backward_cone` / `forward_cone` /
  `defs_die_at` (`ir/stmt/body.py`) are the shared slicing substrate: a rule asks for a cone and judges its
  *properties* (members, external reads, escapes) — construction never fails, so every bail stays a rule-side
  condition. See the dependence-cones section of `compiler/ir/ARCHITECTURE.md`.
- **When generalizing an existing rule, normalize its incidental divergences** (one dtype rule, one index rule)
  and name the behavioral deltas explicitly in the commit — don't preserve two behaviors behind one entry point.

## Resolve the hardware-atom binding once, structurally, at the tile level

The same invariant applies *across* the tile→kernel boundary: the kernel materializer must not re-recognize structure
the tile IR already holds. The **atomize** step (`lowering/tile/_atomize.py`, called from `020_schedule` when it builds
the warp / cooperative option — *not* a standalone pass) resolves the algebra→hardware-atom binding once and stamps it on
the *schedule* (never the op tree, so `op_cache_key`, which digests `lower(op.op)`, stays byte-identical). Resolving it at
option-build time means an atom that **cannot** be bound (e.g. a non-`Load` operand — a computed-cone / demoted matmul)
is rejected at fork construction, alongside `_check_warp_static_k`, instead of failing several passes later:

- a warp `SemiringKernel` → an `AtomBinding` (`ir/tile/binding.py`): the A/B operands bound to roles by which output
  grid axis each operand's OWN leaf `Load` index carries (structural — not a flattened-loop scan), plus `b_trans`, the
  fold accumulator, and the projection epilogue. `_warp` reads the binding instead of `lower()`-ing the `Semiring` and
  pattern-matching the result.
- a cooperative/ILP `MonoidKernel` → a `ReduceBinding`: the `MonoidAtom` (accumulator dtype) + partition widths, with the
  shuffle/tree fold mechanism left **derived** (`ReduceStage.combine`), never stored.

The atom spec is subtyped by kind (`ir/tile/atom.py`: `AtomKind` is the fixed mma cell selected by name; `MonoidAtom`
is the spec-less cooperative combine — its realization is derived, so it carries only a dtype). The Semiring binder
(`bind_contraction`) is node-addressable so warp-flash can later reuse it on flash's nested QK^T / PV; that recursion is
deferred until those inner contractions are structural `Semiring` nodes carrying geometry (see the pass docstring).

## Recognize the schedulable structure once — the tile skeleton

The same "resolve once, never re-recognize" invariant governs the recognize→schedule boundary *inside* tile lowering.
`010_recognize` stamps a **`Skeleton`** (`ir/tile/skeleton.py`) onto the `TileOp` alongside the op tree: a nested tree of
`Scope`s mirroring the carrier, each owning its parallel axes and at most one `ReduceAxis`. The `ReduceAxis` records the
recognized facts a schedule is chosen over — the fold `carrier`, cooperative eligibility, and (for a contraction) the
operand→role binding — so `020_schedule` *reads* them instead of pattern-matching the algebra on every call (it no longer
keeps a `_coop_carrier` predicate or an `isinstance(... Monoid)` re-derivation). Structural facts live on the skeleton;
the menus, occupancy heuristics, and legality filters stay in the scheduler. The skeleton is a schedule-side index, not a
second source of truth: the op tree still owns the combine, and `op_cache_key` digests `lower(op.op)`, never the skeleton
(`TileOp.skeleton` is `compare=False`).

The enabling **normalization**: a contraction's reduce (K) axis carries `carrier = Semiring.as_monoid()` — the
carrier-algebra fact that a SEMIRING is a MONOID with a `⊗` lift — so a contraction's K is structurally identical to any
`Monoid` reduce axis and the Semiring↔Monoid duality never reaches scheduling. Flash (a `Monoid` over a nested `Semiring`)
nests the inner score contraction as a child scope. The builder is **total** (an unbindable contraction stores
`binding=None` rather than raising), so the walk never fails a recognition that the per-cell fallback handles downstream.
