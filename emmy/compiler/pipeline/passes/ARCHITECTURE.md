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
the tile IR already holds. The **atomize** step (`lowering/tile/_atomize.py`, called from the `_schedule` helper inside `010_recognize` when it builds
the warp / register-tiled option — *not* a standalone pass) resolves the algebra→hardware-atom binding once at fork-emit
and feeds it into the `Contraction` structural node (`_schedule._contraction_node`), so materialize reads the operands /
`acc` / epilogue off the node and only `factorize`s. Resolving it at option-build time means an atom that **cannot** be
bound (e.g. a non-`Load` operand — a computed-cone / demoted matmul) is rejected at fork construction, alongside
`_check_warp_static_k`, instead of failing several passes later:

- a warp / register-tiled `CONTRACTION` contraction → the `(a_load, b_load, acc, epilogue)` operand→role facts
  (`_atomize.semiring_binding`): the A/B operands bound to roles by which output grid axis each operand's OWN leaf `Load`
  index carries (structural — read off the annotated loop, not a flattened-loop scan), plus the fold accumulator and the
  projection epilogue. Those facts are stamped straight onto the `Contraction` node at fork-emit (the node is the single
  source of truth — it re-derives `b_trans` off `b_load`); `_factor.factorize` reads them off the node instead of
  `lower()`-ing the contraction and pattern-matching the result. A `STAGE` pin follows the same rule: the option
  builders resolve it against the built node ONCE (`_resolve_warp_stage` / `_resolve_scalar_stage` — transport
  eligibility, the slab K-chunk `bk_elems`, the depth clamps) and stamp the resolved `Stage` (or `None`, gmem-direct)
  on the `TileOp`, so the materializer's one staged driver applies it verbatim, deciding nothing.
- a cooperative / ILP reduce (`PLANAR` / `TWISTED`, or a non-output-tiled `CONTRACTION`) needs **no** binding here — its
  accumulator dtype + the shuffle/tree fold mechanism are **derived** at materialize time (`emit_combine` off the carrier
  + `ReduceStage.combine`), never stored. Its one schedule-time staging decision follows the same
  resolve-once-structurally rule: `_schedule._row_stage` detects the fused norm→linear shared row when the cooperative
  partition is chosen and stamps a `sync` `Stage` naming it (`smem`) on the `TileOp` — a derived schedule field, not a
  knob — so `_factor._tile_reduce_axis` only applies it, never re-detects.

The atom spec is subtyped by kind (`ir/atom.py`: `AtomKind` is the fixed mma cell selected by name; `ScalarAtom`
is the plain scalar fma cell). The contraction binder (`bind_contraction`) is loop-addressable so warp-flash can later
reuse it on flash's nested QK^T / PV; flash's inner score IS now a structural `Contraction` **node** (per-cell
`TilePlan()` today, `source` of the streaming `Reduction` — the `Reduction ⊃ Contraction` composition), so warp-flash is
just that node gaining a warp `TilePlan` — no new path.

**The move catalog** (`lowering/tile/_catalog.py`) is the permitted-move enumeration the schedule emit forks over, keyed
on `AxisRole`: `scalar_tile_moves()` is the legality-guarded scalar register-tile product (`par × reg`, `block_threads ≤
1024`) with per-cell `""` as the conservative option-0, returned by `_schedule._tile_specs` for an unpinned contraction
so `compile` / `tune` explores the tile space (each spec → a structural `Contraction`-node leaf under `TILE@<k_axis>`; an
env pin wins via `Knob.narrow`). Warp / reduce / stage move families and the hierarchical `build_fork_tree` levels (the
MCTS laziness + multi-node flash bundling) fold in next; a flat list suffices for today's single-node scalar product.
