# Composite build unification — status + next push

## Context

The IR is already **composite**: `MonoidReduction(carrier, inner: Contraction | None)` — a carrier fold optionally
wrapped around a contraction. The unification makes the *realization* and the *build* mirror that composite so the
four tile-build moves (`monoid_build`, `chain_build`, `warp_chain_build`, `warp_build`) stop being four hand-written
compositions of the same primitives.

## What landed (branch `feature/unify-carrier-combiners`)

The **realization half is unified** — one `Combiner` (`ir/twist.py`) drives `combine(carrier)` over a tier backend:

- `Combiner` base owns the carrier-generic `combine()` (split → project → seed / declare / rescale / consume /
  normalize → `CombinePhases`) AND the `fold` / `scalar` / `state` / `seed_state` synthesis, written ONCE over three
  tier primitives: `comps` (per-partition SSA suffixes), `_temp` (the `Reassign` temp name), `_reduce` (the
  cross-partition fold). Carrier-shape-generic: twisted (flash) gets the accumulator pipeline; non-twisted (online
  softmax `(m, d)`, pure reduce `(acc)`) is stats-only.
- `MmaTwist` (fragment tier) and `ScalarCombiner` (scalar tier) supply only `comps` / `_temp` / `_reduce`, `pointwise`,
  and the accumulator hooks (`bind_score` / `declare_accum` / `rescale_accum` / `consume_accum` / `normalize_accum`),
  which stay per-tier — structurally divergent (mma realizes the consume from the graph `Mma` cells + rescales fragments
  in place; scalar emits the arithmetic), not a value-representation difference.
- Wired live: the scalar FA-2 flash (`chain_build`) and the serial online-softmax reduce (`monoid_build`) realize via
  `ScalarCombiner`; the warp flash (`warp_chain_build`) via `MmaTwist`. `010_split_register_axes` learned that a
  `Reassign` "defines" its name for replication dataflow (`_repl_defs`) so the FA-2 `O[d]` accumulator replicates.
- New: `loop/recognize/020_recognize_online_softmax` (the `ONLINE_SOFTMAX` knob) fuses the standalone two-pass softmax
  into one online-softmax `(m, d)` `Monoid` pass; `_build._atom_cell` factors `warp_chain_build`'s two near-identical
  per-cell builds (`atomize_cell` + `ko` K-wrap + `AtomTile`).

The **build half is shared at the primitive level**: `atomize_cell`, `split_carrier`, `_rebracket_k`, `_split_axis`, and
now `_atom_cell` (warp tier). The four moves are the honest 2×2 of (single contraction vs chained pair) × (scalar vs
warp), dispatched compositionally by `070_coop_reduce.reduction_build` off `MonoidReduction(inner)`.

## The next push: a single `build_contraction` spanning matmul + flash

Goal: `warp_build` (standalone matmul) and the flash inner cells share ONE contraction-cell builder, so
`build_monoid(reduction)` recursively calls `build_contraction(reduction.inner)` — the build mirrors the composite IR
the way the `Combiner` now mirrors the carrier algebra.

This was investigated and **deferred** — it is a real assembly-tower + K-tower refactor, not duplication removal. Two
concrete walls (read against `_build.warp_build` vs `_build._atom_cell` / `warp_chain_build`):

### Wall 1 — different K strategies (by design)

- `warp_build` (matmul) builds a `_rebracket_k` **tower**: `K → K_o·(bk·fk·atom_k) + K_f + K_i` — the `bk`-serial ×
  `fk` register-tile decomposition an SGEMM needs for reuse — then `atomize_cell(k_name=None)` walks it to the cell.
- The flash inner has no register tiling on the QK^T `K`: a single `ko` loop of `kt = D/atom_k` steps (`_atom_cell`).
- `_rebracket_k(bk=kt, fk=1)` does NOT reproduce the flash form — it emits a degenerate `K_o(1) > K_i(kt)` two-level
  tower, not the single `ko(kt)` loop. Unifying means flash inherits a spurious `K_o` level, or the matmul loses
  `bk`/`fk` tiling — degrading a path to merge code that isn't duplicated.

### Wall 2 — different `AtomTile` creation mechanisms

- `warp_build` never builds an `AtomTile` — it lays **domain axes bound `Role.ATOM`** (`n_a` / `m_a`) and `assemble`'s
  `_wrap_tower` reconstructs the `AtomTile` from those bindings.
- The flash inner (`_atom_cell`) builds the `AtomTile` **inline** in the body.
- So `_atom_cell` can't serve `warp_build` without rewriting how `warp_build` lays its domain *and* how `assemble`
  tier-wraps it.

### Proposed approach (if pursued)

1. **Unify the AtomTile mechanism first.** Pick one — most likely make `warp_build` emit inline `AtomTile`s like the
   flash inner (drop the `Role.ATOM` domain binding), and verify `assemble`'s `_wrap_tower` / `_free_layers` still build
   the WarpTile tower around an inline-`AtomTile` body. `assemble` already handles inline `AtomTile`s, so this is
   a convergence, not new machinery. Gate: matmul e2e byte-identical (`test_ops_vs_torch`, `test_matmul_mma_masked`).
2. **Parameterize `build_contraction(cell, *, atom, k_spec, geometry, ...)` over a `k_spec`** — either a `tower(bk, fk)`
   (matmul) or `steps(kt)` (flash inner). Shared body: `atomize_cell` + the AtomTile wrap; the K-wrap dispatches on
   `k_spec`. This is honest (the K strategies genuinely differ) — `build_contraction` is the named seam, `k_spec` the
   one parameter that varies — not a forced merge.
3. **`build_monoid(reduction, tier)`** then reads: `build_contraction(reduction.inner)` if present (the produce/consume
   cells), `combine(carrier)` via the tier `Combiner`, assemble. `monoid_build` is the `inner=None` case; `chain_build`
   the scalar `inner` case; `warp_chain_build` the warp `inner` case — three call sites of one composition instead of
   three functions.

### Verdict / cost

Payoff is **organizational** (the cell-fusion primitive `atomize_cell` is already shared; this lifts the composition to
mirror the IR), against **real risk** to deployed matmul + flash codegen (the assembly-tower change in step 1).
Estimated multi-day, GPU-gated. Worth doing only with the full `-O3` perf suite (`make bench-kernels`) as a
gate, not folded into other work. Until then, the primitive-level sharing + the compositional `reduction_build` dispatch
is the clean fixed point.

## Verification gates for the next push

- `tests/compiler/e2e/test_ops_vs_torch.py`, `test_matmul_mma_masked.py` (matmul — Wall 1/2 risk).
- `tests/compiler/e2e/test_flash_tensorcore_*.py`, `test_flash_attention.py`, `test_attention_chains.py` (flash).
- The CPU oracles `tests/compiler/passes/test_fragment.py` / `test_scalar_combiner.py` (combiner emission unchanged).
- Deployable perf: `make bench-kernels` (the `-O3` matmul/flash perf must not regress — the whole point of the `bk`/`fk`
  tower is matmul throughput).
