# Reimplement the tile/kernel lowering as one path — red-state-first recovery

## Stance

This is an **aggressive clean-room reimplementation**, not an incremental refactor. The earlier byte-identity-oracle
approach is abandoned. We tear out the duplicated tier × transport implementations, enter a deliberate **red state**, and
recover the suite **one test group per commit**, using the tests as the spec. No commit lands unless its target test
group is green and `make lint` is clean. Nothing touches `main`.

The audit (see history of this file) established the invariant holds at the **dispatch** level (routing keys on carrier
algebra / atom eligibility / dynamic / mma — never a named shape) but is violated at the **code-sharing** level: the same
skeletons are re-implemented per **tier** (scalar / warp-MMA / monoid) and per **transport** (staged smem / gmem-direct /
MMA fragment). The reimplementation makes the *implementation* as single-path as the *routing* already is.

## Target architecture — the single path

New / consolidated modules. Each replaces a family of duplicated functions.

| New home                                               | Subsumes (deleted)                                                                                                                             | Single responsibility                                                                                                                                        |
|--------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lowering/_masking.py`                                 | `kernel/009._clamp_load`, `kernel/_stage_expand._clamp_source_index`, `kernel/005.gmem_guard`+`_masked_k_zero`, `assembly/_slab._kmask_source` | `mask_index(coord, bound, mode={clamp,zero})` + `locate_symbolic_k(reads, reduce_names)` — the one edge-clamp, the one zero-fill, the one symbolic-K locator |
| `enumeration/_build.py::_rebracket_k`                  | `_replace_k_scalar`, `_replace_k_monoid`, `_replace_k_warp`                                                                                    | one K re-bracket, params: `unit`(1/`br`/`atom_k`), `partition`(grid `K_s` / thread `K_c` / none), `fk`, `masking`                                            |
| `enumeration/_build.py::_split_axis`                   | `_split_free_axis`, `_warp_axis`                                                                                                               | one free-axis σ-split, params: `factors`, `placements`, `atom_tier`, `interleave`                                                                            |
| `enumeration/_build.py` move fns (`reduce_decomp`/`free_tile`/`monoid_build`/`warp_build`) | the re-implemented K/axis skeletons inlined in each | rewritten to **compose** `_rebracket_k` + `_split_axis`; they stay distinct per-pass moves (the incremental F3-b fork model is preserved — there is no monolithic `build()`). `chain_build` and the test-only `build_dag` oracle stay |
| `ir/stmt/carrier_algebra.py::classify_merge_program`   | the role-walk inside `_frag_softmax` and `_combine`                                                                                            | one carrier `merge` classifier → `{fold, exp, scale}` roles; two thin geometry emitters consume it                                                           |
| `assembly/_assemble.py` (generic streaming-carry edge) | `assembly/_warp_chain.py`, `assembly/_tower.wrap_carry_tower`, the bespoke `split/005_warp_chain` build                                        | flash assembles like any fused graph: the kv-stream is a scheduled serial-carry axis, the C→A handoff a real `SMEM`/`INLINE` staged edge                     |

What stays distinct (verified non-duplication): `chain_build` (`_build.py:409-473`) is a genuine carrier-restructure
(split carrier + register-promote the P@V `d` axis); it is *called by* `build`, not merged into it. The algebraic
dispatch keys (`_classify.py`, `_moves.legal_decomps`, `_atom.eligible_atoms`) are untouched — they already work.

## Rules of engagement

1. Work in a git worktree off a fresh branch from `main` (`feature/unify-tile-lowering`). Never commit to `main`.
2. Each rung: **delete the duplicated impls + rewire call sites first** (this is what makes the rung's tests go red),
   then implement the unified core until the named test group is green.
3. Commit only on green for that rung's group **and** `make lint` clean. One recovery == one commit. Commit message names
   the rung and the test group recovered. End every commit message with the `Co-Authored-By` trailer.
4. Tests are the spec. Do **not** chase byte-identity with the old kernels; chase the assertions. Where an old test
   encoded an incidental artifact of a deleted path (e.g. a specific helper name), rewrite the test to assert the
   behavior, and say so in the commit.
5. `make test` runs at `-Xcicc -O1` (correctness lane). CUDA-bound rungs (flash e2e, materialize) need a GPU; if the
   working host has none, mark the rung **blocked-on-GPU** and recover the CPU-checkable tests in its group first,
   committing the GPU half separately once on a CUDA box.
6. ARCHITECTURE.md / CLAUDE.md updates ride **in the same commit** as the code that invalidates them (mandatory per
   repo contribution rules) — especially the tile `ARCHITECTURE.md`, which currently documents the deleted per-tier
   functions.
7. A rung that won't green within its scope is **reverted**, not patched around — re-enter red and try a different cut.
   The branch history stays a clean ladder of green commits.

## Red baseline (commit 0)

Land the new module files as signatures + `raise NotImplementedError`, switch the pass call sites to them, and delete
the bodies of the duplicated functions. Run `make test` once and **record the red set** (the list of failing test files)
in the commit body — this is the recovery backlog. The suite is deep red; this is the only intentionally-red commit, and
it exists so each later rung is a measurable green delta against a known baseline.

→ verify: `make lint` clean (stubs are syntactically valid); `make test` red with the recorded failure set.

## Recovery ladder — one commit per rung

Each rung lists the test group that must go green (real files under `tests/compiler/`). Rungs are ordered by dependency:
low-level shared helpers first, so later rungs build on green foundations.

### Rung 1 — Masking core
Implement `_masking.mask_index` + `locate_symbolic_k`; route `kernel/009`, `kernel/_stage_expand`, `kernel/005`,
`assembly/_slab` through them.
→ green: `passes/test_masked_tile.py`, `test_matmul_mma_masked.py`, `test_masked_cooperative_reduce.py`,
`passes/test_lower_atom_tile_guards.py`.

### Rung 2 — K re-bracket unified
Implement `_rebracket_k`; delete `_replace_k_scalar/_monoid/_warp`. Partition placement comes from `legal_decomps`
(grid `K_s` vs thread `K_c`); masking comes from Rung 1. This also closes the latent `_replace_k_scalar` symbolic gap —
today it tiles a symbolic K with a plain `//` at the hint and no mask (safe only because `010_split_demoted` always
routes symbolic-K SEMIRING to the warp consumer); giving the scalar path the `masking` param makes it correct even if
that split ever declines.
→ green: `passes/test_reduce_offers.py`, `passes/test_reduction_rules.py`, `passes/test_decompose_rules.py`,
`test_monoid_reduce_kernel.py`, `test_mma_atomic_free_splitk.py`.

### Rung 3 — Free-axis split unified
Implement `_split_axis` (3-way thread tile and 4-way warp/atom tile as `atom_tier=` + `factors`); delete
`_split_free_axis`/`_warp_axis`. Preserve the masked-interleave option as a param.
→ green: `passes/test_register_tile_rules.py`, `passes/test_thread_offers.py`, `passes/test_strided_coop_rows.py`,
`passes/test_wrap_tower_warp_role.py`.

### Rung 4 — Build moves rewired onto the shared primitives
Rewrite the per-tier move functions to **compose** the Rung 2–3 primitives instead of re-implementing the skeleton:
`reduce_decomp`/`free_tile` (scalar — applied incrementally by `060_reduce_tile`/`090_thread_tile`/`100_register_tile`),
`monoid_build` (`070_coop_reduce`), `warp_build` (`050_warp_build`). The incremental F3-b pass/fork model is preserved —
there is **no** single monolithic `build()`; each pass keeps applying its one move (`010_build` still seeds via
`seed_graph`). `chain_build` stays a distinct function (`070_coop_reduce` selects it when `dag.chain` is present). Keep
`build_dag` — it is the test-only scalar byte-identity oracle that `test_tile_ir_invariants.py` asserts the incremental
pipeline against; update it in lockstep with the rewritten moves so the oracle still holds (this is an internal
consistency check, not the old-kernel byte-identity the Stance abandons).
→ green: `passes/test_matmul_rules.py`, `passes/test_atomize_cell.py`, `passes/test_seed_demoted.py`,
`passes/test_contraction_chain.py`, `passes/test_structural_features.py`, `passes/test_tile_ir_invariants.py`,
`passes/test_tile_naming.py`, `passes/test_cut_offers.py`.

### Rung 5 — Carrier classifier shared
Implement `carrier_algebra.classify_merge_program`; reimplement `_frag_softmax` (fragment geometry) and `_combine`
(lane/smem geometry) as thin emitters over it. Two emitters, one analysis.
→ green: `test_cooperative_combine.py`, `passes/test_frag_softmax.py`, `passes/test_fragment_row_reduce.py`.

### Rung 6 — Slab address + tower walk shared
One cache-axis→slab affine decode used by producer write (`_stage_expand`) and consumer read (`005._mma_src_index`);
one `StageBundle`/compute-phase/reduce tree visitor shared by `005` and `100_materialize_tile`.
→ green: `passes/test_stage_scalar.py`, `passes/test_stage_inputs_mma_probe.py`, `passes/test_fused_edge.py`,
`passes/test_multi_block_assemble.py`, `passes/test_fuse_sibling_register_cells.py`,
`passes/test_materialize_warp_specialize.py`, `passes/test_place_inits_warp_specialize.py`.

### Rung 7 — Flash into the generic assembler (the north star — now in scope)
Teach `enumeration` to emit a streaming-carry `TileGraph` + `Schedule` (kv-stream = scheduled serial-carry axis; C→A
handoff = real staged edge) and `assemble_block`/`_assemble_group` to assemble it. Delete `assembly/_warp_chain.py` and
`_tower.wrap_carry_tower`; collapse `split/005_warp_chain` into an offer/eligibility shim (the hardcoded scope list
shrinks to a real trait — the scheduler simply won't bind a streaming-carry outside its capability). **GPU-bound.**
→ green: `passes/test_streaming_symbolic_chain.py`, `e2e/test_flash_attention.py`, `e2e/test_flash_cooperative_kv.py`,
`e2e/test_flash_tensorcore_generated.py`, `e2e/test_flash_tensorcore_reference.py`,
`ir/stmt/test_reduce_carrier.py`, `passes/test_warp_specialize_deadlock.py`.

### Rung 8 — Integration + parity (final commit)
Whole suite green (`make test`), knob-pin validation intact (`passes/test_knob_pinning.py`,
`passes/test_knob_pin_validation.py`), pipeline guardrails (`pipeline/test_lowering_error_guardrail.py`,
`passes/test_pipeline_semantics.py`). Then **deployable parity**: `deplodock compare <before-dump> <after-dump>` on a
dumped model (the per-kernel `-O3` rows are the stable cross-change signal) — confirm no kernel regressed beyond `--tol`.
Re-bench a symbolic-`seq_len` flash and a masked-N matmul end-to-end (`deplodock run --code ... --bench`) vs eager. Final
ARCHITECTURE.md sweep (tile, kernel, assembly) describing the one path; delete the stale per-tier prose.
→ green: full `make test`; bench parity within tolerance.

## Ladder summary

`commit 0 (red baseline)` → `1 masking` → `2 K-rebracket` → `3 axis-split` → `4 build` → `5 carrier classifier` →
`6 slab+tower` → `7 flash→generic` → `8 integration+parity`. Eight green commits after the red baseline; each is an
independently-revertible recovery; the algebraic dispatch is preserved throughout.
