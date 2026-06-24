# Branch review: `feature/tile-ir-block-dag` vs `main`

Date: 2026-06-23. Reviewer: fan-out of four analysis agents (architecture quality, test-coverage gaps,
documentation consistency, dead-code/shims), synthesized.

## Scope

178 commits, 187 files, **+11,505 / −18,506 lines**. The branch rewrites the **tile lowering pass** (`LoopOp →
TileOp`): the monolithic numbered passes (`010_partition_loops.py` ~1611 LOC, `020_stage_inputs.py` ~1121 LOC,
`050_use_tma.py`, `085_warp_specialize.py`, `_enumeration.py` ~876 LOC, `_split_demoted.py`) are dissolved into:

- `tile/enumeration/` — builds the block-DAG `TileGraph` and searches the `Schedule`
- `tile/assembly/` — assembles the tower (deterministic, post-search)
- `tile/split/` — the structural cut/split fork (`005_split_demoted.py`)
- `ir/tile/ir.py` — the new block-DAG IR

Design principle: **purely algebraic moveset — no shape specializations, dispatch on the carrier algebra**
(`MAP`/`SEMIRING`/`MONOID`/`TWISTED_MONOID`).

## Verdict

**This is a high-quality rewrite that delivers on its central claim.** The tile dir shrank ~37% (9,445 → 5,933
LOC) while *broadening* capability (warp-tier MMA, cooperative reduce, streaming flash, TMA, split-K — previously
spread across the deleted `050_use_tma` + `085_warp_specialize`). The old monoliths were genuinely dissolved, not
relabeled. The algebraic-moveset claim **holds**: there is zero named-shape (`flash`/`sdpa`/`rmsnorm`/`softmax`)
dispatch in control flow anywhere in the new code — every such occurrence is in comments/docstrings. Dead code is
unusually sparse for a rewrite this size.

The problems are localized and fall into three buckets: (1) a **test-coverage cliff** where several live,
miscompile-critical code paths lost their unit tests with no replacement; (2) a **large block of stale documentation**
in the parent/sibling `ARCHITECTURE.md` files and `CLAUDE.md` that still describe deleted subsystems as live; (3) a
short list of dead shims and two over-large functions/modules. None are blockers, but the doc staleness (#2) and the
warp-spec coverage cliff (#1) are actively misleading and should be addressed before merge.

---

## P0 — Address before merge

### Documentation describes deleted subsystems as live (actively misleading)

The new `tile/ARCHITECTURE.md` and `plans/tile-ir-block-dag.md` are accurate. The **parent/sibling docs were not
updated** and now describe the deleted move-composer and numbered passes as current code:

- **`CLAUDE.md:33-39`** — the `DEPLODOCK_MOVE_COMPOSER` paragraph. The env var is gone (no `config.py` entry, no code
  ref); `passes/lowering/tile/partition/`, `walk.py::walk_nest`, and "falls through to the legacy planner" all describe
  deleted code. The move composer *became* the block-DAG `enumeration/` rewrite — it is now the only path, not opt-in.
  Delete the var from the list (line 33) and rewrite/remove the parenthetical.
- **`deplodock/compiler/pipeline/ARCHITECTURE.md`** — four large stale blocks:
  - `:111-177` "The partition planner (`010_partition_loops`)" + the entire `DEPLODOCK_MOVE_COMPOSER` /
    `partition/` / `walk_nest` section (~65 lines of deleted code).
  - `:233-262` "Post-split re-fusion (rules `006_merge_split_glue` – `009_stamp_structural_features`)" — these tile
    rules **do not exist**; `split/` holds only `005_split_demoted.py`. The subsystem was removed.
  - `:847-869` the knob-table "Owning rule" column — nearly every row attributes its knob to a deleted pass
    (`010_partition_loops`, `020_stage_inputs`, `040_use_ring_buffers`, `050_use_tma`, `085_warp_specialize`,
    `017_atomic_free_splitk`). Plus rows for fully-removed knobs (`HOIST_COMPUTE`, `PAD_SMEM`, `GROUP_M`, `ASYNC_COPY`,
    `PIPELINE_STAGES`, `WARPSPEC`). Current knob set lives in `enumeration/_knobs.py` + `knob.py`:
    `BM BN BK BR FM FN FK WM WN SPLITK RING STAGE MMA CUT TMA NOATOMIC ATOM_KIND FKWIN`.
  - `:910-919` the `lowering/tile/` directory-table row — describes the deleted pass chain verbatim
    (`hoist_invariant_compute → use_ring_buffers → use_tma → … → mark_unroll`). Should describe the new
    `split/ → enumeration/ → assembly/` flow.

### Warp-specialization codegen: live code, zero coverage, stale doc refs

The *producer* pass (legacy `085_warp_specialize`) was deleted and the `warp_spec` fork is deferred — so nothing emits
a `WarpSpecialize` today — but the *consumer* lowering is **still live and reachable**:
`kernel/100_materialize_tile.py::emit_warp_specialize` (~line 585, ~100 LOC of mbarrier prologue / producer-consumer
`Cond` split / `SetMaxNReg` / named `bar.sync`) and `kernel/020_place_inits.py:117-133`. Its tests
(`test_materialize_warp_specialize.py` 290L, `test_place_inits_warp_specialize.py` 114L) were deleted **with no
quarantine/xfail breadcrumb** (unlike every other tier in the recovery plan). ~200 LOC of intricate mbarrier codegen
now has zero coverage; the deadlock guard (`test_warp_specialize_deadlock.py`, 146L — guarded a device hang) is also
gone. The tests would still pass against today's code → restore them.

Stale refs to the deleted `tile/085_warp_specialize.py` remain in `assembly/_tower.py` (lines 39, 170) and the tile
`ARCHITECTURE.md`.

### Off-hint masked-MMA accuracy + atom-tile boundary guards: live code, silent-miscompile risk

The masked symbolic-M/N/K mma.sync paths are live (`kernel/_stage_expand.py` `(k < seq_len) ? v : 0` zero-fill
~lines 257-277; per-element store guards in `kernel/005_lower_atom_tile.py`; `split/005_split_demoted.py`). The
accuracy sweep that pinned **straddling runtime sizes** (1, 31, 130, 700) — `test_matmul_mma_masked.py` (581L) — was
deleted. Surviving tests cover dynamic-M **only at tile-divisor sizes** (128/256/512); the off-hint straddling cases
are covered only indirectly via `requires_cuda`-gated qwen e2e. `test_matmul_mma_parity.py` even contains a **dangling
reference** to the deleted file for "off-hint coverage." Likewise the atom-tile guard unit tests
(`test_lower_atom_tile_guards.py`, 215L — pinned `_boundary_guards`/`_unstaged_masked_k`/the OOB+hang gate) are gone;
all four guarded functions remain in `kernel/005_lower_atom_tile.py`. **On a GPU-less CI run these straddling/masked
accuracy paths have no coverage at all.** Recreate the off-hint sweep and the guard unit tests.

---

## P1 — Coverage and layering follow-ups

### Untested live code paths (no replacement coverage)

- **TMA eligibility oracle** — `enumeration/052_transport.py::{tma_eligible,_source_eligible}` (boxDim>256 decline
  ~line 138, alignment decline) and `backend/cuda/_tma.py::encode_tiled` box-range guard (line 122) lost their unit
  tests (`test_use_tma_gates.py` 249L, `test_use_tma.py` 311L, `test_tma_smem_alignment.py` 188L). Fire/don't-fire +
  accuracy covered e2e, but the oracle branches and the `__align__(128)`/swizzle-atom invariant have no replacement
  assertion.
- **Bank-conflict oracle** — `compiler/diagnostics/bank_conflicts.py::{lane_bank_distribution,simulate_graph}` is
  still live (gates `kernel/060_permute_lane_accesses.py`) but lost its cross-validation
  (`test_bank_conflicts.py`, 69L). Zero replacement.
- **`enumeration/_moves.py`** (505 LOC, the central enumeration module) is referenced by **one** test; its offer
  invariants (FK divisor-cleanliness `bk·fk·splitk | K`, register-budget bounds, `FK=1`-first, coop geometry) are
  untested in isolation. The old `test_fk_reduce_enumeration.py` (98L) is gone.
- **2D strided-cooperative rows** — only the 1D case (`BR>1, BN=BM=1`) is tested; the 2D segmented-shuffle case
  (`BN·BM>1` AND `BR>1`) + the pow2≤warp clip lost coverage (`test_strided_coop_rows.py`, 192L).
- Several new auto-discovered passes have no direct test (only transitive e2e):
  `enumeration/{005_tensorize,006_warp_geometry,008_warp_reg,009_warp_build,040_seal_scalar_tier}.py`.

### Loose new-test tolerances

`test_fused_edge.py` uses `atol=1.0, rtol=0.5` (and `3e-1`) — catches gross miscompiles only, weak for the
fused-prologue path the plan itself flags as fragile. Tighten.

### Layering back-edges (tile → kernel) and a circular import

- Three tile modules import *down* into the kernel layer's private `_helpers.py`/`_atom.py` for structural predicates
  (`is_matmul_reduce`, `segmentable_k_extent`, `classify_fragment_epilogue`, `reduce_body_has_coupled_accum`):
  `enumeration/_atom.py:28-29`, `enumeration/_extract.py:101`, `assembly/020_peel.py:30`. Side-effect-free but a
  back-edge in the pass DAG — lift these into a shared loop/IR-vocabulary module.
- `enumeration/_extract.py:585` uses a function-local import of `assembly._fused`'s private `_map_transform` /
  `_split_monoid_producer` (the classic circular-import dodge). Same fix.

### Assembly overrules the chosen transport

`assembly/_slab.py:246-251` silently downgrades TMA→SYNC for a single-stage whole-K slab based on loop shape, even
though the `Schedule` carries `tma_bufs`. The transport choice should be resolved in enumeration and stored, so
assembly stays mechanical (the stated "Fix 1 firewall" intent).

---

## P2 — Dead code, shims, and refactors (all grep-verified)

Delete (zero real callers):

- `ir/tile/ir.py:1295` `trivial_stage_body()` — self-described "deprecated stub during refactor", only ref is its
  own `__all__` entry (`:2331`).
- `ir/tile/ir.py:1324` `_source_pretty()` — superseded by `_source_decl_line`, no caller.
- `enumeration/_iterdag.py:68` `AxisNode.symbolic` — no reader; call sites inline `not axis.extent.is_static`.
- `enumeration/_classify.py:26` `_Regime.k_bounds` — write-only (set at `:46-47`, never read).
- `search/analytic.py:167` `sm_count` param (`noqa: ARG001`) — dead end-to-end through `evaluate_golden` /
  `pick_matmul`; no non-test caller passes it.
- `assembly/__init__.py:3-5` `assemble_block` re-export — everyone imports from the submodule.
- `kernel/_atom.py:26-30` `ATOM_REGISTRY`/`Atom` "back-compat re-exports" — always imported from `ir.tile.ir`
  directly.

Doc/comment fixes:

- `kernel/_helpers.py:10,79-82` — docstrings promise a `thread_tile_of` deprecation alias that does not exist.
- `ir/tile/ir.py:916-918` — `consumer_thread_axes=()` "back-compat" default is unreachable: the live materializer
  arm now **raises** on empty axes (`100_materialize_tile.py:612-613`). Drop the default or fix the docstring.
- `assembly/_tower.py:1-9` — docstring claims "the legacy planner imports them back"; the legacy planner was deleted.
- `kernel/060_permute_lane_accesses.py` — self-described "obsolete (affine-collapse staging subsumes it)"; not
  statically dead (kept for the knob), candidate for follow-up removal; fix the stale `003_`→`030_register_tile`
  comment.
- Stale in-code refs to the deleted `tile/011_lower_atom_cell` pass: `kernel/005_lower_atom_tile.py:3`,
  `ir/stmt/leaves.py:568,591`, `ir/tile/ir.py:1662,1667,1682` (merged into `009_warp_build`).
- `tests/ARCHITECTURE.md:82,205` and `deplodock/compiler/{ir,pipeline/passes}/ARCHITECTURE.md` cite deleted
  files/passes (`test_split_demoted.py`, `test_warp_specialize_deadlock.py`, `010_partition_loops`, `050_use_tma`,
  `021`, `011_lower_atom_cell`) — pass names wrong, behavior descriptions mostly still accurate. Sweep.

Refactor (the one place the rewrite reproduced the monolith at function scale):

- **`enumeration/_extract.py::_fission`** (`:178-435`, 260 LOC, ~36% of the file) — six phases separated only by
  `# ---` comment banners, with the multi-accum path threaded through `if len(accums) > 1` across three islands at
  4-level nesting. Highest-value refactor: extract phase functions; pull multi-accum into its own function.
- `enumeration/_extract.py` (710 LOC) and `_build.py` (649 LOC) are over the 500-line threshold with natural seams
  (cone/fission vs block-extraction; per-tier builders). Lower priority.

Confirm-intent (test-covered forward-looking seams, not dead — don't delete blindly): `_extract.seed_demoted`,
`_partition.monoid_reduce_tilegraph`, `_cut.CutOffer.tier_inline`.

---

## Residual specializations to watch (claim technically met)

The algebraic-moveset claim holds, but two spots hard-code one op family's structure (no name dispatch, but the
closest residue): `_fused.py::_split_monoid_producer` (`:247-272`) positionally pattern-matches the rmsnorm statement
skeleton, and `010_reduce_tile.py:66` `_is_fp16_matmul` is a dtype+structure special case for the half2 window. Both
are legitimate today; revisit as more MONOID/fp16 producer shapes land.

Latent bug: `ir/tile/ir.py:994-996` `BYTES_PER_ELEM = 4` is fp32-only — fp16 smem-byte accounting over-counts 2×
(documented).

---

## Suggested merge checklist

1. Rewrite/remove the stale `CLAUDE.md` MOVE_COMPOSER paragraph + the four `pipeline/ARCHITECTURE.md` blocks (P0).
2. Restore warp-spec materializer/place-inits/deadlock tests; fix stale `085_*` refs (P0).
3. Recreate the off-hint masked-MMA accuracy sweep + atom-tile guard unit tests; fix the dangling
   `test_matmul_mma_parity.py` reference (P0).
4. Add unit tests for `052_transport.tma_eligible`, `encode_tiled`, the bank-conflict oracle, `_moves` offer
   invariants, 2D strided-coop (P1); tighten `test_fused_edge.py` tolerances.
5. Land the dead-code deletions and doc/comment sweeps (P2).
6. Follow-ups: lift structural predicates into a shared module (kill tile→kernel back-edges + circular import);
   resolve TMA/SYNC in enumeration; refactor `_fission`.
