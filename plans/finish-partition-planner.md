# Plan — finalize the partition planner

## Context

`feature/partition-planner` is 8 commits along. The planner already owns matmul register-tile (M4),
matmul K-chunking (M7), and non-matmul chunk-reduce (M8), but both code paths still coexist
behind the `DEPLODOCK_PLANNER` env flag. Partition decisions for **BN, BM, FM, FN, SPLITK,
cooperative-BN** still live in downstream rules (`004_launch_geometry`, `008_register_tile`,
`003_split_matmul_k`). 002, 006, and the legacy fork of 008 are dead code in planner=1 mode
but still ship.

**Goal end-state**:

1. `DEPLODOCK_PLANNER` env flag deleted — planner always runs.
2. All partition knobs (`BK`, `BN`, `BM`, `FM`, `FN`, `SPLITK`) registered + forked in
   `000_partition_planner.py`. Transport knobs (`STAGE`, `TMA`, `TMA_SWIZZLE`, `VW`) stay
   where they are (they reference post-tileify Stage nodes).
3. Post-planner rules are mechanical: they read `Role` tags / knob stamps and rewrite
   accordingly with no further partition decisions or forking.
4. Legacy `002_chunk_matmul_k`, `006_chunk_reduce`, legacy fork in `008_register_tile`
   deleted. `_lift_output_loops`-equivalent prediction done inside the planner.
5. Per-cell Stage coalescing fixed (M11 perf — F·F → 2 coalesced Stages).

Variant count is unchanged — the engine still explores the same cartesian product
(`pipeline.py:424–440`, multiplicative fork composition). What changes is **where** the
decisions live: the planner emits one decision per variant, downstream rules act on those
decisions deterministically.

Per `single-branch-milestones`: each step is one milestone commit on the existing
`feature/partition-planner` branch, gated on `make test` + `make lint` (both env modes
during the env-flag-still-present steps).

---

## Step 1 — Complete the thread-extent predictor (`_lift_output_loops` mirror)

**Why**: M8 currently misses Loops that `001_tileify._lift_output_loops` hoists into
`Tile.axes` from multi-stmt body levels (e.g. SDPA's `head_dim` Loop). On those kernels the
planner mispredicts the slab geometry → chunk-reduce planner branch doesn't fire → legacy
006 keeps owning them.

**Change**: in `000_partition_planner.py`, augment `_predict_thread_extents` (today walks
only the single-stmt outer chain). After the chain breaks, additionally scan the
remaining body for top-level free Loops that (a) aren't `Role.REGISTER` and (b) have a
`Write` somewhere under them whose index references the loop's axis. Mirror
`001_tileify._writes_with_axis` (lines 112–125). Each lifted axis joins the predicted
thread-axes list with full extent (lifted axes stay full-extent — tileify doesn't split
them; only `_plan_partition` does, and only for the inner-most axes in `thread_tile_shape`).

**Files**: `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_planner.py`
(~30 lines).

**Verification**: re-run the M8 firing check on SDPA-512 (`scripts/check_m8.py`-style);
planner's chunk_reduce branch should now fire and `006_chunk_reduce` should not.

---

## Step 2 — M9: cooperative-reduce branch + cooperative BN fork

**Why**: `005_cooperative_reduce` today triggers on the synthetic `t=THREAD` axis injected
by `004_launch_geometry`. Switch its trigger to a planner tag so the cooperative decision
lives in the planner.

**Changes**:

- `000_partition_planner.py`: add `_try_cooperative_reduce(ctx, loop_op)` branch. Mirror
  `004_launch_geometry._cooperative_viable` (lines 105–121) at the LoopOp level — qualifying
  shapes get `Role.COOPERATIVE_STRIDE` on each qualifying reduce Loop. Fork over BN
  candidates from `004_launch_geometry._emit_cooperative_launch` (lines 142–146): heuristic
  `_effective_block_size` first, then the rest of `_TUNE_AXIS_CHOICES ≥ WARP_SIZE`. Stamp
  the `BN` knob per variant.
- `004_launch_geometry.py`: drop the cooperative fork from `_emit_cooperative_launch`.
  When a reduce body has `Role.COOPERATIVE_STRIDE` tags, the launch path becomes
  deterministic — read `BN` from the LoopOp's knobs (forwarded onto the TileOp through
  `_apply_one`) and emit a single variant with the synthetic `t` axis + BIND_BLOCK rebind.
- `005_cooperative_reduce.py`: trigger on `Role.COOPERATIVE_STRIDE` presence in body
  Loops instead of the structural `len(thread_axes) == 1` check. Rewrite logic
  (StridedLoop + Combine) is unchanged.

**Files**: `000_partition_planner.py` (+~100 lines), `004_launch_geometry.py` (~40 lines
removed from `_emit_cooperative_launch`), `005_cooperative_reduce.py` (~10 lines trigger
swap).

**Verification**: RMSNorm + standalone softmax kernels produce identical CUDA between
env=0 and env=1; coop fork variant count matches.

---

## Step 3 — Hoist FM / FN fork into the planner

**Why**: Today the planner's matmul register-tile branch picks one `(FM, FN)` via
`tuning.register_tile_shape` (heuristic only — no fork). The fork happens in
`008_register_tile._enumerate_combos` / `_TUNE_F_CHOICES`. Move that fork into the planner
so 008's legacy fork can be deleted later (M10).

**Change**: in `000_partition_planner.py::_try_matmul_register_tile`, enumerate
`(FM, FN)` candidates from `008_register_tile._TUNE_F_CHOICES` (lines 56–60 — re-export
the constants from `_helpers.py` so neither file imports the other), heuristic first.
Emit one variant per pair, σ-substitute the body for each, stamp `knobs={FM, FN}`.

**Files**: `000_partition_planner.py` (~50 lines), `008_register_tile.py` /
`_helpers.py` (move `_TUNE_F_CHOICES` constant). `006a_register_tile_planned.py` continues
to read FM/FN from knobs (it already does).

**Verification**: matmul kernels produce the same CUDA for each (FM, FN) combo as today's
008-driven path.

---

## Step 4 — Hoist (BN, BM) matmul fork into the planner

**Why**: `004_launch_geometry._matmul_variants` (lines 167–215) forks over
`(BN, BM) ∈ _TUNE_AXIS_CHOICES²`. Move this decision pre-tileify so the planner emits one
LoopOp per `(BN, BM)`, and `004` becomes a deterministic σ-substitute that reads BN/BM
from knobs.

**Change**: planner needs to enumerate `(BN, BM)` against the matmul's outer chain
(post-register-tile + post-K-chunk), clamping per `_matmul_variants._add` (lines 179–192).
For each pair, decide which axes split into BLOCK_outer + THREAD_inner. The pre-tileify
form: just stamp `knobs={BN, BM}` and leave geometry to 004; **004's matmul branch becomes
"read BN/BM from knobs, run `_plan_partition` once, no fork"** (no `_TUNE_AXIS_CHOICES`
loop).

**Files**: `000_partition_planner.py` (~80 lines forking + clamp), `004_launch_geometry.py`
(matmul branch shrinks to single deterministic variant).

**Verification**: matmul TILE_PASSES output identical across the (BN, BM) sweep.

---

## Step 5 — Hoist SPLITK into the planner, strip 003 to the epilogue

**Why**: `003_split_matmul_k._split_tile` (lines 102–155) does two things: (a) the K_o →
K_split × K_o_new σ-split and (b) the atomic-add Write + `Cond(K_split==0, …)` epilogue
rewrite. (a) is a partition decision; (b) is post-split materialization. Per the agent
report, the epilogue logic depends on the already-split K loop's `acc_name`, so factoring
needs care.

**Change**:

- Planner: add `_try_splitk(ctx, loop_op)` — operates on K_o (already split by M7's K-chunk
  branch). Calls `tuning.auto_splitk(synthetic_tile, K_o_extent)`; if > 1, σ-splits K_o →
  K_split × K_o_new. Tag the K_split axis with a new `Role.GRID_BLOCK` (or reuse a stamp
  knob `SPLITK`) so post-tileify `_lift_output_loops` / launch_geometry knows to bind
  K_split as the outermost BLOCK axis.
- `003_split_matmul_k.py`: delete `_split_tile`'s split half (lines 118–130). Keep
  `_rewrite_epilogue`, `_find_accum_name`, `_is_linear_multiplicative_chain`,
  `_extract_simple_residual`. New trigger: presence of a `K_split` axis in `tile.axes`
  with the `GRID_BLOCK` role tag. Rule fires once per Tile, never forks.

**Files**: `000_partition_planner.py` (~60 lines), `003_split_matmul_k.py` (~30 lines
removed + trigger swap). Possibly extend `ir/axis.py::Role` with `GRID_BLOCK` (or
reuse a knob stamp without a new role).

**Verification**: split-K matmuls (`kv_proj`-class, SDPA-shaped) produce identical CUDA
across the SPLITK sweep.

---

## Step 6 — M10: drop env flag + delete legacy paths

**Why**: planner now owns every partition decision (BK, BN, BM, FM, FN, SPLITK,
cooperative-BN, chunk-reduce). Legacy code paths are dead in env=1 mode; delete them and
make planner always run.

**Changes**:

- `000_partition_planner.py`: delete the `if not os.environ.get(_ENABLE_ENV)` guard at
  `rewrite`'s top. Remove `_ENABLE_ENV` constant + the `os` import.
- **Delete `002_chunk_matmul_k.py` entirely.**
- **Delete `006_chunk_reduce.py` entirely.**
- `008_register_tile.py`: move `replicate_along_axis`, `_split_axis`, `_sigma_split`,
  `_sigma_offset` to `_helpers.py`. Delete `_find_slots`, `_thread_split_site`,
  `_reduce_unroll_site`, `_normalize_unroll_site`, `_enumerate_combos`, and the rule's
  `rewrite` entry-point. Delete the file (the legacy fork is the file's reason for
  existing).
- `006a_register_tile_planned.py`: switch the `importlib.import_module` of the 008 module
  to a direct `from ...tile._helpers import replicate_along_axis`.
- `tests/compiler/passes/test_matmul_rules.py`: drop the `"chunk_matmul_k" in fired or
  "partition_planner" in fired` accommodation; assert `partition_planner in fired`.
- `tests/compiler/passes/test_chunk_reduce.py`: drop the `if "partition_planner" not in
  fired: assert "chunk_reduce" in fired` accommodation; assert `partition_planner in
  fired`.
- Other planner-aware tests in `tests/compiler/passes/test_partition_planner_rules.py`
  may need their structural-equivalence references updated.

**Files**: `000_partition_planner.py`, delete `002_chunk_matmul_k.py`,
`006_chunk_reduce.py`, `008_register_tile.py`. Update `006a_register_tile_planned.py`,
`_helpers.py`, plus tests.

**Verification**: `make test` + `make lint` (no env flag set). Single mode now.

---

## Step 7 — Autotune DB schema bump

**Why**: each fork-hoist step changes the `parent_key` topology (planner-stamped knobs
differ from downstream-stamped ones). Stale `lowering` rows in `~/.cache/deplodock/
autotune.db` won't match the new keys. `bench_results` rows are source-hash-keyed and
survive.

**Change**: bump the schema/format token in
`deplodock/compiler/pipeline/search/` (where the lowering cache lives — find the
`SCHEMA_VERSION` or equivalent constant). Document in the M10 commit.

**Files**: schema constant module + commit message.

**Verification**: with a stale DB present, `deplodock tune` re-explores variants instead of
hitting cached terminals.

---

## Step 8 — M11: per-cell Stage coalescing (perf)

**Why**: `006a_register_tile_planned` produces F·F Stages (`x0_smem`, `x0_smem_1`, …)
instead of 2 coalesced Stages (`x0_smem`, `x1_smem`). Costs ~2× smem for FM=FN=2; worse
for bigger F.

**Change**: option (b) from the followup memory (closer to today's working path) —

- `006a_register_tile_planned.py`: defer REGISTER-Loop replication. Keep the
  `Role.REGISTER` Loops in place during staging.
- `007_stage_inputs.py`: include `Role.REGISTER` axes as cache axes when building Stage
  slabs (the cache slab spans BM × BK / BK × BN as in today's post-008 path).
- A new pass (`007c_replicate_register_post_stage`?) unrolls the `Role.REGISTER` body
  *after* staging picks slabs.

**Files**: `006a_register_tile_planned.py`, `007_stage_inputs.py`, possibly one new
post-stage pass.

**Verification**: smem usage on FM=FN=2 matmuls halves; perf-marked tests
(`tests/perf/`) don't regress.

---

## Final verification

After all steps land:

1. `make test` (single mode, no env flag) — 1126 tests pass.
2. `make lint` clean.
3. `deplodock compile --code "..."` on each of: TinyLlama RMSNorm, plain matmul,
   matmul + relu, SDPA-128, SDPA-512, kv_proj — diff the emitted CUDA against the
   pre-refactor recordings (or check for structural equivalence). Most will differ in
   knob naming / IR shape but the post-stage CUDA should be the same.
4. `deplodock tune --code "torch.matmul(torch.randn(256,64), torch.randn(64,256))"`
   walks the full BK × BN × BM × FM × FN cartesian (variant count matches today's
   downstream multiplicative count).
5. Update `MEMORY.md` `partition-planner-followup` entry to "complete"; delete
   `feedback_skip_full_test_suite` referenced gates if no longer needed.

---

## Out of scope (deliberately)

- Hoisting transport knobs (`STAGE`, `TMA`, `TMA_SWIZZLE`, `VW`) into the planner. They
  reference Stage / BufferedStage nodes that don't exist pre-tileify.
- Restructuring `007_stage_inputs`'s STAGE bitmask format.
- Cross-LoopOp (graph-level) partition decisions — the planner remains per-LoopOp.

---

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/000_partition_planner.py` — central.
- `001_tileify.py` (`_lift_output_loops` lines 83–125 — predictor target).
- `002_chunk_matmul_k.py` (delete).
- `003_split_matmul_k.py` (split → planner; epilogue stays).
- `004_launch_geometry.py` (matmul + cooperative forks → planner).
- `005_cooperative_reduce.py` (tag-based trigger).
- `006_chunk_reduce.py` (delete).
- `006a_register_tile_planned.py` (fix import after 008 delete).
- `007_stage_inputs.py` (M11 cache-axis change).
- `008_register_tile.py` (delete; salvage `replicate_along_axis` to `_helpers.py`).
- `_helpers.py` (host for moved helpers + `_TUNE_F_CHOICES`).
- `deplodock/compiler/ir/axis.py` (optional new `Role.GRID_BLOCK` for SPLITK).
- `tests/compiler/passes/test_matmul_rules.py`, `test_chunk_reduce.py`,
  `test_partition_planner_rules.py` — update assertions in M10.
