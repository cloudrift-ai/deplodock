# Register-blocked GEMM: hoist N-invariant compute out of the register tower

## Context

The fused **RMSNorm + vocab projection** kernel (`linear_196` / lm-head, N = 151669) blows up at `FN=32`: each of the
32 register cells re-emits the per-element normalization `v6[k] = add_223[k]·v4·norm_weight[k]` inside its own
K-reduce. That's a ~530-line function with 32 live accumulators that cicc compiles in ~4 s even at `-Xcicc -O1` (over
`tune`'s 2 s budget → `bench_fail`), and at runtime it does the normalization 32× redundantly.

We dug through every existing mechanism that "should" prevent this; none apply:

- **`030_hoist_invariant_compute` / `HOIST_COMPUTE`** only fires on the silu-MLP cone — a SYNC `StageBundle` with one
  *multi-source* `Stage` whose cone sources *share cache axes*. The matmul reduce here has **no `StageBundle`**
  (operands are direct GMEM loads), the cone is **register-invariant compute inside a register-tiled reduce**, and it
  depends on an **outer scalar `v4`**. `HOIST_COMPUTE=1` is verified a no-op (byte-identical kernel).
- **`normalize_body` loop-invariant code motion** (`ir/stmt/normalize.py::hoist_loop_invariants`) doesn't fire because
  (a) `TileOp.__post_init__` calls `normalize_body(..., hoist=False)` (tile/ir.py:1113) — Stage bindings are
  loop-scoped, so hoisting is disabled for tile bodies; (b) the walk only recurses `Loop`/`StridedLoop`/`Cond`, never
  `RegisterTile`; and (c) even if it did, `v6[k]` *varies across the reduce axis K* (nested inside the register tile),
  so it isn't movable as-is — lifting it requires materialization or loop reordering, not code motion.

**Root cause.** `010_partition_loops::_build_split_body` (non-prologue path, lines 1464-1478) wraps the register tower
around the *entire* `{Init(acc); K-reduce; Write}` via `_wrap_tower(layers, new_inner)`. The planner's model is "each
output cell is an independent, self-contained reduction." Then `010_split_register_axes::_replicate_register_tiles`
clones that whole body per cell (σ `N_r→i`, SSA suffix `_<i>`), so the fused prologue is duplicated 32×.

This plan implements the **textbook register-blocked-GEMM nest**: split the `N_r` register tower *around* the K-reduce
so N-invariant compute runs once per K step and the 32 cells share it:

```
N_r REGISTER { Init(acc) }                 # 32 inits, acc_0..31
for k_o: for k_i (reduce):
    v6 = normalize(k)                      # once per k  (N-invariant, hoisted to K-scope)
    N_r REGISTER { Accum(acc, W[k, col(N_r)] * v6) }   # 32 FMAs into the persistent accumulators
N_r REGISTER { Write(C[col(N_r)], acc) }   # 32 writes
```

This removes the duplication **at the source** — fixing both the compile blowup and the 32× redundant FLOPs — and
benefits **every** register-tiled matmul (the shared operand `A[m,k]` / staged-smem reads stop being recomputed per
N cell too), not just the fused RMSNorm. It also generalizes the existing SDPA-prologue hoist (`_build_split_body`
1440-1462), which already pulls *K-independent* N-invariant compute (softmax stats) outside `N_r`; this extends the
same idea to *K-dependent* N-invariant compute, placed inside the K-loop but outside `N_r`.

## Design

### The blocker: per-cell accumulator naming is inferred, not declared

`010_split_register_axes::_replicate_along_axis` (lines 110-166) decides which SSA names get the `_<i>` suffix by a
**per-register-tile-body** keep-analysis: a name is suffixed iff its def-use chain references the register axis *within
that tile's body*. Today `acc` is suffixed because, inside the single all-in-one register tile, `Accum` reads
`W[k, col(N_r)]` → `acc` is keep=True → `Init`/`Accum`/`Write` all become `acc_0..31` consistently.

If we naively split into three towers, each tower's keep-analysis runs independently: the `Init(acc=0)` tower doesn't
reference `N_r`, so `acc` stays one name there, while the `Accum` tower produces `acc_0..31` → **naming mismatch**. So a
correct split needs the accumulator to be recognized as register-tiled *across* the towers.

### 1. First-class register-tiled accumulator → consistent replication

Two candidate mechanisms (prefer **1a**, fall back to **1b** if edge cases bite):

- **1a — body-global keep set per register axis** (`010_split_register_axes`). Before replicating a register axis,
  compute the keep set over the **entire `TileOp` body** (all sibling/cousin register towers that bind that axis +
  the K-reduce between them), not just the current tile body. Then `acc` — register-tiled via the `Accum`'s
  `W[k, col(N_r)]` dep — is keep=True in the `Init` and `Write` towers too, so all three replicate to `acc_0..31`.
  The σ-substitution stays per-tower (`N_r→i`); only the *keep membership* goes global. Minimal IR change.
- **1b — explicit `register_axes` on `Accum`/`Init`** (`ir/stmt/leaves.py:443`). Add `register_axes: tuple[Axis, ...]
  = ()`; the planner stamps it; the replicator force-suffixes any name an `Init`/`Accum` declares register-tiled,
  regardless of per-body dataflow. More explicit/robust, but touches the Stmt vocabulary + structural keys + every
  `Accum` constructor.

### 2. Planner reshape (`010_partition_loops::_build_split_body`)

Add a **blocked** matmul path (new branch alongside the existing non-prologue and prologue paths, lines 1440-1478):

- **Split the reduce body by N-dependence.** Partition the σ-rewritten reduce body into the N-invariant cone (Loads
  + Assigns whose free vars exclude the N axis — `add_223`, `norm_weight`, `v5`, `v6`; tolerating outer scalars like
  `v4`) and the N-dependent tail (the weight Load `B[k, n]` + `Accum`). Reuse the dataflow/cone helpers patterned on
  `030_hoist_invariant_compute::_ssa_dataflow` / `_find_boundary`.
- **Emit the blocked nest**: `Init` wrapped in `RegisterTile(N_r)`; the K-tower (`_wrap_tower([(K_i, STAGE_INNER),
  (K_o, SERIAL_OUTER)], …)` — same builder used at line 1515) whose body is `[N-invariant cone…, RegisterTile(N_r){
  N-dependent Accum }]`; `Write` wrapped in `RegisterTile(N_r)`. `M_r` register tiling composes the same way (outer).
- Keep the existing per-cell path as the fallback (see §4). The blocked path requires a matmul-shape reduce with a
  non-empty N-invariant cone; pointwise / cooperative-reduce / non-matmul reduces fall through unchanged.

### 3. Materialization

`010_split_register_axes` replication already walks arbitrary nested block stmts (it descends `SerialTile` etc. to find
nested `RegisterTile`s — lines 86-95), so three sibling `RegisterTile(N_r)` towers replicate independently. With §1 the
accumulator names line up. Verify `_stage_expand` / `100_materialize_tile` and `020_place_inits` (which already special-
cases `Init` placement) emit the `Init`s before the K-loop and the `Accum`s inside it with the persistent `acc_0..31`.
The K-reduce now runs **once** (not per cell), so `090_mark_unroll`'s `#pragma unroll` decision on the K-inner loop is
no longer multiplied by `FN` — the kernel stays compact.

### 4. Gating + fallback

Gate the blocked path behind a knob so it's an autotune fork and trivially reversible:

- Either **reuse `HOIST_COMPUTE`** (already the "hoist invariant compute" knob; extend its meaning to cover this
  register-blocked shape) or add a sibling `REG_BLOCK` BOOL knob. Emit both polarities when the blocked shape is
  detected; `False` keeps today's per-cell structure.
- The blocked path is strictly better for fused-prologue matmuls and large `FN`; the fork lets the search confirm and
  lets us ship dark/off if a regression shows up.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — `_build_split_body` (1241), tower
  assembly (1440-1478), `_wrap_tower` (483), `_replace_k_loops`, `_classify_fused_prologue` (435).
- `deplodock/compiler/pipeline/passes/lowering/kernel/010_split_register_axes.py` — `_replicate_register_tiles` (59),
  `_replicate_along_axis` keep-analysis (110-166).
- `deplodock/compiler/ir/stmt/leaves.py` — `Accum` / `Init` (≈443) (only if §1b).
- `deplodock/compiler/pipeline/passes/lowering/kernel/_stage_expand.py`, `100_materialize_tile.py`,
  `020_place_inits.py` — Init/Accum placement + materialization.
- `deplodock/compiler/pipeline/passes/lowering/tile/030_hoist_invariant_compute.py` — reuse `_ssa_dataflow` /
  `_find_boundary` for the N-dependence split; align `HOIST_COMPUTE` semantics if reused.

## Phasing (milestone commits on one branch; `make test` green at each)

1. **Register-tiled accumulator (§1).** Implement body-global per-axis keep (1a) in `010_split_register_axes`. Unit-test
   the replicator directly: a hand-built tile with split `Init`/`Accum`/`Write` `RegisterTile(N_r)` towers replicates to
   consistent `acc_0..N`. No planner change yet → no behavior change in production paths.
2. **Planner blocked path (§2), behind the knob, default off.** Detect the N-invariant cone + emit the blocked nest.
   Verify on the `linear_196` repro (`FN=32`): kernel computes the normalization once, drops from ~530 lines, and
   compiles well under 2 s; accuracy matches eager.
3. **Turn the fork on / make it the greedy default for register-tiled matmul** once accuracy + perf are confirmed.
4. **Docs**: `compiler/pipeline/ARCHITECTURE.md` (nesting diagram + the blocked path), `CLAUDE.md` knob bullet if a new
   knob is added.

## Risks

- **Correctness of the persistent multi-accumulator** is the highest risk — it's core matmul codegen. Mitigate with the
  knob (off by default until proven), the replicator unit test, and the accuracy suite at every step.
- **Cone-split precision**: mis-partitioning N-dependent vs N-invariant stmts corrupts results. Use exact free-var
  analysis over the N axis (`N_r` + `N_t` coords), tolerate only outer scalars, and bail to the per-cell path when the
  split is ambiguous (single boundary SSA, like `030`).
- **Interaction with SPLITK / cooperative-K (BR>1) / `M_r` / masked tiles (the lm-head `Cond`)**: start by supporting
  the simplest blocked shape (no SPLITK, `BR=1`) and fall through to per-cell otherwise; widen coverage incrementally.
- **Staged-smem operands**: when `020_stage_inputs` has staged the matmul operands, the N-invariant Loads read smem;
  ensure the hoist keeps the Stage decl in scope (the same constraint that forces `hoist=False` in `normalize_body`).

## Verification

- `./venv/bin/pytest tests/compiler/test_tune_accuracy.py -v` (matmul / rmsnorm / sdpa / gated_mlp) and the e2e tests
  (`tests/compiler/test_e2e_accuracy.py`) — accuracy must hold with the fork both off and on.
- Re-dump the lm-head kernel both ways: `deplodock compile <linear_196.torch.json> --ir cuda` with the knob off vs on —
  confirm the normalization appears once (not ×FN) and the line/accumulator count drops.
- nvcc timing: the on-variant compiles in budget (`-Xcicc -O1`), and `deplodock run --ir … --bench` shows the runtime
  improves (no 32× redundant normalize).
- Full `make test` + `make lint` before each milestone commit.
- End-to-end: `deplodock tune Qwen/Qwen3-Embedding-0.6B --clean --bench` — `linear_196` no longer `bench_fail`s on the
  compile budget, and the per-kernel table shows it benched.
