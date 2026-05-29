# Plan: Replace blocked-GEMM builder with a Kernel-IR dedup pass

## Context

`010_partition_loops.py` carries a specialized "register-blocked GEMM nest" builder
(`_classify_n_dep` + `_split_k_tower_for_block` + `_build_register_blocked_body`, ~135 lines at
`010_partition_loops.py:410–538`) that lifts N-invariant compute out of the per-cell scope BEFORE
the register-tile replicator runs. The blocked layout splits one `RegisterTile(N_r) > [Init,
K-tower, Write]` into three sibling `RegisterTile(N_r)` towers separated by the K-loop, with the
N-invariant cone shared across F_N cells. Without this, the replicator (`010_split_register_axes`)
duplicates the cone F_N times (e.g. on Qwen3 lm_head FN=8 with fused RMSNorm prologue, every
`Load x[m, k]` runs F_N× instead of 1×).

The blocked builder is gated by FIVE disqualifiers (`010_partition_loops.py:941`):
`splitk == 1 and br == 1 and fn > 1 and shape.k_loop is not None and m_bound is None`, plus an
internal "unrecognized stmt → bail" path. Each disqualifier is a "hasn't been generalized yet"
marker; for those shapes the legacy per-cell layout is used and we silently eat the F_N duplication
(critical perf cost when FN is large — 94ms → 3.5ms gap on Qwen3 lm_head per the planner's own
docstring at `010_partition_loops.py:833–839`).

**Architectural insight:** the blocked builder is just loop-invariant code motion on an unrolled
register loop, done at construction time by a specialized analyzer. The same effect is achievable
by **always emitting the per-cell layout** and **running CSE on the replicated body** to fold
identical Loads/Assigns. CSE is content-agnostic (no five disqualifiers), shape-agnostic (works on
SPLITK>1, BR>1, M-mask, fused prologue, FN=1 alike), and catches more dedup opportunities than the
structural classifier (e.g. accidentally-identical user compute). The blocked builder stops being
a layout choice — it becomes "what CSE naturally produces from the unrolled per-cell form."

**Approach:** add a new Kernel-IR pass `011_dedup_replicated.py` between
`010_split_register_axes` and `020_place_inits`. Once the dedup pass produces output structurally
equivalent to today's blocked-then-replicated form, peel off the blocked-builder's disqualifiers
one at a time, verifying perf parity each step. End state: blocked builder deleted, planner
~150 lines smaller, all matmul shapes (including the four previously-disqualified ones) get the
optimal layout.

## Approach: new `011_dedup_replicated.py` pass

### Pass shape

```python
# deplodock/compiler/pipeline/passes/lowering/kernel/011_dedup_replicated.py
PATTERN = [Pattern("root", TileOp)]

def rewrite(ctx, root) -> TileOp | None:
    """Common-subexpression-eliminate redundant Loads / Assigns produced by
    `010_split_register_axes` replicating an N-invariant body F_N times.
    Walks each Body, finds stmts with structurally-identical (op, deps,
    fields), keeps the first occurrence, renames later duplicates' SSA
    names to the survivor across the rest of the Body. Idempotent.
    """
```

### Matching + folding

Two folds, applied iteratively to fixed point per Body:

1. **Load-CSE.** Two `Load(input=I, index=E, ...)` with identical `(I, tuple(e.pretty() for e in E))`
   fold to one. The second one's `name`s alias to the first one's `name`s. Mirrors
   `loop/fusion/020_dedup_loads.py` exactly — same matching key
   (`deplodock/compiler/ir/stmt/normalize.py:606–618`), same alias-table rewrite via
   `Stmt.rewrite(rename)`.

2. **Assign-CSE.** Two `Assign(op=O, args=A, ...)` with identical `(O, A)` after alias rewriting
   fold to one. Needs to run after Load-CSE because the args may need renaming first. Run
   iteratively (Load → Assign → Load → … until fixed point).

Both folds are pure structural equality on frozen dataclasses — Python's `==` already does the
right thing on `Load` / `Assign` after alias normalization.

### Scope

Per-Body, not global. Walk recursively into nested Bodies (Loop, Cond, SerialTile, RegisterTile,
StageBundle) using `Stmt.nested()` + `Stmt.with_bodies()`. At each scope, CSE within the
sibling-list — don't cross scope boundaries (a Load at the K_i scope is not foldable with a Load
at the M_r scope; they have different live-range semantics).

### Pass-order constraints (audited)

| Constraint | Verified |
|---|---|
| Must run AFTER `010_split_register_axes` (need replicated body to exist) | ✓ |
| Must run BEFORE `050_vectorize_loads` (F_N duplicates break consecutive-Load detection) | ✓ |
| Must run BEFORE `080_vectorize_stores` (same) | ✓ |
| Slot `011` is unoccupied; nothing else runs between 010 and 020 | ✓ |
| Dialect at this point is still `TileOp` (KernelOp not until `100_materialize_tile`) | ✓ |
| `030_stamp_types` / `040_demote_to_write_dtype` tolerate either form | ✓ (last-writer-wins) |

### Infrastructure reused

Already exists, no new utilities needed:
- `Stmt.rewrite(rename_ssa, sigma, axis_fn)` — `deplodock/compiler/ir/stmt/base.py:371`
- `Stmt.deps()` / `Stmt.defines()` — `base.py:312, 321`
- `Body.iter()`, `Body.definitions`, `Stmt.nested()`, `Stmt.with_bodies()`
- Frozen-dataclass `__eq__` on every `Stmt` and `Expr` subclass
- `Expr.pretty()` for canonical index keys (Loop-IR dedup uses this same pattern)

## Phased migration

The blocked builder cannot be deleted in one shot — five disqualifier shapes need per-shape
validation (perf parity vs the current blocked output). Phase per PR:

### Phase 1 — Land the dedup pass

Add `011_dedup_replicated.py`. Do NOT touch the planner yet. At this point the pass is a no-op
on existing kernels because the blocked builder is already lifting the invariants out. Verify
no behavior change via `make test` (1377 tests should pass unchanged).

Add unit tests in `tests/compiler/pipeline/test_dedup_replicated.py` that feed the new pass a
synthetic replicated body (mock the replicator's output for a 4-cell unroll) and assert it
produces the deduped form. Tests against the pass in isolation, not the full pipeline.

### Phase 2 — Remove FN=1 disqualifier

Flip the planner's gate so FN=1 matmul kernels also go through the blocked code path. The
blocked builder collapses to a 1-cell-each three-tower form, which after `_wrap_tower`'s size-1
axis filter becomes structurally identical to per-cell layout. Verify:
- `make test` passes
- No CUDA-emit diff on a small matmul kernel (e.g. `torch.matmul(torch.randn(64,128),
  torch.randn(128,64))`)
- `020_stage_inputs` slab-shape inference for inner-RegisterTile Load works (mentioned as a known
  bug at `010_partition_loops.py:932–934`) — fix it here if it breaks, or document why it doesn't
  trigger at FN=1.

> **Investigation note (2026-05-28).** Phase 2 was attempted on `feature/dedup-replicated-pass`
> and reverted; the blocker is deeper than the plan body suggests.
>
> The structural claim "blocked-FN=1 == per-cell after size-1 filter" is true **only after**
> `lowering/kernel/010_split_register_axes` unwraps the `RegisterTile(N_r=1)` wrappers — i.e. at
> Kernel-IR stage. But the staging passes that fail at FN=1 (`020_stage_inputs` →
> `040_use_ring_buffers` → `060_use_async_copy` → `070_pad_smem`) all run **earlier**, in
> `lowering/tile/`, where the blocked path still has the per-tower `RegisterTile(N_r)` wrappers
> visible. So `SerialTile(K_i).is_reduce` returns `False` (the `Accum` is nested inside
> `RegisterTile`, not at the immediate body), and `020_stage_inputs` recurses past K_i instead of
> forming a Stage. **Result on Phase 2 gate flip:** plain-matmul FN=1 produces no `StageBundle`
> at all → `test_lowering_accuracy.py::test_double_buffer_matmul_accuracy` /
> `test_async_copy_matmul_accuracy` / `test_pad_smem_matmul_accuracy` all fail (9 cases).
>
> Cross-check: the **existing FN > 1 blocked path** has the same bug. Running the FN=3 blocked
> shape with `STAGE=111` produces zero `StageBundle`s — the blocked path has never staged its
> matmul K_i. The existing blocked-gemm tests pass because they only assert CUDA accuracy
> (kernel runs slower without smem caching but still correct), not Stage formation.
>
> **Naive is_reduce fix breaks the fused-prologue case.** Changing `SerialTileBase.is_reduce` to
> walk through `RegisterTile` / `Cond` (so K_i with nested Accum is recognised as a reduce)
> makes plain-matmul FN=1 stage correctly, but breaks
> `test_fused_rmsnorm_linear_blocked_prologue`: the fused RMSNorm + linear case has prologue's
> mean-reduce K_i and matmul's K_i as siblings inside `SerialTile(M_r, plain)`, both reading the
> same input `x`. Today's "bug" — matmul K_i not being a reduce — is **load-bearing**: it keeps
> matmul K_i out of staging so the prologue's `x_smem` covers both reads. With the fix, both
> reduces stage → two `x_smem` allocations → smem regression detected by the test.
>
> **Clean Phase 2 fix scope.** Needs either:
> - **Stage-merging across sibling K-towers**: 020 (or a follow-up tile pass) detects that two
>   Stages targeting the same buffer with overlapping (or subset) cache-axis sets can share one
>   smem slab. Then the is_reduce-walks-through fix becomes safe.
> - **Planner-side knowledge**: when the fused prologue + matmul share an input, mark the matmul
>   K_i such that 020 skips staging it (relies on the prologue Stage). Per-cell path doesn't
>   need this because the matmul Load lives at the prologue's K_i scope post-CSE — sharing
>   happens via SSA, not via Stage merge.
>
> Both options require non-trivial design + perf verification on GPU hardware. Phase 1's dedup
> pass is in place and observationally a no-op, so it ships cleanly on its own.

### Phase 3 — Remove SPLITK > 1 disqualifier (matmul_add path)

Flip the planner's gate so SPLITK>1 matmul shapes go through the blocked builder. Two integration
points needed:
- `015_gate_splitk_residual` currently expects the per-cell `[Init, K-tower, Load(r), Assign,
  Write(v)]` shape at the innermost RegisterTile scope. With blocked + SPLITK>1, the Write tower
  becomes its own `RegisterTile(N_r)` — the gate pass needs to recognize that shape too. Likely
  ~30 lines of pattern extension.
- `020_stage_inputs` slab-shape inference for the SPLITK-inner-RegisterTile case (the planner
  notes this as the actual blocker at `010_partition_loops.py:932–934`).

Verify: `test_matmul_with_bias` / `test_linear_with_bias` (the SPLITK+linear-residual e2e tests
that caught PR #169's bug) still pass.

### Phase 4 — Remove remaining disqualifiers (BR > 1, M-mask, fused prologue)

Each is its own integration step against a downstream pass. Most likely order:
- **BR > 1** (cooperative-K): cooperative combine emission needs to fire after the Write tower.
- **M-mask** (overhang): masked-M boundary Cond placement at the right tower scope.
- **Fused prologue** (SDPA P@V): prologue emit as a 4th sibling tower at M_r scope.

Each step: flip one gate condition, run full suite, run perf parity vs blocked output, ship as
its own PR if it touches a downstream pass.

### Phase 5 — Delete the blocked builder

With all five disqualifiers removed, the blocked code path is the unconditional matmul path.
Delete the `if` gate. Then delete `_build_register_blocked_body`, `_split_k_tower_for_block`,
`_classify_n_dep`, and `_BuildSkipped` if unused. Net: ~150 lines out of `010_partition_loops.py`.

Update `tests/compiler/passes/test_partition_planner_rules.py::test_planner_emits_register_blocked_structure`
and `tests/compiler/passes/test_lowering_blocked_gemm.py` — these assert pre-replicator
blocked-tower shape, which won't exist anymore. Rewrite to assert post-CSE codegen shape (count
of Loads / vectorized widths in the final CUDA) or delete if redundant with the new pass's unit
tests.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/kernel/011_dedup_replicated.py` (new, ~120 lines)
- `tests/compiler/pipeline/test_dedup_replicated.py` (new, ~150 lines)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` (delete the blocked
  builder + the disqualifier gate over 4 phases; net ~150 lines out)
- `deplodock/compiler/pipeline/passes/lowering/tile/015_gate_splitk_residual.py` (Phase 3 — extend
  the linear-epilogue matcher to recognize the blocked Write-tower shape)
- `deplodock/compiler/pipeline/passes/lowering/tile/020_stage_inputs.py` (Phase 2/3 — generalize
  slab-shape inference for inner-RegisterTile Load)
- `deplodock/compiler/pipeline/ARCHITECTURE.md` (refresh the `lowering/kernel/` pass-order table
  + the `lowering/tile/` note about the blocked builder when it's removed)
- `tests/compiler/passes/test_partition_planner_rules.py` (rewrite the
  blocked-structure assertion at the end)
- `tests/compiler/passes/test_lowering_blocked_gemm.py` (rewrite or delete — most assertions
  become post-CSE codegen checks)

## Verification

### Per-phase

1. **`make test`** clean at every phase — never ship a phase that regresses any test.
2. **CUDA-emit diff** between pre-phase and post-phase for representative kernels:
   - Plain matmul (FN=8): `deplodock compile --code "torch.matmul(torch.randn(64,512),
     torch.randn(512,64))" --ir cuda`
   - matmul_add at SPLITK>1: `DEPLODOCK_SPLITK=2 deplodock compile --code "torch.addmm(...)"
     --ir cuda`
   - Qwen3 lm_head shape (masked-N, fused RMSNorm prologue): the linear_196 kernel that the
     planner docstring calls out at `010_partition_loops.py:833–839`.

### Perf parity (Phase 5 acceptance)

The smoking gun for this whole refactor is Qwen3 lm_head: planner claims 94ms → 3.5ms (~25×) with
the blocked layout vs per-cell. Same kernel must hit ≤4ms post-refactor:

```bash
./venv/bin/deplodock run --code "<lm_head shape>" --bench
```

Compare median latency pre-refactor vs Phase-5 end state. ≥0.95× of pre-refactor latency on each
of {plain matmul, matmul_add, fused RMSNorm+linear, SDPA P@V} acceptance.

### Lint

`make lint` clean across every phase.

## Notes / risks

- **Idempotence + iteration:** Load-CSE then Assign-CSE then Load-CSE — fixed-point iteration is
  bounded by the number of Assigns (each round folds at least one). For a 32-Load prologue
  replicated FN=8× (256 input Loads), worst case ~10 iterations.

- **SSA uniqueness in the deduped body:** after CSE, every name has exactly one definer. This is
  STRICTLY BETTER for downstream passes than today's "two Loads define `a_0` and `a_1`,
  last-writer-wins in `_build_uses`" silent-assumption pattern in `040_demote_to_write_dtype` /
  `030_stamp_types` (per the exploration findings). CSE doesn't break SSA; it tightens it.

- **`020_stage_inputs` slab-shape inference for inner-RegisterTile Load** is called out as a known
  bug in the planner's own comment at `010_partition_loops.py:932–934`. It blocks Phases 2 and 3.
  Fix as part of those phases — likely the actual planning-side work is this bug, not the dedup
  pass itself.

- **Tests that lock in the blocked tower shape** (`test_planner_emits_register_blocked_structure`,
  `test_lowering_blocked_gemm.py`) are testing the wrong invariant — they assert the planner's
  output structure, not what makes the kernel fast. The new tests should assert post-codegen
  invariants (number of `Load` calls in the emitted CUDA, vectorization widths achieved) which
  are stable across the refactor.

- **Compile-time cost of build-then-fold:** for FN=8 lm_head with 32-Load prologue: 256 Loads
  briefly exist between `010_split_register_axes` and `011_dedup_replicated`. CSE folds in
  O(N²) worst case → ~65k comparisons. Negligible at 10s of microseconds per kernel.
