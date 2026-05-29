# Atomic-free split-K via a two-kernel decomposition

## Context

`SPLITK > 1` today emits `atomicAdd` writes to the output, because the partition planner adds a `K_s` BLOCK
axis (`compiler/pipeline/passes/lowering/tile/010_partition_loops.py:920-921`, :1036-1037`) that does *not*
appear in the output `Write.index` — and `Body.coordination`
(`compiler/ir/stmt/body.py:545`) classifies such writes as atomic-needing. At 2048³ fp32 on RTX 5090, the
cuBLAS gap is **1.63×** (436 µs vs 268 µs) after the BUFCNT / priority work; ncu shows we run **0.75
waves per SM** vs cuBLAS's **3.76 waves**. The cleanest way to crank waves is `SPLITK > 1`, but atomic
contention currently *regresses* that path (`SPLITK=2` benched at 470 µs > 436 µs).

We want: the matmul kernel writes partials to a workspace `partial[S, M, N]` (no atomic — `K_s` is in the
write index), then a second kernel sums along `S` into the original output. This matches CUTLASS's
split-K shape and is what cuBLAS's `_8x4` pipeline relies on.

Cross-CTA reduction without grid-sync infra (no `cudaLaunchCooperativeKernel` in the codebase) must
come from a second kernel launch — the launch boundary itself is the cheapest cross-CTA barrier CUDA
offers (no persistent-kernel framework, no cooperative-groups support to set up).

## Approach: a single tile-IR pass + pre-tiled reduce TileOp

`017_atomic_free_splitk.py` runs after `015_gate_splitk_residual` (which handles the linear-residual
gating, orthogonal to this concern) and before any staging / buffer passes (020+). It declares a single
`ATOMIC_FREE_SPLITK` BOOL knob and forks per shape:

- **False** — original TileOp tagged with `ATOMIC_FREE_SPLITK=False`. Legacy atomicAdd path stays
  intact.
- **True** — a Graph fragment with two TileOps:
  1. The matmul TileOp, with every output `Write` rewired to a workspace name and `Var(K_s)` prepended
     to the index. `Body.coordination.atomic_axes` now returns `∅` for those Writes, so codegen
     emits a plain store.
  2. A sibling **pre-tiled reduce TileOp** consuming the workspace, summing along `S`, writing to the
     original output.

The reduce TileOp is constructed with a fixed schedule (no Forks, no autotune knobs to twiddle): one
CTA per 16×16 output tile, 256 threads per CTA (one thread per output cell), `S` as a fully-unrolled
register-side `SerialTile`. No smem, no `TreeHalve`, no cross-thread coordination — `S ≤ 32` makes a
serial register loop bandwidth-optimal. A boundary `Cond` gates the Write so non-divisor M or N stays
correct (out-of-bounds entries in the workspace stay at `cp.zeros` init).

### Why a pre-tiled TileOp (not a macro op)

A regular `TileOp` with the schedule pinned at construction time:

- skips the tile planner entirely (no autotune knobs exposed → no exploration to "waste"),
- reuses the existing tile→kernel materializer `100_materialize_tile.py` (no new lowering pass),
- keeps the reduce body visible to downstream fusion (if we ever fuse a post-reduce activation),
- adds zero new IR surface (no new dataclass, no new pass to lower it).

The "autotuner re-explores degenerate shapes" objection only applies to a graph-level
`ReduceOp → loop_lifting → tile_planner` path. Direct TileOp construction bypasses both.

### Why `015_gate_splitk_residual` is unchanged

015 gates a fused linear residual (`matmul_add`-shape) on `K_s == 0` so the residual lands in exactly
one of the S accumulators regardless of how they're later combined:

- Atomic path: each CTA atomicAdds `acc_s + (r if K_s==0 else 0)` → `Σ acc_s + r`.
- Atomic-free path: each CTA writes `acc_s + (r if K_s==0 else 0)` to `partial[K_s, m, n]`; the
  reduce kernel sums → `Σ acc_s + r`.

Both paths produce the residual exactly once. 015 fires identically on both branches of the
`ATOMIC_FREE_SPLITK` fork. The only adjustment was factoring `_find_split_k_axis_name` from 015 into
the shared `_splitk_residual.py` helper (now `find_split_k_axis_name`) so both 015 and 017 use one
source of truth.

After 015 fires, the matmul body has **two** Writes — one in the `Cond.body` (residual-bumped) and
one in `Cond.else_body` (bare Accum). 017's rewrite walks the body via `Stmt.nested()` /
`Stmt.with_bodies(...)` and rewires both.

## Files

**New (2 files):**

- `deplodock/compiler/pipeline/passes/lowering/tile/017_atomic_free_splitk.py` — the pass. Declares
  `ATOMIC_FREE_SPLITK = Knob(name, KnobType.BOOL, hints=(False, True))`. Builds a Graph fragment with
  InputOp aliases for the matmul's existing inputs, a matmul TileOp node targeting
  `f"{root.id}__partial"` (Tensor shape `(S, M, N)`), and a reduce TileOp node taking the workspace
  and writing the original output Tensor. Returns `list[TileOp | Graph]` with one entry per BOOL
  candidate.
- `tests/compiler/passes/test_atomic_free_splitk.py` — unit tests covering: SPLITK=1 skip, knob
  idempotence, MMA path skip, fork-emits-two-variants, env-pin True / False, fragment shape
  (workspace Tensor + reduce structure + matmul-Write rewire), and the matmul_add (Cond) residual
  interaction.

**Modified (2 files):**

- `deplodock/compiler/pipeline/passes/lowering/tile/_splitk_residual.py` — added
  `find_split_k_axis_name(op: TileOp) -> str | None`.
- `deplodock/compiler/pipeline/passes/lowering/tile/015_gate_splitk_residual.py` — imports
  `find_split_k_axis_name` from `_splitk_residual` instead of defining it locally.

**No change needed:**

- `compiler/backend/cuda/program.py:_allocate` — `role="scratch"` workspace tensor auto-zeros via
  `cp.zeros` and persists across both launches (single `arrays` dict reused by `_launch`).
- `compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — handles the reduce body
  unchanged (`SerialTile(S, unroll, body=[Load, Accum])` + `Cond(in_bounds, [Write])` is a standard
  register reduce; no smem, no cross-thread coordination).
- `compiler/graph.py:splice` — multi-node fragment splicing handles the matmul + reduce pair; the
  reduce node's id collides with `root.id` at fragment-build time, gets auto-renamed by the splicer,
  then promoted back to `root.id` after the original is removed (rename-to-friendly-name tail).

## Reused infrastructure

| Concern | Existing API | Path |
|---|---|---|
| Knob/Fork declaration | `Knob(name, KnobType.BOOL, hints=(False, True))` + `narrow` | `compiler/pipeline/knob.py` |
| Multi-variant rewrite | `rewrite(...) -> list[Op \| Graph]` (framework forks) | `040_use_ring_buffers.py:54-76` |
| Graph fragment splicing | `Graph.splice(consumed, output)` (auto-rename + alias) | `compiler/graph.py:481` |
| K_s axis lookup | `find_split_k_axis_name(op)` → `Body.coordination.atomic_axes` | `_splitk_residual.py:38` |
| Atomic detection | `Body.coordination.atomic_axes(write)` | `compiler/ir/stmt/body.py:603` |
| Residual gating | `gate_linear_epilogue_on_k_s_zero` (unchanged, fires before 017) | `_splitk_residual.py:102` |
| Workspace allocation | `_allocate` role="scratch" → `cp.zeros` | `compiler/backend/cuda/program.py:270-305` |
| Tile→Kernel lowering | `100_materialize_tile.py` (SerialTile + Accum + Write) | `compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` |

## Verification

End-to-end commands:

```bash
# Unit tests
./venv/bin/pytest tests/compiler/passes/test_atomic_free_splitk.py -v

# Full compiler suite (~1021 tests)
make test

# Lint
make lint

# Accuracy + perf at 2048³ fp32 with atomic-free split-K pinned
DEPLODOCK_KNOBS="BM=16,BN=16,FM=8,FN=8,BK=16,SPLITK=4,BUFCNT=3,STAGE=11,PAD_SMEM=0,ATOMIC_FREE_SPLITK=1" \
  ./venv/bin/python -m deplodock.deplodock run \
  --code "torch.matmul(torch.randn(2048,2048,device='cuda'), torch.randn(2048,2048,device='cuda'))" \
  --bench --warmup 20 --iters 50
# Expected: accuracy passes, 2 kernel launches (matmul + reduce TileOp),
# total latency comfortably under 400 µs (target ~350 µs → ~1.30× cuBLAS).

# Confirm autotune picks the new variant under patience 100
rm -f /tmp/tune_2048_atomicfree.db*
DEPLODOCK_TUNE_DB=/tmp/tune_2048_atomicfree.db ./venv/bin/python -m deplodock.deplodock tune \
  --code "torch.matmul(torch.randn(2048,2048,device='cuda'), torch.randn(2048,2048,device='cuda'))" \
  --patience 100 --bench -q --warmup 10 --iters 30

# Inspect the perf rows
sqlite3 /tmp/tune_2048_atomicfree.db "
SELECT p.latency_us_median,
       json_extract(p.knobs,'\$.ATOMIC_FREE_SPLITK') AFS,
       json_extract(p.knobs,'\$.SPLITK') SPLITK
FROM perf p ORDER BY p.latency_us_median LIMIT 10;
"
```

## Risks / open questions

- **Reduce kernel at very small shapes.** Fixed `BM_RED=BN_RED=16` oversizes the grid for
  `M·N < 64K`. The boundary `Cond` keeps Writes safe, but tiny shapes pay grid-launch overhead.
  Defer dynamic `BM_RED` tuning until a real workload hits it.
- **Downstream pass interference with the reduce TileOp.** Passes 020+ (stage_inputs, ring buffers,
  async copy, TMA, etc.) walk every TileOp by default. The reduce body is minimal (no
  `serial_outer`, no SYNC `StageBundle`, just a register loop + Cond + Write), so each pass's
  structural guards should `RuleSkipped` it cleanly. The `ATOMIC_FREE_SPLITK=True` knob on the
  reduce TileOp is a convenient marker if a future pass needs to opt out explicitly.
- **Reduce kernel reads fp32 from gmem with no smem staging.** For the target 2048³ shape this is
  bandwidth-bound (~50 µs theoretical at 1.5 TB/s); near-optimal. If a later workload shows the
  reduce becoming a bottleneck, add an `020_stage_inputs`-style smem cache as a follow-up.
- **Symbolic / dynamic shapes.** 017 rejects when the output shape isn't fully static
  (`out_shape[i].is_static` must hold). Symbolic-K matmuls (`SPLITK=1` by planner construction)
  hit the SPLITK=1 skip before this gate; symbolic M/N skip cleanly. Atomic-free split-K for
  hint-driven dynamic shapes is a future-work item.

## Milestones

1. **Factor `find_split_k_axis_name` into `_splitk_residual.py`** — pure refactor; 1013 tests pass.
   **Done** (commit `2e6fbab3`).
2. **Add `017_atomic_free_splitk.py` + 8 unit tests** — pass fires correctly, fork variants emit,
   atomic-free Writes resolve to empty `atomic_axes`. `make test` green (1021 tests). **Done**
   (commit `b58df8f9`).
3. **End-to-end accuracy + perf** — **partially done; perf goal blocked.**
   - Atomic-free path is end-to-end correct at `BUFCNT=1` on 2048³ fp32: PASSes accuracy vs eager;
     deplodock runs matmul + reduce in two launches; reduce is 15 µs / 2.7% of total (as predicted
     by the bandwidth model).
   - **Blocking bug at `BUFCNT >= 2`**: `080_pipeline_stages` fires and its rule.txt records the
     correct pipelined form (prologue1 + prologue2 + main-with-bundle + epilogue1 + epilogue2 — 3
     `StageBundle`s), but the end-of-stage tile-IR snapshot for the matmul half of the atomic-free
     fragment carries only the **first** bundle (the K_o-inner steady-state bundle and prologue2 are
     gone). The legacy atomic path at the same knobs keeps all 3 bundles intact, so the loss is
     specific to the atomic-free path's TileOp shape (the Write has K_s in the index → empty
     `atomic_axes`, vs `{K_s}` for the atomic path). With only one prologue bundle alive, the K_o
     loop reads stale / uninitialised smem from slots 1 + 2 on most iterations, producing the
     observed `mean_diff ≈ 75` vs reference at SPLITK=4 BUFCNT=3.
   - The perf target (~350 µs → ~1.30× cuBLAS) requires `BUFCNT >= 2` + ASYNC pipelining. Needs a
     follow-up debug session to find what's dropping the bundles between 080's commit and the
     stage-end snapshot. Likely candidates: a normalize-body pass that doesn't see the atomic-free
     Write's K_s-in-index as a structural marker; a fork resolution that picks a pre-080 candidate
     when the atomic-free TileOp's structural key differs.
