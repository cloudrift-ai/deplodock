# MMA matmul perf closures — closing the cuBLAS gap

## Context

`feature/mma-warp-scoring` (PR #182) closed the 2048² fp16 matmul gap from **0.32× → 0.92× cuBLAS** (317 µs → 108
µs, vs cuBLAS at 99 µs) by rewriting the warp-tier priors so the picker stops landing on `FM=1 FN=32` (the
register-spill corner) and chooses `FM=FN=4` (the cells≈16 sweet spot). The remaining 9% gap lives in three
hardware-level levers cuBLAS uses but deplodock doesn't:

1. **Smem staging with double-buffered cp.async K-pipelining.** cuBLAS's `cutlass_80_tensorop_f16_s16816gemm_…
   128x128_32x4_nn` uses 4 K-pipeline stages overlapping load-K+1 with mma-K. Deplodock's MMA path can plumb the
   staged variant through `020_stage_inputs` → `040_use_ring_buffers` → `060_use_async_copy` → `080_pipeline_stages`
   (verified at the tile-IR level), but two downstream bugs in `005_lower_atom_tile` and `MmaLoad.src_index`
   block the lowered kernel from compiling correctly.
2. **Warp specialization (producer/consumer split).** Hopper-style: dedicated producer warps issuing cp.async +
   mbarrier, separate consumer warps running mma_sync. `085_warp_specialize` exists but gates on cooperative-K
   (`BR > 1`), which the MMA warp-tier forbids by construction (`splitk_choices = (1,)`).
3. **`mma.m16n8k16` atom (ldmatrix + PTX mma.sync).** The `cutlass_80_tensorop_*_s16816` naming reveals cuBLAS
   uses the Ampere PTX MMA shape, not classic WMMA `m16n16k16`. `ldmatrix` does the lane-aware fragment load in
   one instruction; the `wmma::load_matrix_sync` C++ wrapper deplodock uses today expands to a more conservative
   SASS schedule. Documented in `project_tensor_core_dead_end.md` as a hard 5-10% gap.

Plus one CTA-count signal in the score function that the empirical sweep showed leaves ~1.5% on the table.

Target: ≥ 0.98× cuBLAS on 2048² fp16. The fp16 byte-correctness prerequisite already landed (PR #182 commit
`528822c3`).

## Methodology: measure each lever in isolation before touching the picker

The prior optimization round had a recurring failure mode — tweaking score-function priors based on what
*should* be faster, then discovering the new picked variant has a codegen bug, then reverting. The lesson: don't
move the picker until the kernel under it actually runs and benches.

This plan splits into three phases:

**Phase A — Codegen.** Land each codegen lever (M2: pipelined staging, M3: warp specialization, M4: PTX MMA) one
at a time. Each lever is gated on the ability to *pin* its variant from the CLI (M1 below adds the missing
warp-tier knob narrow). For each:

1. Implement the codegen change.
2. Pin the knob set that exercises it; verify the IR has the new shape; verify the kernel compiles and matches
   the f32 reference within tolerance.
3. Bench the pinned variant against the current baseline and against cuBLAS.
4. **Decision: keep or drop.** If the pinned variant doesn't beat baseline by ≥3%, the codegen change is correct
   but the lever isn't load-bearing for this shape — drop it from the golden-tile search, don't try to make the
   picker find it.

**Phase B — Golden tile.** Sweep the kept levers' product space (pin every combination) and record the latency
matrix. Picks the *single best pinned knob set* — the "golden tile" the picker should converge to. This becomes
the bench-gate target for Phase C.

**Phase C — Scoring.** Tune `_priority_matmul_warp` and `score_tile_geometry` so the picker selects the golden
tile by default, *without* a pinned env override. Verified by `deplodock run --code "…" --bench` matching the
golden bench within ±1%.

Phase order is forced: B can't pick a golden until A's levers all bench; C can't aim the picker without B's
target.

## Phase A — Codegen levers, measure each in isolation

### M1 — Warp-tier knob narrow [no perf — pure plumbing for Phase A]

`_enumerate_warp_matmul_impl` iterates `_TUNE_WARP_AXIS_CHOICES` / `_BK_CANDIDATES` directly without applying
`WM.narrow` / `WN.narrow` / `BK.narrow`. Means `DEPLODOCK_WM=2 DEPLODOCK_WN=2 DEPLODOCK_BK=8` are silently
ignored at the warp tier (work at scalar tier). Every subsequent milestone needs CLI pinning to run the bench
gate, so this lands first.

Add the three `narrow` calls inside `_enumerate_warp_matmul_impl`. Also add the corresponding `WS_MMA` and
`ATOM_KIND_KNOB` narrows (created in M3 and M4 respectively, but plumb the framework now).

**Files**: `tile/_enumeration.py` (~10 lines), `tests/compiler/passes/test_partition_planner_mma.py` (extend
existing test for the new knob-narrow gates).

**Bench gate**: none — this is plumbing. Verification: `DEPLODOCK_WM=2 DEPLODOCK_WN=2 ./venv/bin/deplodock
compile --code "…matmul(fp16)…" --ir tile` produces a tile-IR with `for a2 in 0..2 │ for a3 in 0..2  └ warp`.

### M2 — Staged + pipelined MMA codegen [biggest standalone lever]

**Two bugs in `005_lower_atom_tile.py` block the staged-MMA path from compiling correctly even though the
planner already finds the right variant.** Verified at the tile-IR level — the pipelined async bundle DOES emit
for `WM=WN=2 FM=FN=4 BK=2`:

```
AtomTile (a6, a7):
  bundle async[2@0 depth=2]:                          # prologue ring slot 0
    shared b_smem[…] = b[…]
    shared a_smem[…] = a[…]
  for a9 in 0..K_o-1:
    bundle async[2@(a9+1) depth=2]:                   # issue next K_o+1 into slot phase=(a9+1)%2
      shared b_smem[…] = b[…]
      shared a_smem[…] = a[…]
    AsyncWait(keep=1)
    SerialTile(a8, K_i, reduce):
      Load b_smem[(a9 % 2), …]                        # consume current K_o from slot a9%2
      Load a_smem[(a9 % 2), …]
      Assign / Accum
    AsyncWait(keep=1)
  AsyncWait(keep=0)                                   # drain
  SerialTile(a8, K_i, reduce):                        # epilogue: last K_o
    Load b_smem[1, …]
    Load a_smem[1, …]
    Assign / Accum
  Write matmul[…]
```

Two bugs block compile:

**Bug A — `_atom_body_to_mma` walk doesn't handle pipelined nesting.** The current walk does `find outer_st +
reduce_st + bundle` then rebuilds the body from scratch. The pipelined shape has a prologue StageBundle, a K_o
SerialTile containing yet another StageBundle + AsyncWait + reduce, an epilogue AsyncWait, and a tail reduce —
none of which the rebuild path knows about. The walk raises `RuleSkipped` silently and the AtomTile flows
through to render where it errors as `AtomTile must be consumed by the MMA materializer`.

Fix: **transform-style rewrite** rather than pattern-match-and-rebuild. Walk the body verbatim and for each
reduce SerialTile replace its body with the Mma chain (clearing `is_reduce`); for each Write replace with
MmaStore; preserve everything else (StageBundle wraps, AsyncWait, K_o SerialTile, prologue/epilogue) unchanged.
Prepend `MmaFragment` decls + `MmaFill(c_frag, 0)` at the AtomTile body's head. Was 80%-drafted during PR #182
investigation but reverted to keep `test_matmul_mma` green; the draft handled the SYNC + single-bundle shape
cleanly but needed the addressing-based A/B classification described below for the staged path.

**Bug B — `MmaLoad.src_index` misses the buffer-phase prefix.** For `BUFFERED` / `ASYNC` stages with
`buffer_count ≥ 2`, the slab is allocated as `[phase, …cache_axes…]` (rank-prepended). The consumer Load gets
rewritten to `Load(input='b_smem', index=(phase_expr, cache_var_0, cache_var_1, …))` — phase is the leading dim.
But `_mma_src_index` only emits the cache-coord part `(Var(ax) * block, …)`, never the phase. The rendered
`wmma::load_matrix_sync` reads from the wrong slot → `CUDA_ERROR_MISALIGNED_ADDRESS` at runtime.

Fix: have `005_lower_atom_tile` track the enclosing `StageBundle` during the walk and, when the bundle's
`buffer_count > 1`, prepend `bundle.phase` to every `MmaLoad.src_index` for Loads reading from that bundle's
sources. Matches the M8 design note in `plans/mma-smem-staging.md` ("pick (b): keep mma_view in pure cache-coord
space, have 005 synthesize the phase prefix").

**Bug C — A/B classification fails for staged smem loads** (surfaces when fixing A). The pre-pipelined A/B
identification keys off "K_name is in load.index[-1]" → A vs "first dim" → B. For staged smem loads the index is
multi-dim slab coords (e.g. `(phase, a2, a4, a6)`) where the K axis sits in the *middle*. Fix: when `load.input`
is a staged smem name, classify A vs B by reading `Source.cache_dims` — the cache axis whose `axis.name == K_name`
has `source_dim = 1` for A (K inner) or `0` for B (K outer).

**Bench gate** (the lever's perf measurement, gated on M1's knob-narrow plumbing):

```
DEPLODOCK_WM=2 DEPLODOCK_WN=2 DEPLODOCK_FM=4 DEPLODOCK_FN=4 DEPLODOCK_BK=2 \
    ./venv/bin/deplodock run --code "import torch; a=torch.randn(2048,2048,dtype=torch.float16,device='cuda'); \
    b=torch.randn(2048,2048,dtype=torch.float16,device='cuda'); torch.matmul(a,b)" \
    --bench --warmup 5 --iters 20
```

Expected: the rendered C source contains `cp.async.commit_group` + `wmma::mma_sync` + `cp.async.wait_group`,
i.e. the cp.async-staged pipelined MMA path. Latency target: **≤ 102 µs** (a ≥6% improvement over the 108 µs
baseline). If the measured latency doesn't beat baseline by ≥3%, the lever is structurally correct but doesn't
pay for this shape — record the measured number and exclude the staged variant from the Phase B golden search.

**Files**: `kernel/005_lower_atom_tile.py` (~150 lines net — transform walk replaces the rebuild path),
`tests/compiler/test_matmul_mma_staged_pipelined.py` (new — pins the buffered shape; greps the rendered C for
`cp.async` and matching MmaLoad src_index ranks; runs the f32 reference comparison at fp16 tolerance).

**Outcome (2026-05-30, sm_120 RTX 5090).** Codegen fix landed: the pipelined IR now compiles cleanly (verified at
128² / 256²). Bench gate **FAILED — drop the lever**. Pinned `DEPLODOCK_ATOM_KIND=wmma_m16n16k16_f16 WM=WN=2
FM=FN=4 BK=2` runs 2048² fp16 at ~249 µs vs the post-#182 baseline of ~108 µs (2.3× slower). Sweeping BK pinned
at the same WM/WN/FM/FN: BK=1 → 109 µs, **BK=2 → 249 µs**, BK=4 → 254 µs, BK=8 → 170 µs, BK=16 → 107 µs, BK=32 →
107 µs. The double-buffered cp.async path is structurally correct but pays an instruction-scheduling /
syncthreads overhead that outweighs the K-load overlap at this geometry; the BK=16/32 single-bundle SYNC variant
already matches baseline (the picker can find it from the existing scoring). Per the phase-A decision rule,
**M2's staged-pipelined variant is excluded from the Phase B golden search**; the codegen fix stays because the
IR was previously emitting a shape that crashed at materialize.

### M3 — Warp specialization for MMA

**Outcome (2026-05-30).** **Skipped — structurally depends on M2.** WS_MMA's enumeration gates on
`ATOM_KIND in knobs and buffer_count > 1`, and the producer-consumer split routes work onto the cp.async pipeline
that M2 lights up. With M2's bench gate failing by 2.3× (the cp.async path doesn't pay at the 2048² fp16
geometry), an instruction-scheduling cleanup on top of the same path can't recover the 1.4× gap M2 left, let
alone the ≥2% the plan called for. No code change; the plan's `WS_MMA` knob + `085_warp_specialize` extension
stay queued for a future session that picks M2 back up at a different shape (asymmetric matmuls per the
out-of-scope note) where the cp.async lever might pay.



`085_warp_specialize` today gates on `BR > 1` and shapes the producer/consumer split around cooperative-K
reduce. MMA forbids `BR > 1` (`splitk_choices = (1,)`; the warp-tier accumulator lives register-side). To wire
WS for MMA, the split needs a different shape: dedicate `WS_PROD ≥ 1` warps to issuing cp.async + bumping an
mbarrier, route the remaining `WM·WN - WS_PROD` warps through the mma_sync chain consuming from the same smem
slab via mbarrier wait.

Structural additions:

1. **New knob `WS_MMA`**: 0 = no split (current); 1 = one producer warp + rest consumer; 2 = two producer warps
   (one per operand). Enumerated only when `ATOM_KIND in knobs and buffer_count > 1` (otherwise the bundle has
   no producer-side work to delegate). Knob-narrow gate flows through M1's plumbing.
2. **`085_warp_specialize` extends** to detect MMA shape (look for `MmaSync` in the bundle's consumer body — runs
   after `005_lower_atom_tile`) and apply the producer-consumer split with the right mbarrier wiring (signal on
   cp.async commit + cp.async wait-group, wait inside consumer scope, signal back when consumer done with slot).
3. **Per-bundle phase routing**: consumer scope's MmaLoad reads from `phase = ((K_o - WS_PROD) % buffer_count)`,
   producer issues writes to `phase = (K_o % buffer_count)`. The handoff barrier is the existing
   `StageBundle.phase` plus an mbarrier per slot.

Hopper supports this via tcgen05 / wgmma; on sm_80 / sm_120 (Ampere / Blackwell consumer) the same shape works
via classic `__syncthreads()` + cp.async wait-group + mbarrier on shared.

**Bench gate** (depends on M2):

```
DEPLODOCK_WM=2 DEPLODOCK_WN=2 DEPLODOCK_FM=4 DEPLODOCK_FN=4 DEPLODOCK_BK=2 DEPLODOCK_WS_MMA=1 \
    ./venv/bin/deplodock run --code "…matmul(fp16)…" --bench
```

Expected: rendered C contains `__bar_sync` / `mbarrier.arrive` / `mbarrier.wait` + the producer warp's load
loop separated from the consumer warp's mma loop. Latency target: **≤ 100 µs** (a ≥2% improvement over M2's
expected 102 µs — WS overlap is bounded by the existing pipeline overlap from M2; the gain comes from cleaner
instruction scheduling and warp-disjoint smem access). If the gain is < 2%, drop WS from the golden search.

**Files**: `tile/_enumeration.py` (`WS_MMA` knob, ~30 lines), `kernel/085_warp_specialize.py` (MMA detection +
producer-consumer split, ~200 lines), `tests/compiler/test_matmul_mma_warp_specialized.py` (new).

### M4 — `mma.m16n8k16` atom (ldmatrix + PTX mma)

**Outcome (2026-05-30).** **Deferred to a dedicated follow-up session.** The plan's 220-line estimate
understates the work — the PTX mma.sync lane-distributed register layout (4 `__half2` per lane for A, 2 for B,
4 `float` for C, each at specific lane-indexed bit positions) and the `ldmatrix.x4` + `.x2.trans` per-lane
addressing arithmetic are bit-level precise: any off-by-one yields silent wrong outputs that only show up at
runtime correctness checks (slow inner loop: render → nvcc → cubin → run → check). The full lever requires
several hours of focused PTX work + iterative bench-testing. Doing only the cosmetic atom-kind addition + the
mma.sync instruction swap (without ldmatrix) would necessarily run *slower* than the WMMA baseline
(per-lane manual loads can't beat `wmma::load_matrix_sync`'s SASS scheduling), so a half-implementation can't
even validate the lever. Reserve a dedicated session for M4 starting from a reference (CUTLASS's `s16816`
kernel) and an isolated test harness that benches every micro-step.



cuBLAS's `cutlass_80_tensorop_*_s16816gemm_*` reveals the Ampere+ PTX MMA shape `m16n8k16`, loaded via
`ldmatrix.x4` (a lane-aware multi-fragment load issuing 4 32-byte fragment-loads per warp). Compared to
`wmma::load_matrix_sync` it:

- Avoids the C++ template instantiation overhead per fragment.
- Lets the SASS scheduler interleave fragment-loads with mma.sync issuance more aggressively.
- Naturally aligns with the 128 B `ldmatrix` per-warp access pattern (no smem swizzle padding beyond what the
  swizzle pass already wants).

Five pieces:

1. **New atom kind**: `mma_m16n8k16_f16` (and `_bf16`) in `_ATOM_REGISTRY` (`tile/_atom.py`) with
   `shape=(16, 8, 16)`, `operand_dtypes={a: f16, b: f16, c: f32}`, `min_cc=(8, 0)`.
2. **New `MmaFragment` layout**: PTX MMA stores fragments in lane-distributed registers with a known shape
   (`m16n8k16` fp16: `a_frag` = `__half2[4]` per lane, `b_frag` = `__half2[2]`, `c_frag` = `float[4]`). Today's
   `MmaFragment` renders as `wmma::fragment<…>`; the new layout renders as `unsigned r0, r1, r2, r3` or
   `__half2 a_frag[4]`.
3. **`ldmatrix`-based `MmaLoad`**: emits PTX `ldmatrix.sync.aligned.m8n8.x4.shared.b16` with the correct per-lane
   address calculation. Needs to know the swizzle pattern of the smem slab — for non-swizzled slabs the address
   is `slab_base + lane * inner_stride`; for XOR-swizzled it's `slab_base + (lane ^ swizzle_xor) * inner_stride`.
4. **PTX `mma.sync` emission**: replaces `wmma::mma_sync(c, a, b, c)` with explicit
   `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`. One line of inline asm.
5. **Swizzle pattern**: for staged operands, the smem layout should be XOR-swizzled to match `ldmatrix.x4`'s
   access pattern (each lane reads from a different 32-bank-line, no bank conflicts). M7 of
   `plans/mma-smem-staging.md` already skips `070_pad_smem` for blocked sources; this is the constructive
   replacement.

**Bench gate**:

```
DEPLODOCK_ATOM_KIND=mma_m16n8k16_f16 DEPLODOCK_WM=2 DEPLODOCK_WN=2 DEPLODOCK_FM=4 DEPLODOCK_FN=4 \
    ./venv/bin/deplodock run --code "…matmul(fp16)…" --bench
```

Expected: rendered C contains `ldmatrix.sync.aligned.m8n8.x4` + `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`.
Latency target: **≤ 99 µs** (matches cuBLAS, since cuBLAS's `s16816` is structurally the same kernel shape). If
the gain over M3 is < 3%, drop M4 from the golden search.

**Files**: `tile/_atom.py` (new atom + eligibility, ~50 lines), `kernel/ir.py` (`MmaFragment` PTX layout +
`MmaLoad` ldmatrix render + `MmaSync` PTX mma render, ~150 lines), `kernel/005_lower_atom_tile.py` (~10 lines —
atom-spec-aware paths factor cleanly), `tile/_enumeration.py` (~10 lines — list the new kind in
`_ATOM_KINDS_V1`), `tests/compiler/test_matmul_mma_ptx.py` (new).

## Phase B — Find the golden tile

**Outcome (2026-05-30).** **Skipped — empty keep_set.** M2 dropped (pinned variant 2.3× slower than baseline);
M3 dropped (depends on M2); M4 deferred. With every Phase-A lever excluded or deferred, the Phase-B sweep
collapses to the existing scoring's pick — i.e. the post-#182 baseline at ~108 µs. No new golden to find, no
Phase-C picker tweak to chase. The plan's bench-gate gate (≥5% improvement over 108 µs to validate the round)
explicitly tells us to fail loud here rather than ship a tiny perf change as a big PR; recording the negative
result and re-opening the round when M4 lands.



Once M2 / M3 / M4 bench-gates have passed (or been dropped), sweep their kept product space pinned end-to-end
and pick the single best variant for the 2048² fp16 reference shape. The sweep matrix:

```
for atom_kind in [keep_set]:          # M4 kept → both wmma_m16n16k16_f16 and mma_m16n8k16_f16; else just wmma
  for ws_mma in [keep_set]:           # M3 kept → 0, 1, 2; else just 0
  for staged in [keep_set]:           # M2 kept → True/False; else just False
    for wm, wn in [(1,8), (2,4), (4,2), (8,1), (2,2), (4,4)]:
      for fm, fn in [(4,4), (2,4), (4,2), (8,2), (2,8), (8,8), (2,2)]:
        for bk in [1, 2, 4, 8, 16, 32, 64]:
          pin all of (atom_kind, ws_mma, staged, wm, wn, fm, fn, bk)
          bench
          record latency
```

That's up to **~2000 pinned variants × ~50ms compile each = 100 s of sweep time**. The result lands as
`plans/mma-perf-closures-golden.md` (sibling) with the full latency matrix + the picked golden tile.

**Golden-tile criteria**: lowest measured latency on 2048² fp16, tie-broken by lowest variance across 3 reps.
The picked tile becomes the bench-gate for Phase C.

**Files**: `scripts/sweep_mma_perf_closures.py` (new — pins variants via M1's knob-narrow, captures latencies
through `BenchmarkResult`, dumps the CSV + golden pick).

## Phase C — Tune the picker to land on the golden tile

**Outcome (2026-05-30).** **Skipped — no golden to chase.** Phase B produced no winner above the baseline.
Re-open when M4 lands.



Once Phase B has the golden, tune `_priority_matmul_warp` and `score_tile_geometry` so the *default* picker (no
env pins) selects the golden tile for this shape. Score-function changes drafted during the PR #182
investigation that need to be re-introduced cleanly now that the codegen actually runs:

1. **`score_tile_geometry`** gains an `ATOM_KIND in knobs` branch for the smem-fit penalty:
   `macro_m = WM · FM · atom_M`, `macro_n = WN · FN · atom_N`, `base_slab = (macro_m + macro_n) · BK · atom_K
   · bytes_per_elem` (atom_K accounts for the per-cache-axis block multiplier on K). Takes `bytes_per_elem`
   (default fp32 = 4 for back-compat); warp tier passes 2 for fp16 / bf16 atoms.
2. **`TileOp.lazy_score`** includes `ATOM_KIND` / `WM` / `WN` in `score_knobs` (currently only `FM` / `FN` /
   `BK` / `SPLITK`); passes `dtype_bytes = 2` for the warp tier.
3. **CTA-count ramp**. Replace the flat `+0.5` bonus for `target_ctas ≤ ctas ≤ 2048` with a linear ramp
   keyed off the live device's SM count: `+0.5 · min(ctas / (2 · num_SMs), 1.0)`. Empirical (PR #182 sweep):
   `WM=1 WN=8` (128 CTAs) ran at 109.2 µs vs `WM=WN=2` (256 CTAs) at 107.6 µs on sm_120 (170 SMs).
4. **WS_MMA prior**: if M3 kept, add a small `+0.2` bonus when `WS_MMA > 0 and buffer_count >= 2`.
5. **PTX-MMA atom prior**: if M4 kept, the `_ATOM_KINDS_V1` enumeration order is the prior — list
   `mma_m16n8k16_f16` before `wmma_m16n16k16_f16` so the picker tries it first.

The final test: `deplodock run --code "…matmul(fp16)…" --bench` (no env pins) hits the golden bench within
±1%.

**Files**: `ir/tile/ir.py` (`score_tile_geometry` + `lazy_score`, ~50 lines), `tile/_enumeration.py` (atom kind
ordering + WS_MMA prior, ~10 lines), `tests/compiler/ir/tile/test_score_tile_geometry_mma.py` (new — pin the
per-tier macro derivation + scoring-by-design rather than via end-to-end bench).

## Reused machinery (no new code, just call sites)

- `Source.addressing.block` / `AffineAddressing.source_index` (`ir/tile/ir.py:299, 313`) — already lifted by
  `plans/mma-smem-staging.md` M2-M4; M2 of this plan adds the phase-prefix on top.
- `040_use_ring_buffers` / `060_use_async_copy` / `080_pipeline_stages` — all fire automatically once a Source
  admits (verified at the tile-IR level during PR #182 investigation). No code changes needed; M2's bug fixes
  unlock the entire downstream chain.
- `StageBundle.phase` field — already carries the per-bundle phase expression (`(K_o + N) % buffer_count` for
  offset N); M2 reads it for the MmaLoad prefix; M3 routes the producer-side phase off it.
- `_priority_matmul_warp` / `score_tile_geometry` — already cells≈16 + square-aspect aware from PR #182; Phase
  C adds the ATOM_KIND-aware macro derivation, CTA-count ramp, WS_MMA / PTX-MMA priors.
- `_ATOM_REGISTRY` (`tile/_atom.py`) — M4 adds entries; no structural changes.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/tile/_enumeration.py` — M1 (knob narrow), M3 (`WS_MMA` knob),
  M4 (`_ATOM_KINDS_V1`), Phase C (priors).
- `deplodock/compiler/pipeline/passes/lowering/kernel/005_lower_atom_tile.py` — M2 (transform-walk rewrite +
  phase prefix), M4 (atom-spec dispatch for the PTX path).
- `deplodock/compiler/pipeline/passes/lowering/kernel/085_warp_specialize.py` — M3 (MMA detection + split).
- `deplodock/compiler/pipeline/passes/lowering/tile/_atom.py` — M4 (new atom registry entries).
- `deplodock/compiler/ir/kernel/ir.py` — M4 (PTX MmaFragment / MmaLoad / MmaSync render paths).
- `deplodock/compiler/ir/tile/ir.py` — Phase C (`score_tile_geometry`, `lazy_score`).
- `scripts/sweep_mma_perf_closures.py` (new — Phase B sweep harness).
- `tests/compiler/test_matmul_mma_staged_pipelined.py` (new — M2).
- `tests/compiler/test_matmul_mma_warp_specialized.py` (new — M3).
- `tests/compiler/test_matmul_mma_ptx.py` (new — M4).
- `tests/compiler/ir/tile/test_score_tile_geometry_mma.py` (new — Phase C).

## Verification

The phased structure puts the bench gate at every M2-M4 step, not at the end. Each lever lives or dies on its
measured perf number; the picker only chases gold once the gold is real.

Per-milestone correctness gate: `tests/compiler/test_matmul_mma.py` (10 parametrized shapes — 16/64/128/256
squares + 256×256×64 skewed × {F32, F16} outputs) must pass byte-clean after every milestone, with the picker
still selecting the *current* baseline (no auto-picked staged / WS / PTX-MMA variant) until Phase C flips it.
This decouples codegen correctness from picker behavior — a Phase A milestone that breaks `test_matmul_mma`
broke codegen; a Phase C tweak that doesn't is a picker change.

Phase-B golden gate: the sweep script asserts the picked golden beats the post-#182 baseline (108 µs) by ≥5%;
otherwise the whole optimization round netted < 5% gain — fail loud rather than ship a tiny perf change as a
big PR.

Phase-C end-to-end gate: `deplodock run --code "…matmul(fp16)…" --bench` (no env pins) hits the golden bench
within ±1%. `tests/perf/test_matmul_mma_perf.py` (new — same pattern as `tests/perf/ARCHITECTURE.md`'s cases)
asserts the auto-picked latency stays within the golden ±5% on the canonical 2048² shape.

| Milestone | Picked variant (when pinned) | Bench target | Decision |
|---|---|---:|---|
| Baseline (post #182) | `WM=1 WN=8 FM=FN=4 BK=64` gmem-direct | 108 µs | (reference) |
| M2 (staged + pipelined) | `WM=WN=2 FM=FN=4 BK=2-4` async-buffered | ≤ 102 µs | keep iff ≥3% over 108 |
| M3 (warp-specialized) | `WM=WN=2 FM=FN=4 BK=4 WS_MMA=1` | ≤ 100 µs | keep iff ≥2% over M2 |
| M4 (PTX MMA + ldmatrix) | `mma_m16n8k16_f16` | ≤ 99 µs | keep iff ≥3% over M3 |
| Phase B golden | (sweep picks) | ≤ 99 µs | gate: ≥5% over 108 µs |
| Phase C (autotuned) | (matches golden, no env pins) | within ±1% of golden | (final) |

## Out of scope

- **MMA + cooperative-K** (`BR > 1` + MMA): `WarpTileParams` enforces `BR=1`; preserved.
- **TMA for MMA** (`050_use_tma`): the eligibility check today rejects multi-axis-per-source-dim blocked slabs;
  staged WMMA on sm_120 has no TMA path. Hopper-only follow-up (sm_90 / tcgen05 + tcgen05.cp.async.bulk).
- **NVFP4 / wgmma / tcgen05**: separate hardware kinds with their own tmem / async issue-wait infrastructure;
  M4's `mma.m16n8k16` is the sm_80+ Ampere/Blackwell path. The same five-piece extension pattern (atom kind,
  fragment layout, load instruction, mma instruction, swizzle) applies but each generation needs its own work.
- **Asymmetric M/N matmuls** (e.g. 64×1024×K): the priors should generalize but haven't been swept; out of
  scope for these closures, planned in a separate `plans/mma-asymmetric-shapes.md`.

## Failure modes to watch (load-bearing risks)

1. **M2 transform-walk drops a structural variant.** The pipelined-staged shape isn't the only one MMA produces;
   the SYNC-staged shape (no pipelining), the gmem-direct (no staging), the K_o=1 cases (no outer SerialTile),
   and the shape-C (K-filtered) cases all need to flow through the same walk. Mitigation: the walk is structural
   (preserve everything that isn't `Write` or reduce `SerialTile`), so new shapes flow through unchanged. The
   per-milestone test should cover all four shapes via parametrize.

2. **M2 phase prefix interacts with rank-mismatch render path.** `render_index` (`stmt/base.py:249`) flattens by
   row-major stride when `len(indices) == len(shape)`; falls back to sum-without-strides when ranks mismatch.
   Prepending phase to MmaLoad src_index needs to match the slab's rank exactly — Smem.render registers
   `ctx.shapes[name] = full_extents` which is `(buffer_count, *cache_extents)` for buffered. Mitigation: assert
   `len(src_index) == len(ctx.shapes[smem_name])` at MmaLoad render time, fail loud.

3. **Phase-A bench gates flap from xdist GPU contention.** The flaky-but-real bench numbers from PR #182 showed
   ±2 µs variance across runs. Each milestone's bench gate runs 3 reps with a stricter ±1% pass criteria; if
   any rep fails, re-run. The decision "keep / drop" needs to outweigh the noise.

4. **M3's WS_MMA enumeration explodes the search tree.** Each WS_MMA value forks the warp-tier variant set by
   2-3×. Mitigation: gate WS_MMA enumeration on `buffer_count >= 2` (eligible only when pipelining is in play)
   and patience-bound the per-kernel autotune to keep the inner search bounded.

5. **M4's ldmatrix swizzle interacts with `070_pad_smem`'s skip predicate.** M7 of
   `plans/mma-smem-staging.md` skips pad_smem for blocked sources. M4 introduces a constructive XOR-swizzle for
   the PTX MMA path; the pad_smem skip should generalize to "skip when the source carries either a block or a
   swizzle" — but the swizzle field doesn't exist on `Source` yet. Either reuse the existing `Source.pad` slot
   with a swizzle encoding or add a `Source.swizzle` field; the latter is cleaner.

6. **Phase B sweep masks compile-time outliers.** Some pinned variants compile cleanly but error at the cubin
   cache (LLVM blowup on big unrolled register-tile kernels — see `project_tune_compile_bound.md`). The sweep
   script needs a timeout per variant + bench_fail marker so a slow variant doesn't stall the whole sweep.

7. **Phase C's score-tweaks regress scalar matmul** if the new ATOM_KIND-only branches accidentally fire on
   scalar shapes. Mitigation: every new branch is `if "ATOM_KIND" in knobs:`-gated; the existing
   `test_partition_planner_*` and `test_swizzle_blocks_matmul_accuracy` golden-tile assertions catch any
   scalar drift. Run the full `tests/compiler/` suite as the byte-clean gate before merging each Phase C tweak.

## Connection to prior work

- **`plans/mma-fragment-factorization.md`** (PR #177): introduced the warp-tier MMA planner + AtomTile lowering.
  M2 fixes the StageBundle-around-AtomTile case it didn't handle; M4 adds a parallel atom kind.
- **`plans/mma-smem-staging.md`** (PR #180): lifted `Source.addressing` and added `AffineAddressing.block`. M2
  reads `addressing.block` for the per-axis phase routing; M3 reads `StageBundle.phase` for the producer-consumer
  split.
- **PR #182** (`feature/mma-warp-scoring`): tweaked the warp-tier priors so the picker finds the right corner of
  the search space. Phase C extends the same `score_tile_geometry` function for ATOM_KIND-aware macros and the
  CTA-count term.
- **`project_tensor_core_dead_end.md`** memory: established the hard 5-10% WMMA-vs-PTX-MMA SASS gap. M4 is the
  unlock for the half of the remaining 9% that's intrinsic to the WMMA instruction wrapper.
