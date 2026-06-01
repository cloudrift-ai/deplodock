# mma.sync register-tier software pipelining — double-buffered ldmatrix prefetch

## Context

After the smem swizzle landed (`plans/mma-sync-smem-swizzle.md`, S0–S4), the s16816 mma.sync matmul reaches **0.93×** cuBLAS
on 2048² fp16 (RTX 5090, sm_120): 107µs vs cuBLAS 99µs. We already match cuBLAS's *launch geometry* exactly — same
`cutlass_..._128x128_32x4` tile (128×128, K-tile 32, 4 warps / 128 threads, s16816 atom, 256 CTAs). Pushing the knobs toward
cuBLAS's config does **not** close the gap (matching its 4 smem stages via `BUFFER_COUNT=4` regresses us to 124µs / 0.80×).

The gap is the **inner instruction schedule**, proven by the ncu warp-stall breakdown of our golden kernel
(`smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`):

```
wait              (mma.sync pipeline)   3.28   ← dominant
barrier           (mbarrier / sync)     0.41
short_scoreboard  (ldmatrix smem→reg)   0.26
long_scoreboard   (gmem)                0.23   ← well hidden by the TMA ring
```

Our pipelining is **one tier too shallow**. `080_pipeline_stages` software-pipelines the *smem* tier (gmem→smem TMA ring,
`BUFFER_COUNT`) — that works, `long_scoreboard` is near-zero. But the *register* tier (smem→register `ldmatrix`) is **not**
pipelined: per K-step the generated reduce body is `[ldmatrix×12, mma.sync×32]` — **load-then-compute, serially**. The 32
`mma.sync` can't begin until the 12 `ldmatrix` loads land, and the next K-step's loads don't issue until the current mma
chain drains. So the tensor-core pipeline empties at every K-step boundary (the `barrier` + `short_scoreboard` window). With
`BK=2` K-substeps × 64 K_o tiles you pay that bubble ~128× per CTA.

cuBLAS/CUTLASS double-buffers the operand **register fragments**: it issues the *next* K-step's `ldmatrix` into a second
fragment buffer **concurrently** with the *current* K-step's `mma.sync`, so the loads overlap the compute and the mma
pipeline never drains. That overlap is exactly the **~66 extra regs/thread** it spends (232 vs our 166). `095_interleave_loads`
is the *scalar*-FMA analog of this (sinks `Load`+`Assign` clusters within one iteration) but it does **not** touch the
`LdmatrixLoad`/`MmaSyncPtx` chain and does **not** double-buffer across iterations — so register-tier prefetch genuinely does
not exist for the mma.sync path.

Intended outcome: 2048² fp16 from ~107µs (0.93×) toward ~99µs (≈ cuBLAS parity), `wait`/`barrier` stalls reduced, accuracy
unchanged (max_diff = 0), **zero regression with the knob off**.

## The knob (gates the whole feature; autotuner A/Bs it per shape)

`DEPLODOCK_REG_PIPELINE` — `Knob("REG_PIPELINE", KnobType.BOOL, hints=(False, True))`, declared in the new pass. Default
**off** (first hint) so the greedy/DB-less path and every existing test stay byte-identical; the autotuner forks on it
(`True`/`False`) like `PAD_SMEM` / `HOIST_COMPUTE` / `TMA`, measures both, and the goldens record whichever wins per shape.
Register pipelining is **not** universally faster — it costs registers (occupancy) and only pays when the mma chain is long
enough to hide the prefetch — so it must be a measured fork, never forced on. A pin (`DEPLODOCK_REG_PIPELINE=1`) forces it
for bring-up / A-B benching. The knob rides on the warp-tier `TileOp` and is read by the new pass; it joins the warp-fork
level schema in `010_partition_loops` so the autotuner enumerates it alongside `BUFFER_COUNT`.

## The transform

Software-pipeline the K reduce so each step's `ldmatrix` overlaps the previous step's `mma.sync`, with **two operand
fragment buffers** (the accumulator `c_frag` stays single-buffered — it accumulates across K; only the `a`/`b` `RegFragment`
operands double-buffer, which is the register cost):

```
ldmatrix → frag_buf[0]                    # prologue: peel the first load
for k in 0..K-1:
    ldmatrix → frag_buf[(k+1) % 2]        # prefetch next step's operands, issued BEFORE…
    mma.sync(acc, frag_buf[k % 2])        # …this compute — the load latency hides behind the mma chain
mma.sync(acc, frag_buf[(K-1) % 2])        # epilogue: compute the last buffered step
```

`080_pipeline_stages` is the structural template (prologue / steady / epilogue peeling + ring-index rotation) — this applies
the same shape one tier down, on the `RegFragment` chain instead of the `StageBundle` smem ring.

## Constraints that shape the design

- **Operand frags double-buffer; the accumulator does not.** `c_frag` (`float[4]` per cell, 32 cells = 128 regs) accumulates
  in place across all K and must stay single-buffered. Only the `a` frags (`unsigned[4]`×4 M-cells) + `b` frags
  (`unsigned[2]`×8 N-cells) ≈ 32 regs/step double-buffer → +~32 regs/thread (166 → ~198, under the 255 cap; cuBLAS's 232
  carries more working state). Register pressure → occupancy is exactly why this is a measured knob.
- **Two K tiers, not one.** Our K splits `K_o` (smem-staged TMA ring, 64 tiles) × `K_i` (`BK`=2 substeps within a tile). The
  *within-tile* boundary (K_i 0→1, same smem slot) is simple; the *cross-tile* boundary (K_o→K_o+1, new smem slot + mbarrier)
  is the harder, higher-value one (it's where the `barrier` stall lives). M1 does within-tile; M2 spans the K_o boundary.
- **The reduce body is post-005.** `005_lower_atom_tile` emits the `[ldmatrix×12, mma×32]` chain inside the
  `SerialTile(K_i, stage_inner, reduce-cleared)`; `010_split_register_axes` then replicates it per (M_r, N_r) cell. The new
  pass restructures the reduce — decide whether it runs **before** 010 (peel the loop once, replication handles the cells) or
  **after** (operate on the unrolled chain). Before-010 is cleaner (less code to rewrite) but the prologue/epilogue peeled
  `ldmatrix` must survive replication; after-010 sees concrete per-cell frags but a bigger flat chain.
- **Correctness is the precise oracle.** Any peeling / buffer-rotation bug shows up instantly as `max_diff > 0` on the
  accuracy check — iterate on-GPU against it, smallest reproducing shape first.

## Findings (RTX 5090, sm_120, 2048² fp16 — measured during implementation)

**M0 (landed) + M1 (landed)** — `REG_PIPELINE` knob + the new kernel pass `kernel/013_pipeline_mma_regs.py`. The pass is an
op-fork (`list[TileOp]`, one per polarity) mirroring `HOIST_COMPUTE` / `PAD_SMEM` — **not** wired into the
`010_partition_loops` level schema (that was the plan's initial guess; the op-fork is self-contained, lower-risk, and the
inner per-op search explores it identically — confirmed: a bounded `tune` run measures 8 on / 8 off `perf` rows). Default
**off**, byte-identical when off. M1 peels the `K_i` reduce into a double-buffered prologue/steady/epilogue (`__rp1` twin
operand frags; accumulator stays single). Accuracy `max_diff = 0` across 256²–2048², f32 out, rectangular.

**M1 is perf-neutral-to-negative**, and `ncu` says why: the within-tile `K_i` boundary lives in **one basic block**, where
ptxas already overlaps loads/compute (same lesson as `095_interleave_loads`). Batching all 16 `ldmatrix` up-front just adds
LSU pressure — `short_scoreboard` *rises* 0.43→0.91, `barrier` dips 0.82→0.74, `wait` (mma pipeline) stays dominant. At the
cuBLAS-optimal pinned config M1 regresses 107→110µs (BC=3) / 125→129µs (BC=4). So the within-tile lever is the wrong one;
the bubble is the **cross-K_o (smem-tile) boundary**.

**The cuBLAS gap is the register-pipelined inner schedule — proven by an apples-to-apples profile at BC=4** (the
`cutlass_80_tensorop_f16_s16816gemm_128x128_32x4` config cuBLAS actually runs — confirmed via `ncu` kernel name; identical
128×128 tile, K-tile 32, 4 stages, 256 CTAs, 128 threads):

| 2048² fp16, BUFFER_COUNT=4 | ours | cuBLAS |
|---|---|---|
| theoretical + achieved occupancy | 8.33% (1 block/SM) | 8.33% (1 block/SM) |
| occupancy limiter | shared mem (~64 KB) | shared mem (~64 KB) |
| **registers / thread** | **166** | **232** |
| `barrier` stall (per issue active) | 0.40 | 0.05 |
| tensor-pipe active | 42% | 47% |
| latency | 125µs | 99µs |

Same geometry, smem, **and occupancy** — the only difference is the **~66 extra regs/thread cuBLAS spends on operand
double-buffering** (register pipelining), which keeps the mma pipeline fed across the K-step boundary (barrier 0.05, pipe
47%). Ours drains at every `MbarrierWait` boundary (barrier 0.40, pipe 42%). Because BC=4 is **shared-mem-bound at 1
block/SM**, registers 166→255 are **free** (occupancy can't drop) — exactly why cuBLAS picks BC=4 + 232 regs, and why this
is the right config for M2. Our best OFF config is BC=3 (107µs / 0.93×, 2 blocks/SM at 17% occ, where the 2nd block partly
hides the boundary bubble — which is why the gap looks smaller there and masks the lever).

**Orthogonal finding (not REG_PIPELINE):** the trailing `__syncthreads()` after each TMA `MbarrierWait`
(`100_materialize_tile.emit_async_wait`) is a compiler fence; dropping it on the large 128×128 tile is accuracy-clean and
worth ~1.5µs (107→105.6 BC=3, 125→121.5 BC=4) — but it's load-bearing for the smem-ring WAR on small tiles, so it would need
a tile-size gate (or a compiler-only `asm volatile("":::"memory")` fence) and is a separate change.

**M2 status — landed (correct), but perf is marginal — register pipelining can't close the cuBLAS gap on this GPU.**
The transform replaces M1's within-tile peel: it prefetches each K_o tile's *first substep* operands one iteration ahead
into a loop-carried `__rp1` buffer, by **moving** the slot wait to the iteration bottom (one `AsyncWait`+`Sync`/iteration, so
the smem-ring WAR guard is preserved) and priming iteration 0 with a `Cond(K_o==0)` *inside* the loop (so the TMA-group
partitioner doesn't split the ring). Accuracy `max_diff = 0` across 256²–2048².

**The SSA blocker, solved.** A loop-carried `__rp1` (read at the top of iteration *t*, rewritten by the prefetch at the
bottom) is not single-assignment within one loop body, so `normalize_body`'s `topo_sort_siblings` (runs on every `TileOp`
construction) reordered the `mma` *after* the prefetch (def→use edge), computing the wrong tile (`max_diff ≈ 24-32`). Fix:
`ir/stmt/normalize.py` `_LOOP_CARRIED_MARK` — names containing `__rp1` are excluded from the topo def-use graph, so their
read-then-rewrite source order is preserved (single-assignment bodies, i.e. everything else, are unaffected; full compiler
suite green). `rename_ssa_sequential` doesn't touch kernel-IR frag names and `hoist` is off for TileOp bodies, so topo was
the only reorderer.

**Perf (RTX 5090, 2048² fp16, same-run A/B).** M2 reduces the targeted boundary stall — ncu `barrier` 0.63→0.35 (toward
cuBLAS's 0.05), `wait` 3.47→3.14, tensor-pipe 37→37.7% — for a **~1.5%** BC=4 win (OFF→ON), but it does **not** reach
cuBLAS: the kernel is **mma-pipeline-throughput-bound** (`wait` ≈ 3.1 dominates), and register pipelining only removes the
*load/boundary* stalls, not the tensor-pipe `wait`. **Full-tile prefetch** (all substeps, ~205 regs, closer to cuBLAS's
232) *regressed* — the register pressure outweighs the benefit on this mma-bound kernel; the first-substep-only version
(+~22 regs) is what's kept. BC=4 ON still trails our BC=3 best, so the knob stays **off by default**; the autotuner can pick
it where it marginally helps. Closing the last ~7% to cuBLAS needs a better **mma instruction schedule** (interleaving
independent mma chains to hide the tensor-pipe latency), a different lever than load prefetching.

## Milestones (each independently validatable; gate on accuracy + ncu, not pytest loops)

**M0 — knob + pass skeleton (no behavior change).** Add the `REG_PIPELINE` knob + a new kernel pass
`0XX_pipeline_mma_regs.py` that detects the mma.sync reduce and is an **identity no-op** (knob off, or on). Thread the knob
into the `010_partition_loops` warp-fork level schema so it enumerates. *Validate:* full compiler suite + WMMA-removed
mma.sync benches byte-identical; `--ir kernel` unchanged with the knob both off and on (skeleton only). Isolates the
plumbing from the behavior change.

**M1 — within-tile double-buffer (the contained first cut).** With `REG_PIPELINE=1`, peel the `K_i` reduce: allocate a
second operand-fragment set, emit the prologue `ldmatrix`, and in the steady state issue `ldmatrix → buf[(k+1)%2]` before
`mma.sync(buf[k%2])`, with the epilogue mma. *Validate:* accuracy max_diff = 0 (256² → 2048², f16 + f32 out); `--ir kernel`
shows the peeled prologue/steady/epilogue shape; ncu `short_scoreboard` drops at the K_i boundary; re-bench (partial win —
covers the K_i 0→1 boundary, ~half the bubbles).

**M2 — cross-K_o boundary prefetch (the full win).** Extend the prefetch across the smem-stage boundary: issue K_o+1's
first-substep `ldmatrix` (reading the next ring slot, gated on its mbarrier) while the last `mma.sync` of K_o is in flight,
so the tensor cores never idle through the `MbarrierWait`. *Validate:* accuracy max_diff = 0; ncu `barrier` + `wait` stalls
drop; 2048² fp16 `--bench` approaches cuBLAS (≤ ~100µs, ≈ 1.0×).

**M3 — wire into autotune + re-tune goldens.** Confirm `REG_PIPELINE` forks in `deplodock tune`; re-run
`scripts/find_golden_configs.py --shapes square.*.fp16` so the autotuner measures on/off and the goldens record the winner
per shape; update the fp16-square golden records + ratios. Spot-check it didn't regress 512²/1024² (small tiles where the
register cost may not pay).

## Risks

- **Register pressure drops occupancy below the win.** Mitigation: the knob — the autotuner measures on/off per shape and
  only keeps it where it's net-faster. M1 reports the measured reg count (target ~198, under 255; spill kills the win).
- **Restructuring regresses the stable mma.sync mainloop.** Mitigation: M0 keeps the knob off byte-identical; accuracy
  (max_diff = 0) gates every milestone; the feature is purely additive behind the knob.
- **Cross-K_o prefetch races the mbarrier.** Prefetching K_o+1's `ldmatrix` before its `MbarrierWait` reads stale smem.
  Mitigation: M2 gates the prefetch on the mbarrier-ready phase (the same `MbarrierWait` the consumer already issues), and
  the accuracy check catches any ordering bug instantly.
- **Payoff smaller than modeled.** `wait` (mma pipeline, 3.28) dominates — if the tensor cores are already near-saturated,
  removing the boundary bubbles may only reach ~0.96–0.98×, not full parity. If so, the knob still records the gain where it
  helps and costs nothing where it doesn't.

## Critical files

- `deplodock/compiler/pipeline/passes/lowering/kernel/0XX_pipeline_mma_regs.py` — **new** pass: the `REG_PIPELINE` knob + the
  peeling transform (M0–M2).
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — add `REG_PIPELINE` to the warp-fork level
  schema so the autotuner enumerates it (M0).
- `deplodock/compiler/ir/kernel/ir.py` — `RegFragment` (second buffer alloc) / `LdmatrixLoad` / `MmaSyncPtx` if any field
  needs a buffer index for SSA naming (M1).
- `deplodock/compiler/pipeline/passes/lowering/tile/080_pipeline_stages.py` — the prologue/steady/epilogue peeling template
  to mirror (reference only).
- `deplodock/compiler/pipeline/passes/lowering/kernel/095_interleave_loads.py` — the scalar-FMA analog (reference; confirm no
  interaction with the new mma.sync path).
- `scripts/find_golden_configs.py` + `deplodock/compiler/pipeline/search/golden_configs.py` — re-tune + record (M3).

## Verification (end-to-end)

1. Per-milestone gates above (`--ir kernel` shape, `deplodock run` accuracy max_diff = 0, ncu stall breakdown, `--bench`).
2. Correctness: `deplodock run -c "torch.matmul(<fp16>)"` matches eager (max_diff = 0) across square / rectangular / skinny
   and f16 + f32 output, with `REG_PIPELINE=1`.
3. Stalls: ncu `wait` + `barrier` + `short_scoreboard` drop vs the knob-off baseline on 2048² fp16; reg count ≤ ~200/thread
   (no spill).
4. Perf: 2048² fp16 `--bench` with the knob on improves over the 0.93× baseline toward cuBLAS; the autotuned goldens pick the
   winner per shape.
5. No regression: knob off is byte-identical (full suite green, `make lint`); existing `test_matmul_mma*` unchanged.
