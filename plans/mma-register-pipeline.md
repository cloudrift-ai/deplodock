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
