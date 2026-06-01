# mma.sync smem swizzle — TMA-box reshape + ldmatrix consumer XOR

## Status (S0–S4 landed)

**Done (branch `feature/mma-sync-atom`):** S0 per-`Source.swizzle` field, S1 swizzle-atom box reshape, S2 per-source mode
pick (A=B64 / B=B128), S3 consumer XOR. mma.sync 2048² fp16 now **0.93× (107µs)** — beats WMMA 0.87× (the goal), short of
full cuBLAS parity (predicted 0.92–0.95×, since `short_scoreboard` was only 16% of stalls). Bank conflicts **21M → 2.1M**
(~10×). Accuracy max_diff=0 across square / rect / skinny, f16 + f32 out; full compiler suite + `test_matmul_mma_tma.py`
green. Three bugs cost the iteration (all in the project memory): inner source dim is `max(dims)` not `dims[-1]`;
tile-stage `Source.dtype` is None so the swizzle byte-width must come from `match.graph` not `src.dtype`; all CUTLASS
`Swizzle<B,4,3>` modes share element shift 6 (only the mask differs). A descriptor whose swizzle mode mismatches its box
inner byte width HANGS (producer mbarrier deadlock), not a clean encode error.

**Done — S4 (promote via measured autotune).** `010_partition_loops` auto-enumerates mma_sync as a *tunable candidate* on
sm_90+ for large tiles (every extent ≥ 256). The greedy/DB-less default stays WMMA — the static scorer has no TMA-latency
model and would mis-rank the leaner 16×8 atom on small tiles — so mma_sync wins only through measured autotune (the
goldens), and `test_default_picker_lands_on_tma_golden_at_2048_fp16` (asserts WMMA) stays correct unchanged. Single-warp
(WM·WN==1) mma_sync variants are pruned: `020_stage_inputs` skips staging at one warp and mma_sync has no gmem-direct
fallback, so an unstaged AtomTile would crash render. All four fp16-square goldens regenerated onto mma_sync (512² 6.1µs /
1.02×, 1024² 20.5µs, 2048² 106.7µs / 0.93×, 4096² 701µs / 0.95×) — each faster than its old WMMA golden.

## Context

The `mma_m16n8k16_f16` atom (ldmatrix + `mma.sync.aligned`, the s16816 path cuBLAS/CUTLASS use) lowers correctly and, after
the vectorized `RegStore` epilogue, runs at **WMMA parity** on 2048² fp16 (RTX 5090 sm_120): mma.sync 0.86× vs WMMA 0.87× vs
cuBLAS 0.99×. The remaining gap to cuBLAS is one lever: **smem bank conflicts**. Both paths measure ~21M shared-load bank
conflicts over ~25M wavefronts (83% conflict rate, `ncu l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`), which
shows up as `short_scoreboard` ≈ 16% of warp stalls. cuBLAS swizzles its staged smem to eliminate these; we don't.

**Why only mma.sync can take this lever:** `wmma::load_matrix_sync` assumes a plain row-major fragment and reads the slab
opaquely, so the WMMA path *cannot* consume a swizzled slab and is permanently stuck at ~0.87×. The explicit-`ldmatrix`
mma.sync path computes its own per-lane smem address, so it *can* read a swizzled slab — this is the structural reason the
atom exists. Landing the swizzle is what takes mma.sync from "WMMA parity" to "beats WMMA, ≈ cuBLAS".

**Ruled out (measured, not reasoned):**
- *cp.async + padding* (sidestep TMA): 262µs / 0.39× on **both** sm_80 and sm_120, identical ~21M conflicts. TMA is
  structurally required for the fast path; `Source.pad` is TMA-incompatible (TMA writes a contiguous box, can't leave
  inter-row gaps).
- *Flag-flip swizzle*: forcing `050_use_tma._promote` `swizzle=SwizzleMode.B64` →
  `cuTensorMapEncodeTiled failed: CUresult=1` (INVALID_VALUE). **The driver rejects swizzle on the current box shapes.**

**Root cause of the rejection:** TMA requires the descriptor's innermost box dim, in bytes, to *equal* the swizzle width
(32 / 64 / 128 B). Current boxes (knobs `WM2 WN2 FM4 FN8 BK2 BUFFER_COUNT=3`): A `[128,32]` (inner 32 elem = 64 B), B
`[32,128]` (inner **128 elem = 256 B**, over the 128 B max). The smem slabs are already swizzle-atom-shaped
(`x0_smem[3,2,64,32]`, `x1_smem[3,32,2,64]` — the `[2,64]` split is exactly the 128 B atom × 2), but the **TMA descriptor
box is not** tiled to match. So swizzle needs the box reshaped to a swizzle-atom-tiled rank+1 form, then per-operand modes,
then the matching consumer XOR.

Intended outcome: mma.sync 2048² fp16 from ~115µs (0.86×) toward ~100µs (≈ cuBLAS), conflicts ~21M → near-zero, with
accuracy unchanged (max_diff ≈ 0) and **zero regression to the default WMMA/TMA path**.

## Constraints that shape the design

- **Per-operand modes, not per-bundle.** A's inner is 64 B → `SWIZZLE_64B`; B's is 128 B → `SWIZZLE_128B`. But `swizzle`
  lives on `StageBundle` and A + B share one bundle. So swizzle must move to per-`Source` (or per-`Stage`).
- **Shared staging core.** The box logic in `100_materialize_tile.py` and the promotion in `050_use_tma.py` are also on the
  default WMMA/TMA path. Every change must be **gated** so WMMA stays `SwizzleMode.NONE` and byte-identical.
- **Consumer must match the engine exactly.** The XOR the `ldmatrix` address applies must reproduce the TMA hardware swizzle
  (CUTLASS `Swizzle<B,M,S>`); a wrong formula gives wrong results (caught instantly by the accuracy check, but iterated
  semi-blind on-GPU).
- **Correctness is a permutation invariant.** As long as the descriptor swizzle mode and the consumer XOR agree, the data is
  correct regardless of "optimality" — so the accuracy check is a precise oracle for "formula matches".

## Milestones (each independently validatable; gate on accuracy + ncu, not pytest loops)

**S0 — per-`Source` swizzle field (no behavior change).** Add `swizzle: SwizzleMode = NONE` to `Source` (`ir/tile/ir.py`).
Thread it: `050_use_tma._promote` stamps `NONE` on every source (unchanged behavior); `100_materialize_tile.py` reads
`src.swizzle` instead of `bundle.swizzle` for the `TmaDescriptor`. *Validate:* full compiler suite + WMMA/mma.sync benches
byte-identical (every swizzle still NONE). This isolates the plumbing from the behavior change.

**S1 — swizzle-atom-tiled TMA box (the reshape; still `NONE`, so a pure refactor).** In `100_materialize_tile.py`'s box
computation, when a source's inner-row byte width `W ∈ {32,64,128}`, emit the descriptor box as the rank+1 swizzle-atom form
(innermost = `W` bytes, an outer tiling dim for the remainder) so the box already matches the swizzle-ready smem shape
(`[…,2,64]`). Keep `swizzle=NONE` — the reshaped box must encode and run **identically** to today. *Validate:* `--ir kernel`
shows the rank+1 box; `cuTensorMapEncodeTiled` still succeeds; accuracy max_diff ≈ 0; WMMA + mma.sync latency unchanged. This
de-risks the reshape independent of the swizzle bits.

**S2 — pick per-source swizzle mode (encode accepts it).** In `050_use_tma` (or the materializer), set `src.swizzle` from the
inner byte width (`64→B64`, `128→B128`, else `NONE`), **gated** on the kernel's `ATOM_KIND` being an `mma_sync` instruction
(read the atom spec) — WMMA sources stay `NONE`. *Validate:* `cuTensorMapEncodeTiled` now **succeeds** with B64/B128 (the S1
reshape made the box legal); the kernel runs (result is WRONG until S3 — that's expected); conflicts measured to confirm the
producer now swizzles.

**S3 — consumer XOR in `LdmatrixLoad.render` (restore correctness + kill conflicts).** Thread the source's swizzle mode to
the `LdmatrixLoad` (via `_mma_src_index` / `_emit_chain` in `005_lower_atom_tile.py`). In `LdmatrixLoad.render`, apply the
matching XOR to the per-lane smem element offset before `&buf[...]`. Start from the CUTLASS `Swizzle<B,M,S>` parameters for
`SWIZZLE_128B`/`64B` (128B ≈ `Swizzle<3,4,3>`: `off ^= ((off >> 7) & 7) << 4`, in element units adjusted for fp16); iterate
the constants on-GPU using accuracy as the oracle. *Validate:* accuracy max_diff ≈ 0 **and**
`l1tex__data_bank_conflicts…op_ld.sum` drops ~21M → near-zero; re-bench latency.

**S4 — wire into the default + re-tune.** Once correct + faster than WMMA: have `050_use_tma` enable the swizzle by default
for mma_sync TMA bundles (no longer a probe). Re-run the M5 scorer check + the per-op autotune (`deplodock tune … --bench`)
so the picker lands on the swizzled config, and consider promoting `mma_m16n8k16_f16` ahead of WMMA in `_ATOM_KINDS_V1`
(the opt-in gate in `010_partition_loops.py`) now that it's the faster default. Regenerate the fp16 golden configs
(`scripts/find_golden_configs.py --shapes square.*.fp16`).

## Risks

- **Box-reshape regresses the default WMMA/TMA path.** Mitigation: S1 keeps `NONE` and asserts byte-identical encode + latency
  before any swizzle bit flips; the per-source mode (S2) is gated on the mma_sync atom so WMMA sources never reshape-to-swizzle.
- **Consumer XOR formula wrong → wrong results.** Mitigation: S3 iterates on-GPU with the accuracy check as oracle; start from
  the known CUTLASS swizzle constants; validate on a 16×8×16 single-cell known-answer first, then 128² then 2048².
- **A (64B) and B (128B) interact via the shared bundle / phase prefix.** Mitigation: per-`Source` swizzle (S0) keeps them
  independent; validate each operand's conflict count separately (`ncu` per-load) rather than only the aggregate.
- **Swizzle changes smem footprint / alignment.** The atom-tiled smem is already allocated (`[2,64]`); confirm `alloc_extents`
  + `align=128` are unchanged so occupancy (2 blocks/SM) holds.
- **Payoff smaller than modeled.** `short_scoreboard` is 16% of stalls, `wait` (mma pipeline) is 25%. Removing conflicts may
  only reach ~0.92–0.95×, not full parity. If so, stack with M7 (occupancy/depth re-tune) — mma.sync's leaner regs (vs WMMA's
  166) may finally make depth-4 @ 1 block win where it lost on WMMA (`040_use_ring_buffers.py:110`).

## Critical files

- `deplodock/compiler/ir/tile/ir.py` — `Source` (+ `swizzle` field), `SwizzleMode`, `StageBundle` (S0)
- `deplodock/compiler/pipeline/passes/lowering/kernel/100_materialize_tile.py` — TMA box computation + `TmaDescriptor`
  population; the reshape (S1) and per-source swizzle read (S0/S2)
- `deplodock/compiler/pipeline/passes/lowering/tile/050_use_tma.py` — `_promote`; per-source mode pick gated on mma_sync (S2/S4)
- `deplodock/compiler/pipeline/passes/lowering/kernel/005_lower_atom_tile.py` — `_mma_src_index` / `_emit_chain`: thread
  swizzle mode to `LdmatrixLoad` (S3)
- `deplodock/compiler/ir/kernel/ir.py` — `LdmatrixLoad.render`: the consumer XOR (S3)
- `deplodock/compiler/backend/cuda/_tma.py` — `encode_tiled` already takes `swizzle`; confirm box/mode validation (S2)
- `deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py` — opt-in gate; promote atom ahead of WMMA (S4)
- `scripts/find_golden_configs.py` — regenerate fp16 goldens (S4)

## Verification (end-to-end)

1. Per-milestone gates above (`--ir kernel`/`--ir cuda`, `cuTensorMapEncodeTiled` success, `deplodock run` accuracy, `ncu`
   bank-conflict counts, `--bench` latency).
2. Correctness: `deplodock run -c "torch.matmul(<fp16>)"` matches eager (max_diff ≈ 0) across square / rectangular / skinny-N
   and f16 + f32 output, with the swizzle on; 16×8×16 single-cell known-answer passes.
3. Conflicts: `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` drops from ~21M to near-zero on 2048²
   fp16.
4. Perf: 2048² fp16 `--bench` beats the WMMA default (≤ ~0.87×) and approaches cuBLAS; regenerated `square.*.fp16` goldens
   improve and `test_golden_configs.py` invariants still pass.
5. No regression: WMMA path byte-identical through S1, default suite green (`make test`, `make lint`); existing
   `test_matmul_mma_tma.py` (WMMA + mma.sync) unchanged.
