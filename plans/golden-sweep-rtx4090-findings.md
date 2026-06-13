# Golden seed findings — RTX 4090 (sm_89), 2026-06-13 (first-ever 4090 golden file)

- New file: `deplodock/compiler/pipeline/search/goldens/rtx4090_sm89.yaml` (29 matmul shapes), seeded on
  `riftuser@176.124.69.200` (RTX 4090, sm_89, driver 580.65.06, CUDA 12.9, torch 2.12.0+cu130).
- Sweep: `deplodock tune --dataset golden --clean -q` (~30 min) → seed-mode harvest (`run --bench -c snippet`, greedy
  pick) → **diagnosis** → transfer-mode harvest (the recorded 5090 configs benched on the 4090) → per-shape merge of
  the better of {greedy, transferred}. Prior + DB pulled to `/tmp/golden_artifacts/4090/` for offline analysis.
- Branch: `feature/golden-multi-gpu`.
- Headline result: a fresh single-pass tune on sm_89 produces **15/29 degenerate picks** (>1.5× eager, up to 5×). The
  recorded file is **not** that seed — it is the per-shape best of the 4090's own greedy pick and the **transferred
  5090 config benched on the 4090**, which recovers 16 shapes (e.g. o_proj.s128 84→23 µs, down_proj.s128 125→33 µs,
  square.4096 11092→4005 µs). The fp16 squares remain 3–5× and are recorded as scalar fallback (see Finding 1).

## Source of each recorded config (16 transferred 5090 cfg / 13 the 4090's own greedy)

| shape                            | src      | deplo µs | eager µs | d/e  | greedy µs | note                                   |
|----------------------------------|----------|---------:|---------:|-----:|----------:|----------------------------------------|
| square.512                       | greedy   |     13.5 |     10.8 | 1.25 |      13.5 |                                        |
| square.1024                      | greedy   |     71.0 |     45.4 | 1.56 |      71.0 | transfer cfg failed to lower (F3)      |
| square.2048                      | greedy   |    467.5 |    320.0 | 1.46 |     467.5 | greedy beat transfer (652)             |
| square.4096                      | transfer |   4004.9 |   2458.6 | 1.63 |   11092.0 | greedy was STAGE:'10' degenerate (F2)  |
| square.512.fp16                  | greedy   |     17.7 |      5.8 | 3.02 |      17.7 | scalar fallback — no fp16 TC (F1)      |
| square.1024.fp16                 | greedy   |     90.7 |     18.1 | 5.01 |      90.7 | scalar fallback (F1)                   |
| square.2048.fp16                 | greedy   |    512.5 |    115.2 | 4.45 |     512.5 | scalar fallback (F1)                   |
| square.4096.fp16                 | greedy   |   3778.6 |    822.3 | 4.60 |    3778.6 | scalar fallback (F1)                   |
| qwen3_06b.q_proj.s32             | greedy   |      7.9 |      9.9 | 0.80 |       7.9 |                                        |
| qwen3_06b.kv_proj.s32            | greedy   |      6.5 |      6.9 | 0.94 |       6.5 |                                        |
| qwen3_06b.o_proj.s32             | transfer |     11.2 |      9.9 | 1.13 |      12.8 |                                        |
| qwen3_06b.gate_up_proj.s32       | transfer |     12.1 |     11.3 | 1.07 |      34.3 | greedy 2.98× → transfer 1.07×          |
| qwen3_06b.down_proj.s32          | transfer |     15.6 |     13.0 | 1.20 |      33.5 | greedy 2.55×                           |
| qwen3_06b.q_proj.s128            | transfer |     23.6 |     20.2 | 1.17 |      28.7 |                                        |
| qwen3_06b.kv_proj.s128           | transfer |     12.9 |     12.4 | 1.04 |      16.3 |                                        |
| qwen3_06b.o_proj.s128            | transfer |     23.0 |     19.0 | 1.21 |      84.3 | greedy 4.35× → transfer 1.21×          |
| qwen3_06b.gate_up_proj.s128      | transfer |     45.7 |     25.5 | 1.79 |      53.8 | both poor (sm_89, see F4)              |
| qwen3_06b.down_proj.s128         | transfer |     33.1 |     24.5 | 1.35 |     124.5 | greedy 4.89× → transfer 1.35×          |
| qwen3_06b.q_proj.s512            | greedy   |     90.4 |     53.3 | 1.70 |      90.4 | transfer cfg failed to lower (F3)      |
| qwen3_06b.kv_proj.s512           | greedy   |     44.9 |     38.9 | 1.16 |      44.9 |                                        |
| qwen3_06b.o_proj.s512            | transfer |     85.9 |     67.8 | 1.27 |     321.9 | greedy 4.60× → transfer 1.27×          |
| qwen3_06b.gate_up_proj.s512      | transfer |    123.6 |     85.5 | 1.45 |      FAIL | greedy LoweringError (F3)              |
| qwen3_06b.down_proj.s512         | transfer |    143.1 |    113.2 | 1.26 |     447.5 | greedy 3.77×                           |
| square.512.dynM                  | greedy   |     12.8 |     10.8 | 1.19 |      12.8 |                                        |
| qwen3_06b.q_proj.s512.dynM       | transfer |     63.8 |     53.0 | 1.20 |      70.3 |                                        |
| qwen3_06b.kv_proj.s512.dynM      | transfer |     33.4 |     37.0 | 0.90 |      40.5 |                                        |
| qwen3_06b.o_proj.s512.dynM       | greedy   |     62.0 |     67.1 | 0.92 |      62.0 |                                        |
| qwen3_06b.gate_up_proj.s512.dynM | transfer |     91.1 |     85.6 | 1.06 |     111.3 |                                        |
| qwen3_06b.down_proj.s512.dynM    | transfer |     92.3 |    119.4 | 0.77 |      95.9 |                                        |

## Finding 1 — fp16 tensor cores on sm_89 are blocked by an NVRTC sm_89 miscompile of the plain-ldmatrix smem index (P0)

Every fp16 square deploys a **scalar thread-tier** kernel (knobs `BM/BN/FM/FN…`, no `WM/WN/MMA`) and runs 3–5× slower
than cuBLAS HGEMM. The warp/MMA tier is gated to sm_90+ at `passes/lowering/tile/010_partition_loops.py:702`
(`ctx.compute_capability < (9, 0)` ⇒ `eligible = ()`, "pin-only" on sm_80-89). But the gate is not the real blocker —
**pinning the atom (`DEPLODOCK_MMA=mma_m16n8k16_f16`) produces a kernel that faults on the 4090.**

Root-caused with `compute-sanitizer`:

```
Invalid __shared__ read of size 16 bytes at k_matmul_207791+0x16d0
  by thread (96,0,0)  — Access at smem offset 0x2060 (8288 B) is out of bounds
```

`x0_smem` is 8192 B, so byte 8288 is element **~4144** of a 4096-element array. The *source* address for that thread
(warp 3, lane 0) computes **~2576** — well in bounds. The OOB read faults on Ada, wedging the context → the launch
never completes (looks like a hang; confirmed not a deadlock — `synccheck` is clean — and not transport-specific —
it faults identically with cp.async and plain sync staging, `ASYNC_COPY=0`).

It is specifically an **NVRTC sm_89 codegen miscompile of the plain (non-swizzled) ldmatrix index expression**, not a
logic bug:
- The *same* sm_89-lowered kernel (same content hash, `k_matmul_207791`) runs **clean under `compute-sanitizer
  memcheck` on the sm_120 dev box** (`run --target sm_89` — `--target` changes lowering, but NVRTC still compiles for
  the live sm_120 arch, which compiles the index correctly). Only the **native sm_89 cubin** reads OOB.
- The sm_120 fp16 path is an entirely **different kernel** — TMA + mbarrier + warp-specialization + a *swizzled* smem
  layout (`addr ^ (((addr>>6)&3)<<3)`) — so the 5090 / PRO 6000 fp16 goldens never touch the plain-ldmatrix path and
  are correct.

**Tried and ruled out:** hoisting the ldmatrix smem index into an explicit `const int` local before the `&smem[...]`
(forcing single-integer materialization — a common NVRTC-fold workaround) in `LdmatrixLoad.render`. Validated the
sm_120 path stayed correct, deployed to the 4090 with the cubin/kernel caches cleared — **still OOB**. So the
miscompile survives a simple reassociation; reverted (no-op that touched the hot sm_120 codegen for no gain).

**Recommendation (real fix, needs the 4090):** `cuobjdump -sass` / `nvdisasm` the faulting `k_matmul_207791` cubin on
the 4090, map `+0x16d0` to the address computation, and find what NVRTC sm_89 folds wrong (candidate triggers: the
`%`/`/`/`*` mix in the per-lane index, or a CUDA 12.9 nvcc / cupy-cuda12x vs torch-cu130 toolchain-version skew — the
scalar kernels compile fine, so it's specific to the warp-tier index). The workaround likely lands once the exact
trigger is known (e.g. split the index differently, or pin an NVRTC arch/flag). Once the OOB is gone, drop the sm_90+
gate at `010_partition_loops.py:702` for sm_80-89 (the cp.async/sync mma.sync staging the path already emits is valid
Ampere/Ada PTX) and re-tune the fp16 goldens — cuBLAS HGEMM is 4–5× the current scalar fallback, so this is the single
biggest sm_80-89 win available. Until then the fp16 squares are recorded as scalar fallback (ratio ≪ 0.95,
`golden=False`) — the honest deployable number, flagged as a known limitation. The gate stays in place so the
autotuner never deploys the faulting kernel.

## Finding 2 — fresh sm_89 tune produces degenerate picks; the prior mis-searches, and higher patience does not help (P0)

The clean seed deployed `square.4096` at **11.1 ms** (4.45× eager) with `STAGE:'10'` (a non-pipelined single-stage
config) and the s128/s512 projections at 2–5× with `FM:1 FN:1` 1×1-cell thread tiles — pathological arithmetic
intensity. This is the cold **5090-fit `AnalyticPrior`** (`_W_A` weights, code-resident, fit on sm_120 golden data)
mis-pricing sm_89 during early exploration, then the learned half reinforcing the bad region. Crucially, **patience
does not fix it**: re-tuning `down_proj.s128` at `--patience 120` still landed `FM:1 FN:1` at 125.6 µs (4.93×, vs the
124.5 µs seed) — the search keeps converging to the same degenerate region, so it is a *prior/ranking* failure on
sm_89, not a too-short search.

The fix is **transfer**: the recorded 5090 scalar configs are hardware-portable and run 3–5× faster on the 4090 than
its own greedy picks (down_proj.s128 125→33 µs, o_proj.s128 84→23 µs, square.4096 11092→4005 µs). The recorded file
uses them wherever they win.

**Recommendation:** (a) the cold `AnalyticPrior` needs sm_80-89 representation — refit `_W_A`
(`scripts/golden_knob_heuristics.py`) jointly over a 4090 golden set (this file is the first one), or add a
capability-tier feature so the analytic geometry terms don't extrapolate sm_120 occupancy/CTA assumptions onto Ada.
(b) The same golden-seeding recommendation as the 5090 Finding 2 applies doubly here: seeding the inner search with
the (transferred) golden knobs would let `evidence_pick` deploy them directly instead of the degenerate greedy.

## Finding 3 — three shapes hit a hard `LoweringError` with no fallback on sm_89

`gate_up_proj.s512` failed to compile at all under its greedy pick: `LoweringError: 1 node(s) left un-lowered — the
chosen tile shape produced a kernel that failed validate(ctx) and the deterministic compile had no fallback`
(`k:100_materialize_tile` rejected its only lowering). The transferred 5090 config for `square.1024` and
`q_proj.s512` likewise failed to lower on the 4090 (so those two fell back to the 4090 greedy). So on sm_89 the
deterministic compile can paint itself into a corner: a tile shape passes enumeration but fails `validate(ctx)` at
render with no alternative, crashing the whole compile.

**Recommendation:** the deterministic-compile path should fall back to the next-ranked tile when the chosen one fails
`validate(ctx)` rather than raising — a single un-lowerable pick shouldn't make a shape uncompilable. Cite
`passes/lowering/tile/010_partition_loops.py` (the eligibility/validate gate) and the `LoweringError` raise in
`pipeline/pipeline.py`. This is the same class the memory note "clean tune is the real test gate" warns about
(search surfaces unlowerable variant classes that unit tests miss).

## Finding 4 — even transferred, gate_up_proj.s128 and the big fp32 squares stay 1.45–1.79× on sm_89

`gate_up_proj.s128` (1.79×), `square.4096` (1.63×), `square.1024` (1.56×), `square.2048` (1.46×) are the best
available on the 4090 but still well off cuBLAS. The 4090 has ~⅓ the fp32 throughput and far less L2 than the 5090, so
the 5090's large-`FM` tiles (e.g. square.4096 `FM:10`) don't map as well; these want their own sm_89 tuning once
Finding 2's prior issue is addressed. Recorded as-is (honest deployable numbers); not golden-quality.

## Workflow notes

- **A fresh GPU cannot be seeded by `tune` alone today** — the cold prior mis-searches off-sm_120 hardware (Finding 2).
  The transfer approach (bench the existing 5090 goldens on the new card, keep the winners) was essential and should be
  a first-class CLI mode: `tune --dataset golden --transfer-from rtx5090_sm120` or a `--seed-from-golden` flag that
  benches recorded configs on the live GPU and records the per-shape best. It is exactly what produced a usable file
  here, and it generalizes the golden-seeding recommendation.
- **The 4090's intermittent sshd** (rate-limited `255` on rapid reconnects) made remote orchestration flaky; pulling
  the 949 MB DB and re-launching detached jobs needed retry loops. Not a deplodock issue, noted for whoever scripts
  multi-host sweeps — reuse one `ControlMaster` connection.
- **Multi-GPU goldens conflate in `eval`** — the 4090 (cap `8,9`) is at least distinct from the 5090/PRO 6000 (cap
  `12,0`), but `eval`/`tune` still iterate all cards' configs with no live-GPU filter and the names collide
  (`square.512` exists thrice). Filtering the golden consumers to the live `(gpu_name, compute_cap)` is the carried
  recommendation (see the 5090 report and `search/golden.py` docstrings).
- **`tune --dataset golden --clean` retrains the global prior**, so the 4090's `prior.json` / `autotune.db` pulled for
  analysis are the fresh sm_89-tuned artifacts (degenerate-region-trained — see Finding 2), not a curated prior.
