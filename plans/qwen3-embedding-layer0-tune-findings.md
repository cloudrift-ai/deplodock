# Tune findings: Qwen/Qwen3-Embedding-0.6B layer 0 (RTX 5090, sm_120)

**Status:** findings 1–2 (codegen failures / hangs) **fixed** — two TMA eligibility gates in `050_use_tma.py`
plus a defensive box check in `backend/cuda/_tma.py`, locked in by `tests/compiler/passes/test_use_tma_gates.py`.
Findings 3–6 (perf: scalar-tier matmuls) remain open. Findings from a clean tune + -O3 kernel bench on 2026-06-10.
Run: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --clean --bench --dump-dir <dump>` (859 s, 930 benched
variants, 810 ok / 28 `bench_fail` in the tune DB). Numbers below are the `--bench` -O3 re-bench (CUDA-graph
captured); tune-DB latencies quoted for ranking context are -O1.

## Bench results

Full model (one layer, seq_len 32):

| Backend       | Latency (us) | vs Eager |
|---------------|--------------|----------|
| Eager PyTorch | 96           | 1.00x    |
| torch.compile | 45           | 2.14x    |
| Deplodock     | 193          | 0.50x    |

Per-kernel (-O3, sorted by Deplodock latency; mapping to layer ops added from the dump provenance):

| Kernel                                      | Layer op                          | eager | tcompile | deplodock | vs eager |
|---------------------------------------------|-----------------------------------|-------|----------|-----------|----------|
| k_linear_mean_reduce                        | post-attn RMSNorm + gate/up +silu | 70    | 14       | 76        | 0.91x    |
| k_linear_reduce (86a525)                    | down_proj + residual add          | 8     | 8        | 29        | 0.28x    |
| k_sdpa_transpose_unsqueeze_cat_slice_reduce | rotary + QK^T + softmax           | 84    | 8        | 28        | 3.04x    |
| k_linear_reshape_transpose_sdpa_reduce      | P@V (SDPA second matmul)          | 16    | 14       | 26        | 0.64x    |
| k_sdpa_transpose_reshape_linear_reduce      | o_proj                            | 12    | 12       | 10        | 1.21x    |
| k_reshape_linear_mean_reduce (359b55)       | q per-head RMSNorm                | 59    | 8        | 10        | 6.18x    |
| k_linear_reduce (735349)                    | q_proj                            | 6     | 6        | 9         | 0.68x    |
| k_reshape_linear_mean_reduce (61f49f)       | k per-head RMSNorm                | 61    | 8        | 5         | 12.11x   |
| k_linear_reduce (bcc194)                    | v_proj                            | 6     | 6        | 4         | 1.49x    |
| k_mean                                      | input RMSNorm mean                | 55    | 4        | 2         | 36.67x   |

The 193 us total is dominated by four kernels: the fused norm+MLP prologue (76), down_proj (29), rotary+QK^T (28)
and P@V (26) — 159 us of 193. All four are scalar-tier (no tensor cores); see findings 3–5.

## Finding 1 — TMA descriptors with box dim > 256 pass eligibility, then fail at launch (16 bench_fails) — FIXED

**Symptom:** 16× `bench worker error: RuntimeError('cuTensorMapEncodeTiled failed: CUresult=1')` during the tune,
all on the down_proj op (`k_linear_reduce_86a525`, out (32, 1024), K=3072).

**Root cause:** the hardware limit for `cuTensorMapEncodeTiled` is **boxDim[i] <= 256** per dimension.
`_source_eligible` in `deplodock/compiler/pipeline/passes/lowering/tile/050_use_tma.py:294` checks rank, dim
mapping, and 128 B alignment but never bounds the collapsed per-dim box extent. The collapsed M box is `BM*FM`
(consecutive duplicate dims multiply), and the DB split is exact: every TMA-ok variant for this op has
`BM*FM <= 256`; all 16 failures have `BM*FM` in {320 … 768}. `CUresult=1` is `CUDA_ERROR_INVALID_VALUE` from the
driver rejecting the box.

**Why it matters beyond wasted tune budget:** the failure is at *launch* time — the descriptor embeds the device
pointer (`backend/cuda/_tma.py:91`), so nvcc compile succeeds and the bad variant is only caught when it runs. A
greedy `compile`/`run` whose prior picks such a config would crash in deployment, not fall back. (This run's
deployed picks were all `BM*FM <= 256`, so the per-kernel table was unaffected.)

**Fix (landed):** `_source_eligible` now computes the materializer-mirroring `box_per_dim` collapse and declines
any source with a collapsed extent > 256 (`_TMA_MAX_BOX_DIM`), so oversized tiles fall back to cp.async instead
of compiling a kernel that cannot launch. `_tma.py::encode_tiled` validates `box_extents` defensively and names
the offending dim. Tests: `test_use_tma_gates.py` (decline + pinned-raise + at-limit control + GPU run of the
previously-crashing knob family).

## Finding 2 — TMA ring pipeline deadlocks when emitted inside a serial register-tile loop (6 hung kernels) — FIXED

**Symptom:** 6× `HungKernelError("kernel 'k_linear_mean_reduce_23ab9c' did not complete within 1000 ms")`, each
followed by a collateral `bench worker did not accept the request within 6.0s wall budget` for the next variant
(the worker was still being SIGKILL'd — 12 of the 28 bench_fail rows, 6 genuine + 6 collateral).

**Signature:** every genuine hang has `TMA=true, PIPELINE_STAGES=true, FM=2, RING=3, STAGE="110"`; every OK TMA
variant has `FM=1`. Reproduced the codegen (no GPU run) by pinning the failing knobs via `DEPLODOCK_KNOBS` and
compiling the dumped `.torch.json` reproducer with `--ir cuda`.

**Root cause:** in this kernel the per-row norm reduction forces the whole K pipeline *inside* the FM cell loop
(`for a4 in 0..FM`), but the mbarriers are initialized **once at kernel entry**. The pipeline's wait schedule
(`mbarrier_wait_parity(mbar[a5 % 3], a5 / 3 % 2)` plus epilogue waits hardcoded to parity 0) assumes fresh
barriers each time. With K=1024, BK=32 and RING=3, iteration `a4=0` completes 11/11/10 phase rounds on the three
slots, so `a4=1` starts with slots at mixed parities; the parity schedule desyncs and a wait eventually blocks on
a phase that never completes — deadlock. `FM=1` never re-enters the pipeline, hence no hang. Plain matmuls are
immune (their FM cells live inside the K loop); only fused kernels whose prologue reduction nests the pipeline
under a serial outer loop (here: RMSNorm + gate/up matmul) can hit it.

**Fix (landed):** option 1 — `050_use_tma.py::_reenters_pipeline` declines TMA when a promotable bundle's
`serial_outer` is nested inside a serial loop (`SerialTile`/`StridedTile`) with static trip count > 1 (symbolic
treated as > 1), same shape-rejection pattern as the warpspec gate (#217). cp.async (commit/wait_prior, no
cross-iteration phase state) serves the FM>1 design point. Tests: `test_use_tma_gates.py` (decline +
pinned-raise + FM=1 control + GPU run-to-completion of the previously-hanging knob family). The rejected
alternative (carry phase parity across outer iterations) stays open if FM>1 TMA ever looks like a win here.

## Finding 3 — down_proj is locked out of the tensor-core tier by its fused residual add (0.28x vs eager)

`k_linear_reduce_86a525` is `add_7 = add_5 + linear_6(mul_12, W_down)` — fusion pulled the residual add into the
matmul kernel as a post-reduce epilogue (`v1 = add(acc0, in0)`). The MMA eligibility predicate
(`lowering/tile/_atom.py:127`) rejects any Assign consuming the accumulator after the K loop ("no fused
post-reduce epilogue"), because `kernel/005_lower_atom_tile` stores the mma fragment directly. Result: of 74 OK
variants benched for this op, **zero** are warp-tier; the deployed scalar kernel runs 29 us vs 8 us for
cuBLAS-backed eager/tcompile. By contrast q/k/v_proj (plain linears, K=1024) auto-enumerated
`mma_m16n8k16_f16` + TMA + warpspec and land at 1.5x / 0.68x of eager.

The outer fusion search is deterministic today (one terminal), so the tuner cannot un-fuse the add to reach the
warp tier — the fusion pass commits to a kernel the best backend path can't serve.

**Fix candidates:** (a) teach the mma store path to thread the accumulator fragment through a *linear* epilogue
(the SPLITK residual-hoist in `010_partition_loops.py` already classifies `matmul_add` as linear — reuse that
classification); or (b) make the fusion pass aware of the warp-tier eligibility loss (a real fusion fork for the
outer search once it stops being deterministic).

## Finding 4 — fused RMSNorm + gate/up + silu kernel is scalar and re-reads everything (0.91x eager, 5.4x behind torch.compile)

`k_linear_mean_reduce_23ab9c` fuses post-attn RMSNorm + gate_proj + up_proj + silu·mul into one kernel
(out (32, 3072), two chained reductions over 1024). Three compounding problems:

- **No tensor cores:** the gated-MLP reduce body has 4 Loads / 2 Accums (gate and up share the K loop), failing
  the MMA gate's body-purity rule (`_atom.py:106`, canonical 2-Load/1-Assign/1-Accum cell), independent of the
  epilogue rule. Scalar FMA against tcompile's tensor-core gemms is most of the 76 us vs 14 us gap.
- **Redundant traffic:** the winner re-reads the activation row from gmem three times (norm pass + main loop +
  epilogue tiles) and, when FM>1, re-stages the full weight K-strip per cell iteration (the TMA coords in the
  generated CUDA don't depend on `a4`).
- **Search loss:** the FM=2 region of the TMA space hangs (finding 2), so part of the design space was benched
  as `bench_fail @ 2e6 us`.

Worth a fusion-policy look: at this shape, norm+matmul fusion buys one gmem round-trip of (32, 1024) but costs
tensor-core eligibility on a (32, 3072, 1024)-pair of gemms. tcompile's choice (norm separate, gemms on tensor
cores) is 5.4x faster.

## Finding 5 — both SDPA matmul kernels are warp-tier-ineligible by design (prologue rule)

- `k_linear_reshape_transpose_sdpa_reduce_38c877` (P@V fused with softmax stats): 26 us vs 16 eager (0.64x).
- `k_sdpa_transpose_unsqueeze_cat_slice_reduce_b2ab33` (rotary + QK^T + softmax): 28 us — 3.04x vs eager's
  unfused 84 us, but 3.5x behind tcompile's 8 us.

The warp tier requires `not prologue` (`010_partition_loops.py:662`, "M9 extension" — the per-row softmax stats
must reset per register cell). Known/accepted limitation, recorded here because the two kernels are 54 us of the
193 us total; M9 (warp-tier prologue matmuls) is the single biggest remaining lever on this layer after
findings 3–4.

## Finding 6 — minor: q_proj scalar pick beats its own MMA rows at -O1 but loses at -O3 (0.68x)

q_proj (`k_linear_reduce_735349`, out (32, 2048), K=1024) had 27 MMA rows; the -O1 ranking best was an MMA+TMA+WS
variant (10.1 us), yet the deployed -O3 pick lands at 9 us vs eager 6. Not a codegen failure — likely the usual
-O1-vs--O3 rank inversion plus small-M (32 rows) underutilization of the m16 atom. Re-check after finding 3's
fix changes the prior's training mix. Low priority.

## Repro / artifacts

- Tune log: 16× `cuTensorMapEncodeTiled` + 6× `HungKernelError` lines (also recoverable from the tune DB:
  `SELECT knobs FROM perf WHERE status='bench_fail'` — all 28 rows have `TMA: true`).
- Hang codegen repro (no GPU needed, compile-only), from the dump dir of any layer-0 tune:

  ```bash
  DEPLODOCK_KNOBS="BM=8,BN=64,BK=32,FM=2,FN=2,FK=1,RING=3,STAGE=110,SPLITK=1,TMA=1,WARPSPEC=0,\
  PIPELINE_STAGES=1,ASYNC_COPY=0,VECTORIZE_LOADS=1,INTERLEAVE_LOADS=1,GROUP_M=8,PAD_SMEM=0,\
  PERMUTE_LANES=0,HOIST_COMPUTE=0,MMA=0,BR=1" \
    deplodock compile <dump>/07_lowering_cuda.kernels/k_linear_mean_reduce_23ab9c.torch.json --ir cuda
  ```

  In the emitted source: `mbarrier_init` runs once before `for (a4 = 0; a4 < 2; ...)` while every
  `mbarrier_wait_parity` inside assumes fresh phase 0. Flip to `FM=1` and the `a4` loop degenerates to one trip.
- Box-dim repro: same pinning trick with `FM=64` on the down_proj reproducer compiles fine and fails only at
  launch with `CUresult=1`.
