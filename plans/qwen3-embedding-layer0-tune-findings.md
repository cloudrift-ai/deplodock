# Tune findings: Qwen/Qwen3-Embedding-0.6B layer 0 (RTX 5090, sm_120)

**Status:** findings 1–2 (codegen failures / hangs) **fixed** — two TMA eligibility gates in `050_use_tma.py`
plus a defensive box check in `backend/cuda/_tma.py`, locked in by `tests/compiler/passes/test_use_tma_gates.py`.
Finding 3 (down_proj tensor-core lockout) **fixed** — the `matmul_add` residual epilogue now folds into the mma
fragment store (`tests/compiler/test_matmul_mma_residual.py`); the real reproducer went 29 → 9.9 µs tuned
(0.28x → 0.83x vs cuBLAS). Findings 4–6 (gated-MLP purity, SDPA prologue matmuls) remain open. Findings from a
clean tune + -O3 kernel bench on 2026-06-10.
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

## Post-fix clean re-tune (2026-06-10, same command)

- **Zero `bench_fail` rows** across 887 benched variants (was 28) — the boxDim and re-entered-pipeline gates hold
  through a full tune; the ~28 wasted bench slots now go to real variants.
- **down_proj at cuBLAS parity**: 8 us vs eager 8 us (**1.04x**, was 0.28x / 29 us) — the mma residual-epilogue
  fold landed it on `mma_m16n8k16_f16`; q/k/v also tune to MMA picks.
- Caveat for cross-run comparisons: the two SDPA-prologue kernels (still scalar-by-design — finding 5) drew worse
  tuned picks this run (P@V 102 us vs 26 us in the baseline tune; rotary+QK^T 50 vs 28), dragging the full-model
  total to 352 us. Their knob space is untouched by this branch (the `not prologue` warp gate precedes every
  change here); the variance is tune-trajectory / prior-pick reachability on the scalar tier — the
  `eval prior --dataset db` reachability view is the diagnostic for it. Treat per-kernel rows, not the full-model
  total, as the before/after signal for findings 1–3.

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
