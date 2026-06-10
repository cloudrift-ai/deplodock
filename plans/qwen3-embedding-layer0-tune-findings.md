# Tune findings: Qwen/Qwen3-Embedding-0.6B layer 0 (RTX 5090, sm_120)

**Status:** findings 1–2 (codegen failures / hangs) **fixed** — two TMA eligibility gates in `050_use_tma.py`
plus a defensive box check in `backend/cuda/_tma.py`, locked in by `tests/compiler/passes/test_use_tma_gates.py`.
Finding 3 (down_proj tensor-core lockout) **fixed** — pointwise epilogues (residual adds, bias/scale broadcasts,
activation chains) now fold into the mma fragment store, gated by the negative-form
`tile/_atom.classify_fragment_epilogue` (the slice folds unless it has an ineligible op/dependency) and locked in
by `tests/compiler/test_matmul_mma_residual.py`; the real down_proj reproducer went 29 → 8 µs tuned (0.28x →
1.04x, cuBLAS parity). Finding 5 (SDPA prologue matmuls) **machinery landed** — the `SPLIT_CONE` structural fork
(one-sided from PR #219, two-sided for rotary QK^T) makes the warp tier reachable for both SDPA kernels; see the
finding's update for the deployable A/B and the remaining follow-ups. Findings 4 and 6 remain open — note
finding 4's combine (`silu(acc_g)·acc_u`) is now blocked only by the multi-accumulator rule + reduce-body
purity, not the epilogue rule. Findings from a clean tune + -O3 kernel bench on 2026-06-10.
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

## Finding 5 — both SDPA matmul kernels are warp-tier-ineligible by design (prologue rule)

- `k_linear_reshape_transpose_sdpa_reduce_38c877` (P@V fused with softmax stats): 26 us vs 16 eager (0.64x).
- `k_sdpa_transpose_unsqueeze_cat_slice_reduce_b2ab33` (rotary + QK^T + softmax): 28 us — 3.04x vs eager's
  unfused 84 us, but 3.5x behind tcompile's 8 us.

The warp tier requires `not prologue` (`010_partition_loops.py:662`, "M9 extension" — the per-row softmax stats
must reset per register cell). Known/accepted limitation, recorded here because the two kernels are 54 us of the
193 us total; M9 (warp-tier prologue matmuls) is the single biggest remaining lever on this layer after
findings 3–4.

**Update (2026-06-10, `feature/sdpa-prologue-warp-tier`): machinery landed.** Instead of M9 proper (online
softmax in the mma register tower), both kernels reach the warp tier through the `SPLIT_CONE` structural fork
(`005_split_demoted`), plus one bug fix:

- **Eligibility/tagger mismatch fixed** (`tile/_atom.py`): a transposed-B matmul — Q@K^T, both operands with K
  in the LAST index dim — passed `is_atom_eligible` but `011_lower_atom_cell._classify_ab` can never tag it (no
  staging order canonicalizes it), so the unconsumed AtomTile crashed at render. Repro: `run` on the P@V
  reproducer below crashed before this branch. The gate now mirrors the tagger (exactly one K-in-last load);
  such shapes fall to the scalar tier. `tests/compiler/passes/test_partition_planner_mma.py`
  (`test_mma_eligibility_rejects_transposed_b`).
- **P@V** (`0a1109`): the one-sided cut (PR #219) already fired on its softmax prologue; with the crash fixed
  the split's consumer deploys on `mma_m16n8k16_f16`. Full-layer pinned A/B (fresh clean tune, -O3,
  CUDA-graph captured): 33.1 us scalar fused → 14.4 us mma gemm + xn softmax producer.
- **rotary + QK^T** (`b2ab33` / `39b7dc`): new **two-sided** cut in `_split_demoted` — BOTH multiply operands
  are computed cones (rotary on Q and K), so the cut builds two producers; the N-indexed K-side cone
  materializes at `[K, N]` (K out of the last dim), giving the consumer the canonical B layout despite the
  original transposed `[n, k]` access (GQA `head/2` kept as a duplicated leading dim). Reproducer slice:
  38.8 → 7.2 us pinned (eager 82 us). Full layer: 65.8 us fused → 36.4 us split (0.9 xna + 2.3 xnb + 33.2
  consumer). `tests/compiler/passes/test_split_demoted.py` (two-sided structure / numpy / CUDA-mma tests).
- Full-layer pinned A/B total: **179 us greedy-fused → 138 us `DEPLODOCK_SPLIT_CONE=1`** (same trained prior).
  The tuner explores both branches unpinned (339 split-variant perf rows in the clean tune's DB).

**Remaining follow-ups** (why this is "machinery landed", not closed):

1. The QK^T consumer still lowers scalar: its causal-mask epilogue (`v = 0 when (n <= m) / -1e9` Select) is a
   non-Load leaf, which `classify_fragment_epilogue` rightly blocks. Folding axis-only Selects into the
   fragment store (the per-element (row, col) is known at `RegStore` time) is the missing piece for mma here.
2. Greedy `run`/`compile` never deploy a structural option (`policy/greedy._is_structural` — the per-op prior
   can't price a multi-kernel Graph), so unpinned runs keep the fused kernels; landing the split as a greedy
   pick is `plans/structural-forks-in-two-level.md`.
3. Prior pick-reachability on the new `_xn` producer shapes is weak after one tune (the P@V xn softmax drew a
   66.7 us greedy pick; the tune DB holds ~1 us variants of the same op) — the `eval prior --dataset db`
   reachability view is the diagnostic.

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
