# Canonical benchmark runs

Three GPU-prefixed run directories. Each contains the JSON traces, markdown
reports, and (where available) `ncu` diagnostic dumps that back the numbers
in the [companion blog post](https://github.com/cloudrift-ai/deplodock/tree/main/scripts/diagnostics).

| Directory                      | GPU                                                   | Run date   | Notes                                                                                                                                                                                      |
|--------------------------------|-------------------------------------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `rtx5090_2026-04-07_16-42-49/` | NVIDIA GeForce RTX 5090 (sm_120, 170 SMs)             | 2026-04-07 | Local dev box, driver 595.58.03, CUDA 13.2.51, cuBLAS 13.3.0. Single-mode + batched-mode reports. Numbers in the article's headline 5090 tables match this run exactly.                    |
| `pro6000_2026-04-07_21-29-59/` | NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120, 188 SMs) | 2026-04-07 | CloudRift VM, same driver/CUDA/cuBLAS as the 5090 run. Single-mode + batched-mode reports + diagnostics (cuBLAS kernel dump, SASS histogram). Numbers match the article's Pro 6000 tables. |
| `h200_2026-04-07_19-39-24/`    | NVIDIA H200 (sm_90, 132 SMs)                          | 2026-04-07 | Remote SSH host, same driver/CUDA/cuBLAS. Single-mode + batched-mode reports + diagnostics. Numbers match the article's H200 tables.                                                       |

## What's inside each run directory

- `*_b1_*_report_*.md` — Markdown report for the single-batch sweep (sizes 1024–16384)
- `*_b4-8-16_*_report_*.md` — Markdown report for the batched sweep (B=4/8/16, sizes 256–8192)
- `*_*_adaptive.json` — Raw JSON benchmark traces (one per `(batch, size-set)` combination)
- `*_cublas_kernels.txt` — `cuobjdump` of the cuBLAS kernel name catalogue (Pro 6000 + H200 only)
- `*_cublas_loop_vs_strided_*.txt` — `cublasSgemm` loop vs `cublasSgemmStridedBatched` comparison
- `*_sass_analysis_*.md` — SASS opcode histogram of the generated TMA kernel at 8192
- `recipe.yaml` — Snapshot of the recipe used for this run
- `tasks.json` — Bench harness task graph

## Per-arch ncu dispatcher sweep

The full per-arch / per-size cuBLAS dispatcher sweep that supports the
article's central claim — *"the same `libcublas.so` binary picks dramatically
different kernels for the same workload on different GPUs"* — lives in
[`ncu/batched_dispatch_finding.md`](ncu/batched_dispatch_finding.md), with
ncu data captured by [`scripts/diagnostics/ncu_compare.sh`](../../scripts/diagnostics/ncu_compare.sh).

## Reproducing

```bash
deplodock bench recipes/sgemm_cublas_vs_tma --local --filter "deploy.gpu=*5090*"              # RTX 5090 only
deplodock bench recipes/sgemm_cublas_vs_tma --ssh user@host --filter "deploy.gpu=*PRO 6000*"  # Remote Pro 6000
```

The recipe pins driver 595.58.03 and CUDA 13.2 for cloud reproductions.
Note that the bench logs (`*.log`) are intentionally gitignored — the
JSON + Markdown reports above contain all the data.
