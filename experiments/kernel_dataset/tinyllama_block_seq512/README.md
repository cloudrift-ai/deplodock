# TinyLlama Block Kernel Dataset (seq=512, fp32)

Per-kernel sub-graph reproducers extracted from
`scripts/bench_block.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
--seq-len 512 --dtype fp32` (10 hottest kernels covering 99.4% of
block wall time per the v1 timing report).

Each `.json` is a standalone IR sub-graph extracted from the
`04_loop_fusion` stage (post-fusion, pre-tile-lowering) so tuning
passes (BN/BM/F_M/F_N/BK selection, TMA on/off, register-tile shape)
can apply when re-compiled. Load any one with:

```bash
emmy run --bench --warmup 5 --iters 30 --ir <kernel>.json
EMMY_TMA=1 emmy run --bench ... --ir <kernel>.json   # A/B
```

Or profile with ncu directly:

```bash
ncu --metrics sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,...  \
    --kernel-name regex:k_<name> --launch-skip 5 --launch-count 1               \
    emmy run --bench --ir <kernel>.json
```

## Kernels and shapes

| File | Op fusion | M | K | N | Notes |
|---|---|---:|---:|---:|---|
| `k_linear_reduce_reduce.json` | Q proj | 512 | 2048 | 2048 | matmul |
| `k_linear_1_reduce_reduce.json` | K proj | 512 | 2048 | 256 | matmul (GQA, 4 KV heads × 64) |
| `k_linear_2_reduce_reduce.json` | V proj | 512 | 2048 | 256 | matmul (GQA) |
| `k_scaled_dot_product_attention_masked_reduce.json` | Q·Kᵀ + softmax (causal) | 512 | 64 | 512 | per-head SDPA |
| `k_scaled_dot_product_attention_reduce_reduce.json` | SDPA · V | 512 | 512 | 64 | per-head |
| `k_add_3_reduce.json` | O proj + residual | 512 | 2048 | 2048 | matmul + add |
| `k_mul_1_reduce.json` | input RMSNorm | — | — | — | reduce + scale |
| `k_linear_4_reduce.json` | Gate proj | 512 | 2048 | 5632 | matmul |
| `k_mul_8_reduce.json` | Up proj × SiLU(Gate) | 512 | 2048 | 5632 | matmul fused with mul |
| `k_add_5_reduce.json` | Down proj + residual | 512 | 5632 | 2048 | matmul + add |

## Provenance

- Block: `transformers.models.llama.modeling_llama.LlamaDecoderLayer` layer 0
- Hardware reference: RTX 5090 (sm_120)
- Compiler stage: `04_loop_fusion` (sub-graph dump via
  `EMMY_DUMP_DIR=...`)
