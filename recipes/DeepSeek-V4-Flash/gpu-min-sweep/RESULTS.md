# DeepSeek-V4-Flash — minimum-GPU sweep for full 1M context (2026-06-10, 8×H200)

Goal: smallest GPU count that serves the full 1,048,576-token context with good performance.

## Answer: **2× H200 (TP2)** — on the 2026-06-10 vLLM nightly.

Memory is not the constraint (V4 compressed attention). The day-0 `deepseekv4-cu130` image stalled
long-context prefill at TP2 (GPU 0%, workers spin) across all cudagraph modes (default hang = vLLM #40969;
`--enforce-eager` and `cudagraph_mode=PIECEWISE` also stalled). The 2026-06-10 nightly
(`vllm/vllm-openai@sha256:03768d9400bf…b69db8e`) fixes it with default cudagraph.

| Config | GPUs | 1M KV headroom | 200K prefill TTFT | decode TPOT | notes |
|--------|------|----------------|-------------------|-------------|-------|
| TP4 (within 8-GPU DP2, day-0 img) | 4/replica | 14.6× | 18.5 s | 11.2 ms | reference; short probe dodged the #40969 hang |
| TP2 single, day-0 img | 2 | 7.2× | **stall** | — | long prefill never engages GPU (all cudagraph modes) |
| **TP2 single, 2026-06-10 nightly** | **2** | **7.2×** | **24.2 s** | **7.67 ms** | **good perf — the recommended minimum** |
| TP1 | 1 | — | — | — | infeasible: weights ~149 GB > 141 GB single-GPU (OOM) |

### Deployable benchmarks at the recommended config (TP2, 2×H200, 1M ctx, nightly)

Canonical `emmy bench` (short-context throughput, the standard result file):
- **`../2026-06-10_19-06-58_56e5c44d/h200x2_vllm_benchmark.{txt,json}`** — conc 16, 64 prompts, 2000/2000:
  **900 tok/s output, TPOT 16.9 ms, mean TTFT 1.66 s (P99 4.0 s), 0/64 failed, 142 s.**

Long-context characterization (manual `vllm bench serve` probes — emmy bench runs one short-context block,
so these aren't canonical result files; numbers recorded here):
- 200K input, conc 1: TTFT **24.2 s**, decode TPOT **7.67 ms** (0/1 fail).
- 200K input, conc 4 (8 prompts): TTFT **29.7 s** (P99 55 s), TPOT **112 ms**, peak GPU 100%, 0/8 fail —
  concurrent long prefills hold up; decode TPOT rises with concurrent long-context streams (attention cost scales
  with held context). Lever is concurrency-vs-latency at long ctx; step to TP4/TP8 for faster long-prefill.

Earlier short-context throughput on 8×H200 (DP2×TP4, 65,536 ctx, conc 32): 3,414 tok/s total, TPOT 15.2 ms
(see ../2026-06-10_15-39-22_6ed122c6/).

Note: `emmy bench` reads the HF cache path from `config.yaml` `benchmark.model_dir` (not the recipe). On these
GCP H200 VMs `deploy ssh` caches the model at `/mnt/models`, so `config.yaml` was set to `/mnt/models` (temporary)
to avoid a 149 GB re-download — revert once deploy/bench agree on the cache path.
