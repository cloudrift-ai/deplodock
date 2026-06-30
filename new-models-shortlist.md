# New-Model Shortlist — Deployment Candidates

> Generated **2026-06-08** by the `discover-new-models` skill.
> Source: `./venv/bin/python scripts/new_models.py --arena --workers 4` (keyless OpenRouter + HuggingFace + LMArena),
> plus per-candidate web search for hype/specs and HF quant-repo verification.
> **Signals are time-sensitive** (HF `trending`, LMArena Elo) — re-run the script before acting if this is stale.
>
> This is a hand-off feed for the **`benchmark-new-model`** skill: pick a `(model, GPU, gpu_count)` row and pass the
> `hf_id` (or the quant repo) + hardware over. Nothing here has been deployed or benchmarked yet.

## Finalists (demand read)

| Model (`hf_id`) | Size (total / active) | Arch / format novelty | HF trend / Elo (rank) | Demand read |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Flash` | 284B / 13B MoE | CSA+HCA hybrid attn; **native FP4-experts + FP8-attn** (deploy-ready), MIT, 1M ctx | 95 / **1428** (#55) | **Strong** — "beats all open models in Math/STEM/Coding"; best size-for-value flagship |
| `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16` | 550B / 55B MoE | **Hybrid Mamba-Transformer + Latent MoE, NVFP4-pretrained** | **160 (#1)** / — (too new) | **Strong (momentum)** — fastest US open frontier model, 5× throughput; most novel arch for emmy |
| `google/gemma-4-26B-A4B-it` | 25.2B / 3.8B MoE | multimodal (text+img+video), Apache 2.0, 256K ctx | 45 / **1435** (#49) | **Strong (small)** — #1 downloads (12M); near-31B quality on one consumer/Pro card |
| `MiniMaxAI/MiniMax-M2.7` | 230B / 10B MoE | agentic; **official FP8 + NVIDIA NVFP4** | 20 / 1400 (#106) | **Moderate-strong** — open-weight agent model, clean mid-tier fit |
| `ibm-granite/granite-4.1-8b` | 8B dense | Apache 2.0, 131K ctx, enterprise | 5 / 1296 (#197) | **Moderate** — 8B matches IBM's own 32B; reliable consumer-card filler |
| `deepseek-ai/DeepSeek-V4-Pro` | 1.6T / 49B MoE | flagship, MIT, 1M ctx | 136 / **1446** (#31) | **Strong (frontier)** — within 0.2pt of Claude Opus on SWE-bench; needs 8×B200 (NVFP4) |
| `XiaomiMiMo/MiMo-V2.5-Pro` | 1.02T / 42B MoE | SWA/GA hybrid (7× KV savings) | 26 / **1461 (#17, top Elo)** | **Strong (frontier)** — "beats Kimi K2.6, GLM 5.1"; fits 4×B200 (FP4) |
| `moonshotai/Kimi-K2.6` | 1T / — MoE | multimodal agentic | 38 / **1456 (#20)** | **Strong (frontier)** — SWE-bench leader, agent-swarm; fits 4×B200 (NVFP4) |

## Hardware → model matrix (deployment plan)

VRAM rule of thumb: `weight VRAM ≈ total_params(B) × bytes/param` (BF16=2, FP8=1, FP4/AWQ/INT4≈0.5), then **×1.3**
for activations + CUDA graphs + a modest KV cache. `tensor_parallel_size` (TP) must divide the model's attention head
count; prefer 2/4/8.

| Hardware | gpu_count / TP | Model (quant) — repo to pull | Why | Fit (weights ×1.3) |
|---|---|---|---|---|
| **RTX 4090** (24 GB) | 1 | `granite-4.1-8b` (FP8) — community GGUF/FP8 | 8B matching a 32B, Apache 2.0 | ~10 GB, easy |
| **RTX 5090** (32 GB) | 1 | `gemma-4-26B-A4B-it` (FP8) — community quant | hot multimodal MoE, near-31B quality | ~33 GB, tight but fits |
| **RTX PRO 6000** (96 GB) | 1 | `gemma-4-26B-A4B-it` (BF16) — base repo | full-precision multimodal w/ context headroom | ~65 GB of 96 |
| **H100 80GB** | 2 / TP2 | `MiniMax-M2.7` (NVFP4) — `nvidia/MiniMax-M2.7-NVFP4` · **or** `DeepSeek-V4-Flash` (W4A16) — `pastapaul/DeepSeek-V4-Flash-W4A16-FP8` | agentic / coding flagship via TP2 | ~150–200 GB across 2 |
| **H200 141GB** | 2 / TP2 | `DeepSeek-V4-Flash` (native FP4+FP8) — base repo | **top pick** — MIT, 1M ctx, deploy-ready quant | ~210 GB across 2, comfortable |
| **B200** (~180 GB) | 1 | `MiniMax-M2.7` (NVFP4) — `nvidia/MiniMax-M2.7-NVFP4` | fits a 230B agent model on one card | ~150 GB of 180 |
| **B200** | 2 / TP2 | `Nemotron-3-Ultra-550B` (NVFP4) — NVFP4 checkpoint | #1 HF trend, novel Mamba hybrid | ~358 GB across 2 (360) |
| **B200** | 4 / TP4 | `XiaomiMiMo/MiMo-V2.5-Pro` (FP4) — `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` | **top Elo (1461)**, frontier coding | ~663 GB across 4 (720) |
| **B200** | 4 / TP4 | `moonshotai/Kimi-K2.6` (NVFP4) — `nvidia/Kimi-K2.6-NVFP4` | SWE-bench leader, multimodal agent | ~650 GB across 4 (720) |
| **B200** | 8 / TP8 | `deepseek-ai/DeepSeek-V4-Pro` (NVFP4) — `nvidia/DeepSeek-V4-Pro-NVFP4` | 1.6T flagship ≈ Claude Opus, MIT | ~1040 GB across 8 (1440) |
| **B200** | 8 / TP8 | `moonshotai/Kimi-K2.6` (FP8) — `RedHatAI/Kimi-K2.6-FP8-BLOCK` | full-fidelity alt to the 4×B200 NVFP4 run | ~1300 GB across 8 (1440) |

## Top 3 to pursue first

1. **`DeepSeek-V4-Flash` on 2×H200** — best size/value, ships its own FP4+FP8 quant, MIT license → least friction.
2. **`gemma-4-26B-A4B-it` on 1×RTX5090** — cheapest to benchmark, huge demand, single card.
3. **`Nemotron-3-Ultra-550B` on 2×B200** — most novel arch + #1 momentum.
   ⚠️ **Verify engine support first** — hybrid Mamba-Transformer may not be fully supported by vLLM/SGLang yet.

## Watch / revisit (not slotted)

- **`nex-agi/Nex-N2-Pro`, `poolside/Laguna-XS.2`, `tencent/Hy3-preview`** — too quiet / preview-stage right now.
- **`nvidia/Nemotron-3.5-Content-Safety`** — a safety classifier, not a general LLM; skip for throughput benchmarking.
