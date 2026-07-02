---
name: discover-new-models
description: Use this skill when the user asks "what new models should we try / benchmark", "find / discover new (open) models worth exploring", "scout trending models for our GPUs", "which new models fit B200 / H200 / H100 / Pro6000 / 5090 / 4090", "what's new and worth benchmarking", or otherwise wants a shortlist of newly-released, in-demand open-weight models emmy doesn't support yet, each mapped to the hardware it runs on. Produces a ranked, hardware-bucketed shortlist ready to hand to the `benchmark-new-model` skill. Discovery + popularity come from `scripts/new_models.py` (keyless); hype/news from web search; hardware fit from a VRAM calculation.
version: 0.1.0
---

# Discover New Models to Explore

Turn "what new models are worth our GPU hours?" into a concrete, ranked shortlist: newly-released open-weight
models emmy **doesn't support yet**, filtered to the ones with real demand/hype, each tagged with the GPU
configs it fits. The output is a hand-off list for the **`benchmark-new-model`** skill — this skill never
deploys or benchmarks anything itself; it researches and recommends.

Everything here is **keyless and read-only**: `scripts/new_models.py` hits public OpenRouter + HuggingFace
endpoints, and the rest is web search. No servers are touched, no recipes are written.

## Pipeline

```
scripts/new_models.py  →  per-model news/hype search  →  rank by demand  →  VRAM fit calc  →  hardware → model matrix
   (candidates +            (qualitative signal)          (pick finalists)    (params×quant)     (the deliverable)
    HF popularity + Elo)
```

## Step 0 — Confirm scope (only what's unstated)

Ask only if the user hasn't already implied it:

- **Time window** — default the script's ~90 days (`--since` default). Widen with `--since 2026-01-01` if they
  want "this year".
- **Modality** — default include multimodal (shown with a `modality` column); add `--text-only` if they only
  serve text.
- **Target hardware** — default the full fleet (B200, H200, H100, Pro6000, RTX5090, RTX4090). If they name a
  subset, only bucket for those.
- **How many finalists** — default ~5–8, spread across the hardware tiers.

## Step 1 — Get the candidate list (the script)

Run the discovery script with arena enrichment, capturing JSON for parsing and the table for a human view.
Use `--workers 4` to stay gentle on the HF metadata endpoint (it rate-limits bursts; don't re-run in a loop):

```bash
./venv/bin/python scripts/new_models.py --arena --workers 4 --json > /tmp/new_models.json
./venv/bin/python scripts/new_models.py --arena --workers 4          # readable table for the user
```

The script already does the hard part: it lists open-weight models OpenRouter hosts (catalog entries with a
`hugging_face_id`), **excludes families already in `recipes/`**, drops anything older than `--since`, verifies
each on HuggingFace, and ranks by HF momentum. Each JSON row in `models[]` carries:

| Field | Meaning | Use |
|---|---|---|
| `hf_id` | HuggingFace repo id | the model identity; feeds `benchmark-new-model` |
| `created_at` | HF release date | recency |
| `downloads` | HF 30-day pulls | adoption (lagging, size-biased toward small models) |
| `likes` | cumulative HF likes | reputation |
| `trending` | HF trendingScore | **momentum / "hot right now"** (best single demand signal) |
| `elo` / `arena_rank` | LMArena Elo + rank (blank if unrated) | **quality**; blank usually = too new to be rated, not bad |
| `modality` | `text->text`, `text+image->text`, … | multimodal flag |

The table footer also flags stale OpenRouter→HF mappings ("NOT ON HF") and likely arena fuzzy-match misses —
skim those; a miss can mean a model you'd otherwise drop actually has a strong Elo under a slightly different name.

Take the top ~8–12 by `trending` (tie-break `elo`, then `downloads`) into Step 2. The script's full flag list is
documented in `CLAUDE.md` (scripts section).

## Step 2 — Research news & hype per candidate (web search required)

The script gives *quantitative* demand (HF momentum, arena Elo). Layer on *qualitative* mindshare — what people
are actually saying. For each top candidate, web-search in parallel:

- `"<model name>" release` / `"<model name>" benchmark` — official announcement, benchmark claims (MMLU, GPQA,
  LiveCodeBench, SWE-bench, AIME).
- `"<model name>" vs` — head-to-head comparisons (a sign people care).
- Buzz on Reddit (r/LocalLLaMA), Hacker News, X — is it being discussed, or did it land silently?
- The releasing **lab's reputation** (DeepSeek, Qwen, MiniMax, Moonshot/Kimi, Mistral, NVIDIA, Liquid, IBM
  Granite, Allen AI/OLMo) — established labs draw adoption faster.

Distill each into a one-line **demand read**: *strong* (benchmark wins + active discussion + reputable lab),
*moderate*, or *niche/quiet*. Cross-check against the script signals — a model high on HF trend **and** loud
online is a strong pick; high downloads but silent is often a small fine-tuning base, not a flagship.

## Step 3 — Select the most promising

Combine signals into a shortlist. A model is **promising** when it scores on several of:

- High HF `trending` (real, current momentum) — weighted highest for "what's hot".
- High arena `elo` / low `arena_rank` (proven quality) — when present.
- Strong Step-2 hype (mindshare, benchmark wins, reputable lab).
- **Novelty for emmy** — a new architecture / quant / size we haven't benchmarked teaches us more than yet
  another sibling of an existing recipe.

Drop: tiny fine-tuning bases riding download counts, models with no engine support yet (note it, revisit later),
and anything the user explicitly doesn't care about. Aim for a spread of **sizes** so the next step can fill
several hardware tiers (don't pick five 400B MoEs).

## Step 4 — Hardware requirements per finalist

For each finalist, pull from the HF model card / `config.json` (web search or the repo): **total parameters**,
**active parameters** (MoE), and the **available quantizations** (is there an FP8 / AWQ / NVFP4 / INT4 repo, or
only BF16?). Then compute VRAM to fit.

**VRAM rules of thumb** (weights only; per GPU):

```
weight VRAM ≈ total_params(B) × bytes_per_param
    bytes_per_param:  BF16/FP16 = 2  ·  FP8 = 1  ·  AWQ/INT4 ≈ 0.5
min-to-serve ≈ weight VRAM × 1.3        # + CUDA graphs, activations, a small KV cache at modest context
long-context / high-concurrency ≈ weight VRAM × 1.5+  (KV cache grows with context × concurrency)
```

Two traps:
- **MoE VRAM is governed by TOTAL params** (all experts load into memory), not active params — active params
  only drive *throughput*. A 235B-A22B MoE needs ~235B worth of weights resident.
- **Quantization decides the GPU.** The same model in BF16 vs AWQ can be a 4× VRAM difference — always check
  which quant repos actually exist before assigning a tier.

**Multi-GPU:** if it doesn't fit on one card, tensor-parallel across N (`tensor_parallel_size`): per-GPU need
≈ `min-to-serve / N` + per-GPU overhead. N must divide the model's attention head count; prefer 2/4/8.

**Fleet VRAM** (single GPU; multiply by `gpu_count` for multi-GPU nodes):

| GPU (hardware.py name) | VRAM | short |
|---|---|---|
| `NVIDIA B200` | ~180 GB | b200 |
| `NVIDIA H200 141GB` | 141 GB | h200 |
| `NVIDIA H100 80GB` | 80 GB | h100 |
| `NVIDIA RTX PRO 6000 Blackwell {Workstation/Max-Q/Server} Edition` | 96 GB | pro6000 |
| `NVIDIA GeForce RTX 5090` | 32 GB | rtx5090 |
| `NVIDIA GeForce RTX 4090` | 24 GB | rtx4090 |

**Quick fit reference** (weights only — apply ×1.3 for real serving; pick the quant that exists):

| Total params | BF16 (×2) | FP8 (×1) | AWQ/INT4 (×0.5) |
|---|---|---|---|
| ~8B | 16 GB → 1×4090(tight)/5090 | 8 GB → 1×4090 | 4 GB → 1×4090 |
| ~30B | 60 GB → 1×H100/Pro6000 | 30 GB → 1×5090(tight)/Pro6000 | 15 GB → 1×4090(tight)/5090 |
| ~70B | 140 GB → 1×H200/2×H100 | 70 GB → 1×H100/Pro6000 | 35 GB → 1×Pro6000/5090(tight) |
| ~120B | 240 GB → 2×H200/2×B200 | 120 GB → 1×B200/2×H100 | 60 GB → 1×H100/Pro6000 |
| ~235B | 470 GB → 4×B200 | 235 GB → 2×B200/4×H100 | ~120 GB → 1×B200/2×H100 |
| ~400B+ | 800 GB → 8×B200 | 400 GB → 4×B200/8×H100 | ~200 GB → 2×B200/4×H100 |

## Step 5 — Hardware → model matrix (the deliverable)

Present a table mapping each target GPU config to the promising model(s) that fit, with the quant and a one-line
why. Cover the spectrum — small flagships on consumer cards, mid-size on Pro6000/H100, large MoE on H200/B200
(single or tensor-parallel). Example shape:

| Hardware | gpu_count | Recommended model (quant) | Why it's promising | Fit note |
|---|---|---|---|---|
| RTX 4090 / 5090 | 1 | `<8B model>` (AWQ/FP8) | strong small-model Elo, hot on HF | ~Xgb, fits one card |
| RTX PRO 6000 | 1 | `<30–70B>` (FP8) | benchmark wins, reputable lab | ~Xgb of 96 |
| H100 80GB | 1–2 | `<70–120B MoE>` (FP8) | top arena Elo, high HF trend | TP2 for context headroom |
| H200 141GB | 1 | `<120B>` (FP8) | flagship, loud online | fits 1×, long context |
| B200 | 1–8 | `<235–400B+ MoE>` (FP8/NVFP4) | SOTA open, high demand | TPn across the node |

Flag any model with **no engine support yet** or **no suitable quant** as "watch, revisit" rather than slotting it.

## Step 6 — Hand off

For each (model, hardware) pair the user wants to pursue, offer to invoke **`benchmark-new-model`** — pass the
`hf_id` and the chosen GPU + `gpu_count`. That skill does the real work (engine/image research, recipe, validate,
benchmark within time caps). Don't reimplement any of it here.

If the user just wanted the survey, stop at the matrix.

## Common mistakes to avoid

- **Don't rank by downloads alone** — it's lagging and size-biased (tiny fine-tuning bases dominate). `trending`
  + arena `elo` + Step-2 hype together are the demand signal.
- **Don't size MoE by active params.** VRAM follows TOTAL params; active params only set throughput.
- **Don't assign a GPU tier without checking the quant exists.** "Fits H100 at FP8" is meaningless if only a
  BF16 repo is published. Verify the AWQ/FP8/NVFP4 repo on HF first.
- **Don't treat a blank arena Elo as "bad".** It usually means the model is too new for the last weekly arena
  snapshot — lean on HF `trending` + news there.
- **Don't spam the script.** HF rate-limits bursts; use `--workers 4` and re-run sparingly (transient failures
  land in the script's "COULD NOT VERIFY" bucket — wait and re-run, don't hammer).
- **Don't deploy or write recipes in this skill.** Discovery only; `benchmark-new-model` owns deployment.
- **Don't forget overhead.** The fit table is weights-only; real serving needs ~1.3× for activations + KV, more
  for long context / high concurrency.
