---
name: benchmark-new-model
description: Use this skill when the user asks to "add a new model and benchmark it", "benchmark this model", "try out model X on a server", "benchmark <HF model id>", or otherwise wants to onboard a previously-unsupported model into emmy and run benchmarks against it on an existing or freshly provisioned remote GPU server. The skill creates a recipe, web-searches optimal launch parameters and the right engine/image, validates context length / tool / reasoning parsers, and runs benchmarks with hard time caps.
version: 0.1.0
---

# Benchmark a New Model

Onboard a new model into emmy by creating a recipe, validating that it actually launches and serves correctly on the chosen server, then running a benchmark within strict time bounds. The end result is a committed recipe under `recipes/<model>/` plus benchmark output in `recipes/<model>/<timestamp>_<hash>/`.

This skill **assumes** the emmy CLI is the only tool that should touch the remote server. Never `ssh` in by hand to mutate state — that violates the "never modify deployed CloudRift servers" rule. Read-only inspection (logs, `nvidia-smi`, `docker ps`) over SSH is fine.

## Preflight — Load env files

Before any `emmy` command, source the relevant env file(s) so `HF_TOKEN`, `CLOUDRIFT_API_KEY`, `CLOUDRIFT_API_URL`, etc. are exported into the shell. emmy reads these from `os.environ` directly and **does not auto-load `.env`** (no dotenv dep). Without this step, gated HF models fail to download and CloudRift commands hit the wrong cluster or fail to authenticate.

### Which file(s) to source

- **Default** → source plain `.env` only.
- **User specified an additional env file** (e.g. they named a deployment file or pointed at `.env.<something>`) → source `.env` first as the base, then overlay the user-specified file so its values win on conflict. Use the filename the user provided verbatim — never guess one from context.

If the user wants something other than `.env` but didn't say which file, **list candidates** (`ls .env.* 2>/dev/null`) and ask. Don't guess.

### How to source

Run sourcing in the **same Bash call** as the emmy command (Bash sessions don't preserve env across calls). Order matters when overlaying: base first, overlay second, so the overlay wins on conflict.

Default (just `.env`):

```bash
[ -f .env ] && set -a && . ./.env && set +a && emmy deploy ssh --recipe recipes/<name> --ssh <user@host>
```

With a user-specified overlay file `<extra>` (e.g. `.env.local`, or whatever they named):

```bash
set -a && [ -f .env ] && . ./.env; . ./<extra> && set +a && emmy vm create cloudrift --instance-type <type> --ssh-key ~/.ssh/id_ed25519.pub
```

The `[ -f .env ] && . ./.env;` part lets the base file be optional but fails loudly if the overlay is missing — that's intentional, since a missing user-specified file usually means you're about to hit the wrong target.

### Sanity check

After sourcing, confirm the variables you need are set **without echoing values** — secrets land in the transcript otherwise. Use a present/absent check, not parameter expansion of the value:

```bash
[ -n "$HF_TOKEN" ] && echo "HF_TOKEN: set" || echo "HF_TOKEN: MISSING"
[ -n "$CLOUDRIFT_API_URL" ] && echo "CLOUDRIFT_API_URL: set (override)" || echo "CLOUDRIFT_API_URL: default"
[ -n "$CLOUDRIFT_API_KEY" ] && echo "CLOUDRIFT_API_KEY: set" || echo "CLOUDRIFT_API_KEY: MISSING"
```

Do **not** use `echo "HF_TOKEN: ${HF_TOKEN:+set}${HF_TOKEN:-MISSING}"` — when the var is set, `${VAR:-MISSING}` still expands to the **value** (the default only applies when unset), so you get `set<actual-secret>` printed.

If `HF_TOKEN` is missing, gated HF models will 401 silently-ish on download — stop and ask the user how to provide it. If a user-specified overlay was meant to redirect the cluster but `CLOUDRIFT_API_URL` shows as default, you sourced the wrong file (or didn't source the overlay at all) — fix and re-check before running the emmy command.

## Inputs to Confirm

Ask only the ones the user has not already given:

1. **Model** — HuggingFace repo id (e.g. `Qwen/Qwen3-Next-80B-A3B-Instruct`). Required. Validate it exists on HF (web fetch the repo page) before going further; do not guess from a partial name.
2. **Server** — one of:
   - existing host: SSH target `user@host[:port]` (skip provisioning).
   - new server: invoke the **`start-remote-server`** skill first; come back here once it returns the SSH target. Don't reimplement provisioning yourself.
3. **Multimodal** — if the model card lists vision/audio inputs, ask the user explicitly: "do you need multimodal inference benchmarked, or text-only?" Default to text-only if they don't care, since it lowers memory and avoids preprocessing pitfalls.
4. **Recipe name** — propose `recipes/<short-model-name>/` (e.g. `Qwen3-Next-80B-A3B-Instruct`). Only ask if the slug is ambiguous or already exists.

If `recipes/<short-model-name>/` already exists, stop and ask whether to overwrite, version it (`-v2`), or treat the existing one as the starting point.

## Step 1 — Research the Model (Web Search Required)

Before writing a recipe, gather facts. Do this in parallel where possible. **Do not skip this step** — picking the wrong engine or image is the #1 cause of wasted GPU hours.

Search and read for:

- **Architecture & quantization**: dense vs MoE, FP8 / AWQ / GPTQ / BF16, parameter count, active params (for MoE).
- **Context length**: native max in the config (`max_position_embeddings`) and any documented practical cap.
- **Best engine**: check the model's HF README and recent vLLM / SGLang release notes / GitHub issues for "supported as of vX.Y". Prefer:
  - **vLLM** for most well-supported architectures, especially AWQ MoE (vLLM auto-detects; SGLang needs `--quantization moe_wna16`, see `docs/sglang-awq-moe.md`).
  - **SGLang** when the model card or GitHub issues say it lands first there, or the vLLM version doesn't support the arch yet.
  - If both engines support it, default to vLLM. If unsure, note the tradeoff and ask the user.
- **Docker image / version**: a model released in the last ~3 months often needs a **nightly** or specific tagged build:
  - vLLM nightly: `vllm/vllm-openai:nightly` (or a dated tag from Docker Hub).
  - SGLang dev: `lmsysorg/sglang:dev-cu13` or `lmsysorg/sglang:latest`.
  - Pin a specific version in the recipe — don't leave `:latest` permanently. Use `:latest` or `:nightly` only for the validation pass, then pin once it works.
- **Tool & reasoning parsers**: model-family specific. Look up the right `--tool-call-parser` and `--reasoning-parser` values:
  - `qwen3_coder` / `qwen3` for Qwen3 family.
  - `hermes`, `llama3_json`, `mistral`, etc. for others.
  - If the family is unknown, search "vllm tool call parser <model family>" and the vLLM `tool_parsers/` directory.
- **Known issues**: open GitHub issues on vLLM/SGLang mentioning this model often expose the right flags or warn about broken combos. Spend 5 minutes here, save hours later.
- **Multimodal disable flag** (only if user said text-only on a vision model):
  - vLLM: `--limit-mm-per-prompt '{"image":0,"video":0}'` or model-specific flags. For some models, the language-only checkpoint has a separate HF repo — prefer that if it exists.
  - SGLang: `--language-model-only` (already used in `recipes/Qwen3.5-122B-A10B-FP8/recipe.yaml`).

Summarize findings to the user in 5–8 bullets before writing the recipe so they can correct mistaken assumptions cheaply.

## Step 2 — Write the Recipe

Create `recipes/<name>/recipe.yaml`. Mirror the structure of existing recipes (`recipes/Qwen3-Coder-30B-A3B-Instruct-AWQ/recipe.yaml`, `recipes/Qwen3.5-122B-A10B-FP8/recipe.yaml`). **Read the recipe format docs in `README.md`** ("Recipes" section) before writing — named fields go at `engine.llm`, engine-specific go under `engine.llm.vllm` / `engine.llm.sglang`, never duplicate them in `extra_args`.

Initial values — start **conservative**, not optimal:

| Field                                  | Initial value                               | Rationale                              |
|----------------------------------------|---------------------------------------------|----------------------------------------|
| `engine.llm.context_length`            | model's native max (`max_position_embeddings`) | Start at the top, fall back on failure. |
| `engine.llm.max_concurrent_requests`   | 16                                          | Won't OOM the KV cache.                |
| `engine.llm.gpu_memory_utilization`    | 0.9 (vLLM) / 0.85 (SGLang `mem_fraction_static`) | Headroom on first try.            |
| `engine.llm.tensor_parallel_size`      | min GPUs needed to fit the weights          | From research step.                    |
| `benchmark.max_concurrency`            | 16                                          | Match `max_concurrent_requests`.       |
| `benchmark.num_prompts`                | 64                                          | Fast first run.                        |
| `benchmark.random_input_len`           | 2000                                        | Below context, room for output.        |
| `benchmark.random_output_len`          | 2000                                        |                                        |
| `matrices.deploy.gpu`                  | exact name from `emmy/hardware.py`     | Must match — typo = no match.          |
| `matrices.deploy.gpu_count`            | as confirmed                                |                                        |

Place tool / reasoning parser flags in `extra_args`, e.g.:

```yaml
engine:
  llm:
    vllm:
      image: "vllm/vllm-openai:v0.17.0"
      extra_args: "--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
```

For SGLang AWQ MoE, add `--quantization moe_wna16` (see `docs/sglang-awq-moe.md`).

For text-only on a vision model, add the disable flag from research.

## Step 3 — Validate Deployment (Pre-Benchmark)

Deploy without benchmarking first to catch image/parser/quantization issues fast:

```bash
emmy deploy ssh --recipe recipes/<name> --ssh <user@host>
```

Run with `run_in_background: true` and follow logs. Watch for:

- Image pull succeeded.
- Weights loaded (look for `Load weight end` / `Loading weights took` lines).
- Server bound to port (`Uvicorn running on http://0.0.0.0:8000` for vLLM, `Server is fired up` for SGLang).
- No OOM (`CUDA out of memory`, `Not enough memory. Please try to increase --mem-fraction-static`).

If OOM at startup: lower `gpu_memory_utilization` by 0.05, or lower `context_length`, or raise `tensor_parallel_size`. If image-not-found or unknown-arg errors: re-research the engine version. **Tear down and re-deploy after each fix** — running deploys don't reload the recipe:

```bash
emmy deploy ssh --recipe recipes/<name> --ssh <user@host> --teardown
```

### Unfamiliar startup errors → search before guessing

If the boot log shows an error you don't recognize from the table above (assertion failures, kernel-launch errors, `KeyError` in a config loader, `ValueError: Unknown quantization method`, `ImportError`, `RuntimeError: Triton compilation failed`, NCCL hangs, etc.) — **stop and look it up before changing parameters at random**. Random retunes burn GPU minutes and obscure the real cause.

Workflow:

1. **Capture the full traceback** from the container log (last ~40 lines, including the exception type and the file/line where it fired). Strip volatile bits (PIDs, timestamps, paths under `/root/`) so the search query is generic.
2. **Web-search the error**, prefixed with the engine name and model family:
   - `vllm "<exception type>: <key phrase>" <model_family>`
   - `sglang "<exception type>" <model_family>`
   - For NCCL/distributed errors, add the GPU model and TP size.
3. **Check open and recently-closed GitHub issues** on the engine repo:
   - vLLM: `https://github.com/vllm-project/vllm/issues?q=<error+phrase>`
   - SGLang: `https://github.com/sgl-project/sglang/issues?q=<error+phrase>`
   - Recent **closed** issues with a linked PR are often the most useful — they tell you the exact version that fixed it. If the fix is in a release after the image you pinned, bump to a newer image (or a nightly) rather than working around the bug.
4. **Look for documented workarounds** in the issue thread: extra env vars (`VLLM_USE_V1=0`, `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `SGLANG_ENABLE_JIT_DEEPGEMM=0`, `NCCL_P2P_DISABLE=1`, etc.), flag combinations, or alternative quantization methods. Apply the **smallest** change that addresses the root cause; one variable per attempt so you know what fixed it.
5. **Try the other engine** as a fallback only if the issue thread says the model isn't supported in the engine you picked, or if multiple workarounds have failed. Don't switch engines as a first response.
6. **If nothing matches** after a focused 10-minute search: report the full error to the user with the searches you tried and the top issue links you found, and ask whether to keep digging or pivot. Don't silently retry with arbitrary parameters.

Record the eventual fix (env var, flag, image bump) in the recipe's `extra_args` / `extra_env` with a one-line comment pointing to the upstream issue, so the next person doesn't repeat the search.

Once the server is healthy, send a smoke-test request to confirm the OpenAI-compatible API actually answers (don't trust just the boot log):

```bash
curl -s http://<host>:8000/v1/models
curl -s http://<host>:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "<hf-id>",
  "messages": [{"role":"user","content":"say hi in one word"}],
  "max_tokens": 5
}'
```

If the model has tools and/or reasoning, send one request that exercises each:

- **Tools**: a `tools=[...]` request that should trigger a tool call. Verify the response has a structured `tool_calls` field (not the tool call as a raw string in `content`). If parsing is wrong, fix `--tool-call-parser` and re-deploy.
- **Reasoning**: a request to a reasoning model and check the response includes a `reasoning_content` field separate from `content`. If reasoning leaks into `content`, fix `--reasoning-parser` and re-deploy.

Don't proceed to benchmarks until both verifications pass — a benchmark on a misconfigured model is worse than no benchmark.

## Step 4 — Find Max Working Context (Top-Down)

Find the largest context that actually works on this hardware by **starting from the model's native max and reducing on failure**, not by sweeping bottom-up. This is much faster: most models work at their advertised max, and you'll find the cap in 1–2 attempts instead of 4+.

### Procedure

1. **Try the model's native `max_position_embeddings`** (from the HF config) with `max_concurrent_requests: 4` and a single long-input request (`num_prompts: 4`, input ≈ 0.4 × context, output ≈ 0.4 × context).
2. **If it boots and serves a long request cleanly** → that's your verified max. Done.
3. **If it fails**, halve the context (e.g. 262144 → 131072 → 65536 → …) and retry. Stop at the first context that boots and serves cleanly.
4. **If success but tight** (see below) → try one step lower (`÷2`) as well, compare, and prefer the looser one if the throughput hit is small. Some models nominally fit at the max but leave so little KV cache headroom that real benchmark concurrency will OOM.

A context is **tight** if any of these hold at boot or under a single long request:
- vLLM logs `Free memory ... GiB` < ~5% of total VRAM after model load.
- vLLM warns `KV cache will only support X concurrent requests` and X < your planned `max_concurrent_requests`.
- SGLang logs `avail mem=` under ~1 GB after `Load weight end`.
- Single long-input request OOMs mid-decode.

### Failure modes & responses

- **OOM at startup** → context is too large for these GPUs at this `gpu_memory_utilization` and TP. Halve and retry. Don't just bump `gpu_memory_utilization` past 0.95 — that will break under concurrency later.
- **OOM mid-request** → KV cache too small at this context. Halve context, or drop `max_concurrent_requests` to 1 to confirm the context itself fits.
- **Boots but truncated/garbled output at long context** → the model's claimed max is a lie (rare but happens). Halve and retry; don't record the broken value as the verified max.
- **Success but tight** → record the working value, then run one more pass at half that context to have a safe fallback for the real benchmark.

### Example progression (262K native max)

| Attempt | context_length | Result          | Action                     |
|---------|----------------|-----------------|----------------------------|
| 1       | 262144         | OOM at startup  | Halve.                     |
| 2       | 131072         | Boots, tight    | Record + try one lower.    |
| 3       | 65536          | Boots, comfortable | Use this for real bench. |

Run each attempt as a fresh `emmy deploy ssh` (not a multi-variant `bench` matrix) — that lets you read the boot log and decide before paying for the next attempt. Only roll the surviving value(s) into the matrix for Step 5.

## Step 5 — Run the Real Benchmark (Hard 20-Minute Cap)

With the verified context, pick benchmark params that should finish in under 20 minutes per variant. Reasonable defaults for a single GPU:

- `benchmark.max_concurrency`: 32 (small models) → 128 (medium) → 256 (large MoE).
- `benchmark.num_prompts`: 4× `max_concurrency` (so each slot serves ~4 prompts).
- `random_input_len` / `random_output_len`: whatever the user wants to stress, capped so input+output ≤ 80% of `context_length`.

Set `engine.llm.max_concurrent_requests` ≥ `benchmark.max_concurrency`.

Launch:

```bash
emmy bench recipes/<name> --ssh <user@host>
```

**Always** run this with `run_in_background: true` and a wall-clock cap. The internal subprocess timeout is 10800s (3h) — too generous.

### The 20-minute cap is for the benchmark only

The 20-minute budget covers **only the time from "first request fires" to "last request returns"** — it does **not** include Docker image pull, model weight load, server healthcheck wait, or teardown. Those are setup costs that vary wildly with model size (a 400B model can take 10+ minutes just to load weights) and shouldn't count against the budget. Only the request-issuing phase counts.

To time the benchmark phase only, watch the `emmy bench` stdout/log for the marker that `vllm bench serve` has actually started firing requests, then start your timer there:

- **vLLM**: the line `Traffic request rate: ...` followed by `Burstiness factor: ...` and the start of the progress bar (`0%|`) marks request issuance. Tee the bench log on the remote and `tail -f`, or poll `BashOutput` on the local bench process; latch a "started_at" timestamp on the first match of `Traffic request rate` and only then begin the 20-minute clock.
- **End** marker: `============ Serving Benchmark Result ============` in stdout means the bench finished cleanly (cancel the timer).

Implementation approach:

1. Launch `emmy bench` with `run_in_background: true`. Note the wall-clock at launch — but **do not** schedule the kill yet.
2. Poll `BashOutput` until you see `Traffic request rate` (or an explicit progress line). This may take 30s on a small model or 15+ minutes on a large quantized MoE — be patient, that's expected.
3. Record `started_at = now()`. Now `ScheduleWakeup` for 1200s, OR poll on an `until` loop checking both the result-marker and `(now - started_at) > 1200`.
4. If the result marker appears first → success.
5. If 20 minutes elapse from `started_at` first → kill the bench (`KillShell` / `KillBash`), tear down the deploy, and re-tune.
6. Separately, set a **setup-phase ceiling** of ~30 minutes (from launch to first request). If `Traffic request rate` hasn't appeared in that long, something is wrong with deploy/load (image pull stuck, weights download stuck, server crash-looping); abort and inspect logs.

Re-tune triggers and remedies (all measured from the request-issuance start, not bench launch):

| Symptom                                       | Adjust                                               |
|-----------------------------------------------|------------------------------------------------------|
| Request-phase wall-clock > 20 min             | Halve `num_prompts`, then halve again if still slow. |
| Token throughput collapses near end           | `max_concurrency` too high; halve.                   |
| Many request timeouts                         | Lower `max_concurrency` or `random_output_len`.      |
| Server OOM mid-run                            | Lower `context_length` or `max_concurrent_requests`. |
| Bench finishes in <2 minutes with low TPOT    | Bump `num_prompts` next run — under-loaded.          |
| `Traffic request rate` never appears (>30 min) | Setup is stuck; check image pull / weight load logs. |

After the first successful run completes inside 20 minutes (request phase), you have a working baseline. If the user wants a sweep (concurrency / context combinations), expand the matrix and re-run; each variant's request phase must individually fit the 20-minute budget.

## Step 6 — Save & Hand Off

When at least one benchmark variant has completed cleanly:

1. Confirm result file exists in `recipes/<name>/<timestamp>_<hash>/<variant>_benchmark.txt`.
2. Surface key numbers (request throughput, token throughput, mean TTFT, mean ITL) to the user from the output.
3. Pin the docker image version in the recipe (no more `:latest` / `:nightly`) — record exactly which tag was used.
4. Mention the verified max context as a comment in the recipe (one short line — don't pad with explanation).
5. Tell the user the server is **still up and billed** (unless they used `emmy bench` without `--ssh`, which auto-tears down). Show the matching teardown command from the `start-remote-server` skill output.
6. Per `CLAUDE.md` contribution rules: before any commit, update `STYLE.md`, `README.md`, `CLAUDE.md`, `ARCHITECTURE.md` (only those that need changes — usually none for a new recipe), and run `make test` + `make lint`. Open a feature branch; never commit directly to `main`.

## Verification Checklist

Before reporting success, verify:

- [ ] `recipes/<name>/recipe.yaml` exists and `emmy bench --dry-run recipes/<name>` parses it without error.
- [ ] At least one benchmark variant produced a `Serving Benchmark Result` block.
- [ ] If the model has tools, a tool-call request returned structured `tool_calls`.
- [ ] If the model has reasoning, a request returned a populated `reasoning_content`.
- [ ] Multimodal status matches what the user asked for.
- [ ] Docker image tag in the recipe is a pinned version, not a moving tag.
- [ ] No bench variant exceeded the 20-minute wall clock.

If any check fails, report the failure and the raw output instead of claiming success.

## Common Mistakes to Avoid

- **Don't skip the web search.** Models change fast; the right vLLM image for a 6-month-old model is not the right image for a 2-week-old model.
- **Don't go straight to `emmy bench` for a new model.** Validate the deploy first — bench will hide the root cause behind a slow timeout.
- **Don't put named flags in `extra_args`.** `--max-model-len` etc. live as named fields; recipe loader rejects duplicates.
- **Don't leave `:latest` in a committed recipe** — recipes are reproducibility artifacts; pin the tag once it works.
- **Don't sleep through a long bench.** Watch it; abort at 20 minutes and re-tune. The user said so explicitly.
- **Don't ssh in to manually patch the server.** Memory says read-only on CloudRift hosts. If a deploy is broken, fix the recipe and redeploy via `emmy`, don't paper over it on the host.
- **Don't fall back to a different provider/GPU** silently if the requested combo doesn't work. Stop and ask, like `start-remote-server` does.
- **Don't claim the model supports its native max context** without running an input that actually fills the context. Boot success ≠ working context.
