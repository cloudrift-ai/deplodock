# TODO: Deployment & Benchmarking

Issues and improvements for `deplodock deploy` / `deplodock bench` (recipes, cloud provisioning, cache handling).
For matmul/compiler tuning TODOs see [`TODO.md`](TODO.md).

## Recipe `benchmark.model_dir` is silently ignored; bench vs deploy-ssh cache paths diverge

**Found 2026-06-10 while onboarding DeepSeek-V4-Pro (NVFP4, 1.6T) on 8×B200.**

The model-cache mount path used by `deplodock bench` is read from the **global `config.yaml`**
(`benchmark.model_dir`, default `/hf_models`), NOT from the recipe's `benchmark.model_dir` block. See
`deplodock/benchmark/execution.py:115` — `config["benchmark"].get("model_dir", "/hf_models")` where `config` is
`load_config(args.config)` = `config.yaml`, not the recipe. Meanwhile `deploy ssh` defaults `--model-dir /mnt/models`
(`deplodock/commands/deploy/ssh.py`).

Consequences:

- A recipe that sets `benchmark.model_dir:` is **silently ignored** — no error, no warning, no effect. Confirmed: a
  recipe with `benchmark.model_dir: /mnt/models` still mounted `/hf_models` (per the bench log's `path: /hf_models/hub/...`
  line). It parses fine, so it reads as a working knob but isn't one.
- `deplodock bench` against a host already provisioned by `deploy ssh` **re-downloads the whole model** (~851 GB for
  DeepSeek-V4-Pro) into `/hf_models`, because it won't reuse the `/mnt/models` cache that `deploy ssh` populated. The
  long single SSH session during that download is also drop-prone on these VMs (saw `Can't assign requested address` /
  broken pipe mid-download).

**Fix options (pick one or combine):**

1. Make `deplodock bench` honor a recipe-level `benchmark.model_dir` (merge recipe over `config.yaml`) so the cache
   path is configurable per model.
2. Align the defaults so the same path is used by `config.yaml` `benchmark.model_dir` and `deploy ssh --model-dir`
   (both `/hf_models` or both `/mnt/models`), so bench can reuse a deploy-ssh cache.
3. At minimum, **validate/warn** when a recipe sets `benchmark.model_dir` today, so it doesn't masquerade as a working
   setting.

**Workaround used in the meantime:** bench the already-warm `deploy ssh` server directly via a detached container
(`docker run -d --network host -e HF_HOME=/mnt/models -v /mnt/models:/mnt/models ... vllm bench serve --base-url
http://localhost:8000 ...`), writing results to the persistent mount — immune to SSH drops, no re-download.

**Re-confirmed 2026-06-10 on DeepSeek-V4-Flash** (base `deepseek-ai/DeepSeek-V4-Flash`, 2×H200): same `/mnt/models`
(deploy) vs `/hf_models` (bench) divergence. Temporary unblock = set `config.yaml` `benchmark.model_dir: /mnt/models`
(the global value, since the recipe block is ignored), which made `deplodock bench` reuse the cache ("Fetching 74 files
in 0s" instead of a 149 GB re-download).

## `deploy` reports "Failed to download model" when SSH drops, even though the download completed

**Found 2026-06-10, DeepSeek-V4-Flash on a flaky-network GCP H200 spot VM.**

The download runs as `ssh host 'docker run --rm ... hf download ...'` over one long-lived SSH session. When the network
blipped mid-download the SSH client died (`Read from remote host: Operation timed out` / broken pipe) and `deploy ssh`
printed **`Failed to download model`** and aborted — but the `docker run` container kept running detached under
`dockerd` and finished (46/46 shards, 0 `.incomplete`). The reported failure was spurious; a naive retry would
re-download ~149 GB.

**Fix:** before declaring the download failed, reconnect and verify the cache (shard count vs the model's
`*.safetensors.index.json`, no `.incomplete` blobs) and only fetch what's missing. Better: run the download detached
(`docker run -d` / `nohup`) and poll, so a client-side network drop can't kill or "fail" it.

## Kernel/compile cache is ephemeral → every redeploy pays the full recompile (~7 min)

**Found 2026-06-10, DeepSeek-V4 (`deepseekv4-cu130` + nightly), 8×H200.**

First boot JIT-compiles the V4 TileLang/CUDA kernels and captures CUDA graphs (~6-7 min, GPUs idle, logs quiet). The
caches live in the container's **ephemeral layer** (`/root/.tilelang`, `/root/.cache/vllm` ~830 MB); only `/mnt/models`
is mounted. So `deploy`/`bench`'s `docker compose down`→`up` **recompiles from scratch every time** — a sweep across TP
sizes / context lengths paid this tax on each of ~8 redeploys.

**Fix:** mount a persistent compile-cache volume (e.g. `-v /mnt/models/.compile-cache:/root/.cache` plus the tilelang
cache dir) so recompiles survive container recreation — would cut redeploy from ~7 min to ~1-2 min for these models.

## The long silent compile / CUDA-graph phase is indistinguishable from a hang

During that ~7-min compile/capture, `docker compose up --wait` is silent and GPUs read 0% — identical to a real
deadlock (which V4 *also* exhibited at TP2 on the day-0 image). The only way to tell them apart is to SSH in and
`ps`-grep for `nvcc/cc1plus/ptxas/cicc`. **Fix/UX:** surface a heartbeat during the health-wait (tail the engine log for
"Capturing CUDA graph"/compile lines, or just print "waiting for health (Ns elapsed)…") so an operator doesn't misread
normal compilation as a hang.

## `deploy ssh` auto-scale-out fills all host GPUs; pinning a single small instance is non-obvious

On an 8-GPU host, `deploy ssh` (default `data-parallelism` scale-out) sets `dp = detected_gpus // (tp*pp)` and runs
replicas across **all** GPUs. To deploy ONE instance on exactly N GPUs (e.g. to measure the minimum GPU count for a
model), you must pass `--gpu-count N`. **Fix/doc:** document this; "gpu-count" reads like "use this many" but actually
overrides host detection — consider a clearer name/alias.

## DeepSeek-V4 KV pool GROWS with `--max-model-len` — don't size context from a short-ctx reading

**Found 2026-06-10, DeepSeek-V4-Pro NVFP4 on 8×B200 (TP8, gpu-mem-util 0.9).** Counter-intuitive and a real foot-gun
for context sizing: V4's sparse/compressed attention shrinks the per-token KV footprint as `--max-model-len` rises, so
the usable KV-cache token pool *grows* with the configured context. Measured at the same ~47 GiB/GPU:

- `context_length 65536`   → GPU KV cache **696K tokens** (10.6× concurrency @65K)
- `context_length 524288`  → GPU KV cache **3.56M tokens** (6.8× @512K)
- `context_length 1048576` → GPU KV cache **5.0M tokens** (4.77× @1M)

So extrapolating max context from a small-context deploy **under-counts by ~7×** — a 64K reading (696K-token pool) made
the full native 1M look infeasible (696K < 1M) when it actually fits and serves (verified: 1M boots; 128K-input requests
clean, 4/4). The top-down "try native max, halve on OOM" probe (benchmark-new-model Step 4) still works, but the KV log
line read at a *small* context is misleading. **Fix/UX:** when sizing context, deploy at/near the target
`context_length` and read `GPU KV cache size` there — don't trust a 64K reading. A deplodock helper that reports the
KV-pool-at-target-context (or warns that the figure scales with max-len for sparse-attention models) would prevent
under-provisioning.

## `benchmark-new-model` skill improvements (from the DeepSeek-V4-Flash onboarding)

These are SKILL fixes (`.claude/skills/benchmark-new-model/SKILL.md`), recorded here so they aren't lost:

- **Validate with a LONG input, not just short.** The skill's smoke test (`max_tokens 5`, short prompts) passed on a
  deployment whose long-context prefill was completely broken. Add a multi-K-token (tens of K for long-ctx models)
  request that confirms the GPU actually does work, before benchmarking.
- **Prefer a recent nightly over a day-0 image for `<2-month-old` models; bump the image before deep config tuning.**
  The TP2 long-prefill "limitation" was an image bug (cf. vLLM #40969) fixed in a newer nightly — found only after ~5
  redeploys of `enforce-eager`/`PIECEWISE` workarounds. Check sibling `recipes/` for a known-good image digest.
- **Add a silent-hang troubleshooting track** (request accepted, no tokens, GPU 0%, workers spinning): `ps` for
  compilers → it's compiling, wait; else search `"<model> <engine> hang|stall|cudagraph"` issues BEFORE tuning flags.
- **Canonical output is the deliverable.** Don't hand-assemble result files from raw `vllm bench serve` dumps — produce
  `deplodock bench`'s `<timestamp>_<hash>/<variant>_benchmark.{txt,json}`; fix prerequisites (model_dir, SSH keepalive)
  rather than working around bench with manual runs.
- **Document deploy resilience on flaky VMs**: set `ServerAliveInterval` / high `ServerAliveCountMax` in `~/.ssh/config`;
  note that `up -d` and the download container are detached, so an SSH drop ≠ failure — reconnect and poll.
