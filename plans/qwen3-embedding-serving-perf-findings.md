# Qwen3-Embedding-0.6B `deplodock serve` — startup fix + serving perf A/B findings

**Status:** `serve --bench` was broken by a transitive dependency skew (fixed); once running, the deplodock serving
plugin is **correctness-complete but 1–2 orders of magnitude slower per request than stock vLLM**, and the gap **widens
with sequence length**. A per-kernel tune (`tune-model`) is queued to attribute the time kernel-by-kernel.

**Environment:** NVIDIA GeForce RTX 4080 (sm_89, 16 GB), HEAD `72a65e1a`, vLLM 0.22.1, Qwen/Qwen3-Embedding-0.6B,
trunk compute dtype fp16 (bf16 unsupported by the trunk → downcast with warn). **Date:** 2026-06-17.

**Bench commands** (one-shot `start → /health → vllm bench serve --backend openai-embeddings → shutdown`):

```bash
# concurrency 32 (throughput regime; the user's original two runs)
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32                 # deplodock plugin
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32 --stock         # stock vLLM baseline
# concurrency 1 (per-request latency; the apples-to-apples regime)
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32  --max-concurrency 1 [--stock]
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 512 --max-concurrency 1 [--stock]
```

All four bench params default to `--max-concurrency 32 --num-prompts 256 --bench-seed 0`; only the flags shown above
were varied. 256 prompts per run.

---

## Part 1 — startup bug: `serve --bench` never passed `/health` (FIXED)

**Symptom.** The server booted fine (model compiled, weights loaded, `Application startup complete.`), but **every**
HTTP request — including `/health` — returned `500`, so `_wait_for_health` (`commands/serve.py`) polled until the
30-minute timeout, killed the server, and exited 1. The bench never started.

**Root cause.** A transitive dependency skew in the vLLM env, **not** a deplodock bug:

| package | resolved | role in the failure |
|---|---|---|
| `fastapi` | **0.137.0** | introduced `_IncludedRouter` (a `BaseRoute` subclass with no `.path`) into `app.routes` |
| `prometheus-fastapi-instrumentator` | 8.0.0 (latest) | its `_get_route_name` reads `route.path` on every route → `AttributeError` per request |
| `starlette` | 1.3.1 | along for the ride |

vLLM 0.22.1 pins only `fastapi[standard]>=0.115.0` and `prometheus-fastapi-instrumentator>=7.0.0` — **no upper
bound** — so a fresh install pulled FastAPI 0.137.0, which is newer than the instrumentator vLLM ships against. The
instrumentator middleware crashed on every request inside vLLM's own metrics layer; deplodock's plugin was never
implicated.

**Fix.** Cap the `serving` extra at `fastapi[standard]<0.137` (last clean is 0.136.3). Resolves cleanly with vLLM and
deplodock; drop the cap once vLLM ships a `_IncludedRouter`-aware `prometheus-fastapi-instrumentator`.

```toml
serving = ["vllm>=0.22.1,<0.23", "fastapi[standard]<0.137"]   # pyproject.toml
```

Already-broken envs still need a one-time `pip install 'fastapi[standard]<0.137'` (a resolution cap does not downgrade
an existing install).

---

## Part 2 — serving performance A/B (deplodock plugin vs stock vLLM)

### Measured numbers

**Throughput regime — concurrency 32, S=32** (the user's original runs):

| metric | stock vLLM | deplodock | ratio |
|---|---|---|---|
| benchmark duration (256 reqs) | 0.57 s | 23.43 s | 41× |
| request throughput | 447.3 req/s | 10.93 req/s | **41× slower** |
| median E2EL | 32.8 ms | 2908.7 ms | 89× |

**Per-request regime — concurrency 1** (serialization removed; the apples-to-apples latency):

| S (tokens) | stock median E2EL | deplodock median E2EL | gap | stock thru | deplodock thru |
|---|---|---|---|---|---|
| 32  | 4.01 ms  | 94.96 ms   | **24×**  | 194.9 req/s | 10.45 req/s |
| 512 | 10.30 ms | 1057.60 ms | **103×** | 92.5 req/s  | 0.94 req/s  |
| scaling 32→512 | 2.6× | **11.1×** | — | — | — |

### What the data rules in and out

The 41× at concurrency 32 decomposes as **per-request latency (~24×) × the batching stock can do and deplodock
can't (~1.7×)**. deplodock's serving runner processes one sequence at a time (batch axis compile-time fixed at 1,
`serving/ARCHITECTURE.md`), so 32 concurrent requests queue behind each other: 32 × ~91 ms ≈ the observed 2.9 s median.
Stock packs the 32 requests into one ~1024-token prefill.

But serialization is the *smaller* factor. The dominant problem is **per-request latency**, and the concurrency-1
sweep localizes it:

- **Not host overhead.** A fixed ~60 ms mask-upload / cupy↔torch round-trip cannot become 1058 ms. Single-request
  latency growing 11× with sequence length proves the cost is **compute-bound in the kernels**, not I/O.
- **Not serialization.** That is a throughput effect; it does not touch single-request latency, and these runs are at
  concurrency 1.
- **It is kernel performance.** deplodock's compiled kernels for this model's **dynamic-shape** serving path are
  1–2 orders of magnitude slower than stock's, and worsen with S (24× → 103× from S=32 to S=512). Stock's fused
  kernels + warm CUDA graph barely move with sequence length (4 → 10 ms); deplodock explodes (95 → 1058 ms).

### Consistency with known compiler gaps

This matches the documented state of the dynamic-shape path, not a regression:

- **Flash-style symbolic-K attention is future work** (CLAUDE.md, `serving/ARCHITECTURE.md`) — the symbolic-seq
  attention runs as degenerate masked thread tiles, so attention scales poorly.
- The symbolic matmuls / norms are tuned for the symbolic **hint** (`DEFAULT_SEQ_HINT=512`), a *ranking* signal — not
  parity with cuBLAS. At S=32 a hint-sized tile runs mostly masked-off; at S=512 the kernels are sized right yet still
  ~100× off, which is the more damning data point.

The steep-but-sub-S² scaling (11× latency for 16× tokens) means the time is a **mix** of the matmuls and attention,
not attention alone — exact attribution needs a per-kernel profile.

### Bottom line

`deplodock serve` is **correctness- and integration-complete but not performance-competitive today**. The serving
ARCHITECTURE.md frames the deficit as "the gap measures the integration, not kernel quality"; this A/B shows the gap is
much larger than that wording implies **and it lives in the kernels**, not just the batching. There is **no flag or
config** that closes it — the levers are compiler work (flash-style symbolic-K attention) and a deep per-kernel tune of
the dynamic shapes.

### Open follow-up (queued)

Run the `tune-model` flow on Qwen/Qwen3-Embedding-0.6B at the serving shape to turn "it's the kernels" into
"it's *these* kernels, by this many ms" — per-kernel bench vs eager / torch.compile, NCU profile, root-cause writeup.
The whole-model serving path is symbolic `seq_len`; the per-kernel attribution is what tells us whether attention or the
matmuls is the bigger lever (and whether tuning can move them at all, given symbolic-K attention is unimplemented).
