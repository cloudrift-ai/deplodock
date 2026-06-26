# Generative decode perf — findings & the decode-bucket fix

> Post-merge investigation of `deplodock serve --generate` / the per-layer `DeplodockGenRunner` decode latency on
> **TinyLlama-1.1B-Chat** (22 layers), RTX 4080, fp16. Measure-first: we profiled before optimizing, which redirected
> the work away from two ~1% "fixes" and onto the real decode lever (the ~63× kernel). Companion to
> [`generative-inference-support.md`](generative-inference-support.md).

## TL;DR

The generative decode bottleneck is **one kernel** — the symbolic-`num_tokens` matmul tiled for the hint (512) is
pathological at decode (M=1): it computes a full 512-row masked tile when only 1 row is real. A **static small-M
(bucket) program** for the per-layer kernels recovers it: at the **shipped bucket (M=16)** the post drops from
**9.43 ms → 0.53 ms (~17.7×, ≈3.6× cuBLAS)**; an M=8 spot-check probes the floor at 0.29 ms (≈2× cuBLAS). The
9.43 ms → 0.53 ms/layer recovery is what turns the ~221 ms decode step into ~22 ms — the realized **~10× / ~46 tok/s**
(matching the merged ~11× / ~50 tok/s). The fix is a decode-specialized program in the runner — not the device
interleave, not captured-graph replay, not generic autotuning. All times below are CUDA-graph-captured pure-GPU
measurements; see **Methodology**.

## Methodology

How every latency here was produced — so the numbers are reproducible, not anecdotal:

- **Hardware / model:** RTX 4080 (16 GB, driver 595.71.05), fp16, TinyLlama-1.1B-Chat layer 0's carved `post` subgraph
  (`build_attention_split_wrapper` → o_proj + residual + post-norm + gated MLP).
- **deplodock rows:** the committed `CompiledProgram` capture path — `capture_program_graph()` (one CUDA graph over
  every launch at the bound shape) then `time_program_window(N)` (one CUDA-event window around N back-to-back replays,
  divided to per-replay ms). This is the same pure-GPU, dispatch-free measurement `run --bench` / `tune --bench` use, so
  the deplodock numbers are deployable, not dispatch-inflated. The `pre` / `post` programs are built by the exact
  `_compile_split` calls `DeplodockGenRunner.from_model` makes (symbolic = `argnames` set; static decode-bucket =
  `argnames=None`).
- **cuBLAS row:** the same `post` module in torch eager on CUDA, timed with `torch.cuda.Event` over the same window
  after warmup. Labeled "cuBLAS" because the post is matmul-dominated (four linears).
- **Knobs:** WARMUP=50 untimed, WINDOW=200 replays/window, WINDOWS=30, report the **median** per-replay time; seed=0;
  single stream, no concurrent load.
- **Reproduce:** `python scripts/bench_gen_post.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0` (the committed
  layer-0 reproducer; `--bucket 8` reproduces the spot-check floor, `--layer N` another layer).
- **Caveat:** point measurements from a focused layer-0 reproducer. Run-to-run drift is a few %; the cross-step ratios
  (≈63× before, ≈3.6× after) are stable and reconcile with the realized ~10×.

## The measurement chain

**1. Baseline — gen_runner decode step on TinyLlama (deplodock compute only, no attention):**

| | |
|---|---|
| build / startup | 116.9 s, **1.95 GB** cupy workspaces (44 programs = 2 × 22 layers) |
| decode step (T=1) | ~221 ms → **~4.5 tok/s** |
| ↳ `post` kernels | **210.9 ms (95%)** |
| ↳ `pre` kernels | 7.1 ms (3%) |
| ↳ host transfers (numpy↔torch) | **2.9 ms (1.3%)** |

→ **The host-sync interleave is NOT the bottleneck** (1.3%). The device zero-copy interleave would buy ~1%.

**2. Drill-down — captured-graph replay vs uncaptured `run_once` (post layer 0, M=1):**

| | kernels | uncaptured | captured replay | speedup |
|---|---|---|---|---|
| pre | 7 | 0.145 ms | 0.141 ms | — |
| **post** | 3 | **9.856 ms** | **9.429 ms** | **×1.0** |

→ **Capture does nothing (×1.0)** — the post is *not* dispatch-bound. So captured-graph replay is not the win either.

**3. Per-kernel pinpoint + headroom (post layer 0, M=1):**

| post kernel | time |
|---|---|
| `k_linear_fc00c2` | 0.064 ms |
| **`k_linear_mean_reduce_*`** (fused linear + post-norm reduction) | **9.290 ms ← 98%** |
| `k_linear_94e345` | 0.142 ms |
| torch eager (cuBLAS), whole post | **0.144 ms** |

→ One fused kernel is **66× slower than cuBLAS** for the whole post. Everything else is already competitive.

**4. Root cause — static vs symbolic at decode:**

| post @ M=1 | total |
|---|---|
| symbolic (hint 512), run at M=1 | 9.29 ms |
| **static M=1** | **fails to compile** — `add` left as `LoopOp` (the M=1 *demoted-matmul* cold-lowering gap, same as the Phase-0 lm_head) |
| **static M=8** | **0.288 ms** (spot-check floor; ×32 vs symbolic, ~2× cuBLAS — the **shipped** bucket is M=16, ~3.6×; see "Measured impact" below) |

The symbolic kernel tiles the `num_tokens` axis as **one masked tile sized to `DEFAULT_SEQ_HINT=512`**. At decode it
still computes the full 512-row tile (~64× wasted work; 9.29 / 0.16 ≈ 512 / 8). The `pre` kernels are *also* symbolic-M
yet fast — so this is not a systemic "symbolic axis at decode" problem; it is the hint-512 M-tile being wrong for
decode. The tuner can't fix it: the symbolic kernel is benched/tuned at the hint (M=512), where a large M-tile is
correct.

## The fix — a decode-bucket program

Compile a **second set of per-layer programs at a small static M bucket** (default 16) and use them when
`num_tokens ≤ bucket`; pad the activations up to the bucket and slice the real rows back. Safe because the carved
`pre`/`post` compute is **per-token-independent** (pointwise + matmul over the hidden axis) — padding rows are computed
then discarded, and the vLLM attention seam still sees exactly `num_tokens` (the padding is internal to the deplodock
kernel calls). `num_tokens > bucket` (prefill, or large concurrent decode) falls back to the symbolic program, where
the large M-tile is correct.

Why a bucket and not static M=1: static M=1 hits the demoted-matmul cold-lowering gap (above), so the bucket must be a
static M that lowers (≥ ~8). 16 covers single-stream and small-batch decode; larger concurrent batches fall back to
symbolic until multi-bucket / per-`Dim` hints land.

**Measured impact (shipped bucket M=16, captured pure-GPU, RTX 4080):**

| post @ decode | ms | vs cuBLAS |
|---|---|---|
| deplodock symbolic (hint 512) @ M=1 (before) | 9.429 | 63× |
| **deplodock static decode-bucket M=16 (shipped)** | **0.532** | **3.6×** |
| torch eager (cuBLAS) @ M=16 | 0.150 | 1× |

So the deployed post is **9.43 → 0.53 ms/layer (~17.7×)**, ~3.6× cuBLAS — not the ~2× the M=8 spot-check (0.288 ms,
above) suggested. The bucket runs all 16 rows even for a single real decode token, and cuBLAS is latency-flat at this M,
so doubling the bucket (8→16) roughly doubles deplodock's time while cuBLAS barely moves — that 8→16 gap is the whole
difference between "~2× cuBLAS" and "~3.6× cuBLAS". End-to-end this lands the decode step at **22 × 0.53 + pre 7.1 +
host 2.9 ≈ 22 ms ⇒ ~46 tok/s** — the realized **~10×** over the ~4.5 tok/s baseline, matching the merged ~11× / ~50
tok/s (the earlier "~100–140 tok/s" projection over-extrapolated from the M=8 floor; the shipped M=16 is the real
number).

## End-to-end served validation (post-merge)

The numbers above are layer-0 micro-benchmarks. To confirm the win holds for the **real served path**, we benched
`deplodock serve --generate` (the vLLM in-process plugin, `DeplodockGenModel` → `DeplodockGenRunner`) against **stock
vLLM** (`vllm serve`, native — CUDA graphs + cuBLAS + paged attention), same model, on the same GPU. Client: a streaming
`/v1/chat/completions` bench — single-stream decode (median of 3, 128 output tokens) and 16-way concurrency (64
requests). Reproduce: start a server (`deplodock serve <model> --generate …` or `… --generate --stock`), then
`python scripts/bench_gen_serve.py --port <P> --model <model>`.

| TinyLlama-1.1B, fp16, RTX 4080 | deplodock plugin | stock vLLM | gap |
|---|---|---|---|
| **single-stream decode** | **46.5 tok/s** | 273.1 tok/s | ~5.9× slower |
| TTFT | 50 ms | 10 ms | ~5× |
| system throughput (16-way) | 446 tok/s | 3862 tok/s | ~8.7× slower |

**Win confirmed:** served single-stream **46.5 tok/s matches the reconciled ~46 tok/s projection** above — the
decode-bucket makes generative serving usable (was ~4.5 tok/s). The layer-0 micro-projection was right end-to-end.

**Next bottleneck = our kernels, not the seam.** deplodock single-stream is 21.5 ms/token, matching the compute budget
(post 11.7 + pre 7.1 + host 2.9 ≈ 21.7 ms) — so the deplodock-owned kernels dominate; vLLM-side overhead and TTFT are
small. Native vLLM runs the *entire* step in 3.66 ms/token, so deplodock's kernels alone are ~5× vLLM's whole optimized
step. The concurrency gap (8.7×) is *worse* than single-stream (5.9×): the bucket pads every step to M=16 and the
per-layer host round-trip serializes, while vLLM batches concurrent decode into one step.

**Memory constraint surfaced.** The decode-bucket roughly doubled the runner's cupy footprint (up to 4 programs/layer),
so the server needed `--max-num-batched-tokens 1024` + a tuned `--gpu-memory-utilization` to fit TinyLlama on 16 GB
(default 0.9 OOMs — vLLM's KV cache had no room after weights + the deplodock workspaces). This is Top-risk #9, now
concrete; it bites harder on bigger models.

**Takeaway:** generative serving is viable but ~6× off native vLLM, dominated by the decode kernels. The most direct
next lever is tuning the **static** decode-bucket kernels (they're tunable, unlike the symbolic one) — see Follow-ups.

### Update — tuning the decode kernels (the #1 follow-up, applied)

Tuned the static M=16 `post` + `pre` subgraphs (`deplodock tune <subgraph>.json --bench`), which **built the learned
prior from scratch** — the first server run above used the cold `AnalyticPrior` (no `prior.json` existed). Re-benching
the served model with the trained prior:

| TinyLlama-1.1B, fp16, RTX 4080 | cold prior | **tuned prior** | stock vLLM |
|---|---|---|---|
| single-stream decode | 46.5 tok/s | **85.5 tok/s** (1.84×) | 273.1 tok/s |
| TTFT | 50 ms | 29 ms | 10 ms |
| system throughput (16-way) | 446 tok/s | **1007 tok/s** (2.26×) | 3862 tok/s |

Kernel config alone **nearly doubled** single-stream decode and **halved the gap to native vLLM** (5.9× → ~3.2×
single-stream; 8.7× → ~3.8× concurrency). The "~6× gap is mostly architectural" read was **wrong** — the cold prior was
leaving ~2× on the table. The remaining ~3× is the harder, partly-architectural part: CUDA-graph capture over the
deplodock step, killing the per-layer host round-trip, and batched concurrent decode.

(Tooling note: the per-kernel `tune --ir <graph>.json --bench` display couldn't locate most reproducers — content-hash
drift after re-lowering — so the per-kernel before/after table was unusable; the prior update itself is unaffected, and
the served re-bench is the proof.)

### Profiling the tuned step — the remaining gap is overhead-bound

Profiled a tuned decode step at the **runner level** (T=1, all 22 layers, vLLM attention excluded), split into pure GPU
kernel time vs host I/O + dispatch (`scripts/profile_gen_decode.py`, each program's GPU time CUDA-graph-captured):

| bucket=16 (tuned) decode step | ms/step | share |
|---|---|---|
| W — wall | 6.79 | 100% (→ 147 tok/s runner-only) |
| G — GPU kernels | 4.06 | 60% |
| W−G — host numpy↔torch + Python/dispatch | 2.72 | 40% |

- **Kernels are no longer the lopsided cost.** Tuning brought pre+post to ~0.18 ms/layer (was ~0.53 ms/layer for post
  alone, cold). Host/dispatch is now **40%** of the runner step.
- **The host I/O forces `--enforce-eager`**, disabling vLLM's whole-step CUDA graph — so the *served* step (11.7 ms) is
  ~5 ms above the runner step (6.79 ms), most of it vLLM's attention + framework running eager. Device-resident
  interleave doesn't just save the 2.72 ms; it **unblocks capturing the whole step**.
- **Bucket-8 is not a free win.** Re-profiled at bucket=8 → 12.5 ms/step (slower!) because the prior was tuned for the
  M=16 shapes; the M=8 kernels fall back to untuned configs (0.41 vs 0.18 ms/layer). A smaller bucket needs its
  own tune, with uncertain payoff now that M=16 is tuned. Dropped.

**Conclusion:** the gap is now **overhead-bound**. Next lever is device-resident interleave + whole-step CUDA-graph
capture (removes the 40% and unblocks capture); kernel tuning / the M=1 lowering fix is secondary.

## Ruled out / deferred

These were ruled out **pre-tune** (at ~46 tok/s, where kernels were 95%+ of the step). **Post-tune the first two are
re-opened** — the profiling above shows host/dispatch is now 40% of the runner step, so they became the dominant lever
(see Follow-ups). Kept here as a record of how the proportions shifted, not as current guidance.

- **Device zero-copy interleave** — *was* ~1% (host transfers 1.3% of the cold step); **now ~40%** of the tuned runner
  step and it unblocks whole-step capture. Re-opened as the #1 lever.
- **Captured-graph replay** — *was* ~0% on the isolated post kernel (not dispatch-bound at M=1); **re-opened** for the
  whole 22-layer step, where per-launch dispatch + Python orchestration is part of that 40%.
- **Generic autotuning of the symbolic kernel** — benches at the hint, so it can't pick a decode-friendly M-tile (still
  true; the static bucket is the answer, and tuning it is what gave the ~2×).

## Follow-ups

- **Tune the static decode-bucket kernels** — ✅ **applied** (see the Update above): tuning the M=16 `post`/`pre`
  subgraphs nearly doubled served single-stream decode (46.5 → 85.5 tok/s) and halved the gap to native vLLM. Remaining:
  the prior was built from layer-0's two subgraphs only — a fuller sweep (more layers/shapes, other models) + a proper
  golden entry would harden it. The leftover ~3× gap is now the architectural levers below.
- **CUDA-graph capture over the deplodock decode step + kill the per-layer host round-trip** — the now-dominant lever
  after tuning. deplodock runs `--enforce-eager` with per-layer numpy↔torch hops; native vLLM captures the whole step.
  Scoped in [`generative-device-resident-decode.md`](generative-device-resident-decode.md). **Phase A ✅ applied** —
  device-resident decode seam (captured-replay, torch↔cupy zero-copy): served decode **85.5 → 178.7 tok/s (~2×)**, gap
  to vLLM ~3.2× → **~1.5×** (still under `--enforce-eager`). Phase B (drop `--enforce-eager` for whole-step capture)
  remains, with smaller headroom than expected since A's captured-replay already removed most per-program dispatch.
- **Multi-bucket decode** (e.g., 8 / 32 / 128) to cover higher concurrency with tight padding, vs the single 16-bucket
  here. Memory cost: more program sets (Top risk #9).
- **Per-`Dim` hint plumbing** → one symbolic program with a small M-tile, efficient at all widths (the cleanest
  long-term answer; removes the second program set).
- **Fix the M=1 demoted-matmul cold lowering** (also unblocks the `slice_last_logits` lm_head optimization).
- The memory budget (now up to 4 × n_layers programs) — shared scratch / graph-cache cap.
