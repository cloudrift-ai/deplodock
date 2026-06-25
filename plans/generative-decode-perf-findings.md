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

## Ruled out / deferred

- **Device zero-copy interleave** — ~1% (host transfers are 1.3%). Not worth doing for latency.
- **Captured-graph replay** — ~0% on the slow kernel (not dispatch-bound).
- **Generic autotuning of the symbolic kernel** — benches at the hint, so it can't pick a decode-friendly M-tile.

## Follow-ups

- **Multi-bucket decode** (e.g., 8 / 32 / 128) to cover higher concurrency with tight padding, vs the single 16-bucket
  here. Memory cost: more program sets (Top risk #9).
- **Per-`Dim` hint plumbing** → one symbolic program with a small M-tile, efficient at all widths (the cleanest
  long-term answer; removes the second program set).
- **Fix the M=1 demoted-matmul cold lowering** (also unblocks the `slice_last_logits` lm_head optimization).
- The memory budget (now up to 4 × n_layers programs) — shared scratch / graph-cache cap.
